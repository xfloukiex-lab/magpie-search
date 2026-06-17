"""magpie_search MCP server — JSON-RPC 2.0 over stdio.

Exposes magpie_search's transcript SEARCH surface as MCP tools so an agent can
discover and call them like any other tool. Registered in the MCP client
under server name `magpie_search`, so tools surface as `mcp__magpie_search__<verb>`.

magpie_search is a SEARCH tool, not a memory or fact store. It returns what
was *said* in past transcripts — a record to consult and verify, not a
source of ground truth. Treat results as retrieved context, not facts.

Methods served:
  initialize   handshake; server info + protocol version
  tools/list   the search/browse tool catalog (with input schemas)
  tools/call   { name, arguments } -> { content: [...], isError }
  ping         liveness
  shutdown     close the stdin loop

Stdlib only. One JSON object per line on stdin -> one per line on
stdout. stderr carries human-readable trace (ignored by clients).
"""
from __future__ import annotations

import json
import os
import sys
import threading
from typing import Any, Callable

import magpie_search


PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "magpie_search"
SERVER_VERSION = magpie_search.__version__

MAX_LINE = 1 << 20


# ---- tool catalog --------------------------------------------------------
# Search/browse only. The heavy LLM (summarize/rerank/trust) and backup
# surfaces are intentionally NOT exposed as MCP tools — magpie_search-as-a-tool is
# transcript search, per the product framing.

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "search",
        "description": "Search indexed Claude Code transcripts. Returns top-k "
                       "matching message snippets. Results are leads to verify, "
                       "never authoritative fact. Pass 'sources' to fan out across "
                       "multiple backends (e.g. transcripts + files + plugins): "
                       "results are then tagged with source + trust tier and trimmed "
                       "to a token budget.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "search text"},
                "k": {"type": "integer", "description": "max results (default 10)"},
                "mode": {"type": "string", "enum": ["lexical", "semantic", "hybrid"],
                         "description": "ranking strategy (default lexical); "
                                        "single-source only"},
                "project": {"type": "string", "description": "filter to a project slug"},
                "role": {"type": "string", "description": "filter by role (user/assistant)"},
                "dedup": {"type": "boolean", "description": "collapse duplicate clusters"},
                "sources": {"type": "array", "items": {"type": "string"},
                            "description": "multi-source: provider names to fan out "
                                           "across, e.g. ['transcripts','files']"},
                "budget_tokens": {"type": "integer",
                                  "description": "token budget for merged result "
                                                 "(multi-source only)"},
                "min_trust": {"type": "string",
                              "enum": ["fact", "reference", "lead", "stale"],
                              "description": "drop hits below this trust tier "
                                             "(multi-source only)"},
                "scope": {"type": "string",
                          "description": "narrow sources, e.g. a project slug or a "
                                         "subpath (multi-source only)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "recent",
        "description": "Most-recent N messages, optionally scoped to one session "
                       "or project.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "message count (default 50)"},
                "session_id": {"type": "string"},
                "project": {"type": "string"},
            },
        },
    },
    {
        "name": "session",
        "description": "Paginated chronological view of one session's messages.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "limit": {"type": "integer", "description": "default 200"},
                "offset": {"type": "integer", "description": "default 0"},
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "list_sessions",
        "description": "List most-recent sessions for browsing.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {"type": "string"},
                "limit": {"type": "integer", "description": "default 50"},
            },
        },
    },
    {
        "name": "stats",
        "description": "Index health summary (message/session counts, coverage).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "reindex",
        "description": "Run one incremental indexing pass so search is fresh. "
                       "Local-only; reads ~/.claude/projects transcripts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "override projects dir"},
            },
        },
    },
]


# ---- handlers ------------------------------------------------------------
# Explicit per-tool arg mapping (magpie_search functions use keyword-only args;
# blind **kwargs would break on unexpected keys).

# Hard ceiling on result-size params. A single JSON-RPC call must not be
# able to force a multi-hundred-MB result serialization (output DoS against
# the local agent); search() already multiplies k internally for dedup.
_MAX_LIMIT = 1000


class _ParamError(ValueError):
    """A JSON-RPC param failed validation (returned as a tool error, not raised)."""


def _int_param(a: dict[str, Any], name: str, default: int, *, lo: int = 0,
               hi: int = _MAX_LIMIT) -> int:
    """Parse + clamp an integer param. Rejects non-numeric with a clear
    message instead of letting int() raise an opaque ValueError."""
    raw = a.get(name, default)
    try:
        v = int(raw)
    except (TypeError, ValueError):
        raise _ParamError(f"param {name!r} must be an integer, got {raw!r}")
    return max(lo, min(hi, v))


def _h_search(a: dict[str, Any]) -> Any:
    q = a.get("query")
    if not q:
        return {"ok": False, "error": "param 'query' required"}
    sources = a.get("sources")
    if sources is not None and not isinstance(sources, list):
        return {"ok": False, "error": "param 'sources' must be an array of strings"}
    budget = a.get("budget_tokens")
    if budget is not None:
        budget = _int_param(a, "budget_tokens", 2000, lo=1, hi=1_000_000)
    return magpie_search.search(
        q,
        k=_int_param(a, "k", 10, lo=1),
        project=a.get("project"),
        role=a.get("role"),
        mode=a.get("mode", "lexical"),
        dedup=a.get("dedup"),
        sources=sources,
        budget_tokens=budget,
        min_trust=a.get("min_trust"),
        scope=a.get("scope"),
    )


def _h_recent(a: dict[str, Any]) -> Any:
    return magpie_search.recent(
        n=_int_param(a, "n", 50, lo=1),
        session_id=a.get("session_id"),
        project=a.get("project"),
    )


def _h_session(a: dict[str, Any]) -> Any:
    sid = a.get("session_id")
    if not sid:
        return {"ok": False, "error": "param 'session_id' required"}
    return magpie_search.session(sid, limit=_int_param(a, "limit", 200, lo=1),
                         offset=_int_param(a, "offset", 0, lo=0, hi=10_000_000))


def _h_list_sessions(a: dict[str, Any]) -> Any:
    return magpie_search.list_sessions(project=a.get("project"),
                               limit=_int_param(a, "limit", 50, lo=1))


def _h_stats(_a: dict[str, Any]) -> Any:
    return magpie_search.stats()


def _h_reindex(a: dict[str, Any]) -> Any:
    res = magpie_search.index(source=a.get("source"))
    # index_all returns an IndexStats dataclass — coerce to a dict.
    if hasattr(res, "__dict__"):
        return dict(res.__dict__)
    return res


_HANDLERS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "search": _h_search,
    "recent": _h_recent,
    "session": _h_session,
    "list_sessions": _h_list_sessions,
    "stats": _h_stats,
    "reindex": _h_reindex,
}


class MCPServer:
    """Single-process JSON-RPC server. One instance per stdio session."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._methods: dict[str, Callable[[dict[str, Any]], Any]] = {
            "initialize": self._initialize,
            "tools/list": self._tools_list,
            "tools/call": self._tools_call,
            "ping":       self._ping,
            "shutdown":   self._shutdown,
        }

    def _send(self, payload: dict[str, Any]) -> None:
        sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
        sys.stdout.flush()

    def _reply_ok(self, req_id: Any, result: Any) -> None:
        self._send({"jsonrpc": "2.0", "id": req_id, "result": result})

    def _reply_err(self, req_id: Any, code: int, message: str) -> None:
        self._send({"jsonrpc": "2.0", "id": req_id,
                    "error": {"code": code, "message": message}})

    def _initialize(self, _p: dict[str, Any]) -> dict[str, Any]:
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "serverInfo": {"name": f"{SERVER_NAME}-mcp", "version": SERVER_VERSION},
            "capabilities": {"tools": {"listChanged": False}},
        }

    def _tools_list(self, _p: dict[str, Any]) -> dict[str, Any]:
        return {"tools": _TOOLS}

    def _tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name") or ""
        args = params.get("arguments") or {}
        handler = _HANDLERS.get(name)
        if handler is None:
            return {"isError": True,
                    "content": [{"type": "text", "text": f"unknown tool: {name}"}]}
        try:
            result = handler(args)
        except _ParamError as e:
            # Validation errors are safe to echo — they describe the bad param.
            return {"isError": True, "content": [{"type": "text", "text": str(e)}]}
        except Exception as e:  # noqa: BLE001
            # Don't leak internal exception text (paths, SQL fragments) to the
            # client. Full detail only to stderr under MAGPIE_SEARCH_DEBUG.
            if os.environ.get("MAGPIE_SEARCH_DEBUG"):
                sys.stderr.write(f"tools/call {name} error: {type(e).__name__}: {e}\n")
            return {"isError": True, "content": [{"type": "text",
                    "text": "internal error handling tool call"}]}
        return {"isError": False, "content": [{"type": "text",
                "text": json.dumps(result, default=str)}]}

    def _ping(self, _p: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True}

    def _shutdown(self, _p: dict[str, Any]) -> dict[str, Any]:
        self._stop.set()
        return {"ok": True}

    def serve_stdio(self) -> int:
        sys.stderr.write(f"{SERVER_NAME}-mcp {SERVER_VERSION} ready on stdio\n")
        sys.stderr.flush()
        while not self._stop.is_set():
            line = sys.stdin.readline(MAX_LINE + 1)
            if not line:
                break  # EOF
            if len(line) > MAX_LINE:
                self._reply_err(None, -32600, "Request too large")
                while line and not line.endswith("\n"):
                    if self._stop.is_set():
                        return 0
                    line = sys.stdin.readline(MAX_LINE + 1)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                self._reply_err(None, -32700, "Parse error")
                continue
            req_id = msg.get("id")
            method = msg.get("method") or ""
            params = msg.get("params") or {}
            fn = self._methods.get(method)
            if fn is None:
                self._reply_err(req_id, -32601, f"Method not found: {method}")
                continue
            try:
                self._reply_ok(req_id, fn(params))
            except Exception as e:  # noqa: BLE001
                self._reply_err(req_id, -32603, f"Internal error: {type(e).__name__}: {e}")
        return 0


def main() -> int:
    return MCPServer().serve_stdio()


if __name__ == "__main__":
    sys.exit(main())
