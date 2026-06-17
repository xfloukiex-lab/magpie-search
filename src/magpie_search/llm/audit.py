"""audit — append-only JSONL log of every LLM call.

Lives at $MAGPIE_SEARCH_HOME/llm-audit.jsonl (default ~/.magpie-search/llm-audit.jsonl).
Every entry has at minimum:
    ts, role (which worker called us), model, prompt_hash, response_text,
    trust (clean | degraded | untrusted), fallback_fired, probe_results, ms.

The trust monitor (`magpie_search.llm.trust`, or `llm.trust_check` swarm role
via aviary-magpi) tails this file hourly to compute bypass rate,
fallback rate, drift, and surface alerts.

Never deletes entries — old ones are evidence. Rotation, if ever needed,
is the trust monitor's call.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()

# Audit log auto-rotates when it exceeds this size. Old file kept as .jsonl.1
# (single rotation — older histories overwrite). Configurable via env.
_MAX_BYTES = int(
    os.environ.get("MAGPIE_SEARCH_AUDIT_MAX_BYTES")
    or os.environ.get("AVIARY_LLM_AUDIT_MAX_BYTES")
    or str(50 * 1024 * 1024)
)


def _path() -> Path:
    # Resolution: MAGPIE_SEARCH_AUDIT_LOG > AVIARY_LLM_AUDIT_LOG (legacy) > $MAGPIE_SEARCH_HOME/llm-audit.jsonl
    # Empty-string env vars are treated as unset — otherwise `Path("")` resolves
    # to CWD and we'd scatter audit logs across whatever directory the caller
    # was launched from.
    override = os.environ.get("MAGPIE_SEARCH_AUDIT_LOG") or os.environ.get("AVIARY_LLM_AUDIT_LOG")
    if override and override.strip():
        return Path(override)
    base = os.environ.get("MAGPIE_SEARCH_HOME") or os.environ.get("AVIARY_TRANSCRIPTS_DIR")
    if base and base.strip():
        return Path(base) / "llm-audit.jsonl"
    return Path.home() / ".magpie-search" / "llm-audit.jsonl"


def _maybe_rotate(p: Path) -> None:
    """If audit log > _MAX_BYTES, rotate to .jsonl.1 (overwriting previous)."""
    try:
        if not p.exists() or p.stat().st_size <= _MAX_BYTES:
            return
        rotated = p.with_suffix(".jsonl.1")
        if rotated.exists():
            rotated.unlink()
        p.rename(rotated)
    except Exception:
        pass  # Never let rotation poison the caller.


def _hash_prompt(prompt: str) -> str:
    """Stable short ID for grouping. Not security — just dedup/grouping."""
    return hashlib.sha256(prompt.encode("utf-8", errors="replace")).hexdigest()[:12]


def log(event: dict[str, Any]) -> None:
    """Append one audit event. Always best-effort — never raises.

    Caller is responsible for filling: role, model, prompt, response,
    trust, fallback_fired, probe_results (dict), ms.
    """
    record = dict(event)
    record.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    # Redact prompt/response BEFORE they are written. The index-time
    # redactor is best-effort; anything it missed must not be re-persisted
    # verbatim into this long-lived, never-deleted telemetry file. This is
    # the single chokepoint for the audit surface.
    try:
        from ..redactor import redact as _redact
        if isinstance(record.get("prompt"), str):
            record["prompt"] = _redact(record["prompt"])
        if isinstance(record.get("response"), str):
            record["response"] = _redact(record["response"])
    except Exception:
        pass  # never let redaction failure poison telemetry
    if "prompt" in record and "prompt_hash" not in record:
        record["prompt_hash"] = _hash_prompt(record["prompt"])
    # Truncate prompt + response in-line so the log doesn't balloon. Full
    # text recoverable from caller logs if needed; this file is for telemetry.
    if isinstance(record.get("prompt"), str) and len(record["prompt"]) > 400:
        record["prompt"] = record["prompt"][:400] + f"...[trunc {len(record['prompt'])}]"
    if isinstance(record.get("response"), str) and len(record["response"]) > 800:
        record["response"] = record["response"][:800] + f"...[trunc {len(record['response'])}]"

    p = _path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        # In-process lock (avoids reordering between threads in this
        # process) PLUS a cross-process file lock (avoids interleaved
        # writes between cuckoo daemon, the interactive agent, magpie_search CLI,
        # and aviary-magpi plugin). The cross-process lock also
        # serializes against the rotation rename.
        with _LOCK:
            try:
                from aviary.core.safe_io import file_lock as _file_lock
            except Exception:
                _file_lock = None  # aviary not installed — fall back to in-process lock only
            if _file_lock is not None:
                with _file_lock(p, timeout=5.0):
                    _maybe_rotate(p)
                    with p.open("a", encoding="utf-8") as f:
                        f.write(line)
            else:
                _maybe_rotate(p)
                with p.open("a", encoding="utf-8") as f:
                    f.write(line)
    except (OSError, TimeoutError, ValueError, json.JSONDecodeError):
        # Expected failure modes: disk full, permissions, lock-acquire
        # timeout, malformed record. Silently dropped per the
        # never-poison-caller contract — telemetry is best-effort.
        pass
    except Exception as _e:  # noqa: BLE001
        # Unexpected — almost certainly a developer-introduced bug
        # (non-serializable field, broken patch). Still don't propagate
        # (the contract is "never poison the caller"), but surface to
        # stderr under MAGPIE_SEARCH_DEBUG so dev runs can see it. In
        # production this stays quiet.
        if os.environ.get("MAGPIE_SEARCH_DEBUG"):
            import sys as _sys
            import traceback as _tb
            _sys.stderr.write(
                f"audit.log: unexpected error: {type(_e).__name__}: {_e}\n"
            )
            _tb.print_exc(file=_sys.stderr)


def tail(n: int = 100) -> list[dict[str, Any]]:
    """Read last n entries. Trust monitor uses this."""
    p = _path()
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        out = []
        for line in lines[-n:]:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out
    except Exception:
        return []
