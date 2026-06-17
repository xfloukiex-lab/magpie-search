"""cli — `magpie_search <subcommand>` (or `python -m magpie_search <subcommand>`).

Subcommands:
    index   [--source DIR]                — incremental indexing pass
    search  QUERY [--k N] [--project P]   — top-k FTS5 hits with snippets
    recent  [--n N] [--session SID] [--project P]
    session SID [--limit N] [--offset N]
    list    [--project P] [--limit N]     — most-recent sessions
    stats                                 — index size + counts
    backup  [--dry-run] [--show-config]   — back up transcripts (see magpie_search.backup)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import indexer
from .search import search as _search, recent as _recent, session as _session, list_sessions as _list_sessions


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, default=str, ensure_ascii=False))


def _cmd_index(args: argparse.Namespace) -> int:
    src = Path(args.source) if args.source else None
    print(f"Indexing from {src or indexer.claude_projects_dir()} ...")
    stats = indexer.index_all(source_dir=src, progress_every=args.progress_every)
    print(f"Done in {stats.elapsed_s:.1f}s")
    print(f"  files seen:       {stats.files_seen}")
    print(f"  files updated:    {stats.files_updated}")
    print(f"  lines read:       {stats.lines_read}")
    print(f"  messages indexed: {stats.messages_indexed}")
    print(f"  bytes read:       {stats.bytes_read / 1024 / 1024:.1f} MB")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    mode = getattr(args, "mode", "lexical")
    sources = getattr(args, "sources", None)
    if sources:
        # Multi-source ("federated") path — fan out across the named providers.
        src_list = [s.strip() for s in sources.split(",") if s.strip()]
        res = _search(
            args.query, k=args.k, sources=src_list,
            budget_tokens=getattr(args, "budget", None),
            min_trust=getattr(args, "min_trust", None),
            scope=getattr(args, "scope", None) or args.project,
        )
        if args.json:
            _print_json(res)
            return 0 if res.get("ok") else 1
        if not res.get("ok"):
            print(f"error: {res.get('reason')}", file=sys.stderr)
            return 1
        return _print_federated(res, args.query)

    if mode == "rerank":
        # Cross-encoder rerank lives in the optional llm subpackage; import
        # lazily so the base search path never pays its load cost.
        from .llm.reranker import search_rerank
        res = search_rerank(query=args.query, k=args.k,
                            project=args.project, author=args.role)
    else:
        res = _search(
            args.query, k=args.k, project=args.project, role=args.role, mode=mode,
            regex=getattr(args, "regex", False),
        )
    if args.json:
        _print_json(res)
        return 0 if res.get("ok") else 1
    if not res.get("ok"):
        print(f"error: {res.get('reason')}", file=sys.stderr)
        return 1
    hits = res["hits"]
    if not hits:
        print(f"(no matches for {args.query!r})")
        return 0
    print(f"{len(hits)} hits for {args.query!r}:\n")
    for h in hits:
        ts = (h.get("ts") or "")[:19]
        print(f"  [{ts}] {h['project']} / {h['session_id'][:8]}  ({h['role']}/{h['msg_type']})")
        print(f"    {h['snippet']}")
        print()
    return 0


def _cmd_deepweb(args: argparse.Namespace) -> int:
    """Agent-free deep web search: fan several sub-queries at the web provider,
    merge+dedup, return one compact source set for the caller to synthesize."""
    from .deepweb import deep_web_search, expand_queries, corroboration
    queries = list(args.q or [])
    if args.query:
        queries.insert(0, args.query)
    if not queries:
        print("deepweb: give a query and/or --q sub-queries", file=sys.stderr)
        return 2

    # --thorough: rival the old multi-agent skill's coverage at ~1000s of tokens.
    k, kpq, fetch = args.k, args.k_per_query, args.fetch
    fetch_k, ppc = args.fetch_k, args.per_page_chars
    if args.thorough:
        if len(queries) == 1:
            queries = expand_queries(queries[0])   # auto-fan a single question
        k, kpq, fetch, fetch_k, ppc = 12, 8, True, 8, 2000

    hits = deep_web_search(queries, k_per_query=kpq, total_k=k,
                           fetch=fetch, fetch_k=fetch_k, per_page_chars=ppc)
    corr = corroboration(hits)
    if args.json:
        _print_json({"ok": True, "queries": queries, "corroboration": corr,
                     "hits": [h.to_dict() for h in hits]})
        return 0
    print(f"deepweb: {len(queries)} sub-queries -> {len(hits)} merged sources"
          f"{' (+page text)' if fetch else ''}  | "
          f"{corr['distinct_domains']} domains, "
          f"{corr['multi_query_corroborated']} multi-query corroborated\n")
    for h in hits:
        print(f"  [x{len(h.seen_in)}] {h.title}".rstrip())
        print(f"    {(h.content or h.snippet)[:280]}")
        print(f"    {h.url}\n")
    return 0


def _print_federated(res: dict, query: str) -> int:
    hits = res.get("hits", [])
    summary = ", ".join(f"{s}:{n}" for s, n in (res.get("sources") or {}).items())
    print(f"{len(hits)} hits for {query!r}  "
          f"[sources {summary}; ~{res.get('used_tokens')} tok]\n")
    if not hits:
        print("(no matches)")
        return 0
    for h in hits:
        prov = h.get("provenance") or {}
        loc = prov.get("path") or prov.get("session_id") or ""
        if prov.get("heading"):
            loc = f"{loc} :: {prov['heading']}"
        also = f"  (also in: {', '.join(h['also_in'])})" if h.get("also_in") else ""
        print(f"  [{h['trust'].upper()}] {h['source']}  {loc}{also}")
        print(f"    {h.get('text','')}")
        print()
    dropped = res.get("dropped") or {}
    if any(dropped.values()):
        print(f"(dropped — budget:{dropped.get('budget',0)} "
              f"dedup:{dropped.get('dedup',0)} min_trust:{dropped.get('min_trust',0)})")
    return 0


def _cmd_recent(args: argparse.Namespace) -> int:
    res = _recent(n=args.n, session_id=args.session, project=args.project)
    if args.json:
        _print_json(res)
        return 0 if res.get("ok") else 1
    if not res.get("ok"):
        print(f"error: {res.get('reason')}", file=sys.stderr)
        return 1
    print(f"session {res['session_id']} — {res['count']} messages")
    for m in res["messages"]:
        ts = (m.get("ts") or "")[:19]
        text = (m.get("text") or "").replace("\n", " ")[:180]
        print(f"  [{ts}] {m['role']}/{m['msg_type']}: {text}")
    return 0


def _cmd_session(args: argparse.Namespace) -> int:
    res = _session(args.session_id, limit=args.limit, offset=args.offset)
    if args.json:
        _print_json(res)
        return 0 if res.get("ok") else 1
    if not res.get("ok"):
        print(f"error: {res.get('reason')}", file=sys.stderr)
        return 1
    sess = res["session"]
    print(f"session {sess['session_id']} — project={sess.get('project')} "
          f"msgs={sess.get('message_count')}")
    print(f"first_ts={sess.get('first_ts')} last_ts={sess.get('last_ts')}\n")
    for m in res["messages"]:
        ts = (m.get("ts") or "")[:19]
        text = (m.get("text") or "").replace("\n", " ")[:200]
        print(f"  [{ts}] {m['role']}/{m['msg_type']}: {text}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    res = _list_sessions(project=args.project, limit=args.limit)
    if args.json:
        _print_json(res)
        return 0 if res.get("ok") else 1
    if not res.get("ok"):
        print(f"error: {res.get('reason')}", file=sys.stderr)
        return 1
    for s in res["sessions"]:
        last = (s.get("last_ts") or "")[:19]
        print(f"  {last}  {s['project']:<32}  {s['session_id']}  ({s['message_count']} msgs)")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    _print_json(indexer.stats_summary())
    return 0


def _cmd_backup(args: argparse.Namespace) -> int:
    # Delegate to magpie_search.backup — keeps all the backup logic, config loading,
    # and CLI flag handling in one place rather than mirroring it here.
    from . import backup as _bk
    forwarded: list[str] = []
    if args.dry_run:    forwarded.append("--dry-run")
    if args.no_suspend: forwarded.append("--no-suspend")
    if args.show_config: forwarded.append("--show-config")
    return _bk.main(forwarded)


def _cmd_telemetry(args: argparse.Namespace) -> int:
    from . import telemetry
    if args.action == "enable":
        telemetry.enable()
        print("Telemetry ENABLED. Anonymous usage only — never your queries, "
              "file paths, or transcript content.\nDisable anytime: "
              "magpie-search telemetry disable")
    elif args.action == "disable":
        telemetry.disable()
        print("Telemetry disabled.")
    else:
        import json as _json
        print(_json.dumps(telemetry.status(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="magpie-search",
        description="Index, search, and browse Claude Code session transcripts.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("index", help="run an incremental indexing pass")
    sp.add_argument("--source", help="override Claude projects dir")
    sp.add_argument("--progress-every", type=int, default=50)
    sp.set_defaults(func=_cmd_index)

    sp = sub.add_parser("search", help="search the index (or multiple sources)")
    sp.add_argument("query")
    sp.add_argument("--k", type=int, default=10)
    sp.add_argument("--mode",
                    choices=["lexical", "semantic", "hybrid", "rerank", "grep"],
                    default="lexical",
                    help="lexical=keyword (default, fastest); semantic=meaning; "
                         "hybrid=both fused; rerank=hybrid + cross-encoder; "
                         "grep=exact literal/regex match (code, paths, error "
                         "strings — what keyword/semantic miss) "
                         "(semantic/hybrid/rerank need the embedding model)")
    sp.add_argument("--regex", action="store_true",
                    help="with --mode grep: treat the query as a regex "
                         "(default: literal string match)")
    sp.add_argument("--project")
    sp.add_argument("--role", choices=["user", "assistant", "system"])
    sp.add_argument("--sources",
                    help="comma-separated sources for multi-source search, e.g. "
                         "'transcripts,files' (plus any installed provider plugins). "
                         "When set, fans out across providers and returns "
                         "trust-tagged, budgeted results.")
    sp.add_argument("--budget", type=int, default=None,
                    help="token budget for the merged result (multi-source only)")
    sp.add_argument("--min-trust", dest="min_trust",
                    choices=["fact", "reference", "lead", "stale"], default=None,
                    help="drop hits below this trust tier (multi-source only)")
    sp.add_argument("--scope", default=None,
                    help="narrow sources, e.g. a project slug or a subpath like "
                         "'projects/<slug>' (multi-source only)")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=_cmd_search)

    sp = sub.add_parser("deepweb",
                        help="agent-free deep web search: multi-query fan-out, merged+deduped")
    sp.add_argument("query", nargs="?", help="main query")
    sp.add_argument("--q", action="append", help="extra sub-query (repeatable)")
    sp.add_argument("--k", type=int, default=8, help="total merged results (default 8)")
    sp.add_argument("--k-per-query", type=int, default=6, dest="k_per_query",
                    help="results pulled per sub-query before merge (default 6)")
    sp.add_argument("--fetch", action="store_true",
                    help="also fetch+extract the top URLs' page text (depth)")
    sp.add_argument("--fetch-k", type=int, default=4, dest="fetch_k",
                    help="how many top URLs to fetch when --fetch (default 4)")
    sp.add_argument("--per-page-chars", type=int, default=1500, dest="per_page_chars",
                    help="max chars of extracted text per page (default 1500)")
    sp.add_argument("--thorough", action="store_true",
                    help="heavy preset: expand the query, read more pages, add "
                         "corroboration signal (rivals the old deep-research skill)")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=_cmd_deepweb)

    sp = sub.add_parser("recent", help="recent messages from latest (or named) session")
    sp.add_argument("--n", type=int, default=50)
    sp.add_argument("--session")
    sp.add_argument("--project")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=_cmd_recent)

    sp = sub.add_parser("session", help="paginated view of one session")
    sp.add_argument("session_id")
    sp.add_argument("--limit", type=int, default=200)
    sp.add_argument("--offset", type=int, default=0)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=_cmd_session)

    sp = sub.add_parser("list", help="list recent sessions")
    sp.add_argument("--project")
    sp.add_argument("--limit", type=int, default=50)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=_cmd_list)

    sp = sub.add_parser("stats", help="index size + counts")
    sp.set_defaults(func=_cmd_stats)

    sp = sub.add_parser("telemetry",
                        help="opt in/out of anonymous usage telemetry (off by default)")
    sp.add_argument("action", choices=["enable", "disable", "status"])
    sp.set_defaults(func=_cmd_telemetry)

    sp = sub.add_parser("backup", help="back up transcripts to a configurable destination")
    sp.add_argument("--dry-run", action="store_true",
                    help="print every command without executing")
    sp.add_argument("--no-suspend", action="store_true",
                    help="leave VM running afterward (if VM mode)")
    sp.add_argument("--show-config", action="store_true",
                    help="print resolved backup config and exit")
    sp.set_defaults(func=_cmd_backup)

    return p


def _make_stdio_unicode_safe() -> None:
    """Windows consoles default to cp1252 and CRASH printing non-cp1252 chars
    (e.g. '→', smart quotes) that show up constantly in web/transcript
    snippets. Reconfigure stdout/stderr to UTF-8 with replacement so a result
    never dies in the print path. No-op where already UTF-8."""
    import sys
    for stream in (sys.stdout, sys.stderr):
        reconfig = getattr(stream, "reconfigure", None)
        if reconfig is not None:
            try:
                reconfig(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main(argv: list[str] | None = None) -> int:
    _make_stdio_unicode_safe()
    args = build_parser().parse_args(argv)
    if getattr(args, "cmd", None) == "telemetry":
        return args.func(args)
    import time
    from . import telemetry
    telemetry.maybe_first_run_notice()
    t0 = time.monotonic()
    rc, err = 1, None
    try:
        rc = args.func(args)
        return rc
    except Exception as exc:
        err = type(exc).__name__
        raise
    finally:
        _t = telemetry.emit("command", command=getattr(args, "cmd", None),
                            mode=getattr(args, "mode", None), rc=rc,
                            ms=int((time.monotonic() - t0) * 1000), error=err)
        if _t is not None:
            _t.join(2.0)  # give the POST a chance before exit; capped, fail-open


if __name__ == "__main__":
    raise SystemExit(main())
