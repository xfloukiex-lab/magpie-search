"""bakeoff — run each LLM augmentation worker on real history for hand-eval.

Usage:
    python -m magpi.llm.bakeoff               # all three sub-bakeoffs
    python -m magpi.llm.bakeoff rerank        # just reranker
    python -m magpi.llm.bakeoff summarize     # just summarizer (needs Ollama+phi3.5)
    python -m magpi.llm.bakeoff trust         # just trust monitor

Prints, for hand-eval, comparing baseline vs LLM-augmented output.
Nothing is written to disk except the normal audit log + alerts.
"""
from __future__ import annotations

import json
import sys
from typing import Any

from .. import search as _search
from .reranker import search_rerank
from .summarizer import summarize
from .trust import check as trust_check


BAKEOFF_QUERIES = [
    "diagram of system architecture",
    "project config files exclude pyproject",
    "windows update forced reboot",
    "transcript indexer backfill embeddings",
    "scheduled task autostart cuckoo",
]


def _print_hit(h: dict[str, Any], n: int) -> None:
    ts = (h.get("ts") or "")[:19]
    sid = (h.get("session_id") or "")[:8]
    role = h.get("role", "")
    mt = h.get("msg_type", "")
    score = h.get("rerank_score") or h.get("rrf_score") or h.get("distance") or h.get("rank")
    snip = (h.get("snippet") or "").replace("\n", " ")[:150]
    print(f"  {n}. [{ts}] {sid}/{role}/{mt}  score={score}")
    print(f"     {snip}")


def bakeoff_reranker(queries: list[str] = BAKEOFF_QUERIES) -> None:
    print("=" * 78)
    print("BAKE-OFF: RERANKER - baseline (hybrid) vs reranked (cross-encoder)")
    print("=" * 78)
    for q in queries:
        print(f"\nQUERY: {q!r}")
        base = _search.search(q, k=3, mode="hybrid")
        print("\n  -- baseline (hybrid top-3) --")
        for i, h in enumerate(base.get("hits", [])[:3], 1):
            _print_hit(h, i)
        rerank = search_rerank(query=q, k=3, pool=10)
        print("\n  -- reranked top-3 --")
        if not rerank.get("reranked"):
            print(f"    (skipped - {rerank.get('reason')})")
            continue
        for i, h in enumerate(rerank.get("hits", [])[:3], 1):
            _print_hit(h, i)


def bakeoff_summarizer(n_sessions: int = 3) -> None:
    print("\n" + "=" * 78)
    print(f"BAKE-OFF: SUMMARIZER - {n_sessions} recent sessions")
    print("=" * 78)
    lst = _search.list_sessions(limit=n_sessions)
    for s in lst.get("sessions", []):
        sid = s.get("session_id")
        mc = s.get("message_count")
        print(f"\nSESSION {sid[:8]} ({mc} msgs):")
        r = summarize(session_id=sid, n_messages=40)
        trust = r.get("trust")
        print(f"  trust: {trust}")
        print(f"  probes: {r.get('probes')}")
        if r.get("reason"):
            print(f"  reason: {r['reason']}")
        if r.get("summary"):
            print(f"  summary: {r['summary']}")
        else:
            print(f"  summary: <suppressed - {trust}>")


def bakeoff_trust_monitor() -> None:
    print("\n" + "=" * 78)
    print("TRUST MONITOR snapshot")
    print("=" * 78)
    r = trust_check()
    print(json.dumps(r, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if "rerank" in argv or not argv:
        bakeoff_reranker()
    if "summarize" in argv or not argv:
        bakeoff_summarizer()
    if "trust" in argv or not argv:
        bakeoff_trust_monitor()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
