"""reranker — cross-encoder rerank over hybrid search results.

Public:
    search_rerank(query, k=5, pool=10, project=None, author=None) -> dict

Pipeline:
  1. Run hybrid search → `pool` candidates (default 10).
  2. Cross-encoder (`jinaai/jina-reranker-v1-turbo-en`) scores each
     (query, full_message_text) pair.
  3. Reorder candidates by cross-encoder score, return top-k.

Guardrails:
  - Output rowids MUST be a subset of input rowids (probe).
  - If reranker fails or returns non-subset → fall back to hybrid order.

Audit: every call logged to $MAGPIE_SEARCH_HOME/llm-audit.jsonl with score deltas.
No hallucination risk — cross-encoder output is a float similarity score.
"""
from __future__ import annotations

import time
from typing import Any

from .. import indexer as _idx
from ..search import search as _hybrid_search
from . import audit, guardrails


import threading as _threading

_RERANKER = None
_RERANKER_MODEL = "jinaai/jina-reranker-v1-turbo-en"
_RERANKER_LOCK = _threading.Lock()


def _get_reranker():
    """Lazy-load the cross-encoder. Cached at module level after first use.

    Double-checked locking matches `guardrails._get_embedder`: fast path
    when loaded, lock-acquire only on cold-start race. Without it, two
    threads first-calling `search_rerank` could each load the model
    (heavy — model + tokenizer)."""
    global _RERANKER
    if _RERANKER is not None:
        return _RERANKER
    with _RERANKER_LOCK:
        if _RERANKER is not None:
            return _RERANKER
        import os
        from pathlib import Path
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        cache_dir = (
            os.environ.get("MAGPIE_SEARCH_MODELS_DIR")
            or str(Path.home() / ".magpie-search" / "models")
        )
        _RERANKER = TextCrossEncoder(model_name=_RERANKER_MODEL, cache_dir=cache_dir)
    return _RERANKER


def search_rerank(
    *,
    query: str,
    k: int = 5,
    pool: int = 10,
    project: str | None = None,
    author: str | None = None,
) -> dict[str, Any]:
    """Hybrid search then cross-encoder rerank.

    Returns hybrid-shaped output with extra fields:
        reranked: True/False
        pool_size: how many candidates we ran past the cross-encoder
        rerank_ms: how long the rerank pass took
        hits[i].rerank_score: cross-encoder relevance score
    """
    enable_role = "magpie_search.llm.search_rerank"

    # Baseline hybrid — always the floor.
    base = _hybrid_search(query, k=pool, project=project, role=author, mode="hybrid")
    if not base.get("ok") or not base.get("hits"):
        return base

    hits = base["hits"][:pool]
    input_rowids = [h["rowid"] for h in hits]

    # Pull full message text for each candidate (snippets too short to score reliably).
    conn = _idx.connect(read_only=True)
    try:
        placeholders = ",".join("?" * len(input_rowids))
        rows = conn.execute(
            f"SELECT rowid, text FROM messages WHERE rowid IN ({placeholders})",
            input_rowids,
        ).fetchall()
    finally:
        conn.close()
    text_by_rowid = {r["rowid"]: (r["text"] or "")[:2000] for r in rows}
    documents = [text_by_rowid.get(rid, "") for rid in input_rowids]

    t0 = time.time()
    try:
        rerank = _get_reranker()
        scores = list(rerank.rerank(query, documents))
    except Exception as e:
        audit.log({
            "role": enable_role, "model": _RERANKER_MODEL,
            "prompt": f"q={query!r} n_docs={len(documents)}",
            "response": None,
            "trust": "untrusted", "fallback_fired": True,
            "ms": int((time.time() - t0) * 1000),
            "reason": f"reranker error: {type(e).__name__}: {e}",
        })
        base["reranked"] = False
        base["reason"] = f"reranker error, returning hybrid order: {e}"
        return base
    ms = int((time.time() - t0) * 1000)

    paired = sorted(
        zip(hits, scores), key=lambda x: x[1], reverse=True,
    )[:int(k)]
    new_order_rowids = [h["rowid"] for h, _ in paired]

    # Probe: output rowids must be subset of input.
    ok, probe_reason = guardrails.reranker_output_is_subset(
        new_order_rowids, input_rowids,
    )
    if not ok:
        audit.log({
            "role": enable_role, "model": _RERANKER_MODEL,
            "prompt": f"q={query!r}",
            "response": str(new_order_rowids),
            "trust": "untrusted", "fallback_fired": True, "ms": ms,
            "reason": f"probe failed: {probe_reason}",
        })
        base["reranked"] = False
        base["reason"] = f"rerank probe failed: {probe_reason}, returning hybrid order"
        return base

    out = []
    for h, sc in paired:
        h = dict(h)
        h["rerank_score"] = float(sc)
        h.pop("rrf_score", None)
        out.append(h)

    audit.log({
        "role": enable_role, "model": _RERANKER_MODEL,
        "prompt": f"q={query!r} pool={len(documents)} k={k}",
        "response": f"top={new_order_rowids[:5]} scores={[round(s,3) for _,s in paired[:5]]}",
        "trust": "clean", "fallback_fired": False, "ms": ms,
        "probe_results": {"subset_check": True},
    })
    return {
        "ok": True,
        "query": query,
        "mode": "rerank",
        "count": len(out),
        "hits": out,
        "reranked": True,
        "pool_size": len(documents),
        "rerank_ms": ms,
    }
