"""federation — fan out one query across multiple providers, fuse, budget.

The multi-source ("federated") half of magpie-search. Given a list of source
specs, it:

  1. fans out concurrently to each provider (fail-open: a slow/broken provider
     contributes zero, never breaks the call);
  2. filters by `min_trust` if asked;
  3. fuses across providers with trust-weighted Reciprocal Rank Fusion — the
     generalization of magpie's lexical/semantic RRF to trust tiers;
  4. dedups identical content across sources (keeps the highest-trust copy,
     records `also_in`);
  5. trims to a token budget, reporting what was dropped.

Every returned hit is tagged with its source, trust tier, category and
provenance — "categorized by what it is and where it came from."
"""
from __future__ import annotations

import concurrent.futures as _cf
import hashlib
import math
import os
import re
from typing import Any

from .providers import make_provider
from .providers.base import (
    DEFAULT_TRUST_WEIGHTS, TIER_RANK, Hit, Provider, TrustTier,
)

_WS = re.compile(r"\s+")


def _dedup_key(text: str) -> str:
    norm = _WS.sub(" ", (text or "")).strip().lower()
    return hashlib.sha256(norm.encode("utf-8", errors="replace")).hexdigest()


def estimate_tokens(text: str) -> int:
    """Approximate token count. Heuristic by default (no hard dep); ~3.6
    chars/token for English+code, rounded up and conservative so we under-fill
    rather than overflow the agent's window. `MAGPIE_SEARCH_TOKENIZER=tiktoken`
    opts into tiktoken if it's installed."""
    if not text:
        return 0
    if os.environ.get("MAGPIE_SEARCH_TOKENIZER", "").strip().lower().startswith("tiktoken"):
        try:  # pragma: no cover - optional dependency
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    return math.ceil(len(text) / 3.6)


def _coerce_weights(weights: dict | None) -> dict[TrustTier, float]:
    if not weights:
        return dict(DEFAULT_TRUST_WEIGHTS)
    out = dict(DEFAULT_TRUST_WEIGHTS)
    for k, v in weights.items():
        tier = TrustTier.coerce(k, None) if not isinstance(k, TrustTier) else k
        if tier is not None:
            out[tier] = float(v)
    return out


def federated_search(
    query: str,
    sources: list[Any],
    *,
    k: int = 10,
    budget_tokens: int | None = None,
    min_trust: TrustTier | str | None = None,
    scope: Any = None,
    rrf_k: int = 60,
    trust_weights: dict | None = None,
    timeout: float = 5.0,
    per_provider_k: int | None = None,
) -> dict[str, Any]:
    """Search `sources` (provider specs/instances/type-names) and return a
    fused, trust-tagged, budget-trimmed result dict."""
    if not sources:
        return {"ok": False, "reason": "no sources given"}
    k = max(1, int(k))
    fetch_k = per_provider_k if per_provider_k else max(k * 3, 10)
    weights = _coerce_weights(trust_weights)
    min_rank = None
    if min_trust is not None:
        mt = TrustTier.coerce(min_trust, None)
        if mt is not None:
            min_rank = TIER_RANK[mt]

    # --- build providers (a bad spec is reported, not fatal) ----------------
    providers: list[Provider] = []
    errors: dict[str, str] = {}
    for spec in sources:
        try:
            providers.append(make_provider(spec))
        except Exception as e:  # noqa: BLE001
            errors[str(spec)] = f"{type(e).__name__}: {e}"
    if not providers:
        return {"ok": False, "reason": "no usable sources", "errors": errors}

    # --- fan out concurrently, fail-open ------------------------------------
    per_source_raw: dict[str, list[Hit]] = {}
    with _cf.ThreadPoolExecutor(max_workers=min(8, len(providers))) as ex:
        futs = {
            ex.submit(p.search, query, budget_tokens=budget_tokens,
                      scope=scope, k=fetch_k): p
            for p in providers
        }
        for fut, p in futs.items():
            try:
                hits = fut.result(timeout=timeout) or []
            except Exception as e:  # noqa: BLE001 - timeout or provider error
                errors[p.name] = f"{type(e).__name__}: {e}"
                hits = []
            per_source_raw[p.name] = hits

    # --- fuse: trust-weighted RRF on rank-within-provider -------------------
    # GAP-5 guard: the trust weight is ACTUALLY multiplied in (magpie once
    # declared weights but never applied them — see search.py:_search_hybrid).
    dropped_min_trust = 0
    fused: list[Hit] = []
    source_returned: dict[str, int] = {}
    for name, hits in per_source_raw.items():
        kept = 0
        for rank, h in enumerate(hits, 1):
            if min_rank is not None and TIER_RANK[h.trust] < min_rank:
                dropped_min_trust += 1
                continue
            h.rrf_score = weights.get(h.trust, 1.0) / (rrf_k + rank)
            h.dedup_key = _dedup_key(h.text)
            h.tokens = estimate_tokens(h.text)
            fused.append(h)
            kept += 1
        source_returned[name] = kept

    # --- dedup across sources: keep highest-trust, then highest fused -------
    before = len(fused)
    survivors: dict[str, Hit] = {}
    seen_sources: dict[str, set[str]] = {}
    for h in fused:
        key = h.dedup_key
        if key not in survivors:
            survivors[key] = h
            seen_sources[key] = {h.source}
            continue
        seen_sources[key].add(h.source)
        cur = survivors[key]
        if (TIER_RANK[h.trust], h.rrf_score) > (TIER_RANK[cur.trust], cur.rrf_score):
            survivors[key] = h
    deduped: list[Hit] = []
    for key, h in survivors.items():
        # other sources the same content showed up in (corroboration)
        h.also_in = sorted(seen_sources[key] - {h.source})
        deduped.append(h)
    dropped_dedup = before - len(deduped)

    deduped.sort(key=lambda x: x.rrf_score, reverse=True)

    # --- budget + k trim ----------------------------------------------------
    out: list[Hit] = []
    used = 0
    dropped_budget = 0
    for h in deduped:
        if len(out) >= k:
            dropped_budget += 1
            continue
        if budget_tokens is not None and used + h.tokens > budget_tokens and out:
            dropped_budget += 1
            continue
        out.append(h)
        used += h.tokens

    return {
        "ok": True,
        "query": query,
        "count": len(out),
        "sources": source_returned,
        "used_tokens": used,
        "budget_tokens": budget_tokens,
        "dropped": {
            "budget": dropped_budget,
            "dedup": dropped_dedup,
            "min_trust": dropped_min_trust,
        },
        "errors": errors or None,
        "hits": [h.to_dict() for h in out],
    }
