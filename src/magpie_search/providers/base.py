"""providers.base — the Provider contract for multi-source ("federated") search.

A Provider searches ONE backend *live* at call time and returns a list of
normalized `Hit` objects. The federation engine (``magpie_search.federation``)
fans out across the selected providers, tags every hit with its source + trust
tier, fuses/ranks across providers, dedups, and trims to a token budget.

Design rule (mirrors magpie-search's hard line): a Provider SEARCHES a backend;
it never ingests/copies it into magpie's own store. The transcripts provider's
backend happens to be magpie's own index, but every other provider queries its
source live, so results always reflect canonical and magpie never becomes a
stale memory layer.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TrustTier(str, Enum):
    """How much to trust a hit, by where it came from. Ordered high->low.

    str-valued so it serializes to its plain name in JSON (json.dumps).
    """
    FACT = "fact"            # structured/authored truth (a KG, a facts DB)
    REFERENCE = "reference"  # authored canonical docs (memory tree, a docs folder)
    LEAD = "lead"            # "what was said" — verify before trusting (transcripts, diary)
    STALE = "stale"          # known-old / low-confidence

    @classmethod
    def coerce(cls, value: "TrustTier | str | None", default: "TrustTier") -> "TrustTier":
        if value is None:
            return default
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return default


# Rank for min_trust filtering and ordering (higher = more trusted).
TIER_RANK: dict[TrustTier, int] = {
    TrustTier.FACT: 3,
    TrustTier.REFERENCE: 2,
    TrustTier.LEAD: 1,
    TrustTier.STALE: 0,
}

# Default fusion weights — a higher-trust signal should outvote a noisier one.
# This is the same insight magpie's hybrid search applies with lex_weight=2.0 /
# sem_weight=1.0 (see search.py:_search_hybrid), generalized to trust tiers.
DEFAULT_TRUST_WEIGHTS: dict[TrustTier, float] = {
    TrustTier.FACT: 3.0,
    TrustTier.REFERENCE: 2.0,
    TrustTier.LEAD: 1.0,
    TrustTier.STALE: 0.3,
}


@dataclass
class Hit:
    """One normalized search result from any provider.

    `text`     — the snippet/excerpt to surface (MUST already be redaction-safe;
                 providers over raw sources run it through redactor.redact).
    `source`   — the provider INSTANCE name (e.g. "transcripts", "team-kg").
    `trust`    — trust tier for this hit.
    `category` — the provider KIND (e.g. "transcripts", "files", "kg").
    `score`    — provider-local relevance, higher = better. The engine only
                 uses each hit's RANK within its provider, so absolute scale
                 doesn't need to be comparable across providers.
    `provenance` — free-form origin info (path, session_id, ts, entity, ...).
    """
    text: str
    source: str
    trust: TrustTier
    category: str = ""
    score: float = 0.0
    provenance: dict[str, Any] = field(default_factory=dict)
    # Filled by the federation engine, not by providers:
    tokens: int = 0
    dedup_key: str = ""
    also_in: list[str] = field(default_factory=list)
    rrf_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = {
            "text": self.text,
            "source": self.source,
            "trust": self.trust.value,
            "category": self.category,
            "score": round(float(self.score), 6),
            "provenance": self.provenance,
            "tokens": self.tokens,
        }
        if self.also_in:
            d["also_in"] = self.also_in
        if self.rrf_score:
            d["rrf_score"] = round(self.rrf_score, 6)
        return d


class Provider(ABC):
    """Base class for a single-backend search provider.

    Subclasses set class attributes `category` and `default_trust`, and
    implement `search`. Instances carry a `name` (the source label that lands
    on every Hit) and a resolved `trust` tier (config override or default).
    """

    category: str = "generic"
    default_trust: TrustTier = TrustTier.LEAD

    def __init__(self, name: str | None = None, *,
                 trust: TrustTier | str | None = None, **config: Any) -> None:
        self.name = name or self.category
        self.trust = TrustTier.coerce(trust, self.default_trust)
        self.config = config

    @abstractmethod
    def search(self, query: str, *, budget_tokens: int | None = None,
               scope: Any = None, k: int = 10) -> list[Hit]:
        """Return ranked Hits (best first). MUST NOT raise on a bad/empty
        query or missing backend — return [] instead; the federation engine
        treats an exception as a fail-open zero-contribution but providers
        should degrade gracefully on their own where possible."""
        raise NotImplementedError

    def health(self) -> dict[str, Any]:
        return {"name": self.name, "category": self.category, "ok": True}

    def close(self) -> None:  # pragma: no cover - default no-op
        pass
