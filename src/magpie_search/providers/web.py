"""providers.web — live WEB search as a magpie federated source.

Backed by DuckDuckGo via the ``ddgs`` library (no API key). Returns ranked
snippet Hits (title + snippet + url) — a token-efficient "filter the web down to
the relevant bits" feed, NOT an answer machine. The caller reasons over the
snippets and keeps the source URLs.

Trust tier: LEAD. Web results are "what some page said" — a lead to verify, the
same discipline magpie applies to transcripts/diary. Never FACT/REFERENCE.

Per the Provider contract: searches live at call time (never ingested into
magpie's store) and NEVER raises — returns [] on any error / missing dep.
"""
from __future__ import annotations

from typing import Any

from ..redactor import redact
from .base import Hit, Provider, TrustTier


class WebProvider(Provider):
    category = "web"
    default_trust = TrustTier.LEAD
    # Ordered FALLBACK across fast engines: try each, stop at the first that
    # answers. Beats both a single flaky engine (DuckDuckGo returns "No results")
    # AND the library's "auto" (which rotates serially and takes ~4.8s, tripping
    # the federation 5s timeout -> 0 hits). These are all fast (~1.7-2s) and
    # individually reliable; first-hit-wins keeps the common case ~2s.
    default_engines = ("google", "mojeek", "brave", "bing")

    def search(self, query: str, *, budget_tokens: int | None = None,
               scope: Any = None, k: int = 10) -> list[Hit]:
        query = (query or "").strip()
        if not query:
            return []
        try:
            from ddgs import DDGS
        except Exception:
            return []

        engines = self.config.get("backend") or self.config.get("backends") \
            or self.default_engines
        if isinstance(engines, str):
            engines = [engines]

        raw: list = []
        for eng in engines:
            try:
                raw = list(DDGS().text(query, backend=eng, max_results=max(1, k)))
            except Exception:
                raw = []
            if raw:
                break
        if not raw:
            return []

        hits: list[Hit] = []
        for rank, h in enumerate(raw):
            snippet = (h.get("body") or h.get("snippet") or h.get("description") or "").strip()
            if not snippet:
                continue
            title = (h.get("title") or "").strip()
            url = (h.get("href") or h.get("url") or "").strip()
            text = redact(f"{title} — {snippet}".strip(" —"))
            hits.append(Hit(
                text=text,
                source=self.name,
                trust=self.trust,
                category=self.category,
                score=float(len(raw) - rank),   # earlier result = higher score
                provenance={"url": url, "title": title},
            ))
        return hits

    def health(self) -> dict[str, Any]:
        ok = True
        try:
            import ddgs  # noqa: F401
        except Exception:
            ok = False
        return {"name": self.name, "category": self.category, "ok": ok,
                "backend": "duckduckgo/ddgs"}
