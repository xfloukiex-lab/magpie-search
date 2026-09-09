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

import sys as _sys
from typing import Any

from ..redactor import redact
from .base import Hit, Provider, TrustTier

# Which optional dependencies have already been reported missing, so a loop over
# many URLs warns once rather than once per call.
_WARNED: set[str] = set()


def warn_missing_dep(dep: str, extra: str, what: str) -> None:
    """Say ONCE, on stderr, that an optional dependency is absent.

    The provider contract is to return [] and never raise, which is right — a
    missing extra must not take down a federated search that other sources can
    still answer. But an empty result set is ALSO what a genuine no-match looks
    like, so without this the two are indistinguishable and the user has no
    reason to suspect an install problem. Silence here is what made this cost an
    investigation instead of a glance.
    """
    if dep in _WARNED:
        return
    _WARNED.add(dep)
    print(
        f"[magpie-search] {what} is unavailable: the '{dep}' package is not "
        f"installed, so this returns no results. Install it with: "
        f"pip install 'magpie-search[{extra}]'",
        file=_sys.stderr,
    )


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
        # R2 (Aether audit 2026-07-26): the query leaves the machine. magpie's
        # promise is local-first, so a secret pasted into a search ("why does
        # sk-... 401") must not be handed to a third-party engine verbatim.
        # Redact before the call, and say so on stderr — a silent redaction
        # would look like the engine simply returned nothing useful.
        outbound = redact(query)
        if outbound != query:
            print(
                "[magpie-search] web query redacted before outbound search "
                "(a secret-shaped token was removed)",
                file=_sys.stderr,
            )
        query = outbound
        try:
            from ddgs import DDGS
        except Exception:
            warn_missing_dep("ddgs", "web", "web search")
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
                # R1 (Aether audit 2026-07-26): provenance is returned to the
                # caller just like `text`, so it must clear the same bar. A URL
                # can carry a secret in a query string / userinfo (a leaked
                # token echoed back in a search result), and title is
                # page-controlled. Redacting only `text` left a silent hole.
                provenance={"url": redact(url), "title": redact(title)},
            ))
        return hits

    def health(self) -> dict[str, Any]:
        ok = True
        try:
            import ddgs  # noqa: F401
        except Exception:
            ok = False
        out = {"name": self.name, "category": self.category, "ok": ok,
               "backend": "duckduckgo/ddgs"}
        if not ok:
            # Name the cause and the remedy. `ok: False` on its own tells the
            # caller something is wrong but not that it is a one-line fix.
            out["reason"] = "the 'ddgs' package is not installed"
            out["fix"] = "pip install 'magpie-search[web]'"
        return out
