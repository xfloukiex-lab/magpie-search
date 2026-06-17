"""providers.transcripts — the built-in transcripts source.

Thin wrapper over the existing single-source transcript search
(``magpie_search.search``). This is the one provider whose backend is magpie's
own SQLite index; it adds no new storage, just normalizes transcript hits into
`Hit` objects tagged as `lead` ("what was said — verify before trusting").
"""
from __future__ import annotations

from typing import Any

from .base import Hit, Provider, TrustTier


def _project_from_scope(scope: Any) -> str | None:
    if scope is None:
        return None
    if isinstance(scope, str):
        # A bare scope string is treated as a project slug for transcripts.
        return scope or None
    if isinstance(scope, dict):
        return scope.get("project") or scope.get("slug")
    return None


class TranscriptsProvider(Provider):
    category = "transcripts"
    default_trust = TrustTier.LEAD

    def search(self, query: str, *, budget_tokens: int | None = None,
               scope: Any = None, k: int = 10) -> list[Hit]:
        from ..search import search as _search

        project = self.config.get("project") or _project_from_scope(scope)
        mode = self.config.get("mode", "hybrid")
        res = _search(query, k=k, project=project, mode=mode, dedup=True)
        if not res.get("ok"):
            return []
        hits: list[Hit] = []
        raw = res.get("hits", [])
        n = len(raw)
        for i, h in enumerate(raw):
            # Preserve provider-local ordering via a descending score; the
            # snippet is already redacted (transcripts are redacted at index).
            score = h.get("rrf_score")
            if score is None:
                score = (n - i) / float(n) if n else 0.0
            hits.append(Hit(
                text=h.get("snippet") or h.get("text") or "",
                source=self.name,
                trust=self.trust,
                category=self.category,
                score=float(score),
                provenance={
                    "session_id": h.get("session_id"),
                    "project": h.get("project"),
                    "ts": h.get("ts"),
                    "role": h.get("role"),
                    "msg_type": h.get("msg_type"),
                },
            ))
        return hits

    def health(self) -> dict[str, Any]:
        from ..indexer import db_path
        ok = db_path().exists()
        return {"name": self.name, "category": self.category, "ok": ok,
                "reason": None if ok else "index not built yet"}
