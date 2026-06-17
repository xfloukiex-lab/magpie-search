"""providers.vector — generic semantic search over a vector database.

A blank slot: ships with NO database path. The user (their AI / their config)
points it at their own sqlite-vec database; until then it returns nothing. The
product never references any specific store.

Expected sqlite-vec layout (all names configurable):
    a vec0 table  (default "vec")        with an embedding column ("embedding")
    a content table (default "documents") with a text column ("text"),
    joined on rowid. Override any name in config.

Reuses magpie's own embedding model + sqlite-vec loader so query vectors match
the same 384-dim space magpie indexes with. Output is run through redact().
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .base import Hit, Provider, TrustTier
from ..redactor import redact
from .. import embeddings
from ..indexer import _load_vec_extension, vec_available


def _as_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return list(value)


class VectorProvider(Provider):
    category = "vector"
    default_trust = TrustTier.LEAD  # semantic hits are evidence to verify

    def _db(self) -> Path | None:
        db = self.config.get("db")
        return Path(db).expanduser() if db else None

    def search(self, query: str, *, budget_tokens: int | None = None,
               scope: Any = None, k: int = 10) -> list[Hit]:
        if str(self.config.get("driver", "sqlite-vec")).lower() != "sqlite-vec":
            return []  # other drivers (e.g. chroma) deferred
        db = self._db()
        if db is None or not db.exists():
            return []
        qvec = embeddings.embed_one(query)
        if qvec is None:
            return []

        vec_table = self.config.get("vec_table", "vec")
        content_table = self.config.get("content_table", "documents")
        text_col = self.config.get("text_column", "text")
        emb_col = self.config.get("embedding_column", "embedding")
        meta_cols = _as_list(self.config.get("metadata_columns"))

        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            _load_vec_extension(conn)
            if not vec_available(conn):
                return []
            knn = conn.execute(
                f"SELECT rowid, distance FROM {vec_table} "
                f"WHERE {emb_col} MATCH ? ORDER BY distance LIMIT ?",
                (qvec, int(k)),
            ).fetchall()
            if not knn:
                return []
            dist = {r["rowid"]: r["distance"] for r in knn}
            ids = list(dist)
            ph = ",".join("?" * len(ids))
            cols = ", ".join([text_col, *meta_cols])
            rows = conn.execute(
                f"SELECT rowid, {cols} FROM {content_table} WHERE rowid IN ({ph})",
                ids,
            ).fetchall()
        except sqlite3.Error:
            return []
        finally:
            conn.close()

        # order by ascending distance (best first)
        rows = sorted(rows, key=lambda r: dist.get(r["rowid"], 1e9))
        hits: list[Hit] = []
        for r in rows:
            d = dist.get(r["rowid"], 1e9)
            prov: dict[str, Any] = {"rowid": r["rowid"], "distance": round(float(d), 6)}
            for c in meta_cols:
                prov[c] = r[c]
            text = (r[text_col] or "")[:480]
            hits.append(Hit(
                text=redact(text),
                source=self.name,
                trust=self.trust,
                category=self.category,
                score=1.0 / (1.0 + float(d)),
                provenance=prov,
            ))
        return hits

    def health(self) -> dict[str, Any]:
        db = self._db()
        ok = bool(db and db.exists())
        return {"name": self.name, "category": self.category, "ok": ok,
                "db": str(db) if db else None,
                "reason": None if ok else "no vector db configured"}
