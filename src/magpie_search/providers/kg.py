"""providers.kg — generic structured-facts search over a sqlite table.

A blank slot: ships with NO database path. The user points it at their own
sqlite table of facts/records; until then it returns nothing. Tagged `fact`
by default (structured/authored data), so it outranks looser sources in the
merge. Output is run through redact().

Each row is rendered to a line ("col=value | col=value", over the configured
columns), searched lexically via an ephemeral in-memory FTS5 index built per
call — no persistent store, so the source table stays canonical.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .base import Hit, Provider, TrustTier
from ..redactor import redact
from ..search import _sanitize_query, _quote_all

_MAX_ROWS = 20000  # safety cap on how many rows we pull into the ephemeral index


def _as_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return list(value)


class KGProvider(Provider):
    category = "kg"
    default_trust = TrustTier.FACT

    def _db(self) -> Path | None:
        db = self.config.get("db")
        return Path(db).expanduser() if db else None

    def _columns(self, conn: sqlite3.Connection, table: str) -> list[str]:
        cols = _as_list(self.config.get("columns"))
        if cols:
            return cols
        info = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return [r[1] for r in info]

    def search(self, query: str, *, budget_tokens: int | None = None,
               scope: Any = None, k: int = 10) -> list[Hit]:
        db = self._db()
        if db is None or not db.exists():
            return []
        table = self.config.get("table", "facts")

        src = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10.0)
        src.row_factory = sqlite3.Row
        try:
            cols = self._columns(src, table)
            if not cols:
                return []
            col_sql = ", ".join(cols)
            rows = src.execute(
                f"SELECT {col_sql} FROM {table} LIMIT ?", (_MAX_ROWS,)
            ).fetchall()
        except sqlite3.Error:
            return []
        finally:
            src.close()
        if not rows:
            return []

        rendered: list[tuple[str, dict[str, Any]]] = []
        for r in rows:
            parts = []
            prov: dict[str, Any] = {}
            for c in cols:
                v = r[c]
                if v is None:
                    continue
                parts.append(f"{c}={v}")
                prov[c] = v
            text = " | ".join(parts)
            if text:
                rendered.append((text, prov))
        if not rendered:
            return []

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE facts USING fts5(text, tokenize='porter unicode61')"
            )
            conn.executemany(
                "INSERT INTO facts(rowid, text) VALUES(?,?)",
                [(i, t) for i, (t, _) in enumerate(rendered)],
            )
            match = _sanitize_query(query)
            if not match:
                return []
            sql = ("SELECT rowid, bm25(facts) AS rank FROM facts "
                   "WHERE facts MATCH ? ORDER BY rank LIMIT ?")
            try:
                hits_raw = conn.execute(sql, (match, int(k))).fetchall()
            except sqlite3.Error:
                fallback = _quote_all(query)
                if not fallback:
                    return []
                hits_raw = conn.execute(sql, (fallback, int(k))).fetchall()
        finally:
            conn.close()

        out: list[Hit] = []
        n = len(hits_raw)
        for i, hr in enumerate(hits_raw):
            text, prov = rendered[hr["rowid"]]
            out.append(Hit(
                text=redact(text),
                source=self.name,
                trust=self.trust,
                category=self.category,
                score=(n - i) / float(n) if n else 0.0,
                provenance=prov,
            ))
        return out

    def health(self) -> dict[str, Any]:
        db = self._db()
        ok = bool(db and db.exists())
        return {"name": self.name, "category": self.category, "ok": ok,
                "db": str(db) if db else None,
                "reason": None if ok else "no knowledge-graph db configured"}
