"""providers.files — generic live search over a markdown/text directory.

Builds an EPHEMERAL in-memory FTS5 index over the scoped directory on each call
(file parse cached by a content signature; no persistent store is ever written).
This keeps it a tool, not a memory layer: the source files stay canonical and
nothing is copied into magpie's database. Output is run through redactor.redact
so a stray secret in a notes file never surfaces.

Covers the "search the right directory / notes folder" case directly — point it
at a docs tree and scope to a subpath.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .base import Hit, Provider, TrustTier
from ..redactor import redact
from ..search import _sanitize_query, _quote_all  # reuse the FTS5 query hardening

_DEFAULT_GLOBS = ("*.md", "*.markdown", "*.txt")

# Cache parsed chunks per (root, globs) keyed by a content signature so repeated
# calls in one process don't re-read the tree. { key: (signature, [chunk,...]) }
_CHUNK_CACHE: dict[tuple, tuple] = {}


def _norm_globs(value: Any) -> tuple[str, ...]:
    if not value:
        return _DEFAULT_GLOBS
    if isinstance(value, str):
        return tuple(g.strip() for g in value.split(",") if g.strip())
    return tuple(value)


def _scope_path(scope: Any) -> str | None:
    if scope is None:
        return None
    if isinstance(scope, str):
        return scope or None
    if isinstance(scope, dict):
        return scope.get("path") or scope.get("root") or scope.get("subpath")
    return None


def _split_chunks(path: Path, rel: str) -> list[dict[str, Any]]:
    """Split a text/markdown file into paragraph chunks, tracking the nearest
    markdown heading for provenance."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    chunks: list[dict[str, Any]] = []
    heading = ""
    buf: list[str] = []

    def flush() -> None:
        if buf:
            body = "\n".join(buf).strip()
            if body:
                chunks.append({"path": rel, "heading": heading, "text": body})
            buf.clear()

    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            flush()
            heading = s.lstrip("#").strip()
            continue
        if not s:
            flush()
            continue
        buf.append(line)
    flush()
    return chunks


class FilesProvider(Provider):
    category = "files"
    default_trust = TrustTier.REFERENCE

    def _resolve_root(self, scope: Any) -> tuple[Path | None, str | None]:
        """Return (search_root, subpath_filter). Config root is the base; a
        scope path narrows within it, or acts as the root when none is set."""
        cfg_root = self.config.get("root")
        base = Path(cfg_root).expanduser() if cfg_root else None
        sp = _scope_path(scope)
        if base and sp:
            return base, sp.replace("\\", "/").strip("/")
        if sp and not base:
            return Path(sp).expanduser(), None
        return base, None

    def _chunks(self, root: Path, globs: tuple[str, ...]) -> list[dict[str, Any]]:
        files: list[Path] = []
        for g in globs:
            files.extend(root.rglob(g))
        files = sorted(set(p for p in files if p.is_file()))
        # Signature: (relpath, mtime_ns, size) per file — cheap stat-only pass.
        sig_parts = []
        for p in files:
            try:
                st = p.stat()
                sig_parts.append((str(p), st.st_mtime_ns, st.st_size))
            except OSError:
                continue
        sig = hash(tuple(sig_parts))
        key = (str(root), globs)
        cached = _CHUNK_CACHE.get(key)
        if cached and cached[0] == sig:
            return cached[1]
        chunks: list[dict[str, Any]] = []
        for p in files:
            try:
                rel = str(p.relative_to(root)).replace("\\", "/")
            except ValueError:
                rel = str(p)
            chunks.extend(_split_chunks(p, rel))
        _CHUNK_CACHE[key] = (sig, chunks)
        return chunks

    def search(self, query: str, *, budget_tokens: int | None = None,
               scope: Any = None, k: int = 10) -> list[Hit]:
        root, subpath = self._resolve_root(scope)
        if root is None or not root.exists():
            return []
        globs = _norm_globs(self.config.get("globs"))
        chunks = self._chunks(root, globs)
        if subpath:
            sp = subpath.lower()
            chunks = [c for c in chunks if c["path"].lower().startswith(sp)]
        if not chunks:
            return []

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE docs USING fts5("
                "path, heading, text, tokenize='porter unicode61')"
            )
            conn.executemany(
                "INSERT INTO docs(rowid, path, heading, text) VALUES(?,?,?,?)",
                [(i, c["path"], c["heading"], c["text"]) for i, c in enumerate(chunks)],
            )
            match = _sanitize_query(query)
            if not match:
                return []
            sql = (
                "SELECT rowid, path, heading, "
                "snippet(docs, 2, '<<', '>>', '...', 16) AS snippet, "
                "bm25(docs) AS rank FROM docs WHERE docs MATCH ? "
                "ORDER BY rank LIMIT ?"
            )
            try:
                rows = conn.execute(sql, (match, int(k))).fetchall()
            except sqlite3.Error:
                fallback = _quote_all(query)
                if not fallback:
                    return []
                rows = conn.execute(sql, (fallback, int(k))).fetchall()
        finally:
            conn.close()

        hits: list[Hit] = []
        n = len(rows)
        for i, r in enumerate(rows):
            chunk = chunks[r["rowid"]]
            hits.append(Hit(
                text=redact(r["snippet"] or chunk["text"][:240]),
                source=self.name,
                trust=self.trust,
                category=self.category,
                score=(n - i) / float(n) if n else 0.0,
                provenance={
                    "path": r["path"],
                    "heading": r["heading"] or None,
                    "root": str(root),
                },
            ))
        return hits

    def health(self) -> dict[str, Any]:
        root, _ = self._resolve_root(None)
        ok = bool(root and root.exists())
        return {"name": self.name, "category": self.category, "ok": ok,
                "root": str(root) if root else None,
                "reason": None if ok else "root not configured or missing"}
