"""search — read-only query interface over the FTS5 transcript index.

Three queries:
    search(query, k=10, project=None, mode='lexical') -> ranked matches
    recent(n=50, session_id=None, project=None) -> tail of latest session
    session(session_id, limit=200, offset=0) -> paginated session view

Search modes (opt-in via `mode=`):
    'lexical' (default) — FTS5 BM25 ranking. Best for exact terms.
    'semantic'           — cosine K-NN over messages_vec embeddings.
                           Best for concept / paraphrase queries.
    'hybrid'             — RRF fusion of lexical + semantic results.
                           Best general-purpose; ~2x cost.

Every result already passed through `redactor.redact` at index time, so
returned text is safe to surface (no secrets).
"""
from __future__ import annotations

import os
import re
from typing import Any

from . import embeddings
from .indexer import connect, db_path, vec_available


def _dedup_requested(explicit: bool | None) -> bool:
    """Resolve the dedup toggle. `dedup=True/False` overrides the env var.
    `dedup=None` falls back to MAGPIE_SEARCH_DEDUP, which is ON by default
    (operator directive 2026-05-15) — only MAGPIE_SEARCH_DEDUP=0/false/no/off
    disables it."""
    if explicit is not None:
        return bool(explicit)
    val = os.environ.get("MAGPIE_SEARCH_DEDUP", "").strip().lower()
    if val in ("0", "false", "no", "off", "disable", "disabled"):
        return False
    return True


def _annotate_with_dedup(conn, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cluster hits by content_sha256 + annotate with dup_count.

    When the same text (e.g. a file re-read across many sessions) appears
    verbatim N times, a search for it returns N byte-identical hits. This
    collapse returns one (the most recent occurrence) with `dup_count: N`
    so the caller's context budget isn't burned by duplicates.

    Resolves content_sha256 per hit by JOINing messages_meta; falls back
    to in-memory hash if no messages_meta row exists (e.g. content
    indexed BEFORE MAGPIE_SEARCH_DEDUP was first turned on).

    `dup_count` comes from chunk_dedup.count when available — gives the
    TRUE global count, not just count-within-this-result-set. Falls back
    to within-result count if chunk_dedup is empty for that hash."""
    if not hits:
        return hits

    # Lookup content_sha256 + novelty per hit (via messages_meta JOIN).
    rowids = [h.get("rowid") for h in hits if h.get("rowid") is not None]
    sha_by_rowid: dict[int, str] = {}
    novel_by_rowid: dict[int, int] = {}
    if rowids:
        placeholders = ",".join("?" * len(rowids))
        try:
            for r in conn.execute(
                f"SELECT rowid, content_sha256, is_novel_in_session "
                f"FROM messages_meta WHERE rowid IN ({placeholders})",
                rowids,
            ).fetchall():
                sha_by_rowid[r["rowid"]] = r["content_sha256"]
                novel_by_rowid[r["rowid"]] = r["is_novel_in_session"]
        except Exception:
            # Older messages_meta without is_novel_in_session column —
            # fall back to sha-only lookup.
            try:
                for r in conn.execute(
                    f"SELECT rowid, content_sha256 FROM messages_meta "
                    f"WHERE rowid IN ({placeholders})",
                    rowids,
                ).fetchall():
                    sha_by_rowid[r["rowid"]] = r["content_sha256"]
            except Exception:
                pass

    # Fallback hash for hits not in messages_meta (legacy rows).
    import hashlib
    _ws = re.compile(r"\s+")

    def _fallback_hash(text: str) -> str:
        norm = _ws.sub(" ", text or "").strip()
        return hashlib.sha256(norm.encode("utf-8", errors="replace")).hexdigest()

    # Cluster: keep first occurrence per hash (= top BM25 rank).
    clusters: dict[str, dict[str, Any]] = {}
    in_set_counts: dict[str, int] = {}
    for h in hits:
        rid = h.get("rowid")
        sha = sha_by_rowid.get(rid) if rid is not None else None
        if sha is None:
            text = h.get("snippet") or h.get("text") or ""
            sha = _fallback_hash(text)
        in_set_counts[sha] = in_set_counts.get(sha, 0) + 1
        if sha not in clusters:
            novel = novel_by_rowid.get(rid) if rid is not None else None
            h = {**h, "content_sha256": sha}
            if novel is not None:
                h["is_novel_in_session"] = bool(novel)
            clusters[sha] = h

    # Annotate dup_count from chunk_dedup (true global), fallback to
    # in-this-result count.
    shas = list(clusters.keys())
    global_counts: dict[str, int] = {}
    if shas:
        placeholders = ",".join("?" * len(shas))
        try:
            for r in conn.execute(
                f"SELECT sha256, count FROM chunk_dedup "
                f"WHERE sha256 IN ({placeholders})",
                shas,
            ).fetchall():
                global_counts[r["sha256"]] = r["count"]
        except Exception:
            pass

    out = []
    for sha, h in clusters.items():
        h["dup_count"] = global_counts.get(sha, in_set_counts.get(sha, 1))
        out.append(h)
    return out


_FTS_SPECIAL = re.compile(r'[^\w\s"*\-+]')

# Short throwaway words that hurt natural-language retrieval if treated as
# required terms. Kept small — anything ambiguous stays in (e.g. "for" can
# be load-bearing in queries like "look for X"; we exclude it anyway since
# BM25 will still rank results that include it higher).
_FTS_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for",
    "by", "with", "is", "are", "was", "were", "be", "been", "being", "do",
    "does", "did", "have", "has", "had", "this", "that", "these", "those",
    "it", "its", "as", "we", "i", "you", "he", "she", "they", "what",
    "when", "where", "who", "why", "how", "which", "from", "not",
}


def _sanitize_query(q: str) -> str:
    """Make a user-typed query safe for FTS5 MATCH.

    Behavior: punctuation is stripped, stop-words removed, remaining
    tokens joined with `OR` so natural-language queries don't require
    every word to appear (BM25 still ranks all-terms-present higher).
    Operators the user explicitly types — quoted phrases, `*` wildcards,
    leading `-` / `+` — pass through untouched.

    Edge cases:
      - If the user pre-formatted with OR/AND/NEAR/quotes/wildcards, we
        treat it as power-user syntax and leave it alone.
      - Single-token queries skip the OR-join.
    """
    cleaned = _FTS_SPECIAL.sub(" ", q).strip()
    if not cleaned:
        return ""
    # Power-user syntax escape hatch: hand it through verbatim.
    upper = cleaned.upper()
    if any(tok in upper for tok in (" OR ", " AND ", " NEAR ", '"', "*")):
        return cleaned
    if any(t.startswith(("+", "-")) for t in cleaned.split()):
        return cleaned
    tokens = [t for t in cleaned.split()
              if t.lower() not in _FTS_STOPWORDS and len(t) > 1]
    if not tokens:
        # All stop-words — fall back to the raw cleaned form rather than
        # empty (zero-result) query.
        tokens = [t for t in cleaned.split() if t]
        if not tokens:
            return ""

    # Quote every token so FTS5 treats it as a literal string, not a
    # syntax fragment. Without this, a token with an interior hyphen
    # (`mcp-server`, `9-page`, `uniquephrase-abc`) makes FTS5 parse the
    # part after `-` as a NOT/column filter -> "fts5 error: no such
    # column: <x>", hard-crashing the keystone retrieval path (verified
    # 2026-05-16). The power-user escape hatch above already returned
    # for intentional operators / wildcards / quoted phrases, so by here
    # there is no operator we need to preserve. A literal `"` inside a
    # token is doubled per FTS5 string-quoting rules.
    def _q(tok: str) -> str:
        return '"' + tok.replace('"', '""') + '"'

    quoted = [_q(t) for t in tokens]
    if len(quoted) == 1:
        return quoted[0]
    return " OR ".join(quoted)


def _quote_all(q: str) -> str:
    """Force every token to a quoted literal, ignoring operators entirely.

    Used as the safe fallback when the power-user passthrough in
    `_sanitize_query` produces invalid FTS5 (e.g. a bare `*`, `-only`, an
    unbalanced quote, a trailing `OR`). Rather than erroring out on the
    keystone search path, we degrade to a literal search."""
    cleaned = _FTS_SPECIAL.sub(" ", q).strip()
    toks = [t for t in cleaned.split() if t]
    if not toks:
        return ""
    qt = ['"' + t.replace('"', '""') + '"' for t in toks]
    return qt[0] if len(qt) == 1 else " OR ".join(qt)


def search(
    query: str,
    *,
    k: int = 10,
    project: str | None = None,
    role: str | None = None,
    mode: str = "lexical",
    dedup: bool | None = None,
    sources: list | None = None,
    budget_tokens: int | None = None,
    min_trust: str | None = None,
    scope=None,
    regex: bool = False,
) -> dict[str, Any]:
    """Top-k matches. Mode selects ranking strategy.

    mode='lexical'  -> FTS5 BM25 ranking (default; unchanged behavior).
    mode='semantic' -> vector K-NN over messages_vec.
    mode='hybrid'   -> RRF fusion of lexical + semantic.

    `dedup`: if True (or None and MAGPIE_SEARCH_DEDUP=1), cluster results by
    content_sha256 and return one per cluster annotated with `dup_count`.
    Default off; opt-in (cuts context burn ~5x on high-duplication
    searches — text repeated verbatim across many sessions).

    Multi-source ("federated") search:
        `sources`: when given and not just ["transcripts"], the query fans out
        across the named providers (transcripts, files, and any installed
        plugins) and returns a fused, trust-tagged, budget-trimmed result via
        magpie_search.federation.federated_search. `budget_tokens`, `min_trust`
        and `scope` apply to that path. When `sources` is None or ["transcripts"]
        the behavior + return shape below are unchanged (back-compat).

    Returns:
        {"ok": True, "query": ..., "mode": ..., "count": N, "hits": [...]}
        Each hit has: session_id, project, ts, role, msg_type, snippet, rank.
        On hybrid: also rrf_score and (where available) rank_lex / rank_sem.
        With dedup on: also content_sha256 and dup_count.
    """
    # Federated path: only when the caller asks for more than transcripts.
    # Default / ["transcripts"] keep the single-source behavior byte-for-byte.
    if sources is not None and list(sources) != ["transcripts"]:
        from .federation import federated_search
        return federated_search(
            query, list(sources), k=k, budget_tokens=budget_tokens,
            min_trust=min_trust, scope=scope if scope is not None else project,
        )

    if not db_path().exists():
        return {"ok": False, "reason": "index not built yet"}
    # Clamp: SQLite treats a negative LIMIT as "no limit", so a negative/zero
    # k would dump the entire matching set (memory/latency footgun). Floor at 1.
    k = max(1, int(k))
    use_dedup = _dedup_requested(dedup)
    # When dedup is on, over-fetch so the collapse can still return k
    # distinct clusters even when the top results are heavy duplicates.
    fetch_k = k * 5 if use_dedup else k
    if mode == "lexical":
        result = _search_lexical(query, k=fetch_k, project=project, role=role)
    elif mode == "semantic":
        result = _search_semantic(query, k=fetch_k, project=project, role=role)
    elif mode == "hybrid":
        result = _search_hybrid(query, k=fetch_k, project=project, role=role)
    elif mode == "grep":
        result = _search_grep(query, k=fetch_k, project=project, role=role, regex=regex)
    else:
        return {"ok": False, "reason": f"unknown mode: {mode!r}"}
    if not use_dedup or not result.get("ok"):
        return result
    # Apply dedup clustering + top-k trim.
    conn = connect(read_only=True)
    try:
        clustered = _annotate_with_dedup(conn, result.get("hits", []))
    finally:
        conn.close()
    clustered = clustered[:k]
    result["hits"] = clustered
    result["count"] = len(clustered)
    result["dedup"] = True
    return result


def _grep_snippet(text: str, pat: "re.Pattern", width: int = 120) -> str:
    """Excerpt around the first regex match, with the match marked <<...>>."""
    if not text:
        return ""
    m = pat.search(text)
    if not m:
        return _snippet_from_text(text, 240)
    s = max(0, m.start() - width // 2)
    e = min(len(text), m.end() + width // 2)
    pre = "..." if s > 0 else ""
    post = "..." if e < len(text) else ""
    seg = text[s:m.start()] + "<<" + text[m.start():m.end()] + ">>" + text[m.end():e]
    return (pre + seg + post).replace("\n", " ").strip()


def _search_grep(
    query: str,
    *,
    k: int,
    project: str | None,
    role: str | None,
    regex: bool = False,
) -> dict[str, Any]:
    """Exact-match search over message text — the leg FTS5 (token-based) and semantic
    (fuzzy) both miss: code symbols, paths, hashes, exact error strings. By DEFAULT
    the query is a LITERAL string (so `os.environ.get('X')` or `C:\\Users\\...` match
    as typed); pass regex=True to treat it as a Python regex. Case-insensitive. No
    relevance rank — ordered most-recent-first."""
    pattern = query if regex else re.escape(query)
    try:
        pat = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return {"ok": False, "reason": f"invalid regex: {e}"}
    conn = connect(read_only=True)
    try:
        conn.create_function(
            "rematch", 1, lambda s: 1 if (s and pat.search(s)) else 0, deterministic=True)
        sql = ("SELECT rowid, session_id, project, ts, role, msg_type, cwd, text "
               "FROM messages WHERE rematch(text) = 1 ")
        params: list[Any] = []
        if project:
            sql += "AND project = ? "
            params.append(project)
        if role:
            sql += "AND role = ? "
            params.append(role)
        sql += "ORDER BY ts DESC LIMIT ?"
        params.append(int(k))
        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception as e:
            return {"ok": False, "reason": f"grep error: {e}"}
    finally:
        conn.close()
    hits: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        d["snippet"] = _grep_snippet(d.pop("text", ""), pat)
        d["rank"] = 0.0
        hits.append(d)
    return {"ok": True, "query": query, "mode": "grep", "count": len(hits), "hits": hits}


def _search_lexical(
    query: str,
    *,
    k: int,
    project: str | None,
    role: str | None,
) -> dict[str, Any]:
    cleaned = _sanitize_query(query)
    if not cleaned:
        return {"ok": False, "reason": "empty query"}

    sql = (
        "SELECT rowid, session_id, project, ts, role, msg_type, cwd, "
        "       snippet(messages, 6, '<<', '>>', '...', 16) AS snippet, "
        "       bm25(messages) AS rank "
        "FROM messages "
        "WHERE messages MATCH ? "
    )
    params: list[Any] = [cleaned]
    if project:
        sql += "AND project = ? "
        params.append(project)
    if role:
        sql += "AND role = ? "
        params.append(role)
    sql += "ORDER BY rank LIMIT ?"
    params.append(int(k))

    conn = connect(read_only=True)
    used = cleaned
    try:
        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception:
            # The power-user passthrough produced invalid FTS5 syntax
            # (e.g. `*`, `-only`, unbalanced quote, trailing OR). Retry ONCE
            # with every token quoted as a literal before giving up, so a
            # stray operator degrades to a literal search instead of a
            # hard zero-result error on the keystone path.
            fallback = _quote_all(query)
            if fallback and fallback != cleaned:
                params[0] = fallback
                try:
                    rows = conn.execute(sql, params).fetchall()
                    used = fallback
                except Exception as e2:
                    return {"ok": False, "reason": f"fts5 error: {e2}"}
            else:
                return {"ok": False, "reason": "empty query"}
    finally:
        conn.close()

    hits = [dict(r) for r in rows]
    return {"ok": True, "query": used, "mode": "lexical",
            "count": len(hits), "hits": hits}


def _semantic_preflight(conn) -> str | None:
    """Return reason-string if semantic search can't run; None if OK."""
    if not vec_available(conn):
        return "sqlite-vec extension not loaded"
    if not embeddings.available():
        return f"embedding model unavailable: {embeddings.load_error()}"
    try:
        n = conn.execute("SELECT COUNT(*) FROM messages_vec").fetchone()[0]
    except Exception as e:
        return f"messages_vec missing: {e}"
    if n == 0:
        return "messages_vec is empty — run backfill_embeddings"
    return None


def _snippet_from_text(text: str, max_len: int = 240) -> str:
    """Semantic results have no keyword to highlight — return a head excerpt."""
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    return text[:max_len] + ("..." if len(text) > max_len else "")


def _search_semantic(
    query: str,
    *,
    k: int,
    project: str | None,
    role: str | None,
    over_fetch: int = 4,
) -> dict[str, Any]:
    """K-NN over messages_vec; LEFT JOIN messages for metadata + text.

    Over-fetches by `over_fetch * k` then post-filters by project/role so
    the K-NN still gets enough candidates to satisfy the filter."""
    conn = connect(read_only=True)
    try:
        reason = _semantic_preflight(conn)
        if reason:
            return {"ok": False, "reason": reason, "mode": "semantic"}

        qvec = embeddings.embed_one(query)
        if qvec is None:
            return {"ok": False, "reason": "query embed failed"}

        fetch_n = int(k) * (over_fetch if (project or role) else 1)
        knn = conn.execute(
            "SELECT rowid, distance FROM messages_vec "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (qvec, fetch_n),
        ).fetchall()
        if not knn:
            return {"ok": True, "query": query, "mode": "semantic",
                    "count": 0, "hits": []}

        rowid_to_dist = {r["rowid"]: r["distance"] for r in knn}
        ids = list(rowid_to_dist.keys())
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT rowid, session_id, project, ts, role, msg_type, cwd, text "
            f"FROM messages WHERE rowid IN ({placeholders})",
            ids,
        ).fetchall()
    finally:
        conn.close()

    hits = []
    for r in rows:
        d = dict(r)
        if project and d["project"] != project:
            continue
        if role and d["role"] != role:
            continue
        d["snippet"] = _snippet_from_text(d.pop("text", "") or "")
        d["distance"] = rowid_to_dist.get(d["rowid"])
        hits.append(d)

    # vec0 returned in distance order; preserve that. Top-k after filter.
    hits.sort(key=lambda h: h["distance"])
    hits = hits[: int(k)]
    return {"ok": True, "query": query, "mode": "semantic",
            "count": len(hits), "hits": hits}


def _search_hybrid(
    query: str,
    *,
    k: int,
    project: str | None,
    role: str | None,
    rrf_k: int = 60,
    lex_weight: float = 2.0,
    sem_weight: float = 1.0,
) -> dict[str, Any]:
    """Reciprocal Rank Fusion of lexical + semantic, weighted toward lexical.

    score(doc) = w_lex/(rrf_k + rank_lex) + w_sem/(rrf_k + rank_sem)

    Lexical gets 2x weight by default: on a code/transcript corpus the
    embedding model produces a lot of "kinda related" neighbors that
    aren't the specific evidence, so unweighted RRF drags the right
    answer below noisier semantic-popular docs.

    GAP-5 (2026-05-16): the weights were DECLARED here but `_rrf` never
    multiplied by them — fusion was silently 1:1. First full LOCOMO run
    measured the cost: hybrid R@5=0.388 LOST to plain lexical R@5=0.473,
    and an ablation showed equal-weight fusion destroyed the gold-in-top5
    for 44% of the QAs lexical alone got right. The weights are now
    actually applied and the pool widened 2x->6x (a too-shallow pool
    excluded the gold chunk before fusion saw it). Real post-fix LOCOMO
    macro numbers are recorded in
    experiments/2026-05-16-magpie_search-hardening/ — do not put a hand-typed
    number in this docstring again (the prior "conv-0 R@5=0.60" claim
    was fabricated and hid this bug from review).

    Falls back gracefully if the semantic side is unavailable.
    """
    fetch = max(int(k) * 6, 50)
    lex = _search_lexical(query, k=fetch, project=project, role=role)
    sem = _search_semantic(query, k=fetch, project=project, role=role)

    lex_hits = lex.get("hits", []) if lex.get("ok") else []
    sem_hits = sem.get("hits", []) if sem.get("ok") else []

    if not lex_hits and not sem_hits:
        return {
            "ok": False,
            "mode": "hybrid",
            "reason": "no hits in either modality",
            "lex_reason": lex.get("reason"),
            "sem_reason": sem.get("reason"),
        }

    scored: dict[int, dict[str, Any]] = {}
    for rank, h in enumerate(lex_hits, 1):
        rid = h.get("rowid")
        if rid is None:
            continue
        scored.setdefault(rid, {"hit": h, "rank_lex": None, "rank_sem": None})
        scored[rid]["hit"] = {**scored[rid]["hit"], **h}
        scored[rid]["rank_lex"] = rank
    for rank, h in enumerate(sem_hits, 1):
        rid = h.get("rowid")
        if rid is None:
            continue
        scored.setdefault(rid, {"hit": h, "rank_lex": None, "rank_sem": None})
        # Don't clobber lexical snippet (it has keyword highlight) with semantic
        merged = {**scored[rid]["hit"]}
        for kk, vv in h.items():
            merged.setdefault(kk, vv)
        scored[rid]["hit"] = merged
        scored[rid]["rank_sem"] = rank

    def _rrf(entry: dict[str, Any]) -> float:
        # GAP-5 fix: actually apply lex_weight/sem_weight (captured from
        # the enclosing scope). Previously this was 1.0/... for both —
        # equal-weight fusion that let a near-random semantic signal
        # outvote a strong lexical one. See the function docstring.
        s = 0.0
        if entry["rank_lex"] is not None:
            s += lex_weight / (rrf_k + entry["rank_lex"])
        if entry["rank_sem"] is not None:
            s += sem_weight / (rrf_k + entry["rank_sem"])
        return s

    ranked = sorted(scored.values(), key=_rrf, reverse=True)
    hits = []
    for entry in ranked[: int(k)]:
        h = entry["hit"]
        h["rrf_score"] = round(_rrf(entry), 6)
        h["rank_lex"] = entry["rank_lex"]
        h["rank_sem"] = entry["rank_sem"]
        hits.append(h)

    return {
        "ok": True,
        "query": query,
        "mode": "hybrid",
        "count": len(hits),
        "hits": hits,
        "lex_pool": len(lex_hits),
        "sem_pool": len(sem_hits),
    }


def recent(
    *,
    n: int = 50,
    session_id: str | None = None,
    project: str | None = None,
) -> dict[str, Any]:
    """Return the most-recent N messages, optionally scoped to one session.

    If session_id isn't given, returns from whichever session has the most
    recent `last_ts` (optionally filtered by project).
    """
    if not db_path().exists():
        return {"ok": False, "reason": "index not built yet"}
    conn = connect(read_only=True)
    try:
        if session_id is None:
            sql = "SELECT session_id FROM sessions "
            params: list[Any] = []
            if project:
                sql += "WHERE project = ? "
                params.append(project)
            sql += "ORDER BY last_ts DESC LIMIT 1"
            row = conn.execute(sql, params).fetchone()
            if row is None:
                return {"ok": True, "session_id": None, "count": 0, "messages": []}
            session_id = row["session_id"]

        rows = conn.execute(
            "SELECT session_id, ts, role, msg_type, text "
            "FROM messages WHERE session_id = ? "
            "ORDER BY ts DESC LIMIT ?",
            (session_id, max(1, int(n))),  # clamp: negative LIMIT = unbounded in SQLite
        ).fetchall()
    finally:
        conn.close()

    msgs = [dict(r) for r in rows]
    msgs.reverse()  # chronological order for reading
    return {"ok": True, "session_id": session_id, "count": len(msgs), "messages": msgs}


def session(
    session_id: str,
    *,
    limit: int = 200,
    offset: int = 0,
) -> dict[str, Any]:
    """Paginated session view in chronological order."""
    if not db_path().exists():
        return {"ok": False, "reason": "index not built yet"}
    conn = connect(read_only=True)
    try:
        meta = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if meta is None:
            return {"ok": False, "reason": "session not found", "session_id": session_id}
        rows = conn.execute(
            "SELECT ts, role, msg_type, text FROM messages "
            "WHERE session_id = ? ORDER BY ts LIMIT ? OFFSET ?",
            (session_id, max(1, int(limit)), max(0, int(offset))),  # clamp negative LIMIT/OFFSET
        ).fetchall()
    finally:
        conn.close()
    return {
        "ok": True,
        "session": dict(meta),
        "offset": offset,
        "limit": limit,
        "count": len(rows),
        "messages": [dict(r) for r in rows],
    }


def list_sessions(
    *,
    project: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """List most-recent sessions for browsing."""
    if not db_path().exists():
        return {"ok": False, "reason": "index not built yet"}
    conn = connect(read_only=True)
    try:
        sql = "SELECT session_id, project, first_ts, last_ts, message_count FROM sessions "
        params: list[Any] = []
        if project:
            sql += "WHERE project = ? "
            params.append(project)
        sql += "ORDER BY last_ts DESC LIMIT ?"
        params.append(max(1, int(limit)))  # clamp: negative LIMIT = unbounded in SQLite
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return {"ok": True, "count": len(rows), "sessions": [dict(r) for r in rows]}
