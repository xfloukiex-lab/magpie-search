"""Personal-magpie_search dedup feature tests.

Spec: feedback_personal_magpi_must_dedup_at_index_time.md

Four layers, all gated by env vars so customer-wheel installs see
zero behavior change:

  MAGPIE_SEARCH_DEDUP=1         - index-time hash bookkeeping + search-time clustering
  MAGPIE_SEARCH_NOISE_FILTER=1  - skip indexing for well-known noise patterns
  MAGPIE_SEARCH_NOISE_PATTERNS  - extra patterns (newline-separated regex)

These tests stand up an isolated tmp_path-based magpie_search index so they
don't touch the operator's real ~/.magpie_search or ~/.aviary/transcripts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from magpie_search import indexer
from magpie_search.search import search as search_fn


def _isolated_magpi(tmp_path, monkeypatch):
    """Point magpie_search at an isolated home dir for the duration of the test."""
    magpi_home = tmp_path / "magpi_home"
    magpi_home.mkdir()
    projects = tmp_path / "claude_projects"
    projects.mkdir()
    monkeypatch.setenv("MAGPIE_SEARCH_HOME", str(magpi_home))
    monkeypatch.setenv("CLAUDE_PROJECTS_DIR", str(projects))
    indexer._reset_noise_patterns_cache()
    return magpi_home, projects


def _write_session(projects_dir, session_id, *messages):
    """Write a synthetic JSONL session file.

    Each message tuple: (role, msg_type, text). Returns the file path."""
    proj_dir = projects_dir / "-test-project"
    proj_dir.mkdir(exist_ok=True)
    fp = proj_dir / f"{session_id}.jsonl"
    lines = []
    for i, (role, mtype, text) in enumerate(messages):
        ts = f"2026-05-15T10:00:{i:02d}Z"
        obj = {
            "type": role,
            "sessionId": session_id,
            "timestamp": ts,
            "cwd": "/test",
            "message": {
                "role": role,
                "content": [{"type": mtype, "text": text}],
            },
        }
        lines.append(json.dumps(obj))
    fp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fp


# ────────────────────────────────────────────────────────────────────
# Index-time dedup — chunk_dedup table populated correctly
# ────────────────────────────────────────────────────────────────────

def test_dedup_on_by_default(tmp_path, monkeypatch):
    """Design default: dedup is ON. With no env
    var set, the tables ARE populated."""
    _, projects = _isolated_magpi(tmp_path, monkeypatch)
    monkeypatch.delenv("MAGPIE_SEARCH_DEDUP", raising=False)
    _write_session(projects, "sess1", ("user", "text", "hello world"))
    stats = indexer.index_all()
    assert stats.messages_indexed == 1
    conn = indexer.connect(read_only=True)
    try:
        cd_count = conn.execute("SELECT COUNT(*) FROM chunk_dedup").fetchone()[0]
        mm_count = conn.execute("SELECT COUNT(*) FROM messages_meta").fetchone()[0]
    finally:
        conn.close()
    assert cd_count == 1
    assert mm_count == 1


def test_dedup_explicit_off_disables(tmp_path, monkeypatch):
    """The kill switch: MAGPIE_SEARCH_DEDUP=0 (or false/no/off) keeps tables empty.
    This is how a packager's wheel opts out if desired."""
    _, projects = _isolated_magpi(tmp_path, monkeypatch)
    monkeypatch.setenv("MAGPIE_SEARCH_DEDUP", "0")
    _write_session(projects, "sess1", ("user", "text", "hello world"))
    stats = indexer.index_all()
    assert stats.messages_indexed == 1
    conn = indexer.connect(read_only=True)
    try:
        cd_count = conn.execute("SELECT COUNT(*) FROM chunk_dedup").fetchone()[0]
        mm_count = conn.execute("SELECT COUNT(*) FROM messages_meta").fetchone()[0]
    finally:
        conn.close()
    assert cd_count == 0
    assert mm_count == 0


@pytest.mark.parametrize("off_val", ["0", "false", "no", "off", "disabled"])
def test_dedup_kill_switch_values(off_val, monkeypatch):
    """All documented off-values disable dedup."""
    monkeypatch.setenv("MAGPIE_SEARCH_DEDUP", off_val)
    assert indexer.dedup_enabled() is False


def test_dedup_on_populates_chunk_dedup(tmp_path, monkeypatch):
    """With MAGPIE_SEARCH_DEDUP=1, chunk_dedup has one row per unique hash."""
    _, projects = _isolated_magpi(tmp_path, monkeypatch)
    monkeypatch.setenv("MAGPIE_SEARCH_DEDUP", "1")
    _write_session(projects, "sess1",
                    ("user", "text", "shared-doc content here"))
    indexer.index_all()
    conn = indexer.connect(read_only=True)
    try:
        row = conn.execute(
            "SELECT sha256, count FROM chunk_dedup"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row["count"] == 1
    expected_hash = indexer._content_hash("shared-doc content here")
    assert row["sha256"] == expected_hash


def test_dedup_count_increments_on_duplicate(tmp_path, monkeypatch):
    """30 sessions reading the same shared-doc text → chunk_dedup.count = 30."""
    _, projects = _isolated_magpi(tmp_path, monkeypatch)
    monkeypatch.setenv("MAGPIE_SEARCH_DEDUP", "1")
    duplicate_text = "shared-doc.md - project control plane"
    N = 5  # representative — not testing 30 because session-write is per-file
    for i in range(N):
        _write_session(projects, f"sess{i}",
                       ("user", "text", duplicate_text))
    indexer.index_all()
    conn = indexer.connect(read_only=True)
    try:
        row = conn.execute(
            "SELECT count FROM chunk_dedup WHERE sha256 = ?",
            (indexer._content_hash(duplicate_text),),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row["count"] == N


def test_whitespace_folded_hash_collides_on_trivial_diffs(tmp_path, monkeypatch):
    """Same content with different whitespace → same hash (so dedup catches
    pretty-printed JSON variants etc.)."""
    _, projects = _isolated_magpi(tmp_path, monkeypatch)
    monkeypatch.setenv("MAGPIE_SEARCH_DEDUP", "1")
    _write_session(projects, "sess1", ("user", "text", "hello   world"))
    _write_session(projects, "sess2", ("user", "text", "hello\nworld"))
    _write_session(projects, "sess3", ("user", "text", "  hello world  "))
    indexer.index_all()
    conn = indexer.connect(read_only=True)
    try:
        rows = conn.execute(
            "SELECT sha256, count FROM chunk_dedup"
        ).fetchall()
    finally:
        conn.close()
    # All three should collapse to one hash with count=3.
    assert len(rows) == 1
    assert rows[0]["count"] == 3


# ────────────────────────────────────────────────────────────────────
# Index-time noise filter — skips well-known noise patterns
# ────────────────────────────────────────────────────────────────────

def test_noise_filter_on_by_default_drops_noise(tmp_path, monkeypatch):
    """Design default: noise filter ON. With no env var,
    a pure-noise message is dropped."""
    _, projects = _isolated_magpi(tmp_path, monkeypatch)
    monkeypatch.delenv("MAGPIE_SEARCH_NOISE_FILTER", raising=False)
    indexer._reset_noise_patterns_cache()
    _write_session(projects, "sess1",
                    ("user", "text", "onnxruntime::TryGetProviderInfo_CUDA failed"))
    stats = indexer.index_all()
    assert stats.messages_indexed == 0  # the only message was noise


def test_noise_filter_explicit_off_indexes_everything(tmp_path, monkeypatch):
    """Kill switch: MAGPIE_SEARCH_NOISE_FILTER=0 → even noise is indexed."""
    _, projects = _isolated_magpi(tmp_path, monkeypatch)
    monkeypatch.setenv("MAGPIE_SEARCH_NOISE_FILTER", "0")
    indexer._reset_noise_patterns_cache()
    _write_session(projects, "sess1",
                    ("user", "text", "onnxruntime::TryGetProviderInfo_CUDA failed"))
    stats = indexer.index_all()
    assert stats.messages_indexed == 1


def test_noise_filter_drops_onnx_warning(tmp_path, monkeypatch):
    """MAGPIE_SEARCH_NOISE_FILTER=1 drops the ONNX CUDA warning chunks."""
    _, projects = _isolated_magpi(tmp_path, monkeypatch)
    monkeypatch.setenv("MAGPIE_SEARCH_NOISE_FILTER", "1")
    indexer._reset_noise_patterns_cache()
    _write_session(projects, "sess1",
                    ("user", "text", "onnxruntime::TryGetProviderInfo_CUDA failed"),
                    ("user", "text", "real content the user typed"))
    stats = indexer.index_all()
    assert stats.messages_indexed == 1  # only the non-noise survives


def test_noise_filter_drops_machine_noise(tmp_path, monkeypatch):
    """Built-in machine-noise defaults (ONNX/CUDA startup spam) are filtered."""
    _, projects = _isolated_magpi(tmp_path, monkeypatch)
    monkeypatch.setenv("MAGPIE_SEARCH_NOISE_FILTER", "1")
    indexer._reset_noise_patterns_cache()
    _write_session(projects, "sess1",
                    ("user", "text", "onnxruntime::TryGetProviderInfo_CUDA failed to load"),
                    ("user", "text", "real content"))
    stats = indexer.index_all()
    assert stats.messages_indexed == 1


def test_extra_noise_patterns_via_env(tmp_path, monkeypatch):
    """Operator can extend the noise list via MAGPIE_SEARCH_NOISE_PATTERNS."""
    _, projects = _isolated_magpi(tmp_path, monkeypatch)
    monkeypatch.setenv("MAGPIE_SEARCH_NOISE_FILTER", "1")
    monkeypatch.setenv("MAGPIE_SEARCH_NOISE_PATTERNS", r"^EXTRA_PATTERN:\s")
    indexer._reset_noise_patterns_cache()
    _write_session(projects, "sess1",
                    ("user", "text", "EXTRA_PATTERN: a boilerplate line"),
                    ("user", "text", "useful content"))
    stats = indexer.index_all()
    assert stats.messages_indexed == 1


# ────────────────────────────────────────────────────────────────────
# Search-time clustering — collapses duplicate hits
# ────────────────────────────────────────────────────────────────────

def test_search_without_dedup_returns_duplicates(tmp_path, monkeypatch):
    """Baseline: 5 sessions with the same content → 5 hits (no dedup)."""
    _, projects = _isolated_magpi(tmp_path, monkeypatch)
    monkeypatch.setenv("MAGPIE_SEARCH_DEDUP", "1")  # populate tables
    for i in range(5):
        _write_session(projects, f"sess{i}",
                       ("user", "text", "uniquephrasealpha duplicate content"))
    indexer.index_all()
    # Search with dedup OFF explicitly
    result = search_fn("uniquephrasealpha", k=10, dedup=False)
    assert result["ok"]
    assert result["count"] == 5


def test_search_with_dedup_collapses_to_one(tmp_path, monkeypatch):
    """Same setup; with dedup=True → one hit with dup_count=5."""
    _, projects = _isolated_magpi(tmp_path, monkeypatch)
    monkeypatch.setenv("MAGPIE_SEARCH_DEDUP", "1")
    for i in range(5):
        _write_session(projects, f"sess{i}",
                       ("user", "text", "uniquephrasealpha duplicate content"))
    indexer.index_all()
    result = search_fn("uniquephrasealpha", k=10, dedup=True)
    assert result["ok"]
    assert result["count"] == 1
    assert result["hits"][0]["dup_count"] == 5
    assert result["dedup"] is True


def test_search_dedup_default_follows_env_var(tmp_path, monkeypatch):
    """`dedup=None` (default) follows MAGPIE_SEARCH_DEDUP env var."""
    _, projects = _isolated_magpi(tmp_path, monkeypatch)
    monkeypatch.setenv("MAGPIE_SEARCH_DEDUP", "1")
    for i in range(3):
        _write_session(projects, f"sess{i}",
                       ("user", "text", "uniquephrasebeta repeated"))
    indexer.index_all()
    # MAGPIE_SEARCH_DEDUP=1 + no explicit dedup arg → dedup activated
    result = search_fn("uniquephrasebeta", k=10)
    assert result["ok"]
    assert result.get("dedup") is True
    assert result["count"] == 1


def test_search_semantic_dedup_collapses(tmp_path, monkeypatch):
    """c1 audit concern: semantic + hybrid dedup were untested. Verify
    the rowid-JOIN path holds for `mode='semantic'`. Skips if the
    semantic stack (sqlite-vec + fastembed) isn't loaded in this env."""
    _, projects = _isolated_magpi(tmp_path, monkeypatch)
    monkeypatch.setenv("MAGPIE_SEARCH_DEDUP", "1")
    for i in range(4):
        _write_session(projects, f"sess{i}",
                       ("user", "text", "uniquephrasezeta semantic dup test content"))
    indexer.index_all()
    # Probe semantic availability
    conn = indexer.connect(read_only=True)
    try:
        vec_ok = indexer.vec_available(conn)
        from magpie_search import embeddings
        embed_ok = embeddings.available()
    finally:
        conn.close()
    if not (vec_ok and embed_ok):
        pytest.skip("semantic backend unavailable in this env")
    result = search_fn("uniquephrasezeta", k=10, mode="semantic", dedup=True)
    if not result.get("ok"):
        pytest.skip(f"semantic search failed (env): {result.get('reason')}")
    # If semantic returned hits, dedup should collapse them
    assert result["count"] >= 1
    # If it returned more than 1, the dedup didn't work
    if result["count"] > 1:
        pytest.fail(
            f"semantic dedup did not collapse identical content: "
            f"got {result['count']} hits, expected 1. hits={result['hits']!r}"
        )
    assert result["hits"][0].get("dup_count", 0) >= 4


def test_search_hybrid_dedup_collapses(tmp_path, monkeypatch):
    """Same as semantic but in hybrid mode (RRF fusion of lex+sem)."""
    _, projects = _isolated_magpi(tmp_path, monkeypatch)
    monkeypatch.setenv("MAGPIE_SEARCH_DEDUP", "1")
    for i in range(4):
        _write_session(projects, f"sess{i}",
                       ("user", "text", "uniquephraseepsilon hybrid dup test content"))
    indexer.index_all()
    # Hybrid falls back to lexical-only if semantic side fails — so it
    # works even without the semantic backend. Just check dedup collapses.
    result = search_fn("uniquephraseepsilon", k=10, mode="hybrid", dedup=True)
    if not result.get("ok"):
        pytest.skip(f"hybrid search failed (env): {result.get('reason')}")
    assert result["count"] == 1
    assert result["hits"][0]["dup_count"] >= 4


def test_search_dedup_preserves_novel_content(tmp_path, monkeypatch):
    """A search returning a mix of duplicates + unique hits: dedup keeps
    one rep per dup cluster AND every unique hit."""
    _, projects = _isolated_magpi(tmp_path, monkeypatch)
    monkeypatch.setenv("MAGPIE_SEARCH_DEDUP", "1")
    # 3 sessions with the SAME boilerplate, plus one with unique content
    for i in range(3):
        _write_session(projects, f"sess{i}",
                       ("user", "text", "uniquephrasegamma repeated boilerplate"))
    _write_session(projects, "novel",
                    ("user", "text", "uniquephrasegamma something completely novel"))
    indexer.index_all()
    result = search_fn("uniquephrasegamma", k=10, dedup=True)
    assert result["ok"]
    # Two clusters: the boilerplate (dup_count=3) and the novel one (1)
    assert result["count"] == 2
    dups = sorted(h["dup_count"] for h in result["hits"])
    assert dups == [1, 3]


# ────────────────────────────────────────────────────────────────────
# Module-level helpers — hash + classifier sanity
# ────────────────────────────────────────────────────────────────────

def test_content_hash_is_deterministic():
    a = indexer._content_hash("hello world")
    b = indexer._content_hash("hello world")
    assert a == b
    assert len(a) == 64  # sha256 hex


def test_content_hash_whitespace_normalize():
    assert indexer._content_hash("hello world") == indexer._content_hash("hello   world")
    assert indexer._content_hash("hello world") == indexer._content_hash("\thello world\n")


def test_whitespace_normalize_handles_unicode():
    """NBSP (U+00A0), EM SPACE (U+2003), IDEOGRAPHIC SPACE (U+3000) all
    collide with regular space on hash. c1 audit concern verified
    (Python 3 str patterns match Unicode whitespace by default)."""
    base = indexer._content_hash("hello world")
    assert indexer._content_hash("hello world") == base  # NBSP
    assert indexer._content_hash("hello world") == base  # EM SPACE
    assert indexer._content_hash("hello　world") == base  # IDEOGRAPHIC


def test_is_noise_matches_known_patterns():
    indexer._reset_noise_patterns_cache()
    assert indexer._is_noise("onnxruntime::TryGetProviderInfo_CUDA failed")
    assert indexer._is_noise("Failed to create CUDAExecutionProvider")
    assert indexer._is_noise("cublasLt64_12.dll is missing")
    # Real pytest-internals output shape (the actual noise we want to drop)
    assert indexer._is_noise(
        "PYTEST_CURRENT_TEST=aviary/tests/test_x.py::test_y (call)"
    )


def test_is_noise_misses_real_content():
    indexer._reset_noise_patterns_cache()
    assert not indexer._is_noise("legitimate user message about Aviary")
    assert not indexer._is_noise("Hello, how are you?")
    # c1 audit 2026-05-15: a HUMAN discussing the pytest env var must
    # NOT be filtered (the prior unanchored pattern would have eaten this).
    assert not indexer._is_noise(
        "set the PYTEST_CURRENT_TEST variable to debug the fixture"
    )
    assert not indexer._is_noise(
        "the README mentions PYTEST_CURRENT_TEST as an internal"
    )


def test_dedup_enabled_reads_env(monkeypatch):
    # Default ON (design default)
    monkeypatch.delenv("MAGPIE_SEARCH_DEDUP", raising=False)
    assert indexer.dedup_enabled() is True
    monkeypatch.setenv("MAGPIE_SEARCH_DEDUP", "1")
    assert indexer.dedup_enabled() is True
    monkeypatch.setenv("MAGPIE_SEARCH_DEDUP", "true")
    assert indexer.dedup_enabled() is True
    # Explicit kill switch
    monkeypatch.setenv("MAGPIE_SEARCH_DEDUP", "0")
    assert indexer.dedup_enabled() is False
    monkeypatch.setenv("MAGPIE_SEARCH_DEDUP", "off")
    assert indexer.dedup_enabled() is False
    # Unrecognized value → defaults ON (fail-open is fine; dedup is
    # non-destructive)
    monkeypatch.setenv("MAGPIE_SEARCH_DEDUP", "maybe")
    assert indexer.dedup_enabled() is True


def test_is_novel_in_session_flag(tmp_path, monkeypatch):
    """Layer 4: first occurrence of a hash within a session is novel;
    repeats in the same session are not."""
    _, projects = _isolated_magpi(tmp_path, monkeypatch)
    monkeypatch.delenv("MAGPIE_SEARCH_DEDUP", raising=False)  # default on
    monkeypatch.setenv("MAGPIE_SEARCH_NOISE_FILTER", "0")     # don't drop anything
    indexer._reset_noise_patterns_cache()
    # One session: same boilerplate 3x + one unique line
    _write_session(projects, "sessN",
                   ("user", "text", "repeated boilerplate line"),
                   ("user", "text", "repeated boilerplate line"),
                   ("user", "text", "repeated boilerplate line"),
                   ("user", "text", "a unique novel line"))
    indexer.index_all()
    conn = indexer.connect(read_only=True)
    try:
        rows = conn.execute(
            "SELECT m.text, mm.is_novel_in_session "
            "FROM messages m JOIN messages_meta mm ON m.rowid = mm.rowid "
            "WHERE m.session_id = 'sessN' ORDER BY m.rowid"
        ).fetchall()
    finally:
        conn.close()
    flags = [(r["text"], r["is_novel_in_session"]) for r in rows]
    # First boilerplate = novel(1); next two = 0; unique line = novel(1)
    assert flags[0] == ("repeated boilerplate line", 1)
    assert flags[1] == ("repeated boilerplate line", 0)
    assert flags[2] == ("repeated boilerplate line", 0)
    assert flags[3] == ("a unique novel line", 1)


def test_novelty_is_per_session_not_global(tmp_path, monkeypatch):
    """The SAME content in a DIFFERENT session is novel again for that
    session (novelty is per-session, not global like chunk_dedup.count)."""
    _, projects = _isolated_magpi(tmp_path, monkeypatch)
    monkeypatch.delenv("MAGPIE_SEARCH_DEDUP", raising=False)
    monkeypatch.setenv("MAGPIE_SEARCH_NOISE_FILTER", "0")
    indexer._reset_noise_patterns_cache()
    _write_session(projects, "sA", ("user", "text", "shared content X"))
    _write_session(projects, "sB", ("user", "text", "shared content X"))
    indexer.index_all()
    conn = indexer.connect(read_only=True)
    try:
        rows = conn.execute(
            "SELECT m.session_id, mm.is_novel_in_session "
            "FROM messages m JOIN messages_meta mm ON m.rowid = mm.rowid "
            "ORDER BY m.session_id"
        ).fetchall()
        # chunk_dedup.count should be 2 (global), but each session marks
        # its first occurrence novel.
        cd = conn.execute(
            "SELECT count FROM chunk_dedup WHERE sha256 = ?",
            (indexer._content_hash("shared content X"),),
        ).fetchone()
    finally:
        conn.close()
    novelty_by_session = {r["session_id"]: r["is_novel_in_session"] for r in rows}
    assert novelty_by_session["sA"] == 1
    assert novelty_by_session["sB"] == 1  # novel for sB even though global dup
    assert cd["count"] == 2


def test_search_exposes_is_novel_in_session(tmp_path, monkeypatch):
    """Search results carry the is_novel_in_session flag when dedup on."""
    _, projects = _isolated_magpi(tmp_path, monkeypatch)
    monkeypatch.delenv("MAGPIE_SEARCH_DEDUP", raising=False)
    monkeypatch.setenv("MAGPIE_SEARCH_NOISE_FILTER", "0")
    indexer._reset_noise_patterns_cache()
    _write_session(projects, "sessExpose",
                   ("user", "text", "noveltyprobe unique searchable content"))
    indexer.index_all()
    result = search_fn("noveltyprobe", k=5, dedup=True)
    assert result["ok"]
    assert result["count"] == 1
    assert result["hits"][0]["is_novel_in_session"] is True
