"""grep mode tests — literal/regex exact-match search.

grep is the third leg next to lexical (FTS5 token) and semantic (fuzzy): it finds
EXACT strings/patterns those two miss — code symbols, paths, hashes, error text.
Isolated tmp index, so the operator's real store is never touched.
"""
from __future__ import annotations

import json

from magpie_search import indexer
from magpie_search.search import search as search_fn


def _isolated_magpi(tmp_path, monkeypatch):
    home = tmp_path / "magpi_home"; home.mkdir()
    projects = tmp_path / "claude_projects"; projects.mkdir()
    monkeypatch.setenv("MAGPIE_SEARCH_HOME", str(home))
    monkeypatch.setenv("CLAUDE_PROJECTS_DIR", str(projects))
    indexer._reset_noise_patterns_cache()
    return home, projects


def _write_session(projects_dir, session_id, *messages):
    proj_dir = projects_dir / "-test-project"; proj_dir.mkdir(exist_ok=True)
    fp = proj_dir / f"{session_id}.jsonl"
    lines = []
    for i, (role, mtype, text) in enumerate(messages):
        obj = {"type": role, "sessionId": session_id,
               "timestamp": f"2026-05-15T10:00:{i:02d}Z", "cwd": "/test",
               "message": {"role": role, "content": [{"type": mtype, "text": text}]}}
        lines.append(json.dumps(obj))
    fp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fp


def test_grep_finds_exact_symbol(tmp_path, monkeypatch):
    """An exact code symbol with punctuation — the kind FTS5 tokenization mangles."""
    _, projects = _isolated_magpi(tmp_path, monkeypatch)
    _write_session(projects, "s1",
                   ("assistant", "text", "call os.environ.get('GITHUB_PAT') here"),
                   ("assistant", "text", "totally unrelated message about cats"))
    indexer.index_all()
    r = search_fn("os.environ.get('GITHUB_PAT')", k=5, mode="grep")
    assert r["ok"] and r["count"] == 1
    assert "GITHUB_PAT" in r["hits"][0]["snippet"]


def test_grep_regex_pattern(tmp_path, monkeypatch):
    """A real regex (error code) matches; the non-matching message doesn't."""
    _, projects = _isolated_magpi(tmp_path, monkeypatch)
    _write_session(projects, "s1",
                   ("assistant", "text", "failed with Error 126: module not found"),
                   ("assistant", "text", "Error 404 page missing"))
    indexer.index_all()
    r = search_fn(r"Error\s+126", k=5, mode="grep", regex=True)
    assert r["ok"] and r["count"] == 1
    assert "126" in r["hits"][0]["snippet"]


def test_grep_invalid_regex_is_graceful(tmp_path, monkeypatch):
    """A bad pattern returns ok=False with a reason, never raises."""
    _, projects = _isolated_magpi(tmp_path, monkeypatch)
    _write_session(projects, "s1", ("user", "text", "hello"))
    indexer.index_all()
    r = search_fn("(unclosed", k=5, mode="grep", regex=True)
    assert r["ok"] is False and "regex" in r["reason"].lower()


def test_grep_no_match_returns_empty(tmp_path, monkeypatch):
    _, projects = _isolated_magpi(tmp_path, monkeypatch)
    _write_session(projects, "s1", ("user", "text", "hello world"))
    indexer.index_all()
    r = search_fn("zzz_nonexistent_zzz", k=5, mode="grep")
    assert r["ok"] and r["count"] == 0
