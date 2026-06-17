"""Regression tests for the keystone retrieval path (verified bugs,
2026-05-16).

1. `_sanitize_query` must not let a hyphenated token reach FTS5 raw —
   FTS5 parses the part after `-` as a NOT/column filter and hard-errors
   with "no such column: <x>", crashing the whole search role.
2. The aviary-magpi `transcript.search` worker must accept `n=` /
   `limit=` as aliases for `k=` (callers + CLAUDE.md examples use `n=`).
"""
from __future__ import annotations

import pytest

from magpie_search.search import _sanitize_query


@pytest.mark.parametrize("q", [
    "mcp-server",
    "9-page diagram",
    "uniquephrase-abc",
    "a-b-c chained-hyphens",
    "trailing-",
])
def test_hyphenated_query_does_not_emit_raw_fts5_operator(q):
    """No bare hyphen survives into the FTS5 MATCH string outside quotes.
    Quoted tokens make FTS5 treat the hyphen literally."""
    s = _sanitize_query(q)
    # Every emitted term is double-quoted (or it's an empty/escape-hatch
    # passthrough). The crash happens only on an UNQUOTED hyphen token,
    # so assert no unquoted '-' adjacent to word chars remains.
    import re
    # strip quoted spans, then there must be no `\w-\w` left
    unquoted = re.sub(r'"[^"]*"', "", s)
    assert not re.search(r"\w-\w", unquoted), (
        f"unquoted hyphen survived sanitize: query={q!r} -> {s!r}"
    )


def test_sanitize_quotes_plain_tokens():
    s = _sanitize_query("mcp-server crash")
    assert s == '"mcp-server" OR "crash"', s


def test_power_user_syntax_still_passes_through():
    # Explicit operators / wildcards / quotes must NOT be quoted away.
    assert _sanitize_query('foo OR bar') == "foo OR bar"
    assert _sanitize_query('"exact phrase"') == '"exact phrase"'
    assert _sanitize_query("prefix*") == "prefix*"
    assert _sanitize_query("+must -mustnot") == "+must -mustnot"


def test_search_does_not_crash_on_hyphen_live(tmp_path, monkeypatch):
    """End-to-end: a hyphenated query returns ok=True, never an
    'fts5 error: no such column' crash, even on an empty index."""
    import json
    from magpie_search import indexer
    from magpie_search.search import search as search_fn

    mh = tmp_path / "mh"; mh.mkdir()
    pj = tmp_path / "pj"; pj.mkdir()
    proj = pj / "-t"; proj.mkdir()
    (proj / "s.jsonl").write_text(json.dumps({
        "type": "user", "sessionId": "s", "timestamp": "2026-05-16T00:00:00Z",
        "cwd": "/x", "message": {"role": "user", "content": [
            {"type": "text", "text": "the mcp-server handles 9-page reports"}]},
    }) + "\n", encoding="utf-8")
    monkeypatch.setenv("MAGPIE_SEARCH_HOME", str(mh))
    monkeypatch.setenv("CLAUDE_PROJECTS_DIR", str(pj))
    monkeypatch.setenv("MAGPIE_SEARCH_NOISE_FILTER", "0")
    indexer._reset_noise_patterns_cache()
    indexer.index_all()
    r = search_fn("mcp-server", k=5)
    assert r["ok"] is True, f"hyphen query crashed: {r}"
    assert r["count"] >= 1


def test_worker_accepts_n_and_limit_aliases():
    """aviary-magpi transcript.search worker maps n/limit -> k."""
    from aviary_magpi import workers
    import inspect
    sig = inspect.signature(workers.search)
    for p in ("n", "limit", "dedup"):
        assert p in sig.parameters, f"worker.search missing {p!r} param"
