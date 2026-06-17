"""Regression tests for the c1 magpie_search-hardening audit findings (2026-05-16).

CRIT-1: a read with no newline at all (one partial line, common on a
        live JSONL mid-append) used to advance the byte cursor past the
        unindexed content -> next pass sees size==prev_bytes -> silent
        permanent data loss.
HIGH-2: file truncate+reindex deleted `messages` rows but left
        `messages_meta` rows; FTS5 recycles rowids so new content
        inherited an old hash -> dedup clustered the wrong messages.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from magpie_search import indexer


def _iso(db):
    import sqlite3
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def _isolated(tmp_path, monkeypatch):
    mh = tmp_path / "mh"; mh.mkdir()
    pj = tmp_path / "pj"; pj.mkdir()
    monkeypatch.setenv("MAGPIE_SEARCH_HOME", str(mh))
    monkeypatch.setenv("CLAUDE_PROJECTS_DIR", str(pj))
    monkeypatch.setenv("MAGPIE_SEARCH_NOISE_FILTER", "0")
    indexer._reset_noise_patterns_cache()
    return mh, pj


def _row(text, sid="s", ts="2026-05-16T00:00:00Z"):
    return json.dumps({
        "type": "user", "sessionId": sid, "timestamp": ts, "cwd": "/x",
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    })


def test_crit1_partial_line_no_newline_is_not_lost(tmp_path, monkeypatch):
    """A JSONL whose tail is a complete record with NO trailing newline
    must be picked up on a subsequent pass once the newline arrives —
    never silently skipped."""
    mh, pj = _isolated(tmp_path, monkeypatch)
    proj = pj / "-t"; proj.mkdir()
    f = proj / "s.jsonl"

    # Pass 1: one complete line + a second line with NO trailing newline.
    f.write_bytes((_row("first complete line") + "\n" + _row("second no newline yet")).encode())
    s1 = indexer.index_all()
    # Only the first (newline-terminated) line is safely indexable now.
    assert s1.messages_indexed == 1

    # Pass 2: the second line gets its newline (append completes it).
    with f.open("ab") as fh:
        fh.write(b"\n")
    s2 = indexer.index_all()
    # The previously-incomplete line MUST now be indexed (not lost).
    assert s2.messages_indexed == 1
    c = _iso(indexer.db_path())
    n = c.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    got = {r[0] for r in c.execute("SELECT text FROM messages")}
    c.close()
    assert n == 2, f"expected 2 messages, got {n} — CRIT-1 data loss"
    assert any("second no newline yet" in t for t in got)


def test_crit1_pure_partial_then_completes(tmp_path, monkeypatch):
    """A read that is ONE partial line with no newline anywhere (the
    exact CRIT-1 trigger) must not advance the cursor past it."""
    mh, pj = _isolated(tmp_path, monkeypatch)
    proj = pj / "-t"; proj.mkdir()
    f = proj / "s.jsonl"
    f.write_bytes(_row("lonely partial").encode())  # no newline at all
    s1 = indexer.index_all()
    assert s1.messages_indexed == 0  # nothing complete yet
    with f.open("ab") as fh:
        fh.write(b"\n")
    s2 = indexer.index_all()
    assert s2.messages_indexed == 1  # now complete -> indexed, not lost


def test_high2_truncate_reindex_no_meta_orphan(tmp_path, monkeypatch):
    """After a file is truncated and re-indexed, no messages_meta row
    may point at a rowid whose message text doesn't match its hash."""
    mh, pj = _isolated(tmp_path, monkeypatch)
    monkeypatch.setenv("MAGPIE_SEARCH_DEDUP", "1")
    proj = pj / "-t"; proj.mkdir()
    f = proj / "s.jsonl"

    f.write_bytes((_row("original content alpha") + "\n"
                   + _row("original content beta") + "\n").encode())
    indexer.index_all()

    # Truncate + replace with different content (size < prev_bytes path).
    f.write_bytes((_row("brand new gamma") + "\n").encode())
    indexer.index_all()

    c = _iso(indexer.db_path())
    rows = c.execute(
        "SELECT m.rowid, m.text, mm.content_sha256 "
        "FROM messages m JOIN messages_meta mm ON m.rowid = mm.rowid"
    ).fetchall()
    c.close()
    # Every surviving messages_meta row must hash-match its CURRENT text.
    for rowid, text, sha in rows:
        assert sha == indexer._content_hash(text), (
            f"HIGH-2 orphan: rowid {rowid} text={text!r} bound to stale "
            f"hash {sha[:12]} (expected {indexer._content_hash(text)[:12]})"
        )
    # And no messages_meta row should reference a now-deleted rowid.
    c = _iso(indexer.db_path())
    orphans = c.execute(
        "SELECT COUNT(*) FROM messages_meta mm "
        "WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.rowid=mm.rowid)"
    ).fetchone()[0]
    c.close()
    assert orphans == 0, f"{orphans} messages_meta rows orphaned after truncate"
