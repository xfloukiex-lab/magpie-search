"""Regression tests for the Aether audit quick-wins (2026-07-26).

One test per finding, each written to FAIL against the pre-fix code:
  R1  web provider redacts provenance.url / provenance.title
  R2  outbound web query is redacted before it leaves the machine
  R10 FilesProvider will not let `scope` become the search root
  R11 KG/Vector providers reject non-identifier table/column names
  R19 MCP reindex neither accepts nor forwards a caller-supplied `source`
"""
from __future__ import annotations

import sqlite3

import pytest

from magpie_search.providers.base import valid_ident, valid_idents
from magpie_search.providers.files import FilesProvider
from magpie_search.providers.kg import KGProvider
from magpie_search.providers.vector import VectorProvider

# redactor pattern is r"[rs]k_(?:live|test)_[A-Za-z0-9]{16,}".
# Built from parts ON PURPOSE. A literal Stripe-shaped key here trips
# GitHub push protection and every secret scanner a forker runs — a fake
# credential in source is still a real false positive, and training people
# to click "allow this secret" is worse than the inconvenience. Assembled
# at runtime it is byte-identical for the redactor, invisible to scanners.
# Do NOT collapse this back into one string literal.
SECRET = "sk" + "_live_" + "abcdefghijklmnop0123456789"


# ---- R1 / R2 -------------------------------------------------------------

class _FakeDDGS:
    """Stands in for ddgs.DDGS; records the query it was handed."""
    last_query: str | None = None

    def text(self, query, backend=None, max_results=None):
        type(self).last_query = query
        return [{
            "title": f"leak {SECRET}",
            "body": "snippet body",
            "href": f"https://example.com/callback?token={SECRET}",
        }]


@pytest.fixture
def fake_ddgs(monkeypatch):
    import sys, types
    mod = types.ModuleType("ddgs")
    mod.DDGS = _FakeDDGS
    _FakeDDGS.last_query = None
    monkeypatch.setitem(sys.modules, "ddgs", mod)
    return _FakeDDGS


def test_r1_web_provenance_is_redacted(fake_ddgs):
    from magpie_search.providers.web import WebProvider
    hits = WebProvider().search("anything", k=1)
    assert hits, "fixture should yield one hit"
    prov = hits[0].provenance
    assert SECRET not in prov["url"], "R1: secret survived in provenance.url"
    assert SECRET not in prov["title"], "R1: secret survived in provenance.title"
    assert SECRET not in hits[0].text


def test_r2_outbound_query_is_redacted(fake_ddgs, capsys):
    from magpie_search.providers.web import WebProvider
    WebProvider().search(f"why does {SECRET} return 401", k=1)
    sent = fake_ddgs.last_query
    assert sent is not None, "provider never called the engine"
    assert SECRET not in sent, "R2: raw secret was sent to the search engine"
    assert "redacted" in capsys.readouterr().err.lower(), "R2: redaction was silent"


def test_r2_clean_query_is_untouched_and_quiet(fake_ddgs, capsys):
    """Over-redaction would be its own bug — a normal query must pass through."""
    from magpie_search.providers.web import WebProvider
    WebProvider().search("sqlite fts5 tokenizer", k=1)
    assert fake_ddgs.last_query == "sqlite fts5 tokenizer"
    assert capsys.readouterr().err.strip() == ""


# ---- R10 -----------------------------------------------------------------

def test_r10_scope_cannot_become_root_without_config():
    """Unconfigured provider must stay unconfigured, not adopt caller scope."""
    root, sub = FilesProvider()._resolve_root("/etc")
    assert root is None and sub is None, "R10: scope became the search root"


def test_r10_scope_still_narrows_a_configured_root(tmp_path):
    prov = FilesProvider(root=str(tmp_path))
    root, sub = prov._resolve_root("notes")
    assert root == tmp_path, "configured root must be preserved"
    assert sub == "notes", "scope must still narrow within the root"


# ---- R11 -----------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    "facts; DROP TABLE facts",
    'facts" --',
    "facts UNION SELECT 1",
    "1facts",
    "",
    None,
    123,
])
def test_r11_identifier_validator_rejects_injection(bad):
    assert not valid_ident(bad)


@pytest.mark.parametrize("good", ["facts", "_facts", "Facts9", "a_b_c"])
def test_r11_identifier_validator_accepts_plain_names(good):
    assert valid_ident(good)
    assert valid_idents([good])


def _kg_db(tmp_path):
    db = tmp_path / "kg.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE facts (subject TEXT, predicate TEXT, object TEXT)")
    conn.execute("INSERT INTO facts VALUES ('a', 'b', 'c')")
    conn.commit()
    conn.close()
    return db


def test_r11_kg_rejects_bad_table_name(tmp_path, monkeypatch):
    """Must refuse BEFORE touching the DB.

    Asserting only `search() == []` does not prove the guard works: pre-fix,
    the malformed name produced a sqlite3.Error that was caught and turned
    into [] anyway, so such a test passes with or without the fix. The real
    property is that no statement is ever built — so assert the provider
    never opens a connection at all.
    """
    db = _kg_db(tmp_path)
    opened = []
    real_connect = sqlite3.connect
    monkeypatch.setattr(
        sqlite3, "connect",
        lambda *a, **kw: (opened.append(a[0] if a else None), real_connect(*a, **kw))[1],
    )
    prov = KGProvider(db=str(db), table="facts; DROP TABLE facts")
    assert prov.search("a", k=5) == []
    assert not opened, "R11: provider opened the DB with an unvalidated identifier"
    # and the table is of course still intact
    conn = real_connect(db)
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 1
    conn.close()


def test_r11_kg_still_works_on_a_valid_table(tmp_path):
    prov = KGProvider(db=str(_kg_db(tmp_path)), table="facts")
    assert prov.search("a", k=5), "valid config must keep working"


def test_r11_vector_rejects_bad_identifiers(tmp_path, monkeypatch):
    """Same property as the KG test: refuse before opening the DB."""
    db = tmp_path / "v.db"
    sqlite3.connect(db).close()
    opened = []
    real_connect = sqlite3.connect
    monkeypatch.setattr(
        sqlite3, "connect",
        lambda *a, **kw: (opened.append(a[0] if a else None), real_connect(*a, **kw))[1],
    )
    prov = VectorProvider(db=str(db), vec_table="vec; DROP TABLE x")
    assert prov.search("q", k=5) == []
    assert not opened, "R11: provider opened the DB with an unvalidated identifier"


# ---- R19 -----------------------------------------------------------------

def test_r19_reindex_ignores_caller_supplied_source(monkeypatch):
    import magpie_search
    from magpie_search import mcp_server

    seen = {}

    def fake_index(**kwargs):
        seen.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(magpie_search, "index", fake_index)
    mcp_server._h_reindex({"source": "/etc"})
    assert "source" not in seen or seen["source"] is None, \
        "R19: caller-supplied source reached the indexer"


def test_r19_reindex_schema_does_not_advertise_source():
    from magpie_search import mcp_server
    tool = next(t for t in mcp_server._TOOLS if t["name"] == "reindex")
    assert "source" not in tool["inputSchema"].get("properties", {}), \
        "R19: schema still offers a parameter the handler ignores"
