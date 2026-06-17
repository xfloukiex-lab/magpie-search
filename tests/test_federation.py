"""Tests for multi-source ("federated") search: fan-out, trust-weighted fusion,
dedup, budget, fail-open, and redaction on non-transcript output."""
from __future__ import annotations

import sqlite3
import time

import pytest

import magpie_search
from magpie_search.federation import federated_search, estimate_tokens
from magpie_search.providers.base import Hit, Provider, TrustTier


class FakeProvider(Provider):
    """Returns a fixed list of texts as Hits at this provider's trust tier."""
    category = "fake"
    default_trust = TrustTier.LEAD

    def __init__(self, name, texts, *, trust=None, raise_exc=False, sleep=0.0):
        super().__init__(name, trust=trust)
        self._texts = texts
        self._raise = raise_exc
        self._sleep = sleep

    def search(self, query, *, budget_tokens=None, scope=None, k=10):
        if self._raise:
            raise RuntimeError("provider exploded")
        if self._sleep:
            time.sleep(self._sleep)
        return [
            Hit(text=t, source=self.name, trust=self.trust, category=self.category,
                score=1.0)
            for t in self._texts[:k]
        ]


# --------------------------------------------------------------------------- #
# Fan-out + tagging
# --------------------------------------------------------------------------- #

def test_federation_fans_out_and_tags_source_and_trust():
    a = FakeProvider("docs", ["alpha note", "beta note"], trust="reference")
    b = FakeProvider("chat", ["gamma msg"], trust="lead")
    res = federated_search("note", [a, b], k=10)

    assert res["ok"] is True
    assert res["sources"] == {"docs": 2, "chat": 1}
    seen_sources = {h["source"] for h in res["hits"]}
    assert seen_sources == {"docs", "chat"}
    for h in res["hits"]:
        assert h["trust"] in {"reference", "lead"}
        assert "category" in h and "provenance" in h


# --------------------------------------------------------------------------- #
# Trust-weighted ranking — the differentiator, with the GAP-5 guard
# --------------------------------------------------------------------------- #

def test_trust_weight_flips_order_and_is_actually_applied():
    # Both hits arrive at rank 1 of their provider, so the ONLY thing that can
    # separate them is the trust weight. Fact (weight 3.0) must rank above lead
    # (weight 1.0), and its fused score must be ~3x the lead's. If the weights
    # were declared-but-not-applied (magpie's GAP-5 bug class), both scores
    # would be equal (1/61) and the ratio would be 1.0 — this asserts otherwise.
    lead = FakeProvider("chat", ["LEADTEXT"], trust="lead")
    fact = FakeProvider("kg", ["FACTTEXT"], trust="fact")
    res = federated_search("x", [lead, fact], k=10)
    assert res["hits"][0]["source"] == "kg"
    by_src = {h["source"]: h for h in res["hits"]}
    ratio = by_src["kg"]["rrf_score"] / by_src["chat"]["rrf_score"]
    assert 2.5 < ratio < 3.5, (
        f"trust weights must actually be applied (GAP-5 guard); ratio={ratio}"
    )


def test_min_trust_filters_and_reports_dropped():
    lead = FakeProvider("chat", ["l1", "l2"], trust="lead")
    fact = FakeProvider("kg", ["f1"], trust="fact")
    res = federated_search("x", [lead, fact], k=10, min_trust="reference")
    assert all(h["trust"] == "fact" for h in res["hits"])
    assert res["dropped"]["min_trust"] == 2  # both lead hits dropped


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #

def test_budget_trims_to_top_n_within_budget():
    # Three distinct 36-char texts -> ceil(36/3.6)=10 tokens each.
    texts = ["A" * 36, "B" * 36, "C" * 36]
    assert all(estimate_tokens(t) == 10 for t in texts)
    p = FakeProvider("docs", texts, trust="reference")
    res = federated_search("x", [p], k=10, budget_tokens=25)
    assert res["count"] == 2
    assert res["used_tokens"] <= 25
    assert res["dropped"]["budget"] == 1
    # the two kept are the top-2 (provider order preserved by rank).
    assert [h["text"] for h in res["hits"]] == ["A" * 36, "B" * 36]


# --------------------------------------------------------------------------- #
# Cross-source dedup
# --------------------------------------------------------------------------- #

def test_dedup_keeps_highest_trust_and_records_also_in():
    same = "identical solution text"
    lead = FakeProvider("chat", [same], trust="lead")
    fact = FakeProvider("kg", [same], trust="fact")
    res = federated_search("solution", [lead, fact], k=10)
    assert res["count"] == 1
    survivor = res["hits"][0]
    assert survivor["trust"] == "fact"          # highest-trust copy wins
    assert survivor["source"] == "kg"
    assert "chat" in survivor.get("also_in", [])  # corroboration recorded
    assert res["dropped"]["dedup"] == 1


# --------------------------------------------------------------------------- #
# Fail-open: one bad provider never breaks the call
# --------------------------------------------------------------------------- #

def test_failopen_on_provider_exception():
    good = FakeProvider("docs", ["good hit"], trust="reference")
    bad = FakeProvider("broken", ["x"], raise_exc=True)
    res = federated_search("hit", [good, bad], k=10)
    assert res["ok"] is True
    assert [h["source"] for h in res["hits"]] == ["docs"]
    assert "broken" in (res.get("errors") or {})


def test_failopen_on_provider_timeout():
    good = FakeProvider("docs", ["good hit"], trust="reference")
    slow = FakeProvider("slow", ["x"], sleep=0.5)
    res = federated_search("hit", [good, slow], k=10, timeout=0.1)
    assert res["ok"] is True
    assert any(h["source"] == "docs" for h in res["hits"])
    assert "slow" in (res.get("errors") or {})


# --------------------------------------------------------------------------- #
# Files provider — live dir search, redaction, and the public search() surface
# --------------------------------------------------------------------------- #

def test_files_provider_redacts_secrets(tmp_path):
    (tmp_path / "notes.md").write_text(
        "# Deploy\n\ndeployment secret AKIAIOSFODNN7EXAMPLE used here\n",
        encoding="utf-8",
    )
    res = federated_search(
        "deployment secret",
        [{"type": "files", "name": "docs", "root": str(tmp_path), "trust": "reference"}],
        k=5,
    )
    assert res["count"] >= 1
    blob = " ".join(h["text"] for h in res["hits"])
    assert "AKIAIOSFODNN7EXAMPLE" not in blob
    assert "[REDACTED" in blob


def test_files_provider_scope_narrows_subpath(tmp_path):
    (tmp_path / "projects").mkdir()
    (tmp_path / "projects" / "alpha.md").write_text(
        "alpha widget design", encoding="utf-8")
    (tmp_path / "other.md").write_text("widget elsewhere", encoding="utf-8")
    res = federated_search(
        "widget",
        [{"type": "files", "name": "mem", "root": str(tmp_path), "trust": "reference"}],
        k=10, scope="projects",
    )
    paths = {h["provenance"]["path"] for h in res["hits"]}
    assert paths == {"projects/alpha.md"}


def test_public_search_federated_via_dict_spec(tmp_path):
    (tmp_path / "a.md").write_text("federated entry point works", encoding="utf-8")
    res = magpie_search.search(
        "federated",
        sources=[{"type": "files", "name": "docs", "root": str(tmp_path)}],
        budget_tokens=500,
    )
    assert res["ok"] is True
    assert "sources" in res and "docs" in res["sources"]


# --------------------------------------------------------------------------- #
# Generic kg + vector providers — blank slots; user supplies the db
# --------------------------------------------------------------------------- #

def test_available_types_includes_generic_providers():
    from magpie_search.providers import available_types
    assert {"transcripts", "files", "vector", "kg"} <= set(available_types())


def test_unconfigured_kg_and_vector_return_empty():
    # No db configured -> blank slot -> zero hits, never an error.
    res = federated_search(
        "anything",
        [{"type": "vector", "name": "vstore"}, {"type": "kg", "name": "facts"}],
        k=5,
    )
    assert res["ok"] is True
    assert res["count"] == 0
    assert res["sources"] == {"vstore": 0, "facts": 0}


def test_kg_provider_returns_facts_and_redacts(tmp_path):
    db = tmp_path / "facts.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE facts(subject TEXT, predicate TEXT, object TEXT)")
    conn.execute("INSERT INTO facts VALUES('retry','uses','exponential backoff with jitter')")
    conn.execute("INSERT INTO facts VALUES('deploykey','equals','sk-ant-shouldnotappear000000000000')")
    conn.commit()
    conn.close()

    res = federated_search("backoff retry", [{"type": "kg", "name": "facts", "db": str(db)}], k=5)
    assert res["count"] >= 1
    top = res["hits"][0]
    assert top["trust"] == "fact"
    assert "subject" in top["provenance"]

    # secret in a fact value must be redacted on output
    res2 = federated_search("deploykey equals", [{"type": "kg", "name": "facts", "db": str(db)}], k=5)
    blob = " ".join(h["text"] for h in res2["hits"])
    assert "sk-ant-shouldnotappear" not in blob
    assert "[REDACTED" in blob


def _vec_stack_ok() -> bool:
    try:
        from magpie_search import embeddings
        from magpie_search.indexer import _load_vec_extension, vec_available
        if not embeddings.available():
            return False
        c = sqlite3.connect(":memory:")
        _load_vec_extension(c)
        ok = vec_available(c)
        c.close()
        return ok
    except Exception:
        return False


@pytest.mark.skipif(not _vec_stack_ok(), reason="sqlite-vec / embedding model unavailable")
def test_vector_provider_semantic(tmp_path):
    from magpie_search import embeddings
    from magpie_search.indexer import _load_vec_extension

    db = tmp_path / "vec.db"
    conn = sqlite3.connect(str(db))
    _load_vec_extension(conn)
    conn.execute("CREATE TABLE documents(text TEXT)")
    conn.execute("CREATE VIRTUAL TABLE vec USING vec0(embedding float[384])")
    texts = ["the cat sat on the mat",
             "exponential backoff retry strategy for the ingest worker"]
    for i, t in enumerate(texts, start=1):
        conn.execute("INSERT INTO documents(rowid, text) VALUES(?,?)", (i, t))
        conn.execute("INSERT INTO vec(rowid, embedding) VALUES(?,?)",
                     (i, embeddings.embed_one(t)))
    conn.commit()
    conn.close()

    res = federated_search("retry with backoff",
                           [{"type": "vector", "name": "vstore", "db": str(db)}], k=2)
    assert res["count"] >= 1
    assert res["hits"][0]["trust"] == "lead"
    assert "backoff" in res["hits"][0]["text"].lower()  # semantic match wins


def test_back_compat_transcripts_only_keeps_classic_path():
    # sources=["transcripts"] (and sources=None) must take the classic
    # single-source path, never the federated one. The federated result always
    # carries a "dropped" breakdown + a "sources" summary dict; the classic one
    # never does — that's the robust discriminator regardless of index state.
    res = magpie_search.search("anything", sources=["transcripts"])
    assert "dropped" not in res
    assert not isinstance(res.get("sources"), dict)
