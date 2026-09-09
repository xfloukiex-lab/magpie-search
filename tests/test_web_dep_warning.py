"""A missing optional dependency must SAY so, not return an empty list quietly.

The web provider's fail-soft contract is correct — a missing extra must never
break a federated search that other sources can still answer. But an empty
result set is also exactly what a genuine no-match looks like, so before 1.3.1
`--sources web` and `deepweb` returned nothing at all on any clean install and
gave the user no reason to suspect an install problem.

These tests pin the two halves: it still returns [] (never raises), and it now
names the cause and the fix.
"""
import builtins
import sys

import pytest

from magpie_search.providers import web as web_mod
from magpie_search.providers.web import WebProvider


@pytest.fixture(autouse=True)
def _reset_warned():
    """The warn-once set is module state; a leaked entry would make a later
    test pass for the wrong reason."""
    web_mod._WARNED.clear()
    yield
    web_mod._WARNED.clear()


@pytest.fixture
def no_ddgs(monkeypatch):
    """Make `import ddgs` fail the way it does on a clean install."""
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "ddgs" or name.startswith("ddgs."):
            raise ImportError("No module named 'ddgs'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "ddgs", raising=False)


def test_missing_ddgs_still_returns_empty_and_never_raises(no_ddgs):
    assert WebProvider().search("anything", k=3) == []


def test_missing_ddgs_warns_on_stderr_with_the_fix(no_ddgs, capsys):
    WebProvider().search("anything", k=3)
    err = capsys.readouterr().err
    assert "ddgs" in err
    assert "magpie-search[web]" in err


def test_warning_is_emitted_once_not_per_call(no_ddgs, capsys):
    p = WebProvider()
    for _ in range(5):
        p.search("anything", k=3)
    err = capsys.readouterr().err
    assert err.count("magpie-search[web]") == 1


def test_health_names_the_cause_and_the_remedy(no_ddgs):
    h = WebProvider().health()
    assert h["ok"] is False
    assert "ddgs" in h["reason"]
    assert "magpie-search[web]" in h["fix"]


def test_health_is_clean_when_the_dependency_is_present():
    pytest.importorskip("ddgs")
    h = WebProvider().health()
    assert h["ok"] is True
    # A healthy provider must not carry a remedy — that would read as a fault.
    assert "reason" not in h and "fix" not in h


def test_empty_query_short_circuits_without_warning(no_ddgs, capsys):
    """An empty query returns before the import, so it is not evidence about
    the dependency and must not claim to be."""
    assert WebProvider().search("   ", k=3) == []
    assert capsys.readouterr().err == ""
