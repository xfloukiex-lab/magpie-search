"""hardtest_probe_smoke — HARDTEST harness for magpie_search summarizer probes.

Proves the magpie_search probe suite runs end-to-end via the fixture-loader
workflow without trigger-vocabulary leaking into the conversation
prompt. Payloads live in fixtures/*.jsonl; this script loads them by
id and prints only verdicts (never source/summary text).

Run from the magpie_search/ directory:

    python tests/hardtest_probe_smoke.py
    python tests/hardtest_probe_smoke.py --probe self_verify
    python tests/hardtest_probe_smoke.py --id SV_001
    python tests/hardtest_probe_smoke.py --fixture tests/fixtures/probe_smoke.jsonl

Tests live outside src/ (standard src-layout), so there's no
`python -m magpie_search.tests.*` invocation form. Run as a loose script.

Companion files:
- tests/fixtures/probe_smoke.jsonl — fixture set (schema in
  hardtest_glossary.md, schema A)
- tests/hardtest_glossary.md       — phrasing rules + fixture schemas
- ~/memory/feedback_hardtest_safe_conversation_wording.md  — feedback memory

For new probe categories: add an entry to _PROBE_FN + matching
fixtures. For adversarial-payload probes (Canary-style), write a
sibling harness rather than overloading this one.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# Make magpie_search importable when run as a loose script.
_ROOT = Path(__file__).resolve().parent.parent / "src"
if _ROOT.exists():
    sys.path.insert(0, str(_ROOT))

from magpie_search.llm.guardrails import (  # noqa: E402
    summarizer_proper_noun_safety,
    summarizer_identifier_safety,
    summarizer_noun_overlap,
    summarizer_length_ok,
    summarizer_semantic_grounding,
    detect_refusal_drift,
)
# _self_verify is module-private (leading underscore) because it's tightly
# coupled to summarizer.py's prompt template + _llm client. Importing it
# directly is acceptable for test-tool scope; the alternative is calling
# the full summarize() pipeline which is too heavy for unit testing.
from magpie_search.llm.summarizer import _self_verify  # noqa: E402


_PROBE_FN = {
    "proper_noun_safety": lambda f: summarizer_proper_noun_safety(f["summary"], f["source"]),
    "identifier_safety":  lambda f: summarizer_identifier_safety(f["summary"], f["source"]),
    "noun_overlap":       lambda f: summarizer_noun_overlap(f["summary"], f["source"]),
    "length":             lambda f: summarizer_length_ok(f["summary"]),
    "refusal_drift":      lambda f: detect_refusal_drift(f["summary"]),
    "semantic_grounding": lambda f: summarizer_semantic_grounding(f["summary"], f["source"]),
    "self_verify":        lambda f: _self_verify(f["summary"], f["source"]),
}

_REQUIRED_FIELDS = {"id", "probe", "expected_verdict"}
_VALID_VERDICTS = {"clean", "flagged"}


def _validate(fixture: dict, lineno: int) -> str | None:
    """Return error string if fixture is malformed, else None."""
    missing = _REQUIRED_FIELDS - set(fixture)
    if missing:
        return f"line {lineno}: missing required field(s) {sorted(missing)}"
    if fixture["expected_verdict"] not in _VALID_VERDICTS:
        return (f"line {lineno}: id={fixture.get('id')!r} has expected_verdict="
                f"{fixture['expected_verdict']!r}; must be one of {sorted(_VALID_VERDICTS)}")
    probe = fixture["probe"]
    if probe not in _PROBE_FN:
        return f"line {lineno}: id={fixture['id']!r} uses unknown probe {probe!r}"
    # Most probes need summary + source; length/refusal_drift only need summary.
    needs_source = probe not in {"length", "refusal_drift"}
    if "summary" not in fixture:
        return f"line {lineno}: id={fixture['id']!r} missing 'summary' (required for probe {probe!r})"
    if needs_source and "source" not in fixture:
        return f"line {lineno}: id={fixture['id']!r} missing 'source' (required for probe {probe!r})"
    return None


def _is_cache_miss(reason: str | None) -> bool:
    return bool(reason) and "special_tokens_map" in reason


def _load(fixture_path: Path) -> list[dict]:
    out: list[dict] = []
    for lineno, line in enumerate(fixture_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"FATAL: {fixture_path.name} line {lineno}: invalid JSON: {e}")
    return out


def run(fixture_path: Path, *, only_probe: str | None = None, only_id: str | None = None) -> int:
    fixtures = _load(fixture_path)

    # Schema-validate every fixture upfront — fail fast on typos.
    errors = []
    for i, f in enumerate(fixtures, 1):
        err = _validate(f, i)
        if err:
            errors.append(err)
    if errors:
        print("FATAL: fixture schema errors:")
        for e in errors:
            print(f"  - {e}")
        return 2

    # Apply filters.
    filtered = fixtures
    if only_probe:
        filtered = [f for f in filtered if f["probe"] == only_probe]
    if only_id:
        filtered = [f for f in filtered if f["id"] == only_id]
    if not filtered:
        print(f"No fixtures matched filters (probe={only_probe!r}, id={only_id!r}).")
        return 0

    header = f"HARDTEST probe_smoke - {len(filtered)} fixture(s) from {fixture_path.name}"
    if only_probe or only_id:
        header += f" (filtered: probe={only_probe!r}, id={only_id!r})"
    print(header)
    print("=" * 78)

    passed = failed = 0
    per_probe: dict[str, dict] = defaultdict(lambda: {"pass": 0, "fail": 0, "elapsed_ms": 0})
    cache_miss_hinted = False

    for f in filtered:
        fid, probe, expected = f["id"], f["probe"], f["expected_verdict"]
        fn = _PROBE_FN[probe]
        t0 = time.time()
        ok, reason = fn(f)
        elapsed_ms = int((time.time() - t0) * 1000)
        actual = "clean" if ok else "flagged"
        match = "PASS" if actual == expected else "FAIL"
        per_probe[probe]["elapsed_ms"] += elapsed_ms
        if match == "PASS":
            passed += 1
            per_probe[probe]["pass"] += 1
        else:
            failed += 1
            per_probe[probe]["fail"] += 1
        # Deliberately NOT printing source/summary — only ids + verdicts.
        print(f"  {fid:8} [{probe:18}] expected={expected:8} actual={actual:8} {match}  ({elapsed_ms:>5} ms)")
        if match == "FAIL" and reason:
            print(f"    reason: {reason}")
            if _is_cache_miss(reason) and not cache_miss_hinted:
                print(f"    HINT: fastembed cache snapshot incomplete. Try:")
                print(f"          cp ~/.aviary/models/models--qdrant--all-MiniLM-L6-v2-onnx/snapshots/*/special_tokens_map.json \\")
                print(f"             ~/.magpie_search/models/models--qdrant--all-MiniLM-L6-v2-onnx/snapshots/*/")
                cache_miss_hinted = True

    print("=" * 78)
    print(f"PER-PROBE SUMMARY:")
    for probe in sorted(per_probe):
        s = per_probe[probe]
        total = s["pass"] + s["fail"]
        mean_ms = s["elapsed_ms"] // max(total, 1)
        print(f"  {probe:18}  {s['pass']:>2}/{total:<2} pass   mean {mean_ms:>5} ms")
    print(f"RESULT: {passed} passed, {failed} failed, {passed+failed} total")
    return 0 if failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    default_fixture = Path(__file__).resolve().parent / "fixtures" / "probe_smoke.jsonl"
    p = argparse.ArgumentParser(description="HARDTEST probe smoke harness for magpie_search probes.")
    p.add_argument("--fixture", type=Path, default=default_fixture,
                   help=f"path to JSONL fixture file (default: {default_fixture.name})")
    p.add_argument("--probe", type=str, default=None,
                   help=f"only run fixtures for this probe (choices: {sorted(_PROBE_FN)})")
    p.add_argument("--id", type=str, default=None, dest="only_id",
                   help="only run the fixture with this id")
    args = p.parse_args(argv)
    if args.probe and args.probe not in _PROBE_FN:
        p.error(f"unknown --probe {args.probe!r}; choices: {sorted(_PROBE_FN)}")
    if not args.fixture.exists():
        p.error(f"--fixture path does not exist: {args.fixture}")
    return run(args.fixture, only_probe=args.probe, only_id=args.only_id)


if __name__ == "__main__":
    raise SystemExit(main())
