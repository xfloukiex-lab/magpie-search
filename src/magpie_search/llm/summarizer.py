"""summarizer — phi3.5 session summarizer with 6-layer guardrail stack.

Public:
    summarize(session_id, n_messages=80) -> dict

Pipeline:
  1. Pull last N messages of the named session.
  2. Build redacted source text (already redacted at index time).
  3. One phi3.5 call with strict 2-3 sentence prompt.
  4. Six gating probes:
       length, proper_noun_safety, identifier_safety, refusal_drift,
       semantic_grounding (fastembed cosine), self_verify (phi3.5 YES/NO).
  5. Trust label attached. Degraded → summary suppressed.

Conservative by design: false-positive rate (showing wrong info) near zero;
false-negative rate (suppressing real summaries) high.
"""
from __future__ import annotations

import os
import time
from typing import Any

from ..search import recent as _recent
from . import client as _llm
from . import audit, guardrails


_ROLE = "magpie_search.llm.summarize"

_PROMPT_TEMPLATE = """You are summarizing the transcript below. Output ONLY a 2-3 sentence summary (between 20 and 500 characters) of what is explicitly shown in the transcript. No preamble, no markdown, no lists, no quotation marks.

STRICT RULES:
- Reference ONLY what appears in the transcript text below. Do not infer broader project context or events outside this window.
- Name specific files, features, or topics ONLY if they appear verbatim in the transcript.
- If the transcript is short or generic, write a short generic summary — do not pad with assumptions.

Transcript:
{source}

Summary:"""

_VERIFY_PROMPT = """You are a strict fact-checker. Read the source transcript and the candidate summary. Determine if EVERY claim in the summary is directly supported by the source.

Answer with ONE WORD only: YES if every claim is supported, NO if any claim is fabricated, misattributed, or not present in the source. No explanation, no preamble — just YES or NO.

Source:
{source}

Summary:
{summary}

Answer (YES or NO):"""

_MAX_SOURCE_CHARS = 6000

_GATING_PROBES = {
    "length",
    "proper_noun_safety",
    "identifier_safety",
    "refusal_drift",
    "semantic_grounding",
    "self_verify",
}


def _self_verify(summary: str, source: str) -> tuple[bool, str | None]:
    """Second phi3.5 pass acting as strict fact-checker. YES/NO output."""
    prompt = _VERIFY_PROMPT.format(
        source=source[:_MAX_SOURCE_CHARS],
        summary=summary,
    )
    result = _llm.generate(
        prompt,
        role=_ROLE + ".verify",
        num_predict=120,
        temperature=0.0,
        timeout=90.0,
    )
    if not result.get("ok"):
        return False, f"verify call failed: {result.get('reason')}"
    text = (result.get("text") or "").strip().upper()
    if not text.startswith("YES"):
        return False, f"self-verify returned: {text[:40]!r}"
    # The verify prompt demands ONE WORD (YES/NO). So an unqualified pass is
    # bare "YES" + at most trailing punctuation. ANY trailing text ("YES the
    # summary is wrong", "YES, but ...") means the verifier didn't comply and
    # must NOT be trusted — fail closed. The old 4-word hedge denylist let
    # "YES THIS IS FABRICATED" through.
    rest = text[3:].strip(" ,.:;!?\n\t")
    if rest:
        return False, f"self-verify not an unqualified YES: {text[:60]!r}"
    return True, None


def _build_source(session_id: str, n: int = 100) -> tuple[str, int]:
    """Pull last n messages, concat into a redacted source string."""
    r = _recent(n=n, session_id=session_id)
    if not r.get("ok"):
        return "", 0
    msgs = r.get("messages", [])
    parts: list[str] = []
    chars = 0
    for m in msgs:
        text = (m.get("text") or "").strip()
        if not text:
            continue
        role = m.get("role", "?")
        line = f"[{role}] {text}\n"
        if chars + len(line) > _MAX_SOURCE_CHARS:
            break
        parts.append(line)
        chars += len(line)
    return "".join(parts), len(msgs)


def summarize(*, session_id: str, n_messages: int = 80) -> dict[str, Any]:
    """Generate a 2-3 sentence summary of a session, gated by 6 probes."""
    t0 = time.time()

    source, msg_count = _build_source(session_id, n=n_messages)
    if not source:
        return {
            "ok": False, "session_id": session_id, "summary": None,
            "trust": "untrusted", "probes": {},
            "ms": int((time.time() - t0) * 1000),
            "reason": "no source messages",
        }
    if msg_count < 5:
        return {
            "ok": True, "session_id": session_id, "summary": None,
            "trust": "skipped", "probes": {},
            "ms": int((time.time() - t0) * 1000),
            "reason": f"session too short ({msg_count} msgs < 5 min)",
        }

    prompt = _PROMPT_TEMPLATE.format(source=source[:_MAX_SOURCE_CHARS])
    result = _llm.generate(
        prompt,
        role=_ROLE,
        num_predict=180,
        temperature=0.0,
        timeout=90.0,
    )
    if not result.get("ok"):
        return {
            "ok": False, "session_id": session_id, "summary": None,
            "trust": "untrusted", "probes": {},
            "ms": int((time.time() - t0) * 1000),
            "reason": result.get("reason"),
        }

    summary_text = (result.get("text") or "").strip()

    probes: dict[str, bool] = {}
    reasons: list[str] = []

    def _run(name: str, ok: bool, why: str | None) -> None:
        probes[name] = ok
        if not ok and name in _GATING_PROBES:
            reasons.append(f"[{name}] {why or ''}")

    ok, why = guardrails.summarizer_length_ok(summary_text)
    _run("length", ok, why)

    ok, why = guardrails.summarizer_noun_overlap(summary_text, source)
    _run("noun_overlap", ok, why)  # ADVISORY — not in _GATING_PROBES

    ok, why = guardrails.summarizer_proper_noun_safety(summary_text, source)
    _run("proper_noun_safety", ok, why)

    ok, why = guardrails.summarizer_identifier_safety(summary_text, source)
    _run("identifier_safety", ok, why)

    ok, why = guardrails.detect_refusal_drift(summary_text)
    _run("refusal_drift", ok, why)

    raw_threshold = (
        os.environ.get("MAGPIE_SEARCH_SEMANTIC_GROUNDING_THRESHOLD")
        or os.environ.get("AVIARY_LLM_SEM_GROUNDING_THRESHOLD")
        or "0.5"
    )
    try:
        threshold = float(raw_threshold)
    except (ValueError, TypeError):
        threshold = 0.5
    ok, why = guardrails.summarizer_semantic_grounding(
        summary_text, source, threshold=threshold,
    )
    _run("semantic_grounding", ok, why)

    ok, why = _self_verify(summary_text, source)
    _run("self_verify", ok, why)

    gating_results = {p: probes[p] for p in _GATING_PROBES if p in probes}
    all_gating_passed = all(gating_results.values())
    trust = "clean" if all_gating_passed else "degraded"

    audit.log({
        "role": _ROLE + ".probe",
        "model": "phi3.5",
        "prompt": f"session_id={session_id} n={msg_count} source_chars={len(source)}",
        "response": summary_text,
        "trust": trust,
        "fallback_fired": not all_gating_passed,
        "probe_results": probes,
        "ms": int((time.time() - t0) * 1000),
        "reason": " | ".join(r for r in reasons if r) or None,
    })

    if trust != "clean":
        return {
            "ok": True, "session_id": session_id, "summary": None,
            "trust": trust, "probes": probes,
            "ms": int((time.time() - t0) * 1000),
            "reason": " | ".join(r for r in reasons if r) or None,
        }

    return {
        "ok": True, "session_id": session_id, "summary": summary_text,
        "trust": "clean", "probes": probes,
        "ms": int((time.time() - t0) * 1000),
        "reason": None,
        "n_messages_sampled": msg_count,
        "source_chars": len(source),
    }
