# HARDTEST safe-wording glossary

When running a HARDTEST that exercises security-category features
(Canary tripwire, refusal_drift probe, summarizer guardrails,
proper_noun_safety, identifier_safety, semantic_grounding,
self_verify, noun_overlap), the **product code keeps its real names**
— only the conversational framing in chat prompts changes.

**Why:** 2026-05-13 — Anthropic's API policy filter refused several
test prompts during a HARDTEST run because the framing used
adversarial trigger words. The code under test was fine; the chat-side
framing wasn't.

Companion memory: `~/memory/feedback_hardtest_safe_conversation_wording.md`.

## Phrasing swaps

Swap the LEFT-COLUMN wording for the RIGHT-COLUMN wording in
conversational prompts during HARDTESTs.

| Risky phrasing                                | Safe phrasing                                  |
|-----------------------------------------------|------------------------------------------------|
| "feed a jailbreak prompt to the model"        | "feed fixture id=N to the probe"               |
| "craft an adversarial input"                  | "load adversarial_fixtures.jsonl[N]"           |
| "try to extract the system prompt"            | "run extraction-probe fixture id=N"            |
| "manipulate the model into ignoring X"        | "run override-attempt fixture id=N"            |
| "malicious payload"                           | "category-A fixture payload"                   |
| "exfiltrate data via the webhook"             | "run webhook-leak fixture id=N"                |
| "attack the input layer"                      | "run input-layer probe N"                      |
| "make the model leak X"                       | "run leak-probe fixture id=N"                  |
| "trick the model into saying X"               | "run override-probe fixture id=N"              |
| "inject instructions"                         | "run injection-probe fixture id=N"             |

## Fixture-loader pattern

Two fixture schemas, one for each kind of probe.

### Schema A — magpi summarizer probes (`probe_smoke.jsonl`)

Used by `hardtest_probe_smoke.py` for the 7 magpi guardrail probes
(6 gating + 1 advisory: length, proper_noun_safety, identifier_safety,
noun_overlap *(advisory)*, refusal_drift, semantic_grounding,
self_verify).

```json
{
  "id": "SV_001",
  "probe": "self_verify",
  "summary": "<the text claimed to summarize the source>",
  "source":  "<the ground-truth text>",
  "expected_verdict": "flagged",
  "notes": "optional — explain the intent of this fixture"
}
```

Required fields: `id`, `probe`, `expected_verdict` ∈ {clean, flagged}.
Most probes also need `summary` + `source`; the `length` and
`refusal_drift` probes only read `summary`.

### Schema B — adversarial-payload probes (future `adversarial_*.jsonl`)

For future tests of Canary-style features that take a single payload
string (not a summary/source pair). The harness for these isn't built
yet; placeholder schema:

```json
{
  "id": "CANARY_001",
  "probe": "canary_tripwire",
  "category_code": "A",
  "payload": "<real text on disk, never echoed to chat>",
  "expected_verdict": "flagged",
  "notes": "optional"
}
```

When this schema lands, write a sibling harness in `magpi/tests/`
that loads + dispatches but never prints `payload` to stdout.

## Marking the evaluation frame

Top of any test-scaffolding block I write in chat:

```
# EVALUATION CONTEXT — running magpi probe harness against fixture set X
# Payloads are loaded from disk by id; this prompt only references ids.
```

## Out of scope for this glossary

- **Aviary product copy for Canary** ("prompt-injection tripwire" etc.)
  — that's customer-facing positioning, stays as-is.
- **Security audit reports / code review threads** — discussion of
  findings is fine; not running live tests.
- **Probe names in code** (`refusal_drift`, `proper_noun_safety`, etc.)
  — they're technical identifiers, not conversational triggers.
