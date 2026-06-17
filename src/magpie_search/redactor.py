"""redactor — strip secrets from transcript text before indexing.

Hybrid pattern matches industry practice for 2026: high-speed regex pass
for known shapes (keys, addresses, tokens). Optional Presidio NER pass is
gated behind `redact(text, ner=True)` so the base path has no heavy deps.

Each match is replaced with `[REDACTED:<kind>]`. Patterns are tuned to
avoid false positives in conversational text (we deliberately skip generic
6-digit OTP matching without context, and skip email addresses since
discussion of legitimate emails is routine).
"""
from __future__ import annotations

import re
from typing import Pattern


REDACTION_PATTERNS: list[tuple[str, Pattern[str]]] = [
    ("pem_private_key", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY[-A-Z ]*-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY[-A-Z ]*-----"
    )),
    ("jwt", re.compile(
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
    )),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("aws_secret", re.compile(
        r"(?i)aws[_\- ]?secret[_\- ]?access[_\- ]?key[\"'\s:=]+[\"']?([A-Za-z0-9/+=]{40})"
    )),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    # Fine-grained PATs have their own prefix (github_pat_…) — a different
    # shape from the classic gh[pousr]_ above.
    ("github_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    # Slack token bodies legitimately contain underscores; the class must
    # include `_` or the tail after the first `_` leaks in plaintext.
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9_-]{10,}\b")),
    ("stripe_key", re.compile(r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{16,}\b")),
    ("sendgrid_key", re.compile(r"\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\b")),
    ("twilio_sid", re.compile(r"\b(?:AC|SK)[0-9a-fA-F]{32}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("google_oauth_token", re.compile(r"\bya29\.[A-Za-z0-9_-]{20,}\b")),
    ("google_oauth_client_secret", re.compile(r"\bGOCSPX-[A-Za-z0-9_-]{20,}\b")),
    # Developer-registry tokens — common in dev transcripts as bare pastes
    # (the uppercase KEY=value form is also caught by dotenv_secret; these
    # cover bare / prose / lowercase-assignment cases). hf_ is on-point:
    # magpie-search itself pulls models from Hugging Face.
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{34,}\b")),
    ("npm_token", re.compile(r"\bnpm_[A-Za-z0-9]{36}\b")),
    ("pypi_token", re.compile(r"\bpypi-[A-Za-z0-9_-]{16,}\b")),
    ("gitlab_pat", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b")),
    ("slack_webhook", re.compile(r"https://hooks\.slack\.com/services/\S+")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    # Modern OpenAI keys (sk-proj-/sk-svcacct-/sk-admin-, default since 2024)
    # contain a hyphen at offset 3 that the plain openai_key class below
    # can't span; match the scoped formats specifically, first.
    ("openai_scoped_key", re.compile(r"\bsk-(?:proj|svcacct|admin)-[A-Za-z0-9_-]{20,}\b")),
    # Real OpenAI keys are ~48+ chars after the `sk-` prefix; 32 is a
    # safer floor than 20 to avoid matching dev-mock tokens like
    # `sk-fa-mock-token-for-test123`.
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{32,}\b")),
    # Bearer tokens are base64-ish: include / + = or the tail leaks (and a
    # leading special char can defeat the {20,} floor entirely).
    ("bearer_token", re.compile(r"(?i)(\bBearer\s+)([A-Za-z0-9._/+=-]{20,})")),
    ("eth_address", re.compile(r"\b0x[a-fA-F0-9]{40}\b")),
    ("btc_address", re.compile(
        r"\b(?:bc1[a-z0-9]{39,59}|[13][a-zA-HJ-NP-Z0-9]{25,34})\b"
    )),
    # Credentials embedded in a connection string / URL authority:
    # scheme://user:PASSWORD@host . Redact only the password segment.
    ("url_basic_auth", re.compile(
        r"\b([a-zA-Z][\w+.-]*://[^\s:/@]*:)([^\s:/@]+)(@)"  # user may be empty: redis://:pass@
    )),
    # dotenv_secret comes first because it's more specific (line-anchored,
    # uppercase var name). Otherwise api_key_assignment would steal lines
    # like `API_KEY=hunter2longstring` and mis-label them. Var-name floor is
    # 2 chars (`[A-Z]` + `{1,}`) so short secret vars (KEY=, DB=, PWD=, AWS=)
    # are caught; the 20-char value floor suppresses false positives.
    ("dotenv_secret", re.compile(
        r"(?m)^((?:export\s+)?[A-Z][A-Z0-9_]{1,}=)([A-Za-z0-9_/+=.-]{20,})\b"
    )),
    # Passwords specifically: strong passwords use special chars (@!#$%…)
    # and may be short — exactly the ones the alnum-only, 16-floor
    # api_key_assignment below misses. Capture to a whitespace/quote
    # boundary with a low floor. Runs first so it claims password= lines.
    ("password_assignment", re.compile(
        r"(?i)(\b(?:password|passwd|pwd)\b[\"'\s:=]+[\"']?)([^\s\"']{3,})"
    )),
    ("api_key_assignment", re.compile(
        r"(?i)(\b(?:api[_-]?key|api[_-]?secret|access[_-]?token|client[_-]?secret|"
        r"secret[_-]?access[_-]?key|secret[_-]?key|private[_-]?key|"
        r"password|passwd|pwd)[\"'\s:=]+[\"']?)([A-Za-z0-9_/+=.-]{16,})"
    )),
    ("otp_with_context", re.compile(
        r"(?i)(\b(?:otp|verification|verify|2fa|two[\s\-]factor|code)[\s:]+)\b(\d{6,8})\b"
    )),
]

# Patterns whose first capture group is a contextual prefix (e.g. the var
# name, the `Bearer ` keyword, `password=`) that we want to PRESERVE in
# the output so a reader can see WHICH kind of secret was redacted. The
# secret itself (the rest of the match) becomes [REDACTED:<kind>].
_PREFIX_PRESERVING: frozenset[str] = frozenset({
    "bearer_token", "api_key_assignment", "password_assignment",
    "dotenv_secret", "otp_with_context",
})

# Patterns where BOTH a leading group(1) and a trailing group(3) bracket
# the secret (group 2) and must be preserved — e.g. url_basic_auth keeps
# `scheme://user:` and `@` so the redaction is readable in context.
_PREFIX_SUFFIX_PRESERVING: frozenset[str] = frozenset({
    "url_basic_auth",
})


def redact(text: str, *, ner: bool = False) -> str:
    """Return text with secrets replaced by [REDACTED:<kind>] tokens.

    `ner=True` opts into a Presidio NER pass for names/locations/etc. Falls
    back silently if Presidio isn't installed.
    """
    if not text:
        return text
    out = text
    for kind, pat in REDACTION_PATTERNS:
        if kind in _PREFIX_PRESERVING:
            # Preserve the leading capture group (var name / "Bearer " /
            # "password=" / "OTP ") so reader knows WHAT was redacted.
            out = pat.sub(lambda m, k=kind: f"{m.group(1)}[REDACTED:{k}]", out)
        elif kind in _PREFIX_SUFFIX_PRESERVING:
            # Preserve group(1) and group(3); redact only group(2).
            out = pat.sub(
                lambda m, k=kind: f"{m.group(1)}[REDACTED:{k}]{m.group(3)}", out
            )
        else:
            out = pat.sub(f"[REDACTED:{kind}]", out)
    if ner:
        out = _presidio_pass(out)
    return out


import threading as _threading

_PRESIDIO_LOCK = _threading.Lock()
_PRESIDIO_CACHE: dict[str, object] = {}


def _presidio_pass(text: str) -> str:
    """Optional Presidio NER pass — silently no-ops if unavailable.

    `AnalyzerEngine` + `AnonymizerEngine` are heavy (~seconds to construct;
    they load spaCy models). Cache the engines at module level so repeated
    calls (e.g. an index pass) don't pay the construction cost each time."""
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore
        from presidio_anonymizer import AnonymizerEngine  # type: ignore
    except ImportError:
        return text
    with _PRESIDIO_LOCK:
        analyzer = _PRESIDIO_CACHE.get("analyzer")
        anonymizer = _PRESIDIO_CACHE.get("anonymizer")
        if analyzer is None:
            analyzer = AnalyzerEngine()
            _PRESIDIO_CACHE["analyzer"] = analyzer
        if anonymizer is None:
            anonymizer = AnonymizerEngine()
            _PRESIDIO_CACHE["anonymizer"] = anonymizer
    results = analyzer.analyze(text=text, language="en")  # type: ignore[attr-defined]
    if not results:
        return text
    return anonymizer.anonymize(text=text, analyzer_results=results).text  # type: ignore[attr-defined]


def redaction_audit(text: str) -> dict[str, int]:
    """Count which patterns matched. Useful for tests and tuning."""
    counts: dict[str, int] = {}
    for kind, pat in REDACTION_PATTERNS:
        n = len(pat.findall(text))
        if n:
            counts[kind] = n
    return counts
