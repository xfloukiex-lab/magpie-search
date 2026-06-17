"""Redactor regression suite.

The redactor is magpie_search's stated security boundary: transcripts contain
secrets, and anything it misses gets indexed in plaintext, surfaced in search
results / LLM summaries, and backed up to the remote target. An
internal bug-bounty and an independent audit found the redactor had NO
test coverage and leaked several high-value secret classes. This suite pins:

  * one positive case per secret class (must be redacted),
  * partial-leak bypasses found in the audit (full token must be gone),
  * no-false-positive cases (benign text must be untouched),
  * the indexer truncation-before-redaction leak (CRIT-3) end to end.
"""
from __future__ import annotations

import magpie_search.indexer as indexer
from magpie_search.redactor import redact


def _leaks(secret: str, redacted: str) -> bool:
    """True if the secret material survives anywhere in the output."""
    return secret in redacted


# --- positive coverage: every class must be redacted ----------------------

POSITIVE = {
    "pem_private_key":
        "-----BEGIN PRIVATE KEY-----\nMIIBVwIBADANBgkq\nhkiG9w0BAQEF\n-----END PRIVATE KEY-----",
    "jwt": "tok eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcDEFghiJKL",
    "aws_access_key": "AKIAIOSFODNN7EXAMPLE",
    "aws_secret_labeled": 'aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"',
    "github_token": "ghp_" + "A" * 36,
    "github_pat": "github_pat_11ABCDEFG0123456789_abcdefghijklmnop",
    "slack_token": "xoxb-1234567890-secrettail_morestuff",
    "stripe_key": "sk_live_4eC39HqLyjWDarjtT1zdp7dc",
    "sendgrid_key": "SG.aBcDeFgHiJkLmNoP.aBcDeFgHiJkLmNoPqRsTuV",
    "twilio_sid": "AC" + "0" * 32,
    "google_api_key": "AIzaSyD-1234567890abcdefghijklmnopqrstuv",
    "google_oauth": "ya29.a0AfH6SMByourlongtokenstringhere1234567890abcdef",
    "anthropic_key": "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123",
    "openai_proj": "sk-proj-aBcD1234EfGh5678IjKl9012MnOp3456QrSt_uVwX",
    "openai_svcacct": "sk-svcacct-aBcD1234EfGh5678IjKl9012MnOp3456QrSt",
    "openai_classic": "sk-aBcD1234EfGh5678IjKl9012MnOp3456QrStuVwX",
    "bearer": "Authorization: Bearer abcDEF123/xyz+QRS==tuvWXYZ7890abcd",
    "url_basic_auth": "postgres://admin:SuperSecret123@db.example.com:5432/db",
    "redis_auth": "redis://:MyR3disPass@127.0.0.1:6379/0",
    "huggingface_token": "token hf_" + "a" * 36,
    "gitlab_pat": "glpat-abcdefghijklmnopqrstuv",
    "google_oauth_client_secret": "GOCSPX-abcdefghijklmnopqrstuv",
    "npm_token": "npm_" + "B" * 36,
    "pypi_token": "pypi-AgEIcHlwaS5vcmcabcdef-12345",
    "slack_webhook": "post to https://hooks.slack.com/services/T01ABCD/B02EFGH/xXyYzZ1234567890",
    "dotenv_short": "KEY=supersecretvalue1234567890",
    "password_special": "password=P@ssw0rd!verylongpasswordhere",
    "password_short": "password: hunter2",
    "otp": "Your verification code: 123456",
}

# The literal secret fragment that must NOT survive, per case.
LEAKED_FRAGMENT = {
    "slack_token": "secrettail",
    "bearer": "tuvWXYZ7890abcd",
    "url_basic_auth": "SuperSecret123",
    "redis_auth": "MyR3disPass",
    "password_special": "P@ssw0rd",
    "password_short": "hunter2",
    "dotenv_short": "supersecretvalue",
    "openai_proj": "uVwX",
}


def test_each_secret_class_is_redacted():
    failures = []
    for name, text in POSITIVE.items():
        out = redact(text)
        if "[REDACTED:" not in out:
            failures.append(f"{name}: nothing redacted -> {out!r}")
    assert not failures, "Secret classes left unredacted:\n  " + "\n  ".join(failures)


def test_no_partial_secret_leak():
    """The audit's partial-leak bypasses: the secret fragment must be gone."""
    failures = []
    for name, fragment in LEAKED_FRAGMENT.items():
        out = redact(POSITIVE[name])
        if fragment in out:
            failures.append(f"{name}: fragment {fragment!r} survived -> {out!r}")
    assert not failures, "Partial secret leaks:\n  " + "\n  ".join(failures)


# --- negative coverage: benign text must be untouched ---------------------

BENIGN = [
    "Let's discuss the API design for the search module.",
    "http://localhost:8080/api/path",          # URL with port, no creds
    "The meeting is at 10:30 tomorrow.",
    "git clone https://github.com/example/repo.git",
    "I set the timeout to 30 seconds.",
    "version 1.0.0 shipped today",
]


def test_no_false_positives_on_benign_text():
    failures = []
    for text in BENIGN:
        out = redact(text)
        if out != text:
            failures.append(f"{text!r} -> {out!r}")
    assert not failures, "False-positive redactions on benign text:\n  " + "\n  ".join(failures)


def test_empty_and_none_safe():
    assert redact("") == ""
    assert redact("   ") == "   "


# --- CRIT-3: truncation-before-redaction leak (indexer end to end) --------

def test_indexer_redacts_before_truncation_no_key_leak():
    """A PEM key whose END marker falls past the per-block truncation cut
    must still be fully redacted (the verified 974-char leak)."""
    filler = "x" * 7000
    key_body = "B" * 9000
    pem = f"-----BEGIN PRIVATE KEY-----\n{key_body}\n-----END PRIVATE KEY-----"
    tool_result = {
        "type": "tool_result",
        "content": [{"type": "text", "text": filler + "\n" + pem}],
    }
    blocks = indexer._extract_text_from_content([tool_result])
    joined = " ".join(text for _, text in blocks)
    assert "BEGIN PRIVATE KEY" not in joined, "PEM marker leaked past truncation"
    assert "B" * 100 not in joined, "raw key material leaked past truncation"
