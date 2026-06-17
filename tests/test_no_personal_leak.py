"""Build-time scrub gate: the shippable magpie_search package must carry NO
personal/maintainer-internal identifiers. Product/company names are allowed
(VektorGeist = vendor; Aviary/Hummingbird = sibling products a dual-product
customer legitimately integrates with). Only PERSONAL info about the makers is
forbidden.

Rationale: a personal identifier once shipped in a wheel undetected because the
"audited clean" claim was never enforced by a test. This gate makes it
impossible to regress silently.

Two layers of patterns:
  1. GENERIC patterns below — universal personal-data shapes (RFC-1918 LAN IPs,
     PEM private-key headers, `C:\\Users\\<name>` dev paths). These ship publicly
     and protect any fork.
  2. An OPTIONAL maintainer denylist of literal names/handles, one term per line,
     in `tests/leak_denylist.local.txt`. That file is gitignored and never
     published, so the public test never names a private individual. Maintainers
     keep their own copy locally; CI on a clean public clone runs generic-only.
"""
from __future__ import annotations

import pathlib
import re

# Generic personal-data shapes — always enforced, safe to publish.
_GENERIC = [
    r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",  # leaked private key
    # dev home path — but allow documentation placeholders (user/you/username/me/example)
    r"(?:[A-Za-z]:\\Users\\|/home/|/Users/)(?!user[\\/]|you[\\/]|username[\\/]|me[\\/]|example)[A-Za-z0-9._-]+[\\/]",
    r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b",  # RFC-1918 IP
]

_DENYLIST_FILE = pathlib.Path(__file__).with_name("leak_denylist.local.txt")
PKG_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "magpie_search"


def _build_pattern() -> re.Pattern[str]:
    parts = list(_GENERIC)
    if _DENYLIST_FILE.exists():
        for line in _DENYLIST_FILE.read_text("utf-8").splitlines():
            term = line.strip()
            if term and not term.startswith("#"):
                parts.append(term)
    return re.compile("|".join(parts))


def _shippable_files() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for p in PKG_SRC.rglob("*"):
        if p.is_file() and p.suffix in {".py", ".toml", ".md", ".txt", ".cfg"}:
            if "__pycache__" in p.parts:
                continue
            out.append(p)
    return out


def test_no_personal_identifiers_in_shippable_source() -> None:
    forbidden = _build_pattern()
    offenders: list[str] = []
    for f in _shippable_files():
        text = f.read_text("utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            m = forbidden.search(line)
            if m:
                offenders.append(f"{f.relative_to(PKG_SRC)}:{i}: {m.group(0)!r}")
    assert not offenders, (
        "Personal/maintainer-internal identifiers must never ship in magpie_search. "
        "Scrub these (product/company names are fine, personal info is not):\n  "
        + "\n  ".join(offenders)
    )
