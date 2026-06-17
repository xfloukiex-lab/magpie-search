"""Safe subprocess invocation.

magpie_search spawns VM tools (vmrun, VBoxManage) and remote-sync tools (ssh,
rsync, scp) when its nightly backup runs. By default, every subprocess
inherits the parent's full env — including any secrets the operator
has set (API tokens, app passwords, etc.). `safe_env()` strips those
known + pattern-matched secret names so an exec'd binary (or a
compromised tool on PATH) can't read them.


**What DOES flow through to children (intentional, not a leak):**
- `SSH_AUTH_SOCK` and `SSH_AGENT_PID` — required for ssh / rsync / scp
  to use key-based auth via the running ssh-agent. magpie_search's `backup`
  module spawns these tools for remote sync to a remote SSH host, and
  it must inherit the agent socket or the sync fails with no password
  prompt path available in a systemd timer context. Don't add these
  to the denylist without first replacing key-based auth with something
  else.
- All `PATH` / `HOME` / `USER` / `TMPDIR` / `LANG` etc. (tool-discovery
  and locale) — children need them.

Callers that need a specific secret available to a child must re-add it:

    e = safe_env()
    e["GMAIL_APP_PASSWORD"] = os.environ["GMAIL_APP_PASSWORD"]
    subprocess.run([...], env=e, ...)
"""
from __future__ import annotations

import os
import subprocess
from typing import Any


_KNOWN_SECRET_KEYS: frozenset[str] = frozenset({
    "GMAIL_EMAIL",
    "GMAIL_APP_PASSWORD", "GMAIL_PASSWORD",
    "CLOUDFLARE_API_TOKEN",
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "SUDO_PASSWORD",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
})


_SECRET_SUFFIXES: tuple[str, ...] = (
    "_PASSWORD", "_TOKEN", "_SECRET", "_PASSPHRASE", "_API_KEY",
    "_PRIVATE_KEY", "_AUTH_KEY", "_ACCESS_KEY",
)


# Not secrets, but command-execution vectors: these env vars override the
# *remote shell* rsync/git invoke. The backup intends key-agent auth only,
# so a value inherited from a poisoned launch environment must not steer the
# child to an attacker-chosen command. Always stripped.
_COMMAND_OVERRIDE_KEYS: frozenset[str] = frozenset({
    "RSYNC_RSH", "GIT_SSH", "GIT_SSH_COMMAND",
})


def _is_secret(name: str) -> bool:
    upper = name.upper()
    if upper in _KNOWN_SECRET_KEYS or upper in _COMMAND_OVERRIDE_KEYS:
        return True
    return any(upper.endswith(suf) for suf in _SECRET_SUFFIXES)


def safe_env(
    *,
    extra_strip: tuple[str, ...] = (),
    pass_through: tuple[str, ...] = (),
) -> dict[str, str]:
    """Return a copy of os.environ minus known + pattern-matched secrets.

    `extra_strip` removes additional names; `pass_through` re-includes
    names that would otherwise be stripped."""
    pt = {name.upper() for name in pass_through}
    extra = {name.upper() for name in extra_strip}
    out: dict[str, str] = {}
    for k, v in os.environ.items():
        upper = k.upper()
        if upper in pt:
            out[k] = v
            continue
        if upper in extra:
            continue
        if _is_secret(k):
            continue
        out[k] = v
    return out


def safe_run(args: Any, **kwargs: Any) -> subprocess.CompletedProcess:
    """subprocess.run with a default safe env. If caller passes `env=`,
    it's used as-is (trusted)."""
    if "env" not in kwargs or kwargs["env"] is None:
        kwargs["env"] = safe_env()
    return subprocess.run(args, **kwargs)
