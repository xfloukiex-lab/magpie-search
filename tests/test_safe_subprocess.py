"""magpie_search.safe_subprocess — secret-stripping env builder for subprocesses.

Mirrors aviary.core.safe_subprocess; same denylist shape. The same
threat model applies (compromised tool on PATH, postinstall script,
malicious npm dep). magpie_search-specific subprocess sites — vmrun, VBoxManage,
ssh, rsync, scp — all run with env=safe_env() so a compromised binary
can't read secret env vars.
"""
from __future__ import annotations

import pytest

from magpie_search.safe_subprocess import _is_secret, safe_env


def test_known_secret_keys_recognized():
    for k in ["GMAIL_APP_PASSWORD", "AVIARY_NEST_PASSPHRASE",
              "VG_AGENT_TOKEN", "CLOUDFLARE_API_TOKEN", "SUDO_PASSWORD",
              "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]:
        assert _is_secret(k), f"{k!r} must be classified as secret"


def test_pattern_match_catches_future_secrets():
    for k in ["NEW_API_KEY", "MY_PASSWORD", "SERVICE_TOKEN",
              "X_PRIVATE_KEY", "BACKUP_PASSPHRASE"]:
        assert _is_secret(k)


def test_non_secrets_pass_through():
    for k in ["PATH", "HOME", "USERPROFILE", "TMPDIR", "TERM", "LANG"]:
        assert not _is_secret(k)


def test_safe_env_strips_secret(monkeypatch):
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "do-not-leak")
    monkeypatch.setenv("PATH", "/usr/bin")
    env = safe_env()
    assert "GMAIL_APP_PASSWORD" not in env
    assert "PATH" in env


def test_safe_env_pass_through_re_includes(monkeypatch):
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "needed-by-child")
    env = safe_env(pass_through=("GMAIL_APP_PASSWORD",))
    assert env.get("GMAIL_APP_PASSWORD") == "needed-by-child"


def test_backup_module_imports_safe_env():
    """Catch a future revert that removes the safe_env import."""
    import magpie_search.backup as backup_mod
    assert hasattr(backup_mod, "safe_env"), (
        "magpie_search.backup must import safe_env from magpie_search.safe_subprocess. "
        "Subprocess calls in backup.py leak the parent env without it."
    )


def test_backup_subprocess_sites_use_safe_env():
    """Every subprocess.run in backup.py must pass env=safe_env()."""
    import re
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src" / "magpie_search" / "backup.py"
    text = src.read_text(encoding="utf-8")
    run_count = len(re.findall(r"subprocess\.run\(", text))
    env_count = len(re.findall(r"env=safe_env\(\)", text))
    assert run_count > 0, "expected subprocess.run sites in backup.py"
    assert env_count >= run_count, (
        f"{run_count} subprocess.run sites but only {env_count} "
        f"env=safe_env() — env stripping incomplete"
    )
