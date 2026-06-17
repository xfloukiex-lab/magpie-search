"""Backup config injection-guard regression.

The 2026-05-29 audit found ssh/rsync argument-injection + data-exfiltration
vectors: a malicious MAGPIE_SEARCH_BACKUP_SSH_HOST (e.g. `-oProxyCommand=...`)
or SSH_DEST could turn a backup into arbitrary command execution or silent
exfiltration to an attacker host. load_config() now fails closed on unsafe
values. These tests pin that.
"""
from __future__ import annotations

import pytest

import magpie_search.backup as backup


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    # Point the env-file at a nonexistent path so only our env vars matter.
    monkeypatch.setenv("MAGPIE_SEARCH_BACKUP_CONFIG", str(tmp_path / "nope.env"))
    # Clear any inherited backup env so each test starts clean.
    for k in list(__import__("os").environ):
        if k.startswith("MAGPIE_SEARCH_BACKUP_") and k != "MAGPIE_SEARCH_BACKUP_CONFIG":
            monkeypatch.delenv(k, raising=False)
    yield


MALICIOUS_HOSTS = [
    "-oProxyCommand=touch /tmp/pwned",   # ssh option injection -> RCE
    "-Fmalicious_config",
    "host with space",
    "host;rm -rf ~",
    'host"$(id)',
    "-e ssh evil",
]


@pytest.mark.parametrize("host", MALICIOUS_HOSTS)
def test_malicious_ssh_host_rejected(host, monkeypatch):
    monkeypatch.setenv("MAGPIE_SEARCH_BACKUP_SSH_HOST", host)
    monkeypatch.setenv("MAGPIE_SEARCH_BACKUP_SSH_DEST", "~/dest")
    with pytest.raises(backup.BackupConfigError):
        backup.load_config()


@pytest.mark.parametrize("dest", ["-rf /", "dest with space", 'a;b', '--rsync-path=evil'])
def test_malicious_ssh_dest_rejected(dest, monkeypatch):
    monkeypatch.setenv("MAGPIE_SEARCH_BACKUP_SSH_HOST", "backup-host")
    monkeypatch.setenv("MAGPIE_SEARCH_BACKUP_SSH_DEST", dest)
    with pytest.raises(backup.BackupConfigError):
        backup.load_config()


@pytest.mark.parametrize("host", [
    "backup-host",
    "user@host.example.com",
    "user@10.0.0.5:2222",
    "backup-box",
])
def test_valid_ssh_host_accepted(host, monkeypatch):
    monkeypatch.setenv("MAGPIE_SEARCH_BACKUP_SSH_HOST", host)
    monkeypatch.setenv("MAGPIE_SEARCH_BACKUP_SSH_DEST", "~/transcripts/")
    cfg = backup.load_config()  # must not raise
    assert cfg.ssh_host == host.strip()


def test_main_returns_nonzero_on_bad_config(monkeypatch):
    monkeypatch.setenv("MAGPIE_SEARCH_BACKUP_SSH_HOST", "-oProxyCommand=evil")
    monkeypatch.setenv("MAGPIE_SEARCH_BACKUP_SSH_DEST", "~/dest")
    assert backup.main(["--show-config"]) == 2  # fail closed, no crash
