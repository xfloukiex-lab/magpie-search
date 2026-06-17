"""backup — back up local Claude Code transcripts to a configurable destination.

Three modes, picked by what's configured:

  1. **Local folder** (default).  Copies `~/.claude/projects/` to a folder
     on the same machine (default `~/.magpie-search/backups/`).  Zero setup.

  2. **Remote SSH (rsync-over-ssh).**  Set `MAGPIE_SEARCH_BACKUP_SSH_HOST` and
     `MAGPIE_SEARCH_BACKUP_SSH_DEST` to push to a remote box you own — could be
     an Ubuntu server, a NAS that speaks SSH, anything `ssh user@host`
     can reach.

  3. **Remote SSH with VM boot/suspend.**  Same as mode 2, plus boot the
     VM before sync and suspend it after.  Set `MAGPIE_SEARCH_BACKUP_VM_PROVIDER`
     to `vmware` or `virtualbox` and provide the VMX/name.

Config is read from env vars first; missing values are filled in from
`~/.magpie-search/backup.env` (simple KEY=VALUE format, one per line, # for
comments) if it exists.  Nothing about the operator's machine is baked
into the source — every value is overridable.

Pruning is intentionally NOT part of this module.  Copying is one job;
deciding when to delete originals is the user's call.  Run a separate
`magpie_search prune` if/when that ships.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from magpie_search.safe_subprocess import safe_env


# -----------------------------------------------------------------------------
# Config loading
# -----------------------------------------------------------------------------


def _config_path() -> Path:
    """Override via MAGPIE_SEARCH_BACKUP_CONFIG; else `~/.magpie-search/backup.env`."""
    override = os.environ.get("MAGPIE_SEARCH_BACKUP_CONFIG")
    if override:
        return Path(override)
    return Path.home() / ".magpie-search" / "backup.env"


def _read_env_file(p: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE file. Quoted values are stripped of
    matching outer quotes; lines starting with # are ignored."""
    out: dict[str, str] = {}
    if not p.exists():
        return out
    for line in p.read_text("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        out[k.strip()] = v
    return out


def _get(env_file: dict[str, str], key: str, default: str | None = None) -> str | None:
    """Env var wins; otherwise fall through to the config file; else default."""
    v = os.environ.get(key)
    if v:
        return v
    return env_file.get(key, default)


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class BackupConfig:
    src_dir: Path
    # Local-folder mode (always available as fallback)
    local_dest_dir: Path
    # Remote SSH mode (active when ssh_host is set)
    ssh_host: str | None
    ssh_dest: str | None
    # VM-management hooks (active when vm_provider is set AND ssh_host is set)
    vm_provider: str | None       # "vmware" | "virtualbox"
    vm_vmx: Path | None           # vmware: path to .vmx file
    vm_name: str | None           # virtualbox: VM name
    vm_boot_before: bool
    vm_suspend_after: bool
    # Operational
    ssh_ready_timeout_s: int
    log_path: Path
    extras: dict[str, str] = field(default_factory=dict)

    @property
    def mode(self) -> str:
        if self.ssh_host:
            return "ssh+vm" if self.vm_provider else "ssh"
        return "local"


class BackupConfigError(ValueError):
    """Raised when backup config (env / backup.env) is unsafe or malformed."""


import string as _string

# `[user@]host[:port]` — letters/digits/dot/underscore/hyphen only. No part
# may start with '-' (would be parsed as an ssh/rsync OPTION), and the value
# may not contain whitespace or shell/option metacharacters. This is the
# load-bearing guard: an operator (or a poisoned ~/.magpie-search/backup.env)
# must not be able to turn a backup into arbitrary command execution
# (`ssh -oProxyCommand=…`) or silent exfiltration to an attacker host.
_HOST_CHARS = frozenset(_string.ascii_letters + _string.digits + "_.-")
_UNSAFE_CHARS = set(" \t\r\n\"'`;|&$\\<>(){}*?")


def _ok_label(label: str) -> bool:
    """A user or host label: non-empty, no leading '-', allowed chars only."""
    return bool(label) and label[0] != "-" and all(c in _HOST_CHARS for c in label)


def _validate_ssh_host(host: str) -> str:
    h = host.strip()
    user, sep, hostport = h.partition("@")
    if sep:  # had a user@ prefix
        if not _ok_label(user):
            hostport = ""  # force rejection below
    else:
        hostport = h
    host_only, csep, port = hostport.rpartition(":")
    if csep:
        if not port.isdigit():
            host_only = ""  # bad port -> reject
    else:
        host_only = hostport
    if not (sep == "" or _ok_label(user)) or not _ok_label(host_only):
        raise BackupConfigError(
            "MAGPIE_SEARCH_BACKUP_SSH_HOST is not a valid [user@]host[:port] "
            "(letters/digits/.-_ only, no leading '-'); refusing to run a "
            "backup with an unvalidated host (option/command-injection guard)."
        )
    return h


def _validate_token(value: str, var: str) -> str:
    v = value.strip()
    if not v or v.startswith("-") or any(c in _UNSAFE_CHARS for c in v):
        raise BackupConfigError(
            f"{var} is empty, starts with '-', or contains unsafe characters; "
            "refusing to use it in a remote command (injection guard)."
        )
    return v


def load_config() -> BackupConfig:
    env_file = _read_env_file(_config_path())

    src_dir_s = _get(env_file, "MAGPIE_SEARCH_BACKUP_SRC_DIR")
    src_dir = Path(src_dir_s).expanduser() if src_dir_s else Path.home() / ".claude" / "projects"

    local_dest_s = _get(
        env_file, "MAGPIE_SEARCH_BACKUP_DEST_DIR",
        str(Path.home() / ".magpie-search" / "backups"),
    )
    local_dest = Path(local_dest_s).expanduser() if local_dest_s else Path.home() / ".magpie-search" / "backups"

    ssh_host = _get(env_file, "MAGPIE_SEARCH_BACKUP_SSH_HOST")
    ssh_dest = _get(env_file, "MAGPIE_SEARCH_BACKUP_SSH_DEST")
    # Fail-closed validation BEFORE these strings ever reach an ssh/rsync/scp
    # argv. A leading '-' or shell/option metacharacter here is an injection /
    # exfiltration vector (see audit CRIT-1 / HIGH-1).
    if ssh_host is not None:
        ssh_host = _validate_ssh_host(ssh_host)
    if ssh_dest is not None:
        ssh_dest = _validate_token(ssh_dest, "MAGPIE_SEARCH_BACKUP_SSH_DEST")

    vm_provider = (_get(env_file, "MAGPIE_SEARCH_BACKUP_VM_PROVIDER") or "").strip().lower() or None
    if vm_provider not in (None, "vmware", "virtualbox"):
        # Unknown provider — treat as if VM management was not configured
        # rather than silently mis-routing. Real fix is operator's call.
        vm_provider = None

    vmx_s = _get(env_file, "MAGPIE_SEARCH_BACKUP_VM_VMX")
    if vmx_s is not None:
        _validate_token(vmx_s, "MAGPIE_SEARCH_BACKUP_VM_VMX")  # reject option-like / metachars
    vm_vmx = Path(vmx_s).expanduser() if vmx_s else None
    vm_name = _get(env_file, "MAGPIE_SEARCH_BACKUP_VM_NAME")
    if vm_name is not None:
        vm_name = _validate_token(vm_name, "MAGPIE_SEARCH_BACKUP_VM_NAME")
    vm_boot = _truthy(_get(env_file, "MAGPIE_SEARCH_BACKUP_VM_BOOT_BEFORE", "1"))
    vm_susp = _truthy(_get(env_file, "MAGPIE_SEARCH_BACKUP_VM_SUSPEND_AFTER", "1"))

    timeout_s = int(_get(env_file, "MAGPIE_SEARCH_BACKUP_SSH_READY_TIMEOUT_S", "90") or "90")
    log_path = Path(
        _get(env_file, "MAGPIE_SEARCH_BACKUP_LOG", str(Path.home() / ".magpie-search" / "backup.log"))
        or str(Path.home() / ".magpie-search" / "backup.log")
    ).expanduser()

    return BackupConfig(
        src_dir=src_dir,
        local_dest_dir=local_dest,
        ssh_host=ssh_host,
        ssh_dest=ssh_dest,
        vm_provider=vm_provider,
        vm_vmx=vm_vmx,
        vm_name=vm_name,
        vm_boot_before=vm_boot,
        vm_suspend_after=vm_susp,
        ssh_ready_timeout_s=timeout_s,
        log_path=log_path,
        extras=env_file,
    )


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------


def _ts() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log_writer(cfg: BackupConfig):
    """Returns a `log(msg, **fields)` closure. Best-effort — never raises."""
    def log(msg: str, **fields: Any) -> None:
        line = f"{_ts()} {msg}"
        try:
            cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
            with cfg.log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass
        print(line, flush=True)
    return log


# -----------------------------------------------------------------------------
# VM providers (optional)
# -----------------------------------------------------------------------------


def _vmware_vmrun(cfg: "BackupConfig | None" = None) -> Path | None:
    """Find vmrun.exe on Windows (default install path) or `vmrun` on Linux/macOS.

    MED-2 (code review R3): resolve the override via `_get(cfg.extras, ...)`
    (env wins, else backup.env) — same fix as _rsync_bin. A bare
    os.environ lookup silently ignored a backup.env-set path."""
    extras = cfg.extras if cfg is not None else {}
    override = _get(extras, "MAGPIE_SEARCH_BACKUP_VM_VMRUN")
    if override:
        p = Path(override)
        return p if p.exists() else None
    candidates = [
        Path("C:/Program Files (x86)/VMware/VMware Workstation/vmrun.exe"),
        Path("C:/Program Files/VMware/VMware Workstation/vmrun.exe"),
        Path("/usr/bin/vmrun"),
        Path("/usr/local/bin/vmrun"),
    ]
    for c in candidates:
        if c.exists():
            return c
    which = shutil.which("vmrun")
    return Path(which) if which else None


def _vmware_running(vmrun: Path, vmx: Path) -> bool:
    try:
        r = subprocess.run([str(vmrun), "list"], capture_output=True, text=True, timeout=30, env=safe_env())
        return str(vmx) in r.stdout
    except OSError:
        return False


def _vmware_start(vmrun: Path, vmx: Path, dry: bool, log) -> bool:
    cmd = [str(vmrun), "start", str(vmx), "nogui"]
    if dry:
        log(f"DRY: would run {' '.join(cmd)}")
        return True
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=safe_env())
        if r.returncode != 0:
            log(f"vmrun start exit {r.returncode}: {r.stderr.strip()[:300]}")
            return False
        return True
    except OSError as e:
        log(f"vmrun start exception: {type(e).__name__}: {e}")
        return False


def _vmware_suspend(vmrun: Path, vmx: Path, dry: bool, log) -> bool:
    cmd = [str(vmrun), "suspend", str(vmx)]
    if dry:
        log(f"DRY: would run {' '.join(cmd)}")
        return True
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=safe_env())
        if r.returncode != 0:
            log(f"vmrun suspend exit {r.returncode}: {r.stderr.strip()[:300]}")
            return False
        return True
    except OSError as e:
        log(f"vmrun suspend exception: {type(e).__name__}: {e}")
        return False


def _virtualbox_running(name: str) -> bool:
    try:
        r = subprocess.run(
            ["VBoxManage", "list", "runningvms"],
            capture_output=True, text=True, timeout=30, env=safe_env(),
        )
        return f'"{name}"' in r.stdout
    except OSError:
        return False


def _virtualbox_start(name: str, dry: bool, log) -> bool:
    cmd = ["VBoxManage", "startvm", name, "--type", "headless"]
    if dry:
        log(f"DRY: would run {' '.join(cmd)}")
        return True
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=safe_env())
        if r.returncode != 0:
            log(f"VBoxManage startvm exit {r.returncode}: {r.stderr.strip()[:300]}")
            return False
        return True
    except OSError as e:
        log(f"VBoxManage start exception: {type(e).__name__}: {e}")
        return False


def _virtualbox_suspend(name: str, dry: bool, log) -> bool:
    # `savestate` is virtualbox's analogue to vmware's suspend (preserves
    # RAM, resumes fast). `controlvm <name> acpipowerbutton` would shut
    # down cleanly but is slower; we pick the fast option to match vmware.
    cmd = ["VBoxManage", "controlvm", name, "savestate"]
    if dry:
        log(f"DRY: would run {' '.join(cmd)}")
        return True
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=safe_env())
        if r.returncode != 0:
            log(f"VBoxManage savestate exit {r.returncode}: {r.stderr.strip()[:300]}")
            return False
        return True
    except OSError as e:
        log(f"VBoxManage savestate exception: {type(e).__name__}: {e}")
        return False


# -----------------------------------------------------------------------------
# SSH + rsync
# -----------------------------------------------------------------------------


def _rsync_bin(cfg: "BackupConfig | None" = None) -> str | None:
    """Resolve the rsync binary. Override-first (the project's
    "nothing hardcoded, everything overridable" rule — same pattern as
    ssh/vmrun via shutil.which): MAGPIE_SEARCH_BACKUP_RSYNC (env wins, else
    backup.env) > PATH. On Windows, real rsync is an MSYS2/Cygwin
    binary that is rarely on PATH, so the explicit override is the
    supported way to enable verified backups without polluting the
    system PATH with an entire MSYS2 bin dir (which would shadow
    Windows find/sort/etc.).

    NOTE: backup.env keys are NOT exported to os.environ — they live in
    cfg.extras. Resolve via the same `_get` the rest of this module
    uses so a file-set override actually takes effect (a plain
    os.environ lookup silently ignores backup.env)."""
    extras = cfg.extras if cfg is not None else {}
    override = _get(extras, "MAGPIE_SEARCH_BACKUP_RSYNC")
    if override:
        p = Path(override).expanduser()
        return str(p) if p.exists() else None
    return shutil.which("rsync")


def _rsync_src_path(src: Path, cfg: "BackupConfig | None" = None) -> str:
    """Translate a source dir into the path form THIS rsync expects,
    with a trailing slash. An MSYS2/Cygwin rsync treats a leading
    `C:` as a remote host (host:path syntax), so a Windows drive path
    must become a POSIX path. Probe with the `cygpath` that ships
    beside the rsync binary rather than ASSUMING the mount style
    (MSYS2 uses /c/..., Cygwin /cygdrive/c/... — version/config
    dependent; do not hardcode)."""
    s = str(src)
    if os.name != "nt" or not re.match(r"^[A-Za-z]:", s):
        return s.replace("\\", "/").rstrip("/") + "/"
    rb = _rsync_bin(cfg)
    cygpath = None
    if rb:
        cand = Path(rb).with_name("cygpath.exe")
        if cand.exists():
            cygpath = str(cand)
    if cygpath:
        try:
            r = subprocess.run([cygpath, "-u", s], capture_output=True,
                               text=True, timeout=15, env=safe_env())
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip().rstrip("/") + "/"
        except (OSError, subprocess.SubprocessError):
            pass
    # Fallback: best-effort MSYS2-style /c/... (NOT assumed correct —
    # logged by the caller via the transport label if rsync then fails).
    drive = s[0].lower()
    rest = s[2:].replace("\\", "/")
    return f"/{drive}{rest}".rstrip("/") + "/"


def _ssh_host_port(host_spec: str) -> tuple[str, int]:
    """Pull a host out of `user@host[:port]` for the TCP probe."""
    bare = host_spec.split("@", 1)[-1]
    if ":" in bare:
        host, _, port_s = bare.partition(":")
        try:
            return host, int(port_s)
        except ValueError:
            return host, 22
    return bare, 22


def _resolve_ssh_target(host_spec: str, log) -> tuple[str, int]:
    """Resolve `host_spec` the way ssh itself would, then return the
    real (hostname, port) for a raw socket probe.

    Root cause of the 2026-05-15/16 backup outage: `_wait_for_ssh` did
    `socket.create_connection((host, port))` on the bare host string.
    When `MAGPIE_SEARCH_BACKUP_SSH_HOST` is an *ssh-config alias* (e.g.
    `myvm` → HostName 192.0.2.10 in ~/.ssh/config), the raw
    socket does a DNS getaddrinfo on `myvm` — which has no DNS/hosts
    entry — so the probe failed forever even though the VM + sshd were
    fine and `ssh myvm` worked (ssh reads the config; sockets don't).

    Fix: ask ssh to resolve the alias via `ssh -G <host>` and probe the
    HostName/Port it reports. Falls back to the literal host (old
    behavior) only if `ssh -G` is unavailable or returns nothing."""
    host, port = _ssh_host_port(host_spec)
    sh = shutil.which("ssh")
    if not sh:
        return host, port
    try:
        r = subprocess.run([sh, "-G", host], capture_output=True,
                           text=True, timeout=10, env=safe_env())
    except (OSError, subprocess.TimeoutExpired):
        return host, port
    if r.returncode != 0:
        return host, port
    resolved_host, resolved_port = host, port
    for line in r.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        key, val = parts[0].lower(), parts[1].strip()
        if key == "hostname" and val:
            resolved_host = val
        elif key == "port" and val.isdigit():
            resolved_port = int(val)
    if resolved_host != host:
        log(f"ssh-config: {host} -> {resolved_host}:{resolved_port}")
    return resolved_host, resolved_port


def _wait_for_ssh(host_spec: str, timeout: int, log) -> bool:
    host, port = _resolve_ssh_target(host_spec, log)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=3):
                return True
        except OSError:
            time.sleep(2)
    log(f"FAIL: SSH never reachable at {host}:{port} within {timeout}s")
    return False


def _ensure_remote_dir(host: str, dest: str, dry: bool, log) -> bool:
    cmd = ["ssh", host, "mkdir", "-p", dest]
    if dry:
        log(f"DRY: would run {' '.join(cmd)}")
        return True
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=safe_env())
        if r.returncode != 0:
            log(f"ssh mkdir exit {r.returncode}: {r.stderr.strip()[:300]}")
            return False
        return True
    except OSError as e:
        log(f"ssh mkdir exception: {type(e).__name__}: {e}")
        return False


def _rsync_to_remote(src: Path, host: str, dest: str, dry: bool, log,
                     cfg: "BackupConfig | None" = None) -> tuple[bool, int, str]:
    """Returns (ok, source_file_count, transport).

    HIGH-3 (magpie_search-hardening audit 2026-05-16): `file_count` is the
    number of *source* .jsonl files, NOT what was verifiably transferred.
    rsync confirms the transfer (and `--delete` mirrors deletions); the
    scp fallback does neither — it can silently miss files and never
    propagates deletions. The transport label is returned so the caller
    logs an honest message instead of "synced N files" (which falsely
    implied a verified count on the scp path — the same success-log-lies
    class as the backup outage itself)."""
    file_count = sum(1 for _ in src.rglob("*.jsonl"))
    rsync_bin = _rsync_bin(cfg)
    if rsync_bin is None:
        log("rsync not available (set MAGPIE_SEARCH_BACKUP_RSYNC) — falling back "
            "to scp (deletions will NOT propagate; file count is "
            "source-side, not a verified transfer manifest)")
        return _scp_to_remote(src, host, dest, dry, log), file_count, "scp"
    # Windows paths come through as C:\... — rsync wants forward slashes.
    # An MSYS2/Cygwin rsync also parses a leading "C:" as a REMOTE HOST
    # (host:path syntax), so a Windows drive path must be handed over as
    # a /cygdrive-style POSIX path instead.
    src_s = _rsync_src_path(src, cfg)
    # `--` terminates option parsing so the path operands can never be
    # reinterpreted as rsync options (defense-in-depth; host/dest are also
    # validated at config load).
    cmd = [rsync_bin, "-az", "--delete", "--", src_s, f"{host}:{dest}"]
    if dry:
        log(f"DRY: would run {' '.join(cmd)}  ({file_count} jsonl files)")
        return True, file_count, "rsync"
    # Resilience: if rsync fails for ANY reason (a finicky MSYS2 rsync
    # on Windows can die with `dup() in/out/err failed` before it ever
    # connects), fall back to the PROVEN scp path so the backup still
    # succeeds. Before this, no-rsync fell back to scp but a *failing*
    # rsync hard-failed the whole backup — strictly worse than no rsync
    # at all, and exactly the silent-outage class we are removing. rsync
    # is preferred (verified + mirrors deletions); scp is the safety net,
    # never a hard stop. The transport label stays honest.
    def _scp_fallback(why: str) -> tuple[bool, int, str]:
        log(f"rsync failed ({why}) — falling back to scp "
            "(deletions will NOT propagate; source-side count, not a "
            "verified transfer manifest)")
        return _scp_to_remote(src, host, dest, dry, log), file_count, \
            "scp-after-rsync-fail"

    try:
        # stdin=DEVNULL: an MSYS2 rsync inherits Python's (often
        # non-console) stdin and its dup() of std handles fails
        # ("dup() in/out/err failed") before it connects. A real
        # DEVNULL handle avoids that without losing stderr capture.
        r = subprocess.run(cmd, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True,
                            timeout=1800, env=safe_env())
        if r.returncode != 0:
            log(f"rsync exit {r.returncode}: {r.stderr.strip()[:500]}")
            return _scp_fallback(f"exit {r.returncode}")
        return True, file_count, "rsync"
    except (OSError, subprocess.SubprocessError) as e:
        log(f"rsync exception: {type(e).__name__}: {e}")
        return _scp_fallback(f"{type(e).__name__}")


def _scp_to_remote(src: Path, host: str, dest: str, dry: bool, log) -> bool:
    src_s = str(src).replace("\\", "/") + "/."
    cmd = ["scp", "-r", "-q", "--", src_s, f"{host}:{dest}"]
    if dry:
        log(f"DRY: would run {' '.join(cmd)}")
        return True
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=safe_env())
        if r.returncode != 0:
            log(f"scp exit {r.returncode}: {r.stderr.strip()[:300]}")
            return False
        return True
    except OSError as e:
        log(f"scp exception: {type(e).__name__}: {e}")
        return False


def _local_copy(src: Path, dest: Path, dry: bool, log,
                cfg: "BackupConfig | None" = None) -> tuple[bool, int]:
    """Mirror `src/` into `dest/` using rsync if available, else shutil.copytree
    + manual prune. No deletion in shutil fallback — rsync gives parity."""
    file_count = sum(1 for _ in src.rglob("*.jsonl"))
    if dry:
        log(f"DRY: would copy {file_count} files: {src} -> {dest}")
        return True, file_count

    dest.mkdir(parents=True, exist_ok=True)
    rsync_bin = _rsync_bin(cfg)
    if rsync_bin is not None:
        src_s = _rsync_src_path(src, cfg)
        dest_s = _rsync_src_path(dest, cfg)
        cmd = [rsync_bin, "-a", "--delete", src_s, dest_s]
        try:
            r = subprocess.run(cmd, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True,
                                timeout=1800, env=safe_env())
            if r.returncode != 0:
                # MED-1 (code review R3): a non-zero exit (e.g. the MSYS2
                # `dup() in/out/err failed` on Windows) must fall
                # through to the always-works shutil path, NOT
                # hard-fail the backup — same resilience contract as
                # _rsync_to_remote's scp fallback. Returning False here
                # was strictly worse than having no rsync at all.
                log(f"local rsync exit {r.returncode}: "
                    f"{r.stderr.strip()[:500]} — falling back to shutil")
            else:
                return True, file_count
        except (OSError, subprocess.SubprocessError) as e:
            log(f"local rsync exception: {type(e).__name__}: {e} "
                "— falling back to shutil")
        # fall through to shutil
    # shutil fallback — slow on large trees but always works
    try:
        for child in src.iterdir():
            target = dest / child.name
            if child.is_dir():
                shutil.copytree(child, target, dirs_exist_ok=True)
            else:
                shutil.copy2(child, target)
        return True, file_count
    except OSError as e:
        log(f"shutil copy exception: {type(e).__name__}: {e}")
        return False, file_count


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Heartbeat + self-monitoring (GAP-4)
#
# The 2026-05-15/16 outage was invisible for two nights because a failed
# backup only ever wrote to a log file nobody reads — nothing recorded
# "last success was N hours ago" and nothing escalated. The fix has two
# halves: (1) every run writes a durable state file (success OR failure),
# carrying forward the last *successful* timestamp so a failing run can't
# erase the knowledge of when good data last landed; (2) health_check()
# turns that into a machine-checkable verdict a daemon can poll, so the
# SYSTEM detects staleness instead of a human noticing days later.
# -----------------------------------------------------------------------------


def _state_path(cfg: BackupConfig) -> Path:
    """Heartbeat file, beside the log so it travels with the install."""
    return cfg.log_path.parent / ".backup_state.json"


def _read_state(cfg: BackupConfig) -> dict[str, Any]:
    try:
        return json.loads(_state_path(cfg).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_state(cfg: BackupConfig, *, exit_code: int, files: int,
                  transport: str, log, dry: bool = False) -> None:
    """Atomically record this run. Best-effort — never raises (a state
    write failing must not turn a good backup into a reported failure).

    CRIT-1 (code review R2): a `--dry-run` returns exit 0 but transfers
    ZERO bytes. It must NEVER advance `last_success_ts` — doing so makes
    `health_check()` report ok and silences `backup_health` while no real
    backup has happened (the exact "success marker that lies" class as
    the original outage). On dry, the success markers are carried forward
    untouched and only a separate `last_dry_run_ts` is recorded."""
    prev = _read_state(cfg)
    now = _ts()
    real_ok = exit_code == 0 and not dry
    state = {
        "last_attempt_ts": now,
        "last_attempt_exit": exit_code,
        "last_attempt_ok": real_ok,
        "last_attempt_dry": dry,
        "last_files": files,
        "last_transport": transport,
        "mode": cfg.mode,
        # carry forward — a failure OR a dry-run must NOT clobber the
        # last-good marker. Only a real, successful transfer sets it.
        "last_success_ts": now if real_ok else prev.get("last_success_ts"),
        "last_success_files": files if real_ok else prev.get("last_success_files"),
        "last_dry_run_ts": now if dry else prev.get("last_dry_run_ts"),
    }
    try:
        p = _state_path(cfg)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(tmp, p)
    except OSError as e:
        log(f"WARN: could not write backup state: {type(e).__name__}: {e}")


def _parse_ts(s: str | None) -> datetime.datetime | None:
    if not s:
        return None
    try:
        return datetime.datetime.strptime(
            s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
    except (ValueError, TypeError):
        return None


def health_check(*, max_age_hours: float | None = None,
                  cfg: BackupConfig | None = None) -> dict[str, Any]:
    """Machine-checkable backup health. A daemon polls this; if status is
    not 'ok' the SYSTEM escalates — no human-as-failure-detector.

    status:
      ok               last success within max_age_hours
      stale            last success too old (THE outage signature)
      never_succeeded  state exists but no success ever recorded
      no_state         backup has never run / state file missing
    """
    cfg = cfg or load_config()
    if max_age_hours is None:
        try:
            max_age_hours = float(
                os.environ.get("MAGPIE_SEARCH_BACKUP_MAX_AGE_HOURS", "26"))
        except ValueError:
            max_age_hours = 26.0
    st = _read_state(cfg)
    if not st:
        return {"ok": False, "status": "no_state", "age_hours": None,
                "max_age_hours": max_age_hours,
                "reason": f"no backup state at {_state_path(cfg)} — "
                          "backup has never run or never recorded a result"}
    last_ok = _parse_ts(st.get("last_success_ts"))
    now = datetime.datetime.now(datetime.timezone.utc)
    if last_ok is None:
        return {"ok": False, "status": "never_succeeded", "age_hours": None,
                "max_age_hours": max_age_hours,
                "last_attempt_ts": st.get("last_attempt_ts"),
                "last_attempt_exit": st.get("last_attempt_exit"),
                "reason": "backup has run but never succeeded"}
    age_h = (now - last_ok).total_seconds() / 3600.0
    stale = age_h > max_age_hours
    return {
        "ok": not stale,
        "status": "stale" if stale else "ok",
        "age_hours": round(age_h, 2),
        "max_age_hours": max_age_hours,
        "last_success_ts": st.get("last_success_ts"),
        "last_attempt_ts": st.get("last_attempt_ts"),
        "last_attempt_exit": st.get("last_attempt_exit"),
        "last_attempt_ok": st.get("last_attempt_ok"),
        "reason": (f"last successful backup {age_h:.1f}h ago "
                   f"(> {max_age_hours}h threshold)") if stale
                  else f"last successful backup {age_h:.1f}h ago",
    }


def run_backup(*, dry: bool = False, no_suspend: bool = False,
               cfg: BackupConfig | None = None) -> int:
    """Top-level entry. Returns process exit code. Always records a
    heartbeat (success or failure) before returning — see GAP-4."""
    cfg = cfg or load_config()
    log = _log_writer(cfg)
    code, files, transport = _run_backup_impl(
        cfg, log, dry=dry, no_suspend=no_suspend)
    _write_state(cfg, exit_code=code, files=files,
                 transport=transport, log=log, dry=dry)
    return code


def _run_backup_impl(cfg: BackupConfig, log, *, dry: bool,
                     no_suspend: bool) -> tuple[int, int, str]:
    """The actual backup. Returns (exit_code, files, transport) so the
    wrapper can write one heartbeat covering every exit path."""
    t0 = time.time()
    log(f"==== magpie_search backup start (mode={cfg.mode}, dry={dry}) ====")

    if not cfg.src_dir.exists():
        log(f"FAIL: source dir missing: {cfg.src_dir}")
        return (1, 0, "config")

    # --- LOCAL MODE ---
    if cfg.mode == "local":
        ok, n = _local_copy(cfg.src_dir, cfg.local_dest_dir, dry, log, cfg)
        elapsed = int(time.time() - t0)
        log(f"==== local backup done in {elapsed}s, {n} files, ok={ok} ====")
        return (0 if ok else 4, n, "local")

    # --- SSH (and optional VM) MODE ---
    # MED-2 (code review R2): NOT a bare `assert` — `python -O` strips asserts,
    # which would let None ssh_host/dest fall through to a confusing
    # AttributeError deep in _wait_for_ssh instead of failing here.
    if not (cfg.ssh_host and cfg.ssh_dest):
        log("FAIL: ssh mode but ssh_host/ssh_dest unset (config error)")
        return (1, 0, "config")
    started_vm = False
    vmrun: Path | None = None

    if cfg.mode == "ssh+vm":
        if cfg.vm_provider == "vmware":
            if cfg.vm_vmx is None:
                log("FAIL: MAGPIE_SEARCH_BACKUP_VM_PROVIDER=vmware but no MAGPIE_SEARCH_BACKUP_VM_VMX")
                return (1, 0, "config")
            vmrun = _vmware_vmrun(cfg)
            if vmrun is None:
                log("FAIL: vmrun not found — set MAGPIE_SEARCH_BACKUP_VM_VMRUN or install VMware Workstation")
                return (2, 0, "vm")
            if cfg.vm_boot_before and not _vmware_running(vmrun, cfg.vm_vmx):
                log(f"booting VM {cfg.vm_vmx.name}...")
                if not _vmware_start(vmrun, cfg.vm_vmx, dry, log):
                    return (2, 0, "vm")
                started_vm = True
        elif cfg.vm_provider == "virtualbox":
            if not cfg.vm_name:
                log("FAIL: MAGPIE_SEARCH_BACKUP_VM_PROVIDER=virtualbox but no MAGPIE_SEARCH_BACKUP_VM_NAME")
                return (1, 0, "config")
            if cfg.vm_boot_before and not _virtualbox_running(cfg.vm_name):
                log(f"booting VM {cfg.vm_name}...")
                if not _virtualbox_start(cfg.vm_name, dry, log):
                    return (2, 0, "vm")
                started_vm = True

    # SSH reachability probe
    if not dry:
        log(f"waiting for SSH on {cfg.ssh_host} (timeout {cfg.ssh_ready_timeout_s}s)...")
        if not _wait_for_ssh(cfg.ssh_host, cfg.ssh_ready_timeout_s, log):
            _maybe_suspend(cfg, vmrun, started_vm, no_suspend, dry, log)
            return (3, 0, "ssh-unreachable")
    else:
        log("DRY: skipping SSH probe")

    if not _ensure_remote_dir(cfg.ssh_host, cfg.ssh_dest, dry, log):
        _maybe_suspend(cfg, vmrun, started_vm, no_suspend, dry, log)
        return (4, 0, "ssh")

    log("rsyncing transcripts...")
    ok, file_count, transport = _rsync_to_remote(
        cfg.src_dir, cfg.ssh_host, cfg.ssh_dest, dry, log, cfg)
    if not ok:
        log(f"FAIL: {transport} (had {file_count} source files)")
        _maybe_suspend(cfg, vmrun, started_vm, no_suspend, dry, log)
        return (4, file_count, transport)
    if transport == "rsync":
        log(f"synced {file_count} transcript files (rsync, verified, deletions mirrored)")
    else:
        log(f"copied {file_count} source transcript files via {transport} "
            f"(NOT a verified transfer manifest; deletions NOT propagated)")

    _maybe_suspend(cfg, vmrun, started_vm, no_suspend, dry, log)

    elapsed = int(time.time() - t0)
    log(f"==== backup done in {elapsed}s, {file_count} files via {transport} ====")
    return (0, file_count, transport)


def _maybe_suspend(cfg: BackupConfig, vmrun: Path | None, started_vm: bool,
                   no_suspend: bool, dry: bool, log) -> None:
    """Suspend the VM if we booted it and operator wants suspend-after."""
    if not started_vm or no_suspend or not cfg.vm_suspend_after:
        return
    if cfg.vm_provider == "vmware" and vmrun is not None and cfg.vm_vmx is not None:
        log("suspending VM (we booted it)")
        _vmware_suspend(vmrun, cfg.vm_vmx, dry, log)
    elif cfg.vm_provider == "virtualbox" and cfg.vm_name:
        log("suspending VM (we booted it)")
        _virtualbox_suspend(cfg.vm_name, dry, log)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Back up Claude Code transcripts to a configurable destination.",
        epilog="Config: env vars first, then ~/.magpie-search/backup.env. "
               "See magpie_search/README for keys.",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing.")
    p.add_argument("--no-suspend", action="store_true",
                   help="Leave VM running afterward (default: suspend if we booted it).")
    p.add_argument("--show-config", action="store_true",
                   help="Print resolved config and exit (skips backup).")
    args = p.parse_args(argv)
    try:
        cfg = load_config()
    except BackupConfigError as e:
        # Fail closed + loud: an unsafe host/dest/VM value must never silently
        # run a backup (injection / exfiltration guard).
        sys.stderr.write(f"magpie-search backup: config error: {e}\n")
        return 2
    if args.show_config:
        info = {
            "mode": cfg.mode,
            "src_dir": str(cfg.src_dir),
            "local_dest_dir": str(cfg.local_dest_dir),
            "ssh_host": cfg.ssh_host,
            "ssh_dest": cfg.ssh_dest,
            "vm_provider": cfg.vm_provider,
            "vm_vmx": str(cfg.vm_vmx) if cfg.vm_vmx else None,
            "vm_name": cfg.vm_name,
            "vm_boot_before": cfg.vm_boot_before,
            "vm_suspend_after": cfg.vm_suspend_after,
            "config_file": str(_config_path()),
            "config_file_exists": _config_path().exists(),
        }
        print(json.dumps(info, indent=2))
        return 0
    return run_backup(dry=args.dry_run, no_suspend=args.no_suspend, cfg=cfg)


if __name__ == "__main__":
    raise SystemExit(main())
