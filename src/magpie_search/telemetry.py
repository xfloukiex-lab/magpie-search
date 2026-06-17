"""Opt-in, anonymous usage telemetry. OFF by default.

magpie-search sends NOTHING unless you explicitly opt in
(`magpie-search telemetry enable`, or set MAGPIE_SEARCH_TELEMETRY=1). When on, it
sends anonymous usage events — which command ran, search mode, result/hit counts,
latency, error class, and your magpie/python/OS versions — to help us see what to
improve.

It NEVER sends user content. There is a hard firewall in `_clean()`: only numbers,
booleans, and short enum-like tokens (no spaces) pass; any free-text value — a
query, a file path, a snippet, a username — is dropped before send. The collector
also never stores sender IPs. Transport is fire-and-forget on a daemon thread with
a short timeout and fails open, so telemetry can never slow or break the tool.
"""
from __future__ import annotations

import json
import os
import re
import threading
import urllib.request
import uuid
from pathlib import Path

from . import __version__

# The collector endpoint (operator-hosted; override to point elsewhere or to a
# self-hosted collector). Anonymous events POST here only when opted in.
DEFAULT_URL = "https://vektor.taildabcb6.ts.net/v1/ingest"

# A safe value is a number/bool, or a short token with no spaces (mode names,
# error classes, os/arch). Anything else (free text = possible user content) is
# dropped. This is the content firewall.
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.:+\-]{1,48}$")


def _home() -> Path:
    return Path(os.environ.get("MAGPIE_SEARCH_HOME", Path.home() / ".magpie-search"))


def _flag_file() -> Path:
    return _home() / "telemetry_enabled"


def _id_file() -> Path:
    return _home() / "install_id"


def _notice_file() -> Path:
    return _home() / ".telemetry_notice_shown"


def is_enabled() -> bool:
    env = os.environ.get("MAGPIE_SEARCH_TELEMETRY")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    return _flag_file().exists()


def install_id() -> str:
    """Stable anonymous id — a random UUID, never tied to anything identifying."""
    f = _id_file()
    try:
        if f.exists():
            return f.read_text("utf-8").strip()
        f.parent.mkdir(parents=True, exist_ok=True)
        new = uuid.uuid4().hex
        f.write_text(new, "utf-8")
        return new
    except Exception:
        return "anon"


def enable() -> None:
    f = _flag_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("1", "utf-8")


def disable() -> None:
    try:
        _flag_file().unlink()
    except FileNotFoundError:
        pass


def _endpoint() -> str:
    return os.environ.get("MAGPIE_SEARCH_TELEMETRY_URL", DEFAULT_URL)


def _clean(props: dict) -> dict:
    """Content firewall — keep only numbers, bools, and short space-free tokens."""
    out: dict = {}
    for k, v in props.items():
        if v is None:
            continue
        if isinstance(v, bool) or isinstance(v, (int, float)):
            out[k] = v
        elif isinstance(v, str) and _SAFE_TOKEN.match(v):
            out[k] = v
        # everything else (free text, lists, dicts) is intentionally dropped
    return out


def _send(payload: dict) -> None:
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            _endpoint(), data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=3).read()
    except Exception:
        pass  # fail-open: telemetry never raises


def emit(event: str, **props) -> threading.Thread | None:
    """Send one anonymous event if opted in. Non-blocking, never raises.

    Returns the daemon thread (so a short-lived CLI can briefly join it before
    exit, otherwise the POST can be cut off mid-flight) or None when disabled."""
    if not is_enabled():
        return None
    import platform
    payload = {
        "install_id": install_id(),
        "product": "magpie-search",
        "version": __version__,
        "event": str(event)[:48],
        "py": platform.python_version(),
        "os": platform.system(),
        "arch": platform.machine(),
        "props": _clean(props),
    }
    t = threading.Thread(target=_send, args=(payload,), daemon=True)
    t.start()
    return t


def maybe_first_run_notice() -> None:
    """Print a one-time, unobtrusive notice that telemetry exists and is OFF."""
    if is_enabled() or _notice_file().exists():
        return
    try:
        _notice_file().parent.mkdir(parents=True, exist_ok=True)
        _notice_file().write_text("1", "utf-8")
    except Exception:
        return
    import sys
    print("magpie-search collects no telemetry by default. To help improve it, "
          "opt in (anonymous, no query/file content):\n"
          "    magpie-search telemetry enable", file=sys.stderr)


def status() -> dict:
    return {"enabled": is_enabled(), "endpoint": _endpoint(),
            "install_id": install_id() if is_enabled() else "(set on first opt-in)"}
