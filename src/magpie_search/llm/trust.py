"""trust — meta-guardrail watching the LLM audit log for anomalies.

Public:
    check(n_recent=500) -> dict
        Computes bypass rate, fallback rate, probe failures, drift, and
        emergent refusal patterns. Writes alerts to:
          $MAGPIE_SEARCH_HOME/llm-alerts.jsonl  (own log)

For standalone Magpi: run on a schedule (systemd timer / cron).
The companion plugin can register this as `llm.trust_check`
which cuckoo fires hourly.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from . import audit


# Thresholds — env-overridable. Reads are wrapped in try/except so a typo
# in an operator's env var (e.g. "0,5" instead of "0.5") doesn't crash
# module import — which would take down every caller that imports trust,
# including the trust monitor itself.
def _env_float(*names: str, default: float) -> float:
    for n in names:
        v = os.environ.get(n)
        if v:
            try:
                return float(v)
            except (ValueError, TypeError):
                continue
    return default


def _env_int(*names: str, default: int) -> int:
    for n in names:
        v = os.environ.get(n)
        if v:
            try:
                return int(v)
            except (ValueError, TypeError):
                continue
    return default


ALARM_BYPASS_RATE_1H = _env_float(
    "MAGPIE_SEARCH_TRUST_BYPASS_1H", default=0.50,
)
ALARM_FALLBACK_RATE_1H = _env_float(
    "MAGPIE_SEARCH_TRUST_FALLBACK_1H", default=0.30,
)
ALARM_SAME_PROBE_1H = _env_int(
    "MAGPIE_SEARCH_TRUST_SAME_PROBE_1H", default=5,
)
ALARM_DRIFT_DELTA = _env_float(
    "MAGPIE_SEARCH_TRUST_DRIFT", default=0.20,
)
ALARM_MIN_N_1H = _env_int(
    "MAGPIE_SEARCH_TRUST_MIN_N", default=3,
)

# Per-probe threshold overrides. Default global threshold (ALARM_SAME_PROBE_1H)
# is tuned for probes whose failures carry real signal (length,
# proper_noun_safety, identifier_safety, refusal_drift, semantic_grounding).
# Two probes are intentionally noisy and would dominate the alarm channel
# at the global threshold:
#   self_verify — deliberately suppressive (most summaries land degraded per
#                 the no-errors mandate); baseline rate ~5-9/h under normal load.
#   noun_overlap — demoted to advisory, not in the gating set; expected variance.
# Operators can extend or override via MAGPIE_SEARCH_TRUST_PROBE_OVERRIDES, format:
#   "probe_name:threshold,probe_name:threshold"
# e.g. "self_verify:50,length:10"
_DEFAULT_PROBE_OVERRIDES: dict[str, int] = {
    "self_verify": 50,    # ~6x baseline before alarming
    "noun_overlap": 999,  # advisory only, effectively never alarms
}


def _parse_probe_overrides() -> dict[str, int]:
    out = dict(_DEFAULT_PROBE_OVERRIDES)
    raw = os.environ.get("MAGPIE_SEARCH_TRUST_PROBE_OVERRIDES", "").strip()
    if not raw:
        return out
    for pair in raw.split(","):
        if ":" not in pair:
            continue
        k, _, v = pair.partition(":")
        try:
            n = int(v.strip())
        except ValueError:
            continue
        # Clamp to >=1. A threshold of 0 would alarm every single tick
        # (count >= 0 always); negative would alarm even harder. Both are
        # almost certainly operator typos rather than intent.
        if n < 1:
            n = 1
        out[k.strip()] = n
    return out


PROBE_OVERRIDES: dict[str, int] = _parse_probe_overrides()


def _magpi_home() -> Path:
    # `or` treats "" as falsy, but if MAGPIE_SEARCH_HOME is explicitly set to ""
    # via a misconfigured launcher, `os.environ.get` returns "" (truthy
    # for `get(... or ...)` chain? no, "" is falsy) — good. The bug we're
    # guarding against is when both env vars are SET but EMPTY: the prior
    # `Path(base) if base` correctly falls through. Belt-and-braces:
    # require a non-empty string.
    base = os.environ.get("MAGPIE_SEARCH_HOME")
    if base and base.strip():
        return Path(base)
    return Path.home() / ".magpie-search"


def _alert_path() -> Path:
    # Resolution: MAGPIE_SEARCH_ALERTS_LOG > $MAGPIE_SEARCH_HOME/llm-alerts.jsonl
    override = os.environ.get("MAGPIE_SEARCH_ALERTS_LOG")
    if override:
        return Path(override)
    return _magpi_home() / "llm-alerts.jsonl"


def _cuckoo_path() -> Path | None:
    """Standalone build: no external echo path."""
    return None


def _write_alert(record: dict[str, Any]) -> None:
    record = dict(record)
    record.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    record.setdefault("severity", "alert")
    record.setdefault("source", "magpie_search.llm.trust")

    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    paths = [_alert_path()]
    cp = _cuckoo_path()
    if cp:
        paths.append(cp)
    for path in paths:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass


def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _slice(events: list[dict[str, Any]], hours: float) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    out = []
    for ev in events:
        ts = _parse_ts(ev.get("ts"))
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts >= cutoff:
            out.append(ev)
    return out


def _stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(events)
    if n == 0:
        return {"n": 0}
    # Fail-closed: anything that is not explicitly "clean" counts as
    # untrusted — including a record with NO trust field. Every legit
    # writer sets trust (clean/degraded/untrusted/skipped), so a missing
    # value only arises from a torn or forged line; treating it as clean
    # (the old `not in (None, "clean")`) let such lines dilute bypass_rate
    # and mask real untrusted events from the alarm.
    untrusted = sum(1 for e in events if e.get("trust") != "clean")
    fallback = sum(1 for e in events if e.get("fallback_fired"))
    by_role: Counter[str] = Counter()
    by_model: Counter[str] = Counter()
    probe_fails: Counter[str] = Counter()
    for e in events:
        by_role[e.get("role", "?")] += 1
        by_model[e.get("model", "?")] += 1
        probes = e.get("probe_results") or {}
        if isinstance(probes, dict):
            for name, passed in probes.items():
                if not passed:
                    probe_fails[name] += 1
    return {
        "n": n,
        "bypass_rate": round(untrusted / n, 3),
        "fallback_rate": round(fallback / n, 3),
        "by_role": dict(by_role),
        "by_model": dict(by_model),
        "probe_failures": dict(probe_fails),
    }


_KNOWN_PATTERNS = {
    "i'm an ai", "i am an ai", "as an ai", "i cannot", "i can't help",
    "i don't have access", "i apologize", "sorry, but i", "i'm not able to",
    "language model", "as a model", "i don't have", "happy to help",
    "i'm just an", "my knowledge", "i was trained", "training data",
    "as of my last", "i don't know",
}


def _scan_new_refusals(events: list[dict[str, Any]]) -> list[str]:
    import re
    suspect: Counter[str] = Counter()
    for e in events:
        text = (e.get("response") or "").lower()
        for m in re.finditer(r"(?:^|[.!?]\s+)((?:i'm|i\s+\w+|as\s+\w+|sorry,?\s+\w+)[^.!?]{0,40})", text):
            phrase = m.group(1).strip()
            if any(known in phrase for known in _KNOWN_PATTERNS):
                continue
            suspect[phrase[:60]] += 1
    return [p for p, count in suspect.most_common(10) if count >= 3]


def check(*, n_recent: int = 500) -> dict[str, Any]:
    """Audit the LLM call log. Always returns a dict — never raises."""
    t0 = time.time()
    events = audit.tail(n_recent)
    window_1h = _slice(events, hours=1.0)
    window_24h = _slice(events, hours=24.0)

    stats_1h = _stats(window_1h)
    stats_24h = _stats(window_24h)

    drift = {
        "bypass_delta": round(
            stats_1h.get("bypass_rate", 0.0) - stats_24h.get("bypass_rate", 0.0), 3,
        ),
        "fallback_delta": round(
            stats_1h.get("fallback_rate", 0.0) - stats_24h.get("fallback_rate", 0.0), 3,
        ),
    }

    new_refusals = _scan_new_refusals(window_24h)

    alarms: list[str] = []
    if stats_1h.get("n", 0) >= ALARM_MIN_N_1H and stats_1h.get("bypass_rate", 0) >= ALARM_BYPASS_RATE_1H:
        alarms.append(
            f"bypass_rate_1h={stats_1h['bypass_rate']:.0%} ≥ "
            f"{ALARM_BYPASS_RATE_1H:.0%} (n={stats_1h['n']})"
        )
    if stats_1h.get("n", 0) >= ALARM_MIN_N_1H and stats_1h.get("fallback_rate", 0) >= ALARM_FALLBACK_RATE_1H:
        alarms.append(
            f"fallback_rate_1h={stats_1h['fallback_rate']:.0%} ≥ "
            f"{ALARM_FALLBACK_RATE_1H:.0%} (n={stats_1h['n']})"
        )
    for probe, count in stats_1h.get("probe_failures", {}).items():
        threshold = PROBE_OVERRIDES.get(probe, ALARM_SAME_PROBE_1H)
        if count >= threshold:
            alarms.append(f"probe '{probe}' failed {count}× in last 1h")
    if (
        stats_24h.get("n", 0) >= 10
        and drift["bypass_delta"] >= ALARM_DRIFT_DELTA
    ):
        alarms.append(
            f"bypass drift +{drift['bypass_delta']:.0%} vs 24h baseline"
        )
    if new_refusals:
        alarms.append(
            f"new refusal-like patterns seen: {new_refusals[:3]}"
        )

    for line in alarms:
        _write_alert({"alarm": line, "context": {"1h": stats_1h, "24h": stats_24h}})

    return {
        "ok": True,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "window_1h": stats_1h,
        "window_24h": stats_24h,
        "drift": drift,
        "new_refusal_hints": new_refusals,
        "alarms": alarms,
        "alerts_written": len(alarms),
        "elapsed_ms": int((time.time() - t0) * 1000),
    }
