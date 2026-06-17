"""client — Ollama HTTP wrapper with timeout, retry, and audit hookup.

Talks to a local Ollama instance (default :11434, overridable via
$MAGPIE_SEARCH_OLLAMA_HOST or legacy $AVIARY_OLLAMA_HOST).
Every successful or failed call is logged via audit.log.

Public:
    generate(prompt, *, role, model='phi3.5', schema=None, ...) -> dict

Return shape:
    {
      "ok": bool,
      "text": str | None,          # raw model output (post-schema-clean)
      "trust": "clean"|"degraded"|"untrusted",
      "fallback_fired": bool,
      "ms": int,
      "reason": str | None,        # populated on !ok or trust!=clean
    }

The caller is expected to layer worker-specific hallucination probes on
top via guardrails.* — this client only enforces the universal guardrails
(timeout, schema, refusal patterns).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from . import audit

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "phi3.5"
DEFAULT_TIMEOUT = 15.0
DEFAULT_RETRIES = 1

# Refusal patterns — phi3.5 sometimes preambles or refuses; reject those.
_REFUSAL_MARKERS = (
    "i'm an ai",
    "i am an ai",
    "as an ai",
    "i cannot",
    "i can't help",
    "i don't have access",
    "i apologize",
    "sorry, but i",
    "i'm not able to",
)


def _post(host: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    url = host.rstrip("/") + "/api/generate"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body)


def _looks_like_refusal(text: str) -> bool:
    low = text.lower().strip()
    return any(m in low[:200] for m in _REFUSAL_MARKERS)


def generate(
    prompt: str,
    *,
    role: str,
    model: str = DEFAULT_MODEL,
    system: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    temperature: float = 0.0,
    num_predict: int = 256,
    host: str | None = None,
    json_only: bool = False,
) -> dict[str, Any]:
    """One-shot generate. Synchronous. Always returns a dict, never raises.

    role: caller worker identifier — recorded in audit log (e.g.
          'archivist.summarize'). Used by the trust monitor to group calls.
    json_only: if True, response must parse as JSON. Untrusted otherwise.
    """
    host = host or (
        os.environ.get("MAGPIE_SEARCH_OLLAMA_HOST")
        or os.environ.get("AVIARY_OLLAMA_HOST")
        or DEFAULT_HOST
    )
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    if system:
        payload["system"] = system
    if json_only:
        payload["format"] = "json"

    last_err: str | None = None
    t0 = time.time()
    text: str = ""

    for attempt in range(retries + 1):
        try:
            resp = _post(host, payload, timeout=timeout)
            text = (resp.get("response") or "").strip()
            break
        except urllib.error.URLError as e:
            last_err = f"http error: {e}"
        except json.JSONDecodeError as e:
            last_err = f"non-json from ollama: {e}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        # Back off before the next attempt so we don't hammer a host that's
        # still coming up. Skip on the final attempt — no point sleeping
        # before returning the error.
        if attempt < retries:
            time.sleep(min(1.0, 0.2 * (attempt + 1)))

    ms = int((time.time() - t0) * 1000)

    if last_err and not text:
        audit.log({
            "role": role,
            "model": model,
            "prompt": prompt,
            "response": None,
            "trust": "untrusted",
            "fallback_fired": True,
            "ms": ms,
            "reason": last_err,
        })
        return {
            "ok": False, "text": None, "trust": "untrusted",
            "fallback_fired": True, "ms": ms, "reason": last_err,
        }

    # Refusal check — always applied
    if _looks_like_refusal(text):
        audit.log({
            "role": role, "model": model,
            "prompt": prompt, "response": text,
            "trust": "untrusted", "fallback_fired": True, "ms": ms,
            "reason": "refusal pattern",
        })
        return {
            "ok": False, "text": text, "trust": "untrusted",
            "fallback_fired": True, "ms": ms, "reason": "refusal pattern",
        }

    if json_only:
        try:
            json.loads(text)
        except Exception as e:
            audit.log({
                "role": role, "model": model,
                "prompt": prompt, "response": text,
                "trust": "untrusted", "fallback_fired": True, "ms": ms,
                "reason": f"json_only violated: {e}",
            })
            return {
                "ok": False, "text": text, "trust": "untrusted",
                "fallback_fired": True, "ms": ms,
                "reason": f"json_only violated: {e}",
            }

    audit.log({
        "role": role, "model": model,
        "prompt": prompt, "response": text,
        "trust": "clean", "fallback_fired": False, "ms": ms,
    })
    return {
        "ok": True, "text": text, "trust": "clean",
        "fallback_fired": False, "ms": ms, "reason": None,
    }


def health() -> dict[str, Any]:
    """Probe Ollama. Used by trust monitor / boot checks."""
    host = (
        os.environ.get("MAGPIE_SEARCH_OLLAMA_HOST")
        or os.environ.get("AVIARY_OLLAMA_HOST")
        or DEFAULT_HOST
    )
    t0 = time.time()
    try:
        req = urllib.request.Request(host.rstrip("/") + "/api/tags")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        models = [m.get("name", "?") for m in data.get("models", [])]
        return {
            "ok": True, "host": host, "models": models,
            "phi3_available": any(m.startswith("phi3") for m in models),
            "ms": int((time.time() - t0) * 1000),
        }
    except Exception as e:
        return {
            "ok": False, "host": host, "reason": f"{type(e).__name__}: {e}",
            "ms": int((time.time() - t0) * 1000),
        }
