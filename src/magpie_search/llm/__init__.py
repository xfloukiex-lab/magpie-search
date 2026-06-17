"""magpie_search.llm — optional LLM augmentation layer.

Foundation (always available if magpie_search is installed):
    client.generate(prompt, role=..., ...) — Ollama HTTP wrapper
    audit.log(event)                       — JSONL audit log
    audit.tail(n)                          — recent audit entries
    guardrails.summarizer_*(...)           — schema + hallucination probes

Higher-level functions:
    search_rerank(query, k=5, pool=10)     — cross-encoder rerank (no Ollama needed)
    summarize(session_id)                  — phi3.5 session summary
    trust_check(n_recent=500)              — audit log monitor

Conservative-by-design: all gating probes must pass for `trust: clean`.
Otherwise `trust: degraded` and the model output is suppressed.

For the standalone CLI / library, calling these functions is itself the
"enable" signal — no env var gate.
"""
from . import client, audit, guardrails
from .reranker import search_rerank
from .summarizer import summarize
from .trust import check as trust_check

__all__ = [
    "client", "audit", "guardrails",
    "search_rerank", "summarize", "trust_check",
]
