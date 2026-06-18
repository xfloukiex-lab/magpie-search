"""Magpi — searchable archive of Claude Code session transcripts.

Local-first. FTS5 + sqlite-vec + fastembed embeddings. Optional LLM
augmentation (rerank + summarize) via the `magpie_search.llm` subpackage.

Public API:
    magpie_search.search(query, k=10, mode='lexical', ...)  — top-k matches
        (mode: 'lexical' default | 'semantic' | 'hybrid')
    magpie_search.recent(n=50, session_id=None)            — session tail
    magpie_search.session(session_id, limit=200, offset=0) — paginated session view
    magpie_search.list_sessions(limit=50, project=None)    — recent sessions
    magpie_search.stats()                                  — index health
    magpie_search.index(source=None)                       — incremental indexing pass
    magpie_search.backfill(batch_size=64, limit=None)      — fill missing embeddings
    magpie_search.redact(text, ner=False)                  — apply redaction patterns

LLM augmentation (requires `pip install magpie_search[llm]` + Ollama for summarize):
    magpie_search.llm.search_rerank(query, k=5, pool=10)   — cross-encoder rerank
    magpie_search.llm.summarize(session_id)                — phi3.5 session summary
    magpie_search.llm.trust_check(n_recent=500)            — audit log monitor

Storage:
    $MAGPIE_SEARCH_HOME / index.db          (sqlite database)
    $MAGPIE_SEARCH_HOME / llm-audit.jsonl   (audit log)
    $MAGPIE_SEARCH_HOME / llm-alerts.jsonl  (trust monitor output)
    Default $MAGPIE_SEARCH_HOME = ~/.magpie-search
"""
import os as _os
from pathlib import Path as _Path


def _backcompat() -> None:
    """Honor pre-rename (``magpi``) installs without moving any data.

    1. Promote legacy ``MAGPI_*`` env vars to their ``MAGPIE_SEARCH_*``
       equivalents (only if the new name is not already set).
    2. When the new ``~/.magpie-search`` data dir does not exist yet but a
       legacy ``~/.magpi`` does, point the reusable assets (model cache,
       backup config) at the old dir in place — no copy, no move. The
       index location is resolved separately (and is unaffected).
    New installs see neither dir and use ``~/.magpie-search`` cleanly.
    """
    for _k in list(_os.environ):
        if _k.startswith("MAGPI_"):
            _os.environ.setdefault("MAGPIE_SEARCH_" + _k[len("MAGPI_"):], _os.environ[_k])
    _old = _Path.home() / ".magpi"
    _new = _Path.home() / ".magpie-search"
    if _old.exists() and not _new.exists():
        if (_old / "models").exists():
            _os.environ.setdefault("MAGPIE_SEARCH_MODELS_DIR", str(_old / "models"))
        if (_old / "backup.env").exists():
            _os.environ.setdefault("MAGPIE_SEARCH_BACKUP_CONFIG", str(_old / "backup.env"))


_backcompat()

from .redactor import redact, REDACTION_PATTERNS
from .search import search, recent, session, list_sessions
from .federation import federated_search
from .indexer import stats_summary as stats, index_all
from .backfill import backfill as _backfill


def index(*, source=None, progress_every: int = 50):
    """Run one incremental indexing pass.

    `source`: override Claude projects dir (default ~/.claude/projects/).
    Returns the IndexStats dataclass."""
    from pathlib import Path
    src = Path(source) if source else None
    return index_all(source_dir=src, progress_every=progress_every)


def backfill(*, batch_size: int = 64, limit: int | None = None):
    """Fill in embeddings for messages indexed before embedding was on.

    Idempotent + resumable. Returns coverage stats dict."""
    return _backfill(batch_size=batch_size, limit=limit)


__all__ = [
    "redact", "REDACTION_PATTERNS",
    "search", "recent", "session", "list_sessions",
    "federated_search",
    "stats", "index", "backfill",
]
__version__ = "1.2.1"
