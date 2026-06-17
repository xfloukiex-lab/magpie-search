"""indexer — walk Claude Code transcripts and load them into SQLite FTS5.

Source: `~/.claude/projects/<project-dir>/<session-uuid>.jsonl`
Sink:   `~/.magpie-search/index.db` (standalone) or `$MAGPIE_SEARCH_HOME/index.db` (configurable).
        The aviary-magpi plugin sets `MAGPIE_SEARCH_HOME=~/.aviary/transcripts/` to keep
        existing operator data in place.

Incremental: tracks byte offset + line count per source file in
`index_state`, so live sessions (still being appended to) get picked up
on the next pass without re-reading old lines. mtime is intentionally
NOT used as the staleness signal because a live JSONL has unstable mtime.

Concurrency: WAL mode + advisory lockfile (`$MAGPIE_SEARCH_HOME/.indexer.lock`).
A second indexer process exits cleanly; readers (search.*) are unaffected.

Skipped record types: `permission-mode`, `file-history-snapshot`,
`ai-title`, `last-prompt`, `queue-operation`, `attachment`. These are
metadata, not conversation. `thinking` blocks inside assistant messages
are also skipped — internal reasoning, high volume, low search value.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .redactor import redact
from . import embeddings


# --- personal-magpie_search dedup helpers --------------------------------------
# Optional — gated by MAGPIE_SEARCH_DEDUP / MAGPIE_SEARCH_NOISE_FILTER env vars so
# customer installs see no behavior change.

# Whitespace-normalize so trivially-different formattings of the same
# logical content collide on hash (e.g. trailing newlines, indentation
# variants in pretty-printed JSON).
#
# Python 3 `str` patterns match Unicode whitespace by default — this
# includes NO-BREAK SPACE (U+00A0), EM SPACE (U+2003), IDEOGRAPHIC SPACE
# (U+3000), etc. — so the same logical content with NBSP-vs-regular-space
# variants collides on hash. Verified by `test_whitespace_normalize_handles_unicode`.
# The `re.UNICODE` flag is implicit for str patterns in Py3; including it
# here makes the intent explicit so future code review doesn't add `re.ASCII`
# and silently break NBSP folding.
_HASH_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)


def _normalize_for_hash(text: str) -> str:
    return _HASH_WHITESPACE_RE.sub(" ", text or "").strip()


def _content_hash(text: str) -> str:
    """sha256 hex of the whitespace-normalized text. Used as the dedup key."""
    return hashlib.sha256(
        _normalize_for_hash(text).encode("utf-8", errors="replace")
    ).hexdigest()


# Default noise patterns — high-volume, low-search-value content that
# burns context on every `transcript.search`. Tuned to today's actual
# noise sources (ONNX/CUDA startup warnings, repetitive session-start boilerplate).
# Override or extend via $MAGPIE_SEARCH_NOISE_PATTERNS (newline-separated regex
# entries) or by writing to $MAGPIE_SEARCH_HOME/noise.txt.
_NOISE_PATTERNS_DEFAULT: tuple[re.Pattern[str], ...] = (
    re.compile(r"onnxruntime::TryGetProviderInfo_CUDA"),
    re.compile(r"Failed to create CUDAExecutionProvider"),
    re.compile(r"cublasLt64_12\.dll.*missing"),
    # Anchored to the ACTUAL pytest-internals output shape
    # (`PYTEST_CURRENT_TEST=path::test (call)`), NOT a bare mention.
    # audit 2026-05-15: an unanchored substring as a default-on
    # PERMANENT filter would silently eat any transcript where someone
    # *discusses* the env var (README excerpt, debugging chat). The
    # `=<path>::` shape only appears in real pytest env dumps.
    re.compile(r"PYTEST_CURRENT_TEST=\S+::"),
)


_NOISE_PATTERNS_CACHE: tuple[re.Pattern[str], ...] | None = None


def _noise_patterns() -> tuple[re.Pattern[str], ...]:
    """Resolve the active noise-pattern list. Cached per-process; call
    `_reset_noise_patterns_cache()` from tests to force a re-resolve."""
    global _NOISE_PATTERNS_CACHE
    if _NOISE_PATTERNS_CACHE is not None:
        return _NOISE_PATTERNS_CACHE
    extras: list[re.Pattern[str]] = []
    env_blob = os.environ.get("MAGPIE_SEARCH_NOISE_PATTERNS", "")
    if env_blob:
        for line in env_blob.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                extras.append(re.compile(line))
            except re.error:
                continue
    noise_file = transcripts_dir() / "noise.txt"
    if noise_file.exists():
        try:
            for line in noise_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    extras.append(re.compile(line))
                except re.error:
                    continue
        except OSError:
            pass
    _NOISE_PATTERNS_CACHE = (*_NOISE_PATTERNS_DEFAULT, *extras)
    return _NOISE_PATTERNS_CACHE


def _reset_noise_patterns_cache() -> None:
    global _NOISE_PATTERNS_CACHE
    _NOISE_PATTERNS_CACHE = None


def _is_noise(text: str) -> bool:
    for pat in _noise_patterns():
        if pat.search(text):
            return True
    return False


_OFF_VALUES = ("0", "false", "no", "off", "disable", "disabled")


def dedup_enabled() -> bool:
    """ON by default (operator directive 2026-05-15). Explicit kill switch:
    MAGPIE_SEARCH_DEDUP=0 (or false/no/off). Dedup is non-destructive — every
    row is still indexed; duplicates are only collapsed at search time —
    so default-on is safe for customer installs too. The personal-only
    piece of the original spec was the multi-dir-watching bundle, not
    dedup itself (which the spec says 'customers would still benefit')."""
    val = os.environ.get("MAGPIE_SEARCH_DEDUP", "").strip().lower()
    if val in _OFF_VALUES:
        return False
    return True


def noise_filter_enabled() -> bool:
    """ON by default (operator directive 2026-05-15). Kill switch:
    MAGPIE_SEARCH_NOISE_FILTER=0. The default patterns are conservative,
    well-known machine noise (ONNX/CUDA startup warnings, repetitive
    session-start lines, pytest internals) — content with zero search value.
    Operators who want everything indexed set MAGPIE_SEARCH_NOISE_FILTER=0."""
    val = os.environ.get("MAGPIE_SEARCH_NOISE_FILTER", "").strip().lower()
    if val in _OFF_VALUES:
        return False
    return True


# Max bytes read from a transcript file per indexing pass. Bounds the
# decode+split memory spike (~3x the chunk) on very large / long-lived
# sessions; index_all loops per file so one index() call still fully
# catches up. 64 MiB is far larger than any real single JSONL line.
_MAX_READ_BYTES = 64 * 1024 * 1024


# --- paths -------------------------------------------------------------

def _home() -> Path:
    return Path.home()


_DUAL_DB_WARNED = False


def transcripts_dir() -> Path:
    """Default install location.

    Resolution order:
      1. $MAGPIE_SEARCH_HOME              (preferred)
      2. $AVIARY_TRANSCRIPTS_DIR   (legacy — set by aviary-magpi plugin)
      3. ~/.magpie-search                 (standalone default)

    K1 (magpie_search-hardening audit 2026-05-16, hardened 2026-05-16): when
    neither env var is set, this used to silently pick ~/.magpie-search. But the
    aviary-magpi plugin sets MAGPIE_SEARCH_HOME=~/.aviary/transcripts, so the
    swarm indexed/searched a DIFFERENT database than a bare
    `python -m magpie_search …` invocation. Both could hold a full ~85k-message
    index and diverge with no signal — the same
    env-dependent-resolution-with-no-loud-signal class as the SSH-alias
    backup outage. A one-time stderr WARNING was the first mitigation,
    but a warning still needs a human to be watching — the exact
    "human is the failure detector" pattern we're trying to remove.

    Fixed properly: resolution is now DETERMINISTIC. If an Aviary
    operator index exists (~/.aviary/transcripts/index.db), that install
    is authoritative — a bare invocation resolves to the SAME database
    the swarm maintains, so they cannot diverge. We only fall back to
    the standalone ~/.magpie-search when there is no operator install at all
    (a genuine standalone user, who legitimately wants ~/.magpie-search and is
    unaffected). Non-destructive: nothing is deleted; a pre-existing
    standalone ~/.magpie-search/index.db is simply no longer the active DB on a
    box that is also an Aviary operator. A one-time INFO line states
    which DB was chosen so the resolution is observable, not silent."""
    global _DUAL_DB_WARNED
    env = os.environ.get("MAGPIE_SEARCH_HOME") or os.environ.get("AVIARY_TRANSCRIPTS_DIR")
    if env:
        return Path(env)
    standalone = _home() / ".magpie-search"
    legacy = _home() / ".magpi"
    if not standalone.exists() and (legacy / "index.db").exists():
        # Pre-rename standalone install — reuse its index in place.
        standalone = legacy
    aviary_dir = _home() / ".aviary" / "transcripts"
    aviary_idx = aviary_dir / "index.db"
    if aviary_idx.exists():
        # Aviary operator install detected — it is authoritative. Bare
        # invocations now share the swarm's DB instead of forking a
        # divergent standalone copy (fixes both the read AND write path).
        if not _DUAL_DB_WARNED:
            _DUAL_DB_WARNED = True
            extra = ""
            if (standalone / "index.db").exists():
                extra = (f" (a standalone {standalone / 'index.db'} also "
                         "exists but is NOT used here — it is now inert; "
                         "delete it manually if you want the space back)")
            sys.stderr.write(
                "magpie_search: no MAGPIE_SEARCH_HOME set; Aviary operator install "
                f"detected -> using {aviary_idx}{extra}.\n"
            )
        return aviary_dir
    return standalone


def db_path() -> Path:
    return transcripts_dir() / "index.db"


def claude_projects_dir() -> Path:
    """Source of truth for Claude Code JSONL. Override via $CLAUDE_PROJECTS_DIR."""
    env = os.environ.get("CLAUDE_PROJECTS_DIR")
    return Path(env) if env else _home() / ".claude" / "projects"


def lockfile_path() -> Path:
    return transcripts_dir() / ".indexer.lock"


# --- schema ------------------------------------------------------------

_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages USING fts5(
    session_id UNINDEXED,
    project UNINDEXED,
    ts UNINDEXED,
    role UNINDEXED,
    msg_type UNINDEXED,
    cwd UNINDEXED,
    text,
    tokenize = 'porter unicode61'
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    project TEXT,
    file_path TEXT,
    first_ts TEXT,
    last_ts TEXT,
    message_count INTEGER DEFAULT 0,
    indexed_at TEXT
);

CREATE TABLE IF NOT EXISTS index_state (
    file_path TEXT PRIMARY KEY,
    bytes_read INTEGER NOT NULL DEFAULT 0,
    lines_indexed INTEGER NOT NULL DEFAULT 0,
    last_indexed_at TEXT NOT NULL
);

-- Personal-magpie_search dedup tables (per the "personal-magpie_search must dedup at index time" spec).
-- Empty + harmless on customer installs that never set MAGPIE_SEARCH_DEDUP=1; populated only
-- when the indexer + search code paths see the flag. Migration-safe via IF NOT EXISTS.
CREATE TABLE IF NOT EXISTS messages_meta (
    rowid INTEGER PRIMARY KEY,
    content_sha256 TEXT NOT NULL,
    is_novel_in_session INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS chunk_dedup (
    sha256 TEXT PRIMARY KEY,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project);
CREATE INDEX IF NOT EXISTS idx_sessions_last_ts ON sessions(last_ts DESC);
CREATE INDEX IF NOT EXISTS idx_messages_meta_sha ON messages_meta(content_sha256);
"""

# Vec0 virtual table for semantic search. Created lazily (requires
# sqlite-vec extension load) so the rest of the indexer keeps working
# if the extension is unavailable. messages_vec.rowid mirrors
# messages.rowid (FTS5 implicit rowid) — same row, two indexes.
_VEC_SCHEMA = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS messages_vec USING vec0(
    embedding float[{embeddings.EMBED_DIM}]
);
"""


# --- db helpers --------------------------------------------------------

def connect(path: Path | None = None, *, read_only: bool = False) -> sqlite3.Connection:
    p = path or db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    if read_only:
        uri = f"file:{p}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=10.0)
    else:
        conn = sqlite3.connect(str(p), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    try:
        _load_vec_extension(conn)
    except Exception:
        pass
    return conn


def _load_vec_extension(conn: sqlite3.Connection) -> bool:
    """Best-effort sqlite-vec extension load. Returns True on success.
    Failures are silent so the indexer stays usable without semantic."""
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return True
    except Exception:
        return False


def vec_available(conn: sqlite3.Connection) -> bool:
    """True iff sqlite-vec is loaded on this connection."""
    try:
        conn.execute("SELECT vec_version()").fetchone()
        return True
    except sqlite3.OperationalError:
        return False


def _migrate_messages_meta(conn: sqlite3.Connection) -> None:
    """Layer-4 migration: add is_novel_in_session to a messages_meta
    that predates the column (created by the first dedup build earlier
    today, before layer 4). SQLite has no ADD COLUMN IF NOT EXISTS, so
    probe PRAGMA table_info first. Idempotent."""
    try:
        cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(messages_meta)"
        ).fetchall()}
    except sqlite3.OperationalError:
        return  # table doesn't exist yet; _SCHEMA will create it fresh
    if cols and "is_novel_in_session" not in cols:
        conn.execute(
            "ALTER TABLE messages_meta "
            "ADD COLUMN is_novel_in_session INTEGER NOT NULL DEFAULT 1"
        )


def init_db(conn: sqlite3.Connection) -> None:
    with conn:
        conn.executescript(_SCHEMA)
        _migrate_messages_meta(conn)
        if vec_available(conn):
            conn.executescript(_VEC_SCHEMA)


@contextmanager
def advisory_lock() -> Iterator[None]:
    """Single-writer lock via PID file. Second indexer raises."""
    lf = lockfile_path()
    lf.parent.mkdir(parents=True, exist_ok=True)
    if lf.exists():
        try:
            existing_pid = int(lf.read_text().strip() or "0")
        except (OSError, ValueError):
            existing_pid = 0
        if existing_pid and _pid_alive(existing_pid):
            raise RuntimeError(f"indexer already running (pid {existing_pid})")
        # stale lock
        try:
            lf.unlink()
        except OSError:
            pass
    lf.write_text(str(os.getpid()))
    try:
        yield
    finally:
        try:
            lf.unlink()
        except OSError:
            pass


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        h = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return False
        ctypes.windll.kernel32.CloseHandle(h)  # type: ignore[attr-defined]
        return True
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False


# --- record extraction -------------------------------------------------

_SKIPPED_TYPES = {
    "permission-mode", "file-history-snapshot", "ai-title",
    "last-prompt", "queue-operation", "attachment",
}


@dataclass
class IndexedMessage:
    session_id: str
    project: str
    ts: str
    role: str
    msg_type: str
    cwd: str
    text: str


def _extract_text_from_content(content: Any) -> list[tuple[str, str]]:
    """Return list of (msg_type, text). Skips thinking blocks."""
    if isinstance(content, str):
        return [("text", content)] if content.strip() else []
    if not isinstance(content, list):
        return []
    out: list[tuple[str, str]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        t = item.get("type", "text")
        if t == "thinking":
            continue
        if t == "text":
            txt = item.get("text", "")
            if txt.strip():
                out.append(("text", txt))
        elif t == "tool_use":
            name = item.get("name", "?")
            inp = item.get("input", {})
            # Compact, searchable representation. Redact BEFORE truncating:
            # truncating first can split a multiline secret (e.g. a PEM key)
            # so its end-marker falls past the cut and the regex no longer
            # matches, leaking key material into the index.
            try:
                blob = json.dumps(inp, ensure_ascii=False)
            except (TypeError, ValueError):
                blob = repr(inp)
            out.append(("tool_use", f"[tool:{name}] {redact(blob)[:4000]}"))
        elif t == "tool_result":
            content_val = item.get("content", "")
            if isinstance(content_val, list):
                parts: list[str] = []
                for c in content_val:
                    if isinstance(c, dict) and c.get("type") == "text":
                        parts.append(c.get("text", ""))
                txt = "\n".join(parts)
            else:
                txt = str(content_val)
            if txt.strip():
                out.append(("tool_result", redact(txt)[:8000]))  # redact before truncate
        else:
            # Unknown block type — index a stub for visibility. Redact before
            # truncating (see tool_use note above).
            out.append((f"block:{t}", redact(json.dumps(item, ensure_ascii=False))[:2000]))
    return out


def _project_name_from_path(p: Path) -> str:
    """`.../projects/<project-slug>/<session-id>.jsonl` -> `<project-slug>`.

    Claude Code stores transcripts under `~/.claude/projects/<slug>/` where
    the slug is a flattened form of the original cwd (e.g. `/home/user/repo`
    becomes `-home-user-repo`)."""
    try:
        parts = p.parts
        idx = parts.index("projects")
        return parts[idx + 1] if idx + 1 < len(parts) else "?"
    except ValueError:
        return p.parent.name


def parse_line(line: str, file_path: Path) -> list[IndexedMessage]:
    line = line.strip()
    if not line:
        return []
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return []
    t = obj.get("type", "")
    if t in _SKIPPED_TYPES or t not in ("user", "assistant", "system"):
        return []

    sid = obj.get("sessionId") or file_path.stem
    ts = obj.get("timestamp", "")
    cwd = obj.get("cwd", "")
    project = _project_name_from_path(file_path)

    msg = obj.get("message")
    if not isinstance(msg, dict):
        return []
    role = msg.get("role", t)
    content = msg.get("content")
    blocks = _extract_text_from_content(content)
    out: list[IndexedMessage] = []
    for mtype, raw_text in blocks:
        redacted = redact(raw_text)
        out.append(IndexedMessage(
            session_id=sid,
            project=project,
            ts=ts,
            role=role,
            msg_type=mtype,
            cwd=cwd,
            text=redacted,
        ))
    return out


# --- indexing pass -----------------------------------------------------

@dataclass
class IndexStats:
    files_seen: int = 0
    files_updated: int = 0
    lines_read: int = 0
    messages_indexed: int = 0
    bytes_read: int = 0
    dedup_backfilled: int = 0
    elapsed_s: float = 0.0


def index_all(*, source_dir: Path | None = None, progress_every: int = 50) -> IndexStats:
    """Walk all JSONL under source_dir, index new content. Returns stats."""
    src = source_dir or claude_projects_dir()
    if not src.exists():
        raise FileNotFoundError(f"Claude projects dir not found: {src}")

    transcripts_dir().mkdir(parents=True, exist_ok=True)
    stats = IndexStats()
    started = time.monotonic()

    with advisory_lock():
        conn = connect()
        try:
            init_db(conn)
            jsonl_files = sorted(src.rglob("*.jsonl"))
            stats.files_seen = len(jsonl_files)
            for i, fp in enumerate(jsonl_files, 1):
                # _index_one_file reads at most _MAX_READ_BYTES per call
                # (bounded memory). Loop until this file is fully caught up —
                # the same connection sees its own index_state cursor advance,
                # so each pass resumes where the last stopped. Bounded by a
                # generous per-file pass cap as an infinite-loop backstop.
                file_lines = file_msgs = file_bytes = 0
                file_updated = False
                try:
                    for _pass in range(100000):
                        delta = _index_one_file(conn, fp)
                        if delta is None:
                            break
                        lines, messages, bytes_added = delta
                        file_lines += lines
                        file_msgs += messages
                        file_bytes += bytes_added
                        if lines:
                            file_updated = True
                        if bytes_added <= 0:
                            break  # no further progress (caught up / incomplete tail)
                except Exception as e:
                    print(f"  ! {fp.name}: {e}", file=sys.stderr)
                    continue
                if file_updated:
                    stats.files_updated += 1
                stats.lines_read += file_lines
                stats.messages_indexed += file_msgs
                stats.bytes_read += file_bytes
                if progress_every and i % progress_every == 0:
                    print(f"  ... {i}/{len(jsonl_files)} files, "
                          f"{stats.messages_indexed} msgs indexed")
        finally:
            conn.close()

        # Self-healing dedup catch-up. The live path only writes
        # messages_meta when dedup_enabled() is true AT INDEX TIME; any
        # rows indexed while the flag was off (e.g. before dedup went
        # on-by-default, or a pass that set MAGPIE_SEARCH_DEDUP=0) are permanently
        # meta-less unless re-swept. backfill_dedup is idempotent,
        # batched, and returns instantly when nothing is missing (a
        # single COUNT), so running it every pass makes the dedup
        # metadata complete-by-construction instead of a decaying
        # one-shot. Indexing has already committed; a catch-up failure
        # must NOT fail the pass and must NOT be swallowed silently — log
        # it loudly.
        #
        # HIGH-4 (code review R2): this runs INSIDE the advisory lock (the
        # indexing conn is already closed above, but the lock is still
        # held). backfill_dedup opens its own connection and runs DDL
        # (init_db); doing that while another process's index_all writes
        # under WAL caused contention that surfaced as a caught
        # exception -> a missed backfill on busy 85k boxes. Holding the
        # advisory lock here serializes it with every other indexer, so
        # there is no competing writer. backfill_dedup does NOT re-acquire
        # the lock (it only connect()s), so no reentry / RuntimeError.
        if dedup_enabled():
            try:
                bf = backfill_dedup()
                if bf.get("ok"):
                    stats.dedup_backfilled = int(bf.get("backfilled", 0) or 0)
                    if stats.dedup_backfilled:
                        print(
                            f"  dedup catch-up: backfilled "
                            f"{stats.dedup_backfilled} meta-less messages",
                            file=sys.stderr,
                        )
                else:
                    print(
                        f"  ! dedup catch-up skipped: {bf.get('reason')}",
                        file=sys.stderr,
                    )
            except Exception as e:
                print(f"  ! dedup catch-up FAILED (indexing OK): {e}",
                      file=sys.stderr)

    stats.elapsed_s = time.monotonic() - started
    return stats


def _index_one_file(conn: sqlite3.Connection, fp: Path) -> tuple[int, int, int] | None:
    """Index any new bytes in `fp`. Returns (lines, messages, bytes_added)."""
    cur = conn.execute(
        "SELECT bytes_read, lines_indexed FROM index_state WHERE file_path=?",
        (str(fp),),
    )
    row = cur.fetchone()
    prev_bytes = row["bytes_read"] if row else 0
    prev_lines = row["lines_indexed"] if row else 0

    size = fp.stat().st_size
    if size == prev_bytes:
        return (0, 0, 0)
    if size < prev_bytes:
        # File was truncated/replaced — reindex from start.
        prev_bytes = 0
        prev_lines = 0
        # Capture rowids first so we can cascade the wipe into messages_vec.
        # vec0 has no FK, so we delete by explicit rowid list.
        doomed = [r[0] for r in conn.execute(
            "SELECT rowid FROM messages WHERE session_id IN "
            "(SELECT session_id FROM sessions WHERE file_path=?)",
            (str(fp),),
        ).fetchall()]
        conn.execute(
            "DELETE FROM messages WHERE session_id IN "
            "(SELECT session_id FROM sessions WHERE file_path=?)",
            (str(fp),),
        )
        conn.execute("DELETE FROM sessions WHERE file_path=?", (str(fp),))
        if doomed:
            # HIGH-2 (magpie_search-hardening audit 2026-05-16): FTS5 reuses
            # rowids after delete. If we drop the messages rows but leave
            # their messages_meta rows, a freshly-indexed message can be
            # assigned a recycled rowid that still maps (via messages_meta)
            # to the OLD content's hash — dedup then clusters the wrong
            # messages. Wipe messages_meta for doomed rowids too.
            conn.executemany(
                "DELETE FROM messages_meta WHERE rowid = ?",
                [(r,) for r in doomed],
            )
            if vec_available(conn):
                conn.executemany(
                    "DELETE FROM messages_vec WHERE rowid = ?",
                    [(r,) for r in doomed],
                )

    new_messages = 0
    new_lines = 0
    bytes_after = prev_bytes
    first_ts: str | None = None
    last_ts: str | None = None
    session_id: str | None = None
    project: str | None = None

    with fp.open("rb") as f:
        f.seek(prev_bytes)
        buf = f.read(_MAX_READ_BYTES)  # bounded read: avoid a 3x RAM spike on a multi-GB session
    if not buf:
        return (0, 0, 0)
    # Decode safely; trailing partial line (if any) is held back.
    text = buf.decode("utf-8", errors="replace")
    lines = text.split("\n")
    # Drop the trailing element: if buf ends with "\n" it's the empty
    # string after the last newline; if it doesn't, it's an incomplete
    # final line we must NOT index yet. Either way `lines[:-1]` is the
    # set of complete lines.
    complete_lines = lines[:-1]

    # Advance the byte cursor to exactly the end of the last COMPLETE
    # line. CRIT-1 (magpie_search-hardening audit 2026-05-16): the old code
    # only corrected the cursor `if ... and complete_lines`. When a read
    # had NO newline at all (one long partial line, common on a live
    # JSONL mid-append, especially with a non-UTF-8 byte), complete_lines
    # was empty so the correction was skipped and bytes_after stayed at
    # EOF — permanently skipping that content (next pass sees
    # size == prev_bytes and returns early). Silent data loss. Compute
    # the cursor from the raw bytes unconditionally:
    if buf.endswith(b"\n"):
        bytes_after = prev_bytes + len(buf)        # every line complete
    else:
        last_nl = buf.rfind(b"\n")                  # raw bytes, UTF-8-safe
        if last_nl >= 0:
            bytes_after = prev_bytes + last_nl + 1  # up to last complete line
        else:
            bytes_after = prev_bytes                # no complete line — re-read next pass

    msg_rows: list[IndexedMessage] = []
    for ln in complete_lines:
        new_lines += 1
        items = parse_line(ln, fp)
        if not items:
            continue
        if first_ts is None:
            first_ts = items[0].ts
        last_ts = items[-1].ts
        session_id = items[0].session_id
        project = items[0].project
        msg_rows.extend(items)

    # Personal-magpie_search noise filter — drop messages matching well-known
    # high-volume / low-search-value patterns BEFORE indexing. Gated by
    # MAGPIE_SEARCH_NOISE_FILTER=1 so customer installs are unaffected.
    if noise_filter_enabled() and msg_rows:
        msg_rows = [m for m in msg_rows if not _is_noise(m.text)]

    if msg_rows:
        # Snapshot rowid ceiling BEFORE inserts. With the advisory lock
        # held we're the only writer, so the FTS5 rowids assigned by
        # executemany are deterministically prev_max+1 .. prev_max+N —
        # which we then reuse as messages_vec.rowid below to keep the
        # two indexes joinable without a separate id column.
        prev_max = conn.execute(
            "SELECT IFNULL(MAX(rowid), 0) FROM messages"
        ).fetchone()[0]
        conn.executemany(
            "INSERT INTO messages (session_id, project, ts, role, msg_type, cwd, text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(m.session_id, m.project, m.ts, m.role, m.msg_type, m.cwd, m.text)
             for m in msg_rows],
        )
        new_messages = len(msg_rows)
        rowids = list(range(prev_max + 1, prev_max + 1 + len(msg_rows)))
        # HIGH-1 (magpie_search-hardening audit 2026-05-16): messages_vec /
        # messages_meta rowids are derived from the assumption that FTS5
        # assigned exactly prev_max+1 .. prev_max+N contiguously. The
        # advisory lock makes us the sole writer so this holds today,
        # but it's an undocumented invariant — if FTS5 ever assigns a
        # non-contiguous range, the vec/meta rows would silently bind to
        # the WRONG messages (same mis-map class as HIGH-2). Assert it
        # loudly instead of trusting it silently: cheap (one MAX query)
        # and converts a silent corruption into an immediate failure.
        _max_after = conn.execute(
            "SELECT IFNULL(MAX(rowid), 0) FROM messages"
        ).fetchone()[0]
        if _max_after != prev_max + len(msg_rows):
            conn.rollback()
            raise RuntimeError(
                f"FTS5 rowid invariant violated in {fp.name}: expected max "
                f"{prev_max + len(msg_rows)} after inserting {len(msg_rows)} "
                f"rows from base {prev_max}, got {_max_after}. Aborting this "
                "file to avoid mis-mapping messages_vec/messages_meta."
            )

        # Personal-magpie_search dedup bookkeeping — gated by MAGPIE_SEARCH_DEDUP=1 so
        # customer wheel is unchanged.
        existing_hashes: set[str] = set()
        hashes: list[str] = []
        if dedup_enabled():
            hashes = [_content_hash(m.text) for m in msg_rows]
            # Pre-check which hashes were already in chunk_dedup so we
            # can skip embedding the duplicates. Batched to avoid N
            # queries on a large insert.
            if hashes:
                # Batch the IN(...) lookup: a single pass can hold far more
                # than SQLite's ~32766-variable limit (a 64 MiB read of short
                # JSONL lines), which would raise "too many SQL variables" and
                # skip the whole file. Chunk well under the limit.
                for _i in range(0, len(hashes), 900):
                    batch = hashes[_i:_i + 900]
                    placeholders = ",".join("?" * len(batch))
                    already = conn.execute(
                        f"SELECT sha256 FROM chunk_dedup WHERE sha256 IN ({placeholders})",
                        batch,
                    ).fetchall()
                    existing_hashes.update(r[0] for r in already)
            # Layer 4 — per-session novelty. A chunk is novel-in-session
            # if its (session, hash) pair hasn't been seen before, either
            # earlier in THIS batch or in a prior incremental pass of the
            # same session (live sessions get re-indexed as they grow).
            prior_session_hashes: set[str] = set()
            if session_id:
                try:
                    prior_session_hashes = {
                        r[0] for r in conn.execute(
                            "SELECT DISTINCT mm.content_sha256 "
                            "FROM messages_meta mm "
                            "JOIN messages m ON m.rowid = mm.rowid "
                            "WHERE m.session_id = ?",
                            (session_id,),
                        ).fetchall()
                    }
                except sqlite3.OperationalError:
                    prior_session_hashes = set()
            seen_in_session = set(prior_session_hashes)
            novelty: list[int] = []
            for h in hashes:
                if h in seen_in_session:
                    novelty.append(0)
                else:
                    novelty.append(1)
                    seen_in_session.add(h)
            # messages_meta: one row per messages row, joinable for
            # search-time clustering + novelty-aware ranking.
            conn.executemany(
                "INSERT INTO messages_meta(rowid, content_sha256, is_novel_in_session) "
                "VALUES (?, ?, ?)",
                [(rowids[i], hashes[i], novelty[i]) for i in range(len(msg_rows))],
            )
            # chunk_dedup: upsert with count++ on conflict.
            for h, m in zip(hashes, msg_rows):
                conn.execute(
                    "INSERT INTO chunk_dedup(sha256, first_seen_at, last_seen_at, count) "
                    "VALUES (?, ?, ?, 1) "
                    "ON CONFLICT(sha256) DO UPDATE SET "
                    "  last_seen_at=excluded.last_seen_at, "
                    "  count=chunk_dedup.count + 1",
                    (h, m.ts or "", m.ts or ""),
                )

        if vec_available(conn) and embeddings.available():
            # Skip embedding for content whose hash already had an
            # embedding from a prior occurrence — saves the most
            # expensive cost in the pipeline. When dedup is OFF this
            # filter is a no-op (existing_hashes is empty).
            if dedup_enabled() and existing_hashes:
                embed_pairs = [
                    (rowids[i], msg_rows[i].text)
                    for i in range(len(msg_rows))
                    if hashes[i] not in existing_hashes
                ]
            else:
                embed_pairs = list(zip(rowids, [m.text for m in msg_rows]))
            if embed_pairs:
                try:
                    vecs = embeddings.embed_batch([t for _, t in embed_pairs])
                    if vecs:
                        conn.executemany(
                            "INSERT INTO messages_vec(rowid, embedding) VALUES (?, ?)",
                            [(rid, v) for (rid, _), v in zip(embed_pairs, vecs)],
                        )
                except Exception as e:
                    print(f"  ! embed {fp.name}: {e}", file=sys.stderr)

    if session_id is not None:
        conn.execute(
            "INSERT INTO sessions (session_id, project, file_path, first_ts, last_ts, "
            "message_count, indexed_at) VALUES (?, ?, ?, ?, ?, ?, datetime('now')) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "  last_ts=COALESCE(excluded.last_ts, sessions.last_ts), "
            "  message_count=sessions.message_count + excluded.message_count, "
            "  indexed_at=excluded.indexed_at",
            (session_id, project, str(fp), first_ts, last_ts, new_messages),
        )

    conn.execute(
        "INSERT INTO index_state (file_path, bytes_read, lines_indexed, last_indexed_at) "
        "VALUES (?, ?, ?, datetime('now')) "
        "ON CONFLICT(file_path) DO UPDATE SET "
        "  bytes_read=excluded.bytes_read, "
        "  lines_indexed=index_state.lines_indexed + excluded.lines_indexed, "
        "  last_indexed_at=excluded.last_indexed_at",
        (str(fp), bytes_after, new_lines),
    )
    conn.commit()
    return (new_lines, new_messages, bytes_after - prev_bytes)


def backfill_dedup(*, batch: int = 5000, progress=None) -> dict[str, Any]:
    """Populate messages_meta + chunk_dedup for already-indexed messages.

    The dedup tables only ever got rows for content indexed AFTER the
    feature went on. The 85k-message backlog had zero dedup data, so the
    "context saving" was unmeasurable (~1.08x observed vs the 5x design
    target). This walks every messages row lacking a messages_meta row,
    computes its content hash + per-session novelty, and fills both
    tables. Idempotent (only touches un-backfilled rows), resumable
    (batched commits), memory-bounded (streams by rowid).

    Novelty: processed in (session_id, rowid) order; the first time a
    hash appears within a session = novel(1), repeats = 0. Existing
    messages_meta rows for a session are honored so a partial backfill
    + resume stays correct.
    """
    p = db_path()
    if not p.exists():
        return {"ok": False, "reason": "index not built yet"}
    conn = connect()
    try:
        init_db(conn)
        total_missing = conn.execute(
            "SELECT COUNT(*) FROM messages m "
            "WHERE NOT EXISTS (SELECT 1 FROM messages_meta mm WHERE mm.rowid=m.rowid)"
        ).fetchone()[0]
        if total_missing == 0:
            return {"ok": True, "backfilled": 0, "already_complete": True}

        # Seed per-session seen-hashes from any rows ALREADY in
        # messages_meta (correctness across resume / mixed state).
        seen: dict[str, set[str]] = {}
        for r in conn.execute(
            "SELECT m.session_id AS sid, mm.content_sha256 AS h "
            "FROM messages_meta mm JOIN messages m ON m.rowid=mm.rowid"
        ):
            seen.setdefault(r["sid"], set()).add(r["h"])

        done = 0
        while True:
            rows = conn.execute(
                "SELECT m.rowid AS rid, m.session_id AS sid, m.text AS txt "
                "FROM messages m "
                "WHERE NOT EXISTS (SELECT 1 FROM messages_meta mm "
                "                  WHERE mm.rowid=m.rowid) "
                "ORDER BY m.session_id, m.rowid "
                "LIMIT ?", (batch,),
            ).fetchall()
            if not rows:
                break
            meta_rows = []
            for r in rows:
                h = _content_hash(r["txt"] or "")
                ss = seen.setdefault(r["sid"], set())
                novel = 0 if h in ss else 1
                if novel:
                    ss.add(h)
                meta_rows.append((r["rid"], h, novel))
                conn.execute(
                    "INSERT INTO chunk_dedup(sha256, first_seen_at, last_seen_at, count) "
                    "VALUES (?, '', '', 1) "
                    "ON CONFLICT(sha256) DO UPDATE SET count=chunk_dedup.count+1",
                    (h,),
                )
            conn.executemany(
                "INSERT INTO messages_meta(rowid, content_sha256, is_novel_in_session) "
                "VALUES (?, ?, ?)", meta_rows,
            )
            conn.commit()
            done += len(rows)
            if progress:
                progress(done, total_missing)

        cd = conn.execute("SELECT COUNT(*) FROM chunk_dedup").fetchone()[0]
        occ = conn.execute("SELECT IFNULL(SUM(count),0) FROM chunk_dedup").fetchone()[0]
        return {
            "ok": True,
            "backfilled": done,
            "chunk_dedup_uniques": cd,
            "total_occurrences": occ,
            "dedup_ratio": round(occ / cd, 3) if cd else 1.0,
        }
    finally:
        conn.close()


def stats_summary() -> dict[str, Any]:
    """Quick read-only stats for monitoring."""
    p = db_path()
    if not p.exists():
        return {"ok": False, "reason": "index not built yet", "db_path": str(p)}
    conn = connect(read_only=True)
    try:
        msgs = conn.execute("SELECT COUNT(*) AS n FROM messages").fetchone()["n"]
        sess = conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"]
        last = conn.execute(
            "SELECT MAX(last_indexed_at) AS t FROM index_state"
        ).fetchone()["t"]
        size = p.stat().st_size
        embedded = 0
        if vec_available(conn):
            try:
                embedded = conn.execute(
                    "SELECT COUNT(*) AS n FROM messages_vec"
                ).fetchone()["n"]
            except sqlite3.OperationalError:
                embedded = 0
    finally:
        conn.close()
    return {
        "ok": True,
        "db_path": str(p),
        "db_size_bytes": size,
        "messages": msgs,
        "embedded": embedded,
        "embed_coverage": (embedded / msgs) if msgs else 0.0,
        "sessions": sess,
        "last_indexed_at": last,
        "semantic_available": embeddings.available(),
    }
