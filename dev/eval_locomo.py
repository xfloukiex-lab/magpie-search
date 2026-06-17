"""eval_locomo — LOCOMO retrieval benchmark for transcript_indexer.

Measures Recall@K of our search modes against the SNAP Research LOCOMO
benchmark (10 long-term conversations, ~200 QA each, evidence references
to specific dia_ids).

Pipeline:
  1. Load locomo10.json from ~/.magpie-search/eval/locomo10.json.
  2. Build a temp SQLite DB with the SAME schema as production (FTS5 +
     vec0 + redactor + embedder) — proves the actual codepath, no shims.
  3. For each conversation: index all turns, embed them, then run each
     QA's question through search() in three modes and check whether the
     evidence dia_id appears in the top-K hits.
  4. Report Recall@1, Recall@5, Recall@10 per mode.

Stores dia_id in the `cwd` UNINDEXED column so it doesn't pollute ranking
but is retrievable from each hit.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from . import embeddings, indexer, redactor


LOCOMO_PATH = Path.home() / ".magpie-search" / "eval" / "locomo10.json"


def load_locomo(path: Path = LOCOMO_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def index_conversation_into_temp_db(
    db_path: Path,
    conv: dict,
    conv_idx: int,
) -> int:
    """Build a FTS5 + vec0 index for ONE conversation. Returns msg count."""
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    indexer._load_vec_extension(conn)
    indexer.init_db(conn)

    sid = f"locomo-{conv_idx}"
    session_data = conv.get("conversation", {})

    rows: list[tuple] = []
    texts: list[str] = []
    for key, val in session_data.items():
        if not key.startswith("session_") or key.endswith("_date_time"):
            continue
        if not isinstance(val, list):
            continue
        for turn in val:
            if not isinstance(turn, dict):
                continue
            speaker = turn.get("speaker", "?")
            dia_id = turn.get("dia_id", "")
            raw = turn.get("text", "")
            if not raw or not dia_id:
                continue
            redacted = redactor.redact(raw)
            # cwd column carries dia_id (UNINDEXED — doesn't affect FTS ranking)
            rows.append((sid, f"locomo-{conv_idx}", dia_id, speaker, "text", dia_id, redacted))
            texts.append(redacted)

    if not rows:
        conn.close()
        return 0

    prev_max = conn.execute("SELECT IFNULL(MAX(rowid), 0) FROM messages").fetchone()[0]
    conn.executemany(
        "INSERT INTO messages (session_id, project, ts, role, msg_type, cwd, text) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    if indexer.vec_available(conn):
        vecs = embeddings.embed_batch(texts)
        if vecs:
            rowids = range(prev_max + 1, prev_max + 1 + len(rows))
            conn.executemany(
                "INSERT INTO messages_vec(rowid, embedding) VALUES (?, ?)",
                list(zip(rowids, vecs)),
            )
    conn.commit()
    conn.close()
    return len(rows)


def search_temp(db_path: Path, query: str, k: int, mode: str) -> list[dict]:
    """Mode-aware search against a temp DB. Returns list of {dia_id, ...}."""
    # We bypass the global db_path() machinery by setting the env var.
    # MED-4 (c1 R2): os.environ is PROCESS-global — this is safe only
    # because run() processes conversations strictly sequentially. Do
    # NOT parallelize the conv loop without passing db_path explicitly
    # into search(); two threads would race on MAGPI_HOME and silently
    # search the wrong temp DB.
    saved = os.environ.get("MAGPI_HOME")
    os.environ["MAGPI_HOME"] = str(db_path.parent)
    # Force the search module to re-resolve db_path() against the env var
    try:
        # `from . import search` resolves to the re-exported *function*
        # (magpi/__init__.py), not the module — `search_mod.search` was an
        # AttributeError, so this harness never actually ran. Import the
        # symbol directly. search() calls indexer.db_path() per-call, so the
        # MAGPI_HOME override set above is still honored.
        from .search import search as _search_fn
        result = _search_fn(query, k=k, mode=mode)
    finally:
        if saved is None:
            os.environ.pop("MAGPI_HOME", None)
        else:
            os.environ["MAGPI_HOME"] = saved
    if not result.get("ok"):
        return []
    return result.get("hits", [])


def evaluate_conversation(
    db_dir: Path,
    conv: dict,
    conv_idx: int,
    *,
    k_max: int = 10,
    qa_limit: int | None = None,
) -> dict[str, Any]:
    """Run all QAs against the indexed conversation. Return per-mode recalls."""
    qas = conv.get("qa", [])
    if qa_limit:
        qas = qas[:qa_limit]

    modes = ("lexical", "semantic", "hybrid")
    # buckets[mode][k] = count of QAs where evidence was found in top-k
    buckets: dict[str, dict[int, int]] = {m: {1: 0, 5: 0, 10: 0} for m in modes}
    answered: dict[str, int] = {m: 0 for m in modes}

    db_path = db_dir / "index.db"

    for qa in qas:
        question = qa.get("question", "")
        evidence = set(qa.get("evidence") or [])
        if not question or not evidence:
            continue
        for mode in modes:
            hits = search_temp(db_path, question, k=k_max, mode=mode)
            if not hits:
                continue
            answered[mode] += 1
            found_at: int | None = None
            for rank, h in enumerate(hits, 1):
                if h.get("cwd") in evidence:
                    found_at = rank
                    break
            if found_at is not None:
                for k in (1, 5, 10):
                    if found_at <= k:
                        buckets[mode][k] += 1

    total = sum(1 for qa in qas if qa.get("question") and qa.get("evidence"))
    out = {"conv_idx": conv_idx, "total_qas": total, "answered": answered, "recall": {}}
    for mode in modes:
        out["recall"][mode] = {
            f"R@{k}": (buckets[mode][k] / total) if total else 0.0
            for k in (1, 5, 10)
        }
    return out


def run(
    *,
    qa_limit_per_conv: int | None = None,
    conv_limit: int | None = None,
) -> dict[str, Any]:
    if not LOCOMO_PATH.exists():
        return {"ok": False, "reason": f"dataset not found: {LOCOMO_PATH}"}
    if not embeddings.available():
        return {"ok": False, "reason": f"embeddings unavailable: {embeddings.load_error()}"}

    conversations = load_locomo()
    if conv_limit:
        conversations = conversations[:conv_limit]

    started = time.monotonic()
    per_conv: list[dict] = []

    for i, conv in enumerate(conversations):
        # Fresh temp db per conversation so cross-conv evidence can't leak.
        with tempfile.TemporaryDirectory(prefix=f"locomo-{i}-") as td:
            td_path = Path(td)
            n = index_conversation_into_temp_db(td_path / "index.db", conv, i)
            if n == 0:
                continue
            r = evaluate_conversation(
                td_path, conv, i, qa_limit=qa_limit_per_conv,
            )
            r["msgs_indexed"] = n
            per_conv.append(r)
            # NIT-2 (c1 R2): progress -> stderr so stdout stays clean
            # JSON for `subprocess.check_output(...)` callers.
            print(f"  conv {i}: {n} msgs, {r['total_qas']} qas — "
                  f"lex R@5={r['recall']['lexical']['R@5']:.3f}  "
                  f"sem R@5={r['recall']['semantic']['R@5']:.3f}  "
                  f"hyb R@5={r['recall']['hybrid']['R@5']:.3f}",
                  file=sys.stderr, flush=True)

    # Macro-average across conversations (each conv contributes equal weight)
    def _avg(mode: str, k: int) -> float:
        vals = [c["recall"][mode][f"R@{k}"] for c in per_conv]
        return sum(vals) / len(vals) if vals else 0.0

    summary = {
        "macro": {
            mode: {f"R@{k}": round(_avg(mode, k), 4) for k in (1, 5, 10)}
            for mode in ("lexical", "semantic", "hybrid")
        },
        "n_conversations": len(per_conv),
        "total_qas": sum(c["total_qas"] for c in per_conv),
        "elapsed_s": round(time.monotonic() - started, 1),
    }
    return {"ok": True, "summary": summary, "per_conv": per_conv}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-limit", type=int, default=None,
                    help="Max QA per conversation (default: all)")
    ap.add_argument("--conv-limit", type=int, default=None,
                    help="Max conversations to evaluate (default: all 10)")
    args = ap.parse_args()

    result = run(qa_limit_per_conv=args.qa_limit, conv_limit=args.conv_limit)
    print()
    print(json.dumps(result.get("summary", {}), indent=2, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
