"""backfill_embeddings — populate messages_vec for already-indexed messages.

Idempotent and resumable: each iteration selects unembedded rows
(LEFT JOIN messages_vec WHERE vec.rowid IS NULL), batches them through
the embedder, writes vectors back. Crash anywhere = pick up next run.

Usage:
    python -m magpie_search.backfill [--batch 64] [--limit N]

Estimate: ~5-15 min for 70k messages on CPU (depends on text length).
"""
from __future__ import annotations

import argparse
import sys
import time

from . import embeddings, indexer


def backfill(batch_size: int = 64, limit: int | None = None) -> dict:
    if not embeddings.available():
        return {"ok": False, "reason": f"embeddings unavailable: {embeddings.load_error()}"}

    with indexer.advisory_lock():
        conn = indexer.connect()
        try:
            if not indexer.vec_available(conn):
                return {"ok": False, "reason": "sqlite-vec extension did not load"}
            indexer.init_db(conn)

            total_done = 0
            batches_done = 0
            started = time.monotonic()
            last_log = started

            while True:
                rows = conn.execute(
                    "SELECT m.rowid AS rid, m.text AS txt "
                    "FROM messages m "
                    "LEFT JOIN messages_vec v ON v.rowid = m.rowid "
                    "WHERE v.rowid IS NULL "
                    "LIMIT ?",
                    (batch_size,),
                ).fetchall()
                if not rows:
                    break

                texts = [r["txt"] or "" for r in rows]
                rowids = [r["rid"] for r in rows]
                vecs = embeddings.embed_batch(texts)
                if not vecs:
                    return {"ok": False, "reason": "embed_batch returned None mid-run"}

                conn.executemany(
                    "INSERT INTO messages_vec(rowid, embedding) VALUES (?, ?)",
                    list(zip(rowids, vecs)),
                )
                conn.commit()
                total_done += len(rows)
                batches_done += 1

                now = time.monotonic()
                if now - last_log >= 5.0:
                    rate = total_done / (now - started)
                    print(f"  ... {total_done:,} embedded "
                          f"({rate:.0f}/s, batch {batches_done})", flush=True)
                    last_log = now

                if limit and total_done >= limit:
                    break

            elapsed = time.monotonic() - started
            total_msgs = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            total_vec = conn.execute("SELECT COUNT(*) FROM messages_vec").fetchone()[0]
            return {
                "ok": True,
                "newly_embedded": total_done,
                "batches": batches_done,
                "elapsed_s": round(elapsed, 1),
                "rate_per_s": round(total_done / max(elapsed, 0.001), 1),
                "total_messages": total_msgs,
                "total_embedded": total_vec,
                "coverage": round(total_vec / total_msgs, 4) if total_msgs else 0.0,
            }
        finally:
            conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64,
                    help="Embed batch size (default: 64)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after N messages (default: all)")
    args = ap.parse_args()
    result = backfill(batch_size=args.batch, limit=args.limit)
    import json
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
