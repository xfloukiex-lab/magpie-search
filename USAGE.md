# Magpie Search — Features & Usage (v1.0.0)

Local-first search over your Claude Code session transcripts. Everything runs
on your machine; nothing is sent to any external service.

## What it does (features)

**Search**
- **Full-text (FTS5/BM25)** keyword search over every message in your
  transcripts — the default, fast and offline.
- **Semantic + hybrid search** (Python/MCP API): vector nearest-neighbour via
  `sqlite-vec` + `fastembed` embeddings, and a hybrid mode that fuses keyword +
  semantic results (RRF). Finds things by meaning, not just exact words.
- **Browse**: recent messages, a paginated single-session view, and a list of
  recent sessions.

**Indexing**
- **Incremental**: tracks a byte cursor per file, so live sessions get picked up
  on the next pass without re-reading. Bounded memory even on huge sessions.
- **Deterministic location**: shares the Aviary operator index if present, else
  uses `~/.magpie-search`.
- Optional **dedup** + **noise filtering** to cut repeated/low-value content.

**Security (the part that matters for transcripts)**
- **Secret redaction at ingest**: API keys, tokens, private keys, JWTs, AWS/GCP
  creds, GitHub/Slack/Stripe/Twilio/SendGrid keys, OpenAI keys (incl. `sk-proj-`),
  database/connection-string passwords, bearer tokens, OTPs, crypto addresses —
  scrubbed to `[REDACTED:<kind>]` **before** anything is written to the index.
- Redaction also runs before the LLM audit log is written.
- Optional Presidio NER pass for names/PII (`pip install` extra).

**LLM augmentation (optional, needs Ollama for summaries)**
- Cross-encoder **reranker** for sharper top results.
- **Summarizer** (phi3.5) with a 7-probe anti-hallucination guardrail stack.
- **Trust monitor** over the augmentation audit log (fail-closed).

**Backup**
- Back up your transcripts to: a local folder (default), a remote box over
  SSH (rsync, scp fallback), or a VM it boots/suspends around the sync.
- Hardened: remote host/dest/VM args are validated fail-closed (no command
  injection / exfiltration via a poisoned config).

**Integrations**
- **MCP server** (`magpie-search-mcp`): exposes search/recent/session/list/
  stats/reindex as tools over JSON-RPC stdio — wire it into any MCP client.
- **Aviary plugin** (`aviary-magpi`): registers `transcript.*` roles into the
  Aviary swarm.

## How to use it

**Install** (from the wheel):
```
pip install magpie-search-1.0.0-py3-none-any.whl
```

**Index your transcripts** (run once, then re-run anytime to catch up):
```
magpie-search index
```

**Search**:
```
magpie-search search "rate limiting bug"          # keyword search
magpie-search search "auth flow" --k 20           # more results
magpie-search search "deploy" --project myrepo    # filter by project
magpie-search search "error" --role assistant     # filter by role
magpie-search search "topic" --json               # machine-readable
```

**Browse**:
```
magpie-search recent                 # tail of the latest session
magpie-search session <session-id>   # full paginated session
magpie-search list                   # recent sessions
magpie-search stats                  # index size + counts
```

**Back up transcripts**:
```
magpie-search backup --dry-run       # show what it would do
magpie-search backup                 # run it
magpie-search backup --show-config   # print resolved config
```
Configure backup via env vars or `~/.magpie-search/backup.env`, e.g.
`MAGPIE_SEARCH_BACKUP_SSH_HOST=user@host`, `MAGPIE_SEARCH_BACKUP_SSH_DEST=~/transcripts/`.

**As an MCP server** (for an agent/IDE):
```
magpie-search-mcp        # speaks JSON-RPC 2.0 over stdio
```
Tools: `search`, `recent`, `session`, `list_sessions`, `stats`, `reindex`.

**Semantic / hybrid search** (Python API):
```python
import magpie_search
magpie_search.search("how did we fix the cursor drift", k=10, mode="hybrid")
magpie_search.search("vector stuff", mode="semantic")
```

## Configuration (env)
All optional; sensible defaults.
- `MAGPIE_SEARCH_HOME` — data dir (default `~/.magpie-search`; an existing
  `~/.magpi` is reused automatically).
- `MAGPIE_SEARCH_MODELS_DIR` — embedding-model cache.
- `MAGPIE_SEARCH_DEDUP=0` / `MAGPIE_SEARCH_NOISE_FILTER=0` — turn those off.
- `MAGPIE_SEARCH_DEBUG=1` — surface internal errors to stderr.
- Legacy `MAGPI_*` names are still honored (deprecated).

## Compatibility
- Python 3.10–3.12; Windows, Linux, macOS.
- The old `magpi` / `magpi-mcp` commands still work as deprecated aliases.
