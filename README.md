<p align="center">
  <img src="https://raw.githubusercontent.com/xfloukiex-lab/magpie-search/main/assets/magpie-logo.png" alt="Magpie Search" width="180">
</p>

<h1 align="center">Magpie Search</h1>

<p align="center">
  <a href="https://pypi.org/project/magpie-search/"><img alt="PyPI" src="https://img.shields.io/pypi/v/magpie-search?color=7c5cff&cacheSeconds=300"></a>
  <a href="https://pypi.org/project/magpie-search/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/magpie-search?color=22d3ee&cacheSeconds=300"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue?cacheSeconds=300"></a>
</p>

<p align="center"><b>A federated search engine — a search engine an AI reaches for when it needs to find something true to reason over.</b></p>

---

Ever had your computer reboot on you, or a power outage hit mid-session? Every
thread your agent was holding — gone. Now you have the tool to get it back.
**Never forget what your agent lost again.** Magpie indexes everything your AI
has ever worked through, locally, so a crash is a hiccup instead of amnesia.

## What Magpie is

A normal search engine looks in one place. Magpie takes one question and fans it
across everything that matters at once — the AI's entire conversation history,
the files on the machine, a structured knowledge graph, a vector store, the live
web, even YouTube — and pulls the answer back from wherever it actually lives.
**Six sources, one call.**

And it searches each one the right way. It can grep for an exact string or regex
when you know the precise token — a file path, an error, a line of code. It can
search by keyword. It can search by meaning, so it finds the thing even when the
words don't match. It can do all of that at once.

Then it does the part that makes it trustworthy: it fuses everything into a
single ranked answer, and every result carries a **trust tier** — `fact >
reference > lead > stale`. The solid sources rise, the loose ones are marked as
leads to verify, duplicates collapse, and it's all trimmed to fit so it never
floods the AI's context. Ask it to go deep and it expands one question into many,
reads the pages, and tells you how many independent sources agree — a full
research sweep without an army of agents.

It runs **entirely on the machine**. No server, no account, and no telemetry
unless you turn it on. The AI's transcripts and files never leave. It plugs into
whatever AI is running over **MCP**, so the agent can reach all six sources the
instant it needs them.

It is a tool for an AI.

## What's inside

At its core is a local index of the AI's transcripts: a SQLite database with two
structures built side by side —

- an **FTS5** full-text index (BM25 keyword ranking), and
- a **vector index** (`sqlite-vec`) of 384-dim embeddings produced locally by a
  small `all-MiniLM-L6-v2` model.

Everything is **redacted at ingest** — a scrubber strips ~30 classes of secrets
(keys, tokens, private keys, connection strings) before a single byte hits the
index.

On top of that index sit the **five search modes**:

| Mode | What it does |
|---|---|
| `grep` | literal / regex match (exact tokens: paths, errors, code) |
| `lexical` | FTS5 / BM25 keyword |
| `semantic` | embedding K-NN, cosine distance in the vector index |
| `hybrid` | lexical + semantic fused by RRF |
| `rerank` | hybrid, then a cross-encoder (jina-reranker) re-scores each candidate |

Around that sits the **federation layer** — the part that makes it federated:

- A **provider plugin system.** Six backends (transcripts, files, knowledge
  graph, vector, web, YouTube), each returns `Hit` objects tagged with a trust
  tier.
- A **fan-out**: one query goes to all providers concurrently (≤8 workers), each
  with a 5-second timeout that **fails open** — a slow source contributes nothing
  rather than blocking the call.
- **Trust-weighted RRF fusion** — Reciprocal Rank Fusion where each source's rank
  is multiplied by its trust weight (`fact ×3, reference ×2, lead ×1, stale
  ×0.3`), damping constant 60. This is the math that merges six heterogeneous
  sources into one honest ranking.
- **Cross-source dedup** by content hash — the same fact found in three places
  collapses to one hit, tagged with where else it appeared (corroboration).
- A **token-budget trim**, so the merged set never overflows the calling AI's
  context.

And it exposes all of this to an AI over an **MCP server** — the tools it hands
an agent are exactly: `search`, `recent`, `session`, `list_sessions`, `stats`,
`reindex`. Note what's *not* in that list: nothing that writes an answer.

## Why that is not RAG

RAG = Retrieval-Augmented **Generation**. It's a two-stage pipeline, and the
defining stage is the second one: a retriever finds chunks → they're stuffed into
a prompt → a language model **generates** the prose answer. The "G" is the whole
point of the name; without a generator writing the answer, it isn't RAG.

Magpie has no G:

1. **There is no generator anywhere in the search path.** Nothing in Magpie
   composes a natural-language answer. The closest thing to a model — the
   cross-encoder reranker — outputs a relevance number per result and reorders
   the list. It scores; it never writes a sentence.
2. **It stops at "here are the ranked hits."** A RAG owns the prompt assembly and
   the model call. Magpie returns the fused, trust-ranked results and hands them
   back through MCP. What the AI does next — whether it even generates anything —
   is the AI's job, outside Magpie.
3. **Its retriever is more than a RAG's retriever, not less.** A textbook RAG
   retriever is one vector store: embed the query, top-k by cosine, done.
   Magpie's retrieval is six sources, five modes, trust-weighted fusion,
   cross-source dedup. It's a far more capable "R" — but it's still only the R.

Plug Magpie into an AI and the pair can form a RAG — Magpie is the R, the AI you
bring is the G. But Magpie by itself ships only the R, and a stronger R than
usual. It finds and ranks the truth; it never generates the answer.

## Deep web search — research breadth without the token bill

The expensive part of "deep research" is **reasoning**, and the multi-agent
approach pays for it N times over — one full LLM context per agent, often
*millions* of tokens for a single question. But reasoning doesn't need to fan
out; one capable model already in context can synthesize. Only the **searching**
needs breadth — and searching the web is pure retrieval, **zero LLM tokens**.

`magpie-search deepweb` is built on that asymmetry. It fires several sub-queries
at the web in parallel, fuses them by trust-weighted RRF + dedup-by-URL into one
compact, token-budget-trimmed source set, optionally reads the top pages' text
(still token-free), and reports how many independent domains corroborate the
result — an agent-free version of the verification a research swarm pays agents
to do.

So you get the breadth, page-reading, and corroboration of a multi-agent deep
search, but your model only pays for a **single synthesis pass** over a trimmed
result set.

**Token cost, measured — one deep question:**

| Approach | Tokens the model pays |
|---|---|
| Multi-agent deep-research swarm (N agents each read pages into their own context) | **~2,000,000** |
| `magpie-search deepweb --thorough` (6 angles → 12 sources, 12 full pages read) | **~1,050** |

That's **~2,000× fewer tokens** — about 1/2000th the cost — because the searching
and page-reading are pure retrieval (**zero model tokens**); your model only does
the final synthesis pass over the trimmed, corroborated set.

```bash
# one question, several angles, read the top pages — all token-free retrieval
magpie-search deepweb "the question" --q "another angle" --q "a third angle" --thorough
```

The model in your loop then does one synthesis pass over the merged, corroborated
set. That's the whole saving: the breadth is free, you pay only for the answer.

---

## Install

```bash
pip install magpie-search
```

Or install the latest straight from source (pulls all dependencies):

```bash
pip install "git+https://github.com/xfloukiex-lab/magpie-search.git"
```

Optional — add the local-LLM features (the cross-encoder reranker runs on the
base install; the session summarizer needs Ollama):

```bash
# 1. Install Ollama (free, runs entirely locally) — https://ollama.com/download
# 2. Pull the model magpie-search uses
ollama pull phi3.5
```

Python 3.10+ on Windows, macOS, and Linux.

## Quickstart

```bash
magpie-search index                               # build the index (incremental)
magpie-search search "that retry backoff thing"   # keyword search
magpie-search search --mode hybrid "..."          # keyword + semantic, fused
magpie-search search --mode rerank "..."          # + cross-encoder rerank
magpie-search stats                               # sanity-check the index
```

## Connect it to your AI (MCP)

Magpie speaks the Model Context Protocol, so any MCP-capable agent can call it.
Point your client at the bundled server:

```jsonc
// e.g. an MCP client config
{
  "mcpServers": {
    "magpie": { "command": "magpie-search-mcp" }
  }
}
```

The agent then has `search`, `recent`, `session`, `list_sessions`, `stats`, and
`reindex` available — federated, trust-ranked, context-budgeted.

## CLI reference

| Command | What |
|---|---|
| `magpie-search index` | Incremental indexing pass over `~/.claude/projects/` |
| `magpie-search search "q"` | Search — `--mode grep\|lexical\|semantic\|hybrid\|rerank` |
| `magpie-search recent --n 30` | Latest 30 messages of the newest session |
| `magpie-search session SESSION-ID` | Full transcript of one session |
| `magpie-search list` | Recent sessions |
| `magpie-search stats` | Index size, last-indexed time, row counts |
| `magpie-search backup` | Back up `~/.claude/projects/` to a configurable destination |

Add `--help` to any command for full options.

## Python API

```python
import magpie_search

results = magpie_search.search("retry backoff", mode="hybrid", k=5)
for h in results["hits"]:
    print(h["trust"], h["source"], h["snippet"])

# LLM features (needs Ollama + phi3.5)
import magpie_search.llm
ranked  = magpie_search.llm.search_rerank(query="retry backoff", k=3, pool=10)
summary = magpie_search.llm.summarize(session_id="abc-123", n_messages=80)
```

## Backup

`magpie-search backup` copies your transcript tree to a destination of your
choice — a local folder (default, zero config), a remote SSH target (NAS / home
server), or a remote SSH target with VM boot/suspend. Configure it in
`~/.magpie-search/backup.env`:

```env
MAGPIE_SEARCH_BACKUP_SSH_HOST=user@nas.local
MAGPIE_SEARCH_BACKUP_SSH_DEST=~/claude-transcripts/
```

Useful flags: `--dry-run`, `--no-suspend`, `--show-config`. Backup copies; it
never deletes originals.

## Configuration

Everything is environment-variable driven with sensible defaults.

| Var | Default | What |
|---|---|---|
| `MAGPIE_SEARCH_HOME` | `~/.magpie-search` | Data directory (DB, models, logs) |
| `MAGPIE_SEARCH_MODELS_DIR` | `$MAGPIE_SEARCH_HOME/models` | fastembed model cache |
| `MAGPIE_SEARCH_OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `MAGPIE_SEARCH_TOKENIZER` | heuristic | Set to `tiktoken` for precise budget counting |
| `MAGPIE_SEARCH_AUDIT_LOG` | `$MAGPIE_SEARCH_HOME/llm-audit.jsonl` | Per-call audit log |

The summarizer passes through a 6-probe guardrail stack (length,
proper-noun-safety, identifier-safety, refusal-drift, semantic-grounding,
self-verify); all six must pass for `trust: clean`. Any failure suppresses the
summary and returns `trust: degraded` — quiet over wrong. Raw messages stay
accessible via `magpie-search session SESSION-ID`.

## Privacy

Magpie Search is a local tool. No server, no account, no auto-update, no crash
reporter, and **no telemetry unless you explicitly opt in** (see below). Your
transcripts, the index, the audit log, the model cache, and the backups all live
on your machine.

**Opt-in telemetry.** Telemetry is **off by default** — magpie sends nothing
until you run `magpie-search telemetry enable` (or set
`MAGPIE_SEARCH_TELEMETRY=1`). When on, it sends only **anonymous usage**: which
command ran, search mode, result/hit counts, latency, error class, and your
magpie/python/OS versions, tagged with a random install id. It **never** sends
your queries, file paths, results, transcript content, username, or IP — a
hard content firewall in `telemetry.py` drops anything that isn't a number or a
short enum token. Disable anytime with `magpie-search telemetry disable`; check
state with `magpie-search telemetry status`. The only
network calls it ever makes are: your local Ollama server (LLM features), your
own backup target (only when you run `backup`), and a one-time model download
from Hugging Face on first run. Verify it yourself with `tcpdump`, Wireshark, or
a network-blocked sandbox.

## Scheduling

Run `magpie-search index` (and optionally `backup`) on a schedule. Ready-made
units live in [`installers/`](installers/) for systemd (Linux), launchd (macOS),
and Task Scheduler (Windows).

## Troubleshooting

- **"rsync not on PATH"** — falls back to `scp -r`. On Windows, install
  [Git for Windows](https://git-scm.com/download/win), which ships rsync.
- **Search returns nothing** — run `magpie-search stats`; if `last_indexed_at` is
  null, run `magpie-search index`.
- **Summarizer always `degraded`** — that's the false-positive guard working as
  designed. Raw transcripts remain available via `session SESSION-ID`.

---

## About

**Magpie Search is built by [VektorGeist LLC](https://vektorgeist.com).**

We build local-first tools for people who run their own AI. Magpie is the search
core; our agent platform is at **[vektorgeist.com](https://vektorgeist.com)**.

- Website: **[vektorgeist.com](https://vektorgeist.com)**
- Contact: **floukie@vektorgeist.com**
- Issues & contributions: open an issue or PR on this repository.

## License

Licensed under the **Apache License 2.0** — see [LICENSE](LICENSE).
Copyright © 2026 VektorGeist LLC.

*"Magpie Search" and the magpie mark are trademarks of VektorGeist LLC. The code
is open under Apache-2.0; the brand and name are reserved.*
