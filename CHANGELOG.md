# Changelog

All notable changes to magpi are recorded here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
[SemVer](https://semver.org/) (pre-1.0 — minor bumps may include breaking
changes when called out).

> **Note on the name.** Pre-public, the package was named `magpie-search`. It
> was renamed to **`magpi`** before any public distribution. All version
> entries below describe the package under its current name; historical
> commits/scripts referencing `magpie-search` are equivalent. Env vars renamed
> from `MAGPIE_SEARCH_*` to `MAGPI_*`; data dir from `~/.magpie-search/` to `~/.magpi/`;
> CLI command from `magpie-search` to `magpi`.

## [1.0.0] — 2026-05-29

First stable release. **Renamed `magpi` → `magpie-search`** and hardened to
v1 via a dual-track security bug-bounty (an in-process multi-agent pass plus
an independent audit run on a separate Linux machine). 70 tests, green on
Windows and Linux (Python 3.10–3.12).

### Renamed
- Distribution/import: `magpi` → `magpie-search` / `magpie-search`.
- CLI: `magpie-search` (+ `magpie-search-mcp`). The old `magpi` / `magpi-mcp`
  commands are kept as deprecated aliases.
- Env prefix `MAGPI_*` → `MAGPIE_SEARCH_*`; data dir `~/.magpi` → `~/.magpie-search`.
- **Back-compat:** legacy `MAGPI_*` env vars are auto-promoted at import, and an
  existing `~/.magpi` install is reused in place (model cache, backup config,
  index) — no data migration needed.

### Security (redactor — the product's secret boundary)
- **Fixed key-material leak:** index blocks were truncated *before* redaction,
  splitting multiline secrets (PEM keys) so they bypassed the pattern. Redaction
  now runs before truncation.
- **New secret classes covered:** database/service connection-string credentials
  (`scheme://user:pass@host`), Stripe, GitHub fine-grained PATs, SendGrid,
  Twilio, Google API keys + OAuth (`ya29.`), OpenAI scoped keys
  (`sk-proj-`/`sk-svcacct-`/`sk-admin-`).
- **Closed partial-leak bypasses:** Slack token underscores, base64 bearer
  tokens (`/ + =`), short/special-char passwords, short env-var names.
- **Audit log:** prompt/response are now redacted before being written to
  `llm-audit.jsonl`.
- Added `test_redactor.py` (was previously untested).

### Security (backup / subprocess)
- `ssh`/`rsync`/`scp` host, dest, and VM args are validated fail-closed at
  config load (reject leading `-`, whitespace, shell/option metacharacters) —
  closes argument-injection + silent-exfiltration via a poisoned `backup.env`.
  `--` option-terminators added to transfer commands. Remote-shell env
  overrides (`RSYNC_RSH`, `GIT_SSH*`) are stripped from the child env.

### Robustness
- Search: FTS5 power-user queries that produce invalid syntax (`*`, `-foo`,
  unbalanced quote, trailing `OR`) now degrade to a literal search instead of
  erroring; negative/zero `k`/`n`/`limit` are clamped (were unbounded full-table
  dumps).
- Indexer: bounded per-pass file read (no multi-GB RAM spike on huge sessions);
  `IN (...)` dedup lookups are batched under SQLite's variable limit.
- LLM trust monitor fails closed on missing/forged `trust` fields; summarizer
  self-verify and proper-noun/identifier guards are token/word-boundary exact.
- MCP server clamps result-size params, validates numeric params, and no longer
  leaks internal exception text to clients.

## [0.4.1] — 2026-05-18

Security + ship-hygiene release (customer-edition readiness).

### Security
- Scrubbed operator-personal identifiers (VM host/IP, operator/agent
  handles) that had leaked into source docstrings/comments in 0.3.0–0.4.0.
  Comment/string-only — behavior-neutral. Added a build-time regression
  gate (`tests/test_no_personal_leak.py`) that fails the build if any
  personal identifier returns; product/company names remain allowed.

### Packaging
- Internal dev/benchmark tools (`eval_locomo`, `llm/bakeoff`) moved out of
  the shippable package — no longer included in the customer wheel.

### Tests
- `the companion plugin` companion plugin: added a real test suite (previously
  untested) — role-registration wiring, env-path `setdefault` semantics,
  LLM gating, and the personal-config allowlist injection guard.

## [0.4.0] — 2026-05-16

Hardening release. Driven by a real 2-night silent backup outage and a
"close every gap, measure don't assume" review. Every change below was
verified end-to-end and independently audited (Queen → c1 → c2 hive).

### Fixed

- **Backup outage root cause.** `_wait_for_ssh` did a raw socket probe
  on the configured host, which fails forever when it's an ssh-config
  alias (DNS can't resolve an ssh-only alias). Now resolves via
  `ssh -G` first (`_resolve_ssh_target`). Proven: real backup to an
  aliased host now connects in ~1s.
- **Search crashed on hyphenated queries** (FTS5 treated `-` as an
  operator). Tokens are now FTS5-quoted.
- **The LOCOMO benchmark had never run once** — an import shadowing bug
  (`from . import search` resolved to the re-exported function) made
  the harness `AttributeError` on the first query. Any prior quoted
  recall number was therefore fabricated. Fixed; full benchmark now
  runs (10 conv / 1982 QA).
- **Hybrid search was worse than plain keyword search.** The
  `lex_weight`/`sem_weight` parameters were declared and documented but
  never applied in the RRF fusion — it was silently equal-weight, so a
  near-random semantic signal outvoted strong lexical hits ~44% of the
  time. Weights now applied + candidate pool widened 2×→6×. Measured:
  hybrid macro R@5 0.388→0.460, R@10 0.503→0.590 (now beats lexical).
  Residual R@1/R@5 gap is the embedding model — a separate, documented
  ceiling, NOT claimed fixed.
- **`--dry-run` wrote a lying "success" backup heartbeat** (would make
  health report OK with zero bytes transferred). Dry runs now never
  advance the last-success marker.

### Added

- **Dedup is self-healing.** The backlog-dedup metadata used to decay
  (only live-indexed rows got it); `index_all` now runs an idempotent
  catch-up under the advisory lock every pass, so coverage is
  complete-by-construction.
- **Deterministic index location.** When no env var is set and an
  an operator index exists, it is authoritative — a bare
  `python -m magpi` and the swarm can no longer silently diverge onto
  two ~85k DBs. True standalone installs still use `~/.magpi`.
- **Backup self-monitoring (GAP-4).** Every run writes a durable
  heartbeat (failure-safe: a failed/dry run cannot clobber the
  last-good marker). `health_check()` + a `backup.health` swarm role
  detect staleness and self-alert into the operator alert channel,
  throttled. The system now flags a silent outage itself instead of a
  human noticing days later.
- **`MAGPI_BACKUP_RSYNC` override** + cygpath-probed Windows→POSIX path
  translation, so verified rsync can be enabled without polluting the
  system PATH. rsync failure now falls back to the proven scp path
  instead of hard-failing the backup (a failing rsync used to be
  strictly worse than no rsync). Note: an MSYS2 rsync on Windows hits a
  runtime `dup()` bug; scp fallback covers it. Linux operators get
  verified rsync.

## [0.2.0] — 2026-05-14

### Added

- **`magpi.backup` — configurable transcript backup.** New module replacing
  the operator-specific `nightly_sync`. Three modes, picked by what's
  configured:
  - **Local folder** (default): copies `~/.claude/projects/` to
    `~/.magpi/backups/`. Zero setup.
  - **Remote SSH** (rsync-over-ssh): set `MAGPI_BACKUP_SSH_HOST` and
    `MAGPI_BACKUP_SSH_DEST` to push to any reachable SSH target — NAS,
    home server, VM, anything `ssh user@host` can reach.
  - **Remote SSH with VM boot/suspend**: above plus `MAGPI_BACKUP_VM_PROVIDER`
    (`vmware` or `virtualbox`), VMX path or VM name, and the script handles
    the boot-before / suspend-after dance for you.
  - Config sources, in priority order: env vars, then `~/.magpi/backup.env`
    (simple `KEY=VALUE` format).
  - CLI: `python -m magpi.backup [--dry-run] [--no-suspend] [--show-config]`.
- **`MAGPI_DEBUG=1`** — surfaces unexpected audit-log write failures to
  stderr so dev runs see broken patches that would otherwise be swallowed
  by the "never poison the caller" contract.
- **Threading locks on lazy-init globals** (`_EMBEDDER`, `_RERANKER`) —
  prevents double-load of the 80MB embedding model under concurrent first
  calls. Double-checked locking, cheap fast path.
- **Presidio engine caching** — `_presidio_pass` caches `AnalyzerEngine` /
  `AnonymizerEngine` at module level. Heavy spaCy-model construction now
  happens once instead of per-call.

### Changed

- **`magpi.nightly_sync` is now a 3-line compat shim** that delegates to
  `magpi.backup`. Existing Windows Task Scheduler / cron entries calling
  `python -m magpi.nightly_sync` keep working — the operator-specific
  config has moved into `~/.magpi/backup.env`.
- **Cross-process file locking on `llm-audit.jsonl`** — concurrent writers
  (cuckoo daemon, interactive sessions, magpi CLI, companion plugin)
  now serialize through a `file_lock`. Closes a window where torn JSONL
  lines could land and be silently dropped by `trust.tail()`. Requires
  the companion package to be importable; falls back to in-process lock only if
  not available.
- **Redactor preserves contextual prefix** on four pattern kinds:
  `Bearer xxx` → `Bearer [REDACTED:bearer_token]`,
  `API_KEY=xxx` → `API_KEY=[REDACTED:dotenv_secret]`,
  `password=xxx` → `password=[REDACTED:api_key_assignment]`,
  `OTP: xxx` → `OTP: [REDACTED:otp_with_context]`.
  Previously the whole match including the prefix was replaced, losing
  the auditable label.
- **Redactor pattern order** — `dotenv_secret` now runs before
  `api_key_assignment` so line-anchored `API_KEY=...` gets the more
  specific label.
- **Redactor `openai_key` length floor** raised from `{20,}` to `{32,}` —
  fewer false-positives on dev-mock tokens like `sk-mock-...`.
- **`summarizer._self_verify` rejects hedged YES** — `YES, but...`,
  `YES however...`, `YES although...`, `YES except...` no longer pass.
  Closes the soft-hallucination escape where the verifier acknowledged
  a problem but the summarizer accepted the answer anyway.
- **`summarizer_proper_noun_safety`** — sentence-initial exemption now
  only applies when the lowercase form is a stopword/article (`The`,
  `And`). Fabricated proper nouns at the start of a sentence (`Redis was
  used`) are no longer silently exempted.
- **`summarizer_noun_overlap` docstring** corrected — default threshold is
  `0.55` (was previously documented as `0.6` while the code used `0.55`).
- **`trust.py` per-probe threshold overrides clamp** — values `< 1` are
  promoted to `1`. Prevents an operator typo (`self_verify:0`) from
  triggering alarms on every tick.
- **`trust.py` and `summarizer.py` env-var reads** are now wrapped in
  `try/except` — a typo like `MAGPI_TRUST_BYPASS_1H=0,5` (comma instead
  of dot) no longer crashes module import.
- **Empty-string env vars** (`MAGPI_HOME=""`, `MAGPI_AUDIT_LOG=""`,
  `MAGPI_MODELS_DIR=""`) no longer resolve to CWD. Treated as unset,
  falling through to the documented defaults.
- **`audit.py` exception handling** narrowed from blanket `except
  Exception` to specific expected types (`OSError`, `TimeoutError`,
  `ValueError`, `json.JSONDecodeError`). Unexpected exceptions still
  don't propagate (never-poison-caller contract preserved) but surface
  to stderr under `MAGPI_DEBUG=1` so developer-introduced bugs are
  visible.
- **`client.py` retry loop** now sleeps `min(1.0, 0.2 * (attempt+1))`
  seconds between attempts. Previously retried back-to-back in ~0ms
  against a host that was still coming up.
- **`indexer.py` partial-line byte cursor** now uses raw `buf.rfind(b"\n")`
  instead of re-encoding decoded text. Closes a drift window where
  `errors="replace"` would substitute `U+FFFD` (3 bytes encoded) for
  single bad bytes and the cursor would advance past unread content
  on the next pass.

### Fixed

- Operator-identifying strings removed from shipping docstrings (no
  hard-coded usernames or names in the package source).

### Security

- The audit-log multi-writer race fix (file locking, above) closes a
  small window where torn lines could mask `untrusted` events from the
  trust monitor — telemetry now reliably reflects the population.

### Configuration changes

Existing operators with custom env-var overrides will see no change in
behavior. New env vars (all optional):

| Var | Default | Notes |
|---|---|---|
| `MAGPI_BACKUP_SRC_DIR` | `~/.claude/projects` | What to back up. |
| `MAGPI_BACKUP_DEST_DIR` | `~/.magpi/backups` | Local destination (used when no SSH target is set). |
| `MAGPI_BACKUP_SSH_HOST` | unset | `user@host` or an SSH-config alias. Set this to enable remote mode. |
| `MAGPI_BACKUP_SSH_DEST` | unset | Remote path (e.g. `~/transcripts/`). |
| `MAGPI_BACKUP_VM_PROVIDER` | unset | `vmware` or `virtualbox`. Enables VM boot/suspend hooks. |
| `MAGPI_BACKUP_VM_VMX` | unset | (vmware) path to `.vmx` file. |
| `MAGPI_BACKUP_VM_NAME` | unset | (virtualbox) registered VM name. |
| `MAGPI_BACKUP_VM_BOOT_BEFORE` | `1` | Boot before sync. |
| `MAGPI_BACKUP_VM_SUSPEND_AFTER` | `1` | Suspend after sync (only if we booted it). |
| `MAGPI_BACKUP_SSH_READY_TIMEOUT_S` | `90` | How long to wait for SSH after boot. |
| `MAGPI_BACKUP_LOG` | `~/.magpi/backup.log` | Plain-text log. |
| `MAGPI_BACKUP_CONFIG` | `~/.magpi/backup.env` | Where the env-file lives. |
| `MAGPI_DEBUG` | unset | Surfaces unexpected audit-log errors to stderr. |

## [0.1.0] — 2026-05-13

### Added

- Initial extraction from `an internal transcript_indexer` into a
  standalone Python package.
- `magpi.indexer` — JSONL → SQLite (FTS5 + sqlite-vec) ingest with
  byte-cursor incremental indexing and per-file advisory locking.
- `magpi.search` — hybrid lexical + semantic search with reciprocal
  rank fusion.
- `magpi.llm` — phi3.5 summarizer with 7-probe guardrail stack
  (length, proper_noun_safety, identifier_safety, refusal_drift,
  noun_overlap, semantic_grounding, self_verify), reranker via
  `jinaai/jina-reranker-v1-turbo-en`, trust monitor.
- `magpi.redactor` — regex pass for keys / tokens / addresses /
  passwords / OTPs, optional Presidio NER pass.
- `magpi.nightly_sync` — operator-specific VM-rsync workflow
  (replaced in 0.2.0 by the generic `magpi.backup` module).
- `the companion plugin` plugin — exposes magpi roles into the operator
  swarm (transcript.*, archivist.summarize, llm.trust_check).
