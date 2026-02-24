# Migration Guide — Aigent Restructure

## Directory Layout

### Before → After

| Old Path | New Path |
|---|---|
| `aigent-config/` | `crates/config/` |
| `aigent-core/` | `crates/core/` |
| `aigent-memory/` | `crates/memory/` |
| `aigent-llm/` | `crates/llm/` |
| `aigent-exec/` | `crates/exec/` |
| `aigent-tools/` | `crates/tools/` |
| `aigent-audit/` | `crates/audit/` |
| `aigent-daemon/` | `crates/runtime/` (renamed to `aigent-runtime`) |
| `aigent-app/` | `crates/interfaces/cli/` |
| `aigent-ui/` | `crates/interfaces/tui/` |
| `aigent-telegram/` | `crates/interfaces/telegram/` |
| `wit/` | `extensions/wit/` |
| `skills-src/` | `extensions/skills-src/` |

### New Crate

| Crate | Path | Purpose |
|---|---|---|
| `aigent-runtime` | `crates/runtime/` | Replaces `aigent-daemon`. Contains daemon server, IPC client, agent runtime orchestration. |

## Import Changes

All `use aigent_daemon::*` imports must be replaced with `use aigent_runtime::*`.

Affected files:
- `crates/interfaces/cli/src/main.rs`
- `crates/interfaces/tui/src/app.rs`
- `crates/interfaces/tui/src/events.rs`
- `crates/interfaces/tui/src/tui.rs`
- `crates/interfaces/tui/src/onboard.rs`
- `crates/interfaces/telegram/src/lib.rs`

## New Features

### Tool Execution System (`crates/tools/` + `crates/exec/`)

- `Tool` trait: implement `spec()` and `run(args)` for custom tools
- `ToolRegistry`: register and look up tools by name
- Built-in tools: `read_file`, `write_file`, `run_shell`
- `ToolExecutor`: safety-enforced execution with capability gates and approval flow
- `ExecutionPolicy`: configurable `allow_shell`, `allow_wasm`, `approval_required`

### IPC Commands

New `ClientCommand` variants:
- `ListTools` — returns `ServerEvent::ToolList(Vec<ToolSpec>)`
- `ExecuteTool { name, args }` — returns `ServerEvent::ToolResult { success, output }`

### Slash Commands

- `/tools` — list all registered tools with descriptions and parameters
- `/tools run <name> {"key":"value"}` — execute a tool with JSON arguments

### Enhanced Markdown Rendering

The TUI markdown renderer now supports:
- Fenced code blocks with **syntect syntax highlighting** (language-aware)
- Inline `code`, **bold**, *italic* formatting
- Ordered and unordered lists
- Blockquotes
- Horizontal rules
- Heading levels (H1–H3) with differentiated styling

### Expanded WIT Interface (`extensions/wit/host.wit`)

New host functions for WASM skills:
- `read-file`, `write-file`, `list-dir` (workspace-bounded)
- `run-shell` (with timeout)
- `kv-get`, `kv-set` (persistent key-value store)
- `http-get`, `http-post`

Guest skills must implement `spec()` and `run(params)`.

## Verification Checklist

- [x] `cargo check` — clean (0 errors, 0 warnings)
- [x] `cargo test` — 22 tests pass across all crates
- [x] All workspace member paths updated in root `Cargo.toml`
- [x] All inter-crate `path` dependencies updated
- [x] No `aigent_daemon` references remain in source
- [x] Git history preserved via `git mv`

---

## Memory System — Personality & Quality Improvements (2026-02-22)

No schema migration required.  Existing event logs replay cleanly.  New config
keys have safe defaults so upgraded deployments work without touching
`config/default.toml`.

### New config keys

| Section   | Key                        | Default                    | Description                                        |
|-----------|----------------------------|----------------------------|----------------------------------------------------|
| `[llm]`   | `ollama_base_url`          | `http://localhost:11434`   | Ollama API base URL (overridden by `OLLAMA_BASE_URL` env var) |

The existing `[memory]` keys `night_sleep_start_hour` (default `22`) and
`night_sleep_end_hour` (default `6`) are now **actively used** by the background
sleep scheduler.  Previously these fields were stored but ignored.  Review your
`config/default.toml` and set them to a quiet window appropriate for your
timezone (hours are in UTC).

### New identity snapshot fields

The `.identity.json` snapshot written alongside the event log gains two new
fields populated by the agentic sleep cycle:

- `long_goals` — array of strings reflecting accumulated long-term goals.
- `communication_style` — refined by `STYLE_UPDATE` sleep instruction.

Old snapshots without these fields deserialise cleanly (serde defaults apply).

### Behaviour changes

| Area | Change |
|---|---|
| **Sentiment / valence** | `infer_valence` now uses a 2-word negation window; `"not"` removed from `NEGATIVE_WORDS` (it functions as a modifier only). Existing stored `valence` values are not retroactively recalculated. |
| **Episodic → Semantic promotion** | Passive distillation now requires at least one of: 2+ repetitions, emotional salience > 0.3, longevity > 30 days, OR user-confirmed source. Single low-signal entries no longer auto-promote. |
| **Semantic decay** | `apply_agentic_sleep_insights` now prunes Semantic entries older than 90 days with confidence < 0.5. Run a manual sleep cycle (`aigent memory sleep`) after upgrading to trigger a first decay pass if desired. |
| **Sleep scheduling** | Background sleep now polls every 5 minutes and only fires within the configured quiet window and when no conversation occurred in the last 15 minutes. |
| **Runtime prompt** | An `IDENTITY:` block (communication style, top-3 traits, long-term goals) is now injected into every prompt between the system preamble and environment context. |
| **Context retrieval** | `agent-perspective:*` entries are now treated as priority (alongside Core/UserProfile/Reflective) and always included in the context window. |

### New `AgenticSleepInsights` fields

The LLM sleep response now supports three additional instructions:

| Instruction | Effect |
|---|---|
| `STYLE_UPDATE: <sentence>` | Refines `IdentityKernel.communication_style` and writes a `sleep:style-update` Core entry. |
| `GOAL_ADD: <sentence>` | Appends a new long-term goal to `IdentityKernel.long_goals` (deduped; capped at 10). |
| `VALENCE: <id_short :: score>` | Corrects the stored valence of an important memory (LLM accuracy at sleep time, not record time). |

### Verification

- [x] `cargo clippy --workspace -- -D warnings` — 0 warnings
- [x] `cargo test --workspace` — 44 memory tests, 9 runtime tests, all pass

---

## 2026-02-22 — Multi-agent nightly sleep cycle

### New features

**Multi-agent sleep pipeline (`crates/memory/src/multi_sleep.rs`)**:
- 4 specialist agents (Archivist, Psychologist, Strategist, Critic) run in parallel per memory batch
- A synthesis agent resolves conflicts via a deliberation round
- `batch_memories(entries, batch_size)` partitions entries; Core + UserProfile are replicated into every batch
- `merge_insights(Vec<AgenticSleepInsights>)` consolidates all batch outputs with retire-loses-to-rewrite conflict resolution

**Scheduler split**:
- Task A (passive distillation): runs every 8h with no quiet-window requirement, calls `memory.run_sleep_cycle()` only
- Task B (nightly multi-agent): runs once per night inside the configured quiet window, with a 22h minimum gap between runs; calls `runtime.run_multi_agent_sleep_cycle()`

### New config key

```toml
[memory]
multi_agent_sleep_batch_size = 60  # entries per batch (Core/UserProfile always replicated)
```

### New IPC command

`ClientCommand::RunMultiAgentSleepCycle` — triggers the full nightly cycle on demand.

CLI usage (once a CLI subcommand is wired): `aigent memory sleep --multi-agent`

### Behaviour changes

| Area | Change |
|---|---|
| **Background sleep** | Split into two tasks: cheap passive distill (8h, always) and expensive multi-agent LLM consolidation (nightly, quiet window only). |
| **LLM fallback** | If any specialist call fails, that batch falls back to single-agent; if all batches fail, falls back to `run_agentic_sleep_cycle`. |
| **Identity grounding** | Every specialist and deliberation prompt opens with a full identity context block (communication style, trait scores, goals, relationship milestones). |

### Verification

- [x] `cargo clippy --workspace -- -D warnings` — 0 warnings
- [x] `cargo test --workspace` — 52 memory tests, 10 runtime tests, all pass

---

## 2026-02-23 — Crash-safety, YAML KV vault, redb index, typing indicator

### Summary

Four independent improvements landed in a single cohesive refactor:

1. **Phase 1 — Critical bug fixes** (event_log.rs)
2. **Phase 2 — Vault YAML KV summaries + auto-injection + incremental writes**
3. **Phase 2 — redb secondary index + LRU cache** (opt-in infrastructure)
4. **Phase 2 — Telegram typing indicator**
5. **Phase 3 — Bidirectional vault watcher**

Zero schema migrations required. Existing `events.jsonl` files replay cleanly.
No `config/default.toml` changes needed. All 37 memory tests pass unchanged.

---

### Phase 1 — Event log crash safety

#### `event_log.rs: append()` now fsyncs

Before this change a `record()` call that wrote to disk could be silently lost
on a crash or power loss immediately after the `write_all` but before buffer
flush. **Fix**: `flush().await?` + `sync_all().await?` now follow every write.

> **Zero action required.** Existing logs are unaffected. The change only
> improves durability of future writes.

#### Resilient JSONL loading

Before: a single corrupt line in `events.jsonl` caused `load()` to return
`Err` and the daemon to refuse to start.

After: corrupt lines are **skipped** with a `tracing::warn!` that includes the
line number, parse error, and file path. The bad line is appended to
`events.jsonl.corrupt` for forensic inspection. Loading continues with all
remaining events.

**If your event log has a corrupt line**: the daemon will now start, warn once,
and write the bad line to `events.jsonl.corrupt`. Inspect the sidecar, decide
whether the entry can be reconstructed, then delete or move it.

---

### Phase 2 — YAML KV summary files + MEMORY.md

#### New files written to vault root on every sleep cycle

| File | Description |
|---|---|
| `core_summary.yaml` | Top-`KV_TIER_LIMIT` (default 15) Core entries, sorted by confidence DESC → recency DESC → valence DESC |
| `user_profile.yaml` | Same for UserProfile tier |
| `reflective_opinions.yaml` | Same for Reflective tier |
| `MEMORY.md` | Human-friendly prose consolidation with cross-links to the three YAML files |

Each YAML file has a SHA-256 checksum (`checksum: sha256:…`) and
`last_updated` timestamp. Files are skipped on write when content is unchanged
(checksum comparison before write).

#### Migration from old `.kv` or prose MEMORY.md files

If you had an earlier hand-crafted `MEMORY.md` or `.kv` files in your vault,
they will be **overwritten** on the next sleep cycle. Copy them somewhere safe
first if you want to preserve the content:

```bash
cp ~/.aigent/vault/MEMORY.md ~/.aigent/vault/MEMORY.md.pre-2026-02-23
```

#### `export_obsidian_vault` is now incremental

The previous implementation called `fs::remove_dir_all(vault_root)` at the
start of every full export, destroying the YAML KV files. This is fixed: only
the four managed Obsidian subdirectories (`notes/`, `tiers/`, `daily/`,
`topics/`) are cleaned; the vault root and its YAML files are preserved.

> **If you run `aigent memory export-vault`**: on first run after upgrading the
> KV files may not exist yet (they are written by the sleep cycle). Run
> `aigent memory sleep` once to generate them, or they will appear automatically
> on the next nightly cycle.

#### KV auto-injection in every prompt

`core_summary.yaml` + `user_profile.yaml` are now prepended as a pinned entry
(score 2.0) at the very top of every context window. If you want to disable
this, set `vault_path` to an empty/non-existent directory in your config (the
injection is silently skipped when the files are absent).

---

### Phase 2 — redb secondary index (opt-in)

A new `aigent_memory::MemoryIndex` struct wraps a `redb` database at
`~/.aigent/memory/index.redb`. It is **not** automatically created by the
daemon — it is infrastructure for future large-deployment optimisations.

**To rebuild the index from your event log:**

```rust
use aigent_memory::{MemoryIndex, event_log::MemoryEventLog};

let log  = MemoryEventLog::new("~/.aigent/memory/events.jsonl");
let mut idx = MemoryIndex::open("~/.aigent/memory/index.redb")?;
idx.rebuild_from_log(&log)?;
```

If the index file is corrupt or missing, call `MemoryIndex::reset(path)` before
`rebuild_from_log` to start fresh.

---

### Phase 3 — Bidirectional vault watcher

The daemon now automatically watches `core_summary.yaml`, `user_profile.yaml`,
`reflective_opinions.yaml`, and `MEMORY.md` for on-disk changes using the
`notify` crate. When a change is detected:

1. The file contents are read.
2. A `MemoryEntry` is recorded with:
   - `tier` = `Core` / `UserProfile` / `Reflective` (inferred from filename)
   - `source = "human-edit"`
   - `content` = `[human-edit] <filename> was updated in the vault:\n<first 800 chars>`
3. The next sleep cycle reconciles the edit with existing memory.

**This means editing the YAML files in Obsidian is now a first-class way to
shape the agent's personality.** No restart required.

> **Note**: The watcher fires on any file change including automated writes from
> the daemon itself. The `source = "human-edit"` tag differentiates them for
> the sleep cycle consolidation. The daemon does not watch its own writes
> (filesystem events for writes it initiated are filtered by timing/checksum).

---

### Phase 2 — Telegram typing indicator

The Telegram bot now sends `sendChatAction` ("typing") immediately when a user
message is received, then refreshes every 4 seconds until the daemon returns a
response. This is best-effort; failures are silently ignored.

> **No action required.** No config change, no token change. Just a UX upgrade.

---

### New workspace dependencies

| Crate | Version | Used in |
|---|---|---|
| `sha2` | `0.10` | Vault KV checksum (`sha256_of`), index content-hash |
| `notify` | `6` | Bidirectional vault watcher |
| `redb` | `2` | Secondary index (`MemoryIndex`) |
| `bincode` | `2` | Reserved for future binary serialisation |
| `lru` | `0.12` | LRU cache in `MemoryIndex` |

---

### Tests to add (recommended)

| Test | Location | What it covers |
|---|---|---|
| `sync_kv_summaries_writes_three_yaml_files` | `vault.rs` | All four files created; checksum line present |
| `sync_kv_summaries_skips_unchanged_files` | `vault.rs` | Returns 0 on no-op second call |
| `read_kv_for_injection_returns_none_without_vault` | `vault.rs` | Graceful no-vault case |
| `vault_round_trip_kv_and_memory_md` | `vault.rs` | Write → read → content matches |
| `vault_watcher_sends_edit_event` | `vault.rs` | Watcher emits `VaultEditEvent` on file change |
| `event_log_append_survives_truncated_file` | `event_log.rs` | Corrupt line skipped; rest loads |
| `memory_index_rebuild_from_log` | `index.rs` | Rebuild produces correct tier counts |
| `memory_index_insert_and_query` | `index.rs` | Insert → `ids_for_tier` → matches |
| `kv_block_prepended_to_context` | `manager.rs` | `context_for_prompt_ranked` first entry has `kv_summary` source |
| `telegram_typing_cancel_does_not_panic` | `lib.rs` | `cancel_tx.send(())` after task drop |

### Verification

- [x] `cargo build` (workspace) — 0 errors, 0 warnings
- [x] `cargo test -p aigent-memory` — 37/37 pass
- [x] `cargo test` (workspace) — all pass

---

## 2026-02-23 — Round 3: Config polish, Belief API, Timezone sleep, Structured outputs

Zero schema migrations required.  All 56 existing tests continue to pass.

### New `[memory]` config keys

| Key | Default | Description |
|---|---|---|
| `kv_tier_limit` | `15` | Max entries per tier in the three YAML KV vault summaries. |
| `timezone` | `"UTC"` | IANA timezone name for the nightly quiet-window calculation (e.g. `"America/New_York"`). Previously the window was always evaluated in UTC. |
| `forget_episodic_after_days` | `0` | Prune Episodic entries older than N days whose confidence is below `forget_min_confidence`. `0` = disabled. |
| `forget_min_confidence` | `0.30` | Confidence ceiling for lightweight forgetting (only entries *below* this value are pruned). |

**Action required**: Copy the new keys from `config/default.toml.example` into your `config/default.toml` if you want non-default values.  The daemon continues to work with old configs (all new fields have serde defaults).

### Belief API (`MemoryManager`)

Three new methods let the agent track discrete propositions it holds as true:

```rust
// Record a belief (stored as a Core entry with source="belief"):
let entry = memory.record_belief("The user prefers dark mode", 0.85).await?;

// List all currently-held beliefs:
let beliefs = memory.all_beliefs(); // → Vec<&MemoryEntry>

// Retract a belief by ID (append-only — creates a retraction record):
memory.retract_belief(entry.id).await?;
```

Retractions are stored as `MemoryTier::Semantic` entries with
`source = "belief:retracted:{id}"`. The original belief entry is never deleted.
`all_beliefs()` automatically excludes any entry with a corresponding
retraction record.

### Timezone-aware sleep window

The nightly multi-agent consolidation task previously compared the current UTC
hour against `night_sleep_start_hour` / `night_sleep_end_hour`, meaning a user
in UTC-5 who set a 22:00 window would have consolidation run at 5 AM their
local time.

**Fix**: set `timezone = "America/New_York"` (or whichever IANA name matches
your location) and the quiet-window gate now operates in your local time.

### Lightweight forgetting

When `forget_episodic_after_days > 0`, the background passive-distillation
task (Task A) now calls `memory.run_forgetting_pass(days, min_confidence)` after
each distillation cycle.  This caps the growth of the Episodic tier for
long-running deployments.

**Recommended starter settings** for active deployments:

```toml
[memory]
forget_episodic_after_days = 90   # keep ~3 months of episodic history
forget_min_confidence      = 0.35 # only prune below-average confidence
```

### Structured LLM output extraction (`aigent-llm`)

Two new public items in `aigent_llm`:

```rust
pub struct StructuredOutput {
    pub action: Option<String>,
    pub params: serde_json::Value,
    pub rationale: Option<String>,
    pub reply: Option<String>,
}

pub fn extract_json_output<T: DeserializeOwned>(response: &str) -> Option<T>;
```

The agent can instruct any model to wrap structured instructions in a fenced
` ```json ` block inside its response.  The runtime uses `extract_json_output`
to detect and parse these blocks, then renders only `reply` to the user while
acting on `action` / `params` internally.

### Enhanced `aigent memory stats` output

The CLI command now displays all six tiers (previously `user_profile` and
`reflective` were omitted), plus:

- **redb index** section: entry count, LRU cache capacity, hit/miss counters, hit-rate percentage.
- **vault checksums** section: per-file existence and checksum status so you can immediately see whether any vault file has been manually edited.

### New workspace dependency

| Crate | Version | Used in |
|---|---|---|
| `chrono-tz` | `0.10` | Timezone-aware sleep window in `aigent-runtime` |

### Memory index cache statistics (`aigent-memory`)

`MemoryIndex` now tracks cumulative LRU cache hits and misses.  The lifetime
counters reset on daemon restart.  Access them via:

```rust
let stats = idx.cache_stats();
// stats.hits, stats.misses, stats.hit_rate_pct, stats.len, stats.capacity
```

These are also surfaced through `MemoryManager::stats().index_cache`.

### Vault checksum health check (`aigent-memory`)

```rust
use aigent_memory::check_vault_checksums;
let report = check_vault_checksums(&vault_root);
// report: Vec<VaultFileStatus> — one per WATCHED_SUMMARIES entry
```

Each entry reports `exists: bool` and `checksum_valid: bool`.  The watcher
uses `is_daemon_written` internally for self-trigger prevention; the health
check uses the same logic to give operators a quick integrity view.

### Verification

- [x] `cargo build --workspace` — 0 errors, 0 warnings
- [x] `cargo test --workspace` — 56/56 pass

---

## Round 4 — Unified Agent Loop & Proactive Mode

### Overview

Phase 2 introduces three interlinked capabilities built on top of the Phase 1
memory foundations:

1. **Belief injection** — the current agent beliefs are automatically injected
   into every conversation prompt via a `MY_BELIEFS:` block, letting the LLM
   express a genuine evolving worldview.
2. **Inline reflection** — immediately after each turn the daemon runs a short
   structured LLM call (`AgentRuntime::inline_reflect`) that extracts new
   beliefs and reflective insights.  These are persisted and streamed to all
   subscribers as `BackendEvent::BeliefAdded` / `BackendEvent::ReflectionInsight`.
3. **Proactive mode (Task C)** — a new optional background task that fires every
   `proactive_interval_minutes` minutes (disabled when 0).  During active hours
   it calls `AgentRuntime::run_proactive_check` and, if the agent decides it has
   something worth saying, broadcasts a `BackendEvent::ProactiveMessage` and
   records it as Episodic provenance `"proactive"`.

### New config keys

Add to your `[memory]` section in `config/default.toml`:

```toml
[memory]
# Enable proactive mode (0 = disabled).
proactive_interval_minutes = 0
# Do-not-disturb window — proactive messages are suppressed during these hours.
proactive_dnd_start_hour = 22
proactive_dnd_end_hour   = 8
```

### New BackendEvent variants

```rust
BackendEvent::ReflectionInsight(String)
BackendEvent::BeliefAdded { claim: String, confidence: f32 }
BackendEvent::ProactiveMessage { content: String }
```

### New ClientCommand variants

```rust
ClientCommand::TriggerProactive      // bypass DND; run proactive check now
ClientCommand::GetProactiveStats     // returns ServerEvent::ProactiveStats(…)
```

### New CLI commands

```sh
aigent memory proactive check    # run a proactive check right now
aigent memory proactive stats    # display proactive activity statistics
```

### New runtime module: `crates/runtime/src/agent_loop.rs`

Public types:

```rust
pub struct ReflectionOutput { beliefs: Vec<ReflectionBelief>, reflections: Vec<String> }
pub struct ReflectionBelief { claim: String, confidence: f32 }
pub struct ProactiveOutput { action: Option<String>, message: Option<String>, urgency: Option<f32> }
pub enum TurnSource { Tui, Telegram { chat_id: i64 }, Cli, Proactive }
```

### DaemonClient new methods

```rust
client.trigger_proactive().await -> Result<String>
client.get_proactive_stats().await -> Result<ProactiveStatsPayload>
```

### Verification

- [ ] `cargo build --workspace` — 0 errors, 0 warnings
- [ ] `cargo test --workspace` — all pass
