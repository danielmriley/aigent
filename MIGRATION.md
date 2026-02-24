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

---

## 2026-02-23 — Cleanup Round (daemon concurrency, DND timezone, belief cap, exhaustive events)

### Concurrency safety in `SubmitTurn` and background tasks

The `SubmitTurn` IPC handler previously held the `DaemonState` mutex for the
entire duration of the LLM call (sometimes 30+ seconds).  This blocked
`GetStatus`, `GetMemoryPeek`, the vault watcher, and any other concurrent
connection.  Task A (passive distillation) had the same issue.

**Fix applied**: all long-running LLM / distillation operations now follow the
consistent pattern already established by `RunSleepCycle` and `RunMultiAgentSleepCycle`:
1. Acquire the lock briefly, clone the runtime, take the `MemoryManager`, release the lock.
2. Do the LLM work without holding any lock.
3. Re-acquire the lock to restore `MemoryManager` and update bookkeeping.

The per-turn auto-sleep is now a fire-and-forget `tokio::spawn` so it never
blocks the response to the client.

No user-visible behavior change; `respond_and_remember_stream`, `inline_reflect`,
and all public APIs are unchanged.

### DND / quiet-window helper

Extracted a shared `is_in_window(now, tz, start_hour, end_hour) -> bool` helper
that is used by both Task B (nightly multi-agent quiet window) and Task C
(proactive DND window).  This eliminates a duplicate midnight-wrap calculation
and makes both windows respect the same `[memory] timezone` config key.

### Belief injection cap (`max_beliefs_in_prompt`)

A new `[memory]` key controls how many beliefs are injected into each prompt:

```toml
[memory]
max_beliefs_in_prompt = 5   # default; 0 = unlimited
```

Beliefs are sorted by confidence (descending) before the cap is applied so the
most strongly-held beliefs always survive truncation.  Add this key to your
`config/default.toml` to keep your prompt size under control as beliefs accumulate.

### Exhaustive `BackendEvent` handling in Telegram

The Telegram response handler's `_ => {}` wildcard was replaced with explicit
match arms for every `BackendEvent` variant.  The new Phase-2 variants
(`ReflectionInsight`, `BeliefAdded`, `ProactiveMessage`) are logged at
`debug` level; all lifecycle/meta events remain no-ops in the per-request
response path (they are still delivered to persistent `Subscribe` connections).

### Verification

- [ ] `cargo build --workspace` — 0 errors, 0 warnings
- [ ] `cargo test --workspace` — all pass

---

## 2026-02-24 — Phase 3: Tool Use + Cleanup Round 2

### Cleanup items

#### 1. Proactive shutdown safety (`DaemonState::proactive_handle`)

Task C (proactive mode) now stores its `AbortHandle` in `DaemonState`.
On daemon shutdown the handle is aborted before the final flush-and-sleep so
Task C can never fire mid-exit.

```rust
// DaemonState gains:
proactive_handle: Option<tokio::task::AbortHandle>,

// Shutdown (run_unified_daemon):
let handle = state.lock().await.proactive_handle.take();
if let Some(h) = handle { h.abort(); info!("proactive task stopped"); }
```

#### 2. Proactive cooldown (`proactive_cooldown_minutes`)

New `[memory]` config key (default `5`).  Skips a proactive check if a message
was sent within the last N minutes, preventing bursts when the daemon becomes
very active.

```toml
[memory]
proactive_cooldown_minutes = 5   # 0 = no cooldown
```

#### 3. Richer belief injection sorting

Beliefs are now sorted by a composite score before the `max_beliefs_in_prompt`
cap is applied:

```
score = confidence × 0.6  +  recency_factor × 0.25  +  valence × 0.15
recency_factor = 1 / (1 + days_since_created)
```

The most relevant, recent, and positive beliefs always surface first regardless
of raw confidence.

#### 4. README / capabilities matrix updated

- Telegram command parity: `✅ Complete`
- Proactive mode entry: cooldown and graceful shutdown noted
- `Belief injection into prompts`: updated to reflect composite scoring

---

### Phase 3: Full Tool Use & Action Capabilities

#### New tools (aigent-tools, src/builtins.rs)

| Tool | Description | Data path |
|---|---|---|
| `calendar_add_event` | Append event to JSON calendar store | `.aigent/calendar.json` |
| `web_search` | DuckDuckGo Instant Answer API | live HTTP |
| `draft_email` | Save draft as plain-text file | `.aigent/drafts/` |
| `remind_me` | Append reminder for proactive surfacing | `.aigent/reminders.json` |

All four are **approval-exempt** by default (`approval_exempt_tools` in config).

#### ExecutionPolicy: per-tool allow/deny lists

```toml
[safety]
tool_allowlist = []                    # empty = all tools permitted
tool_denylist  = []                    # explicit block list
approval_exempt_tools = ["calendar_add_event", "remind_me", "draft_email", "web_search"]
```

These map to three new fields on `ExecutionPolicy`:

```rust
pub tool_allowlist: Vec<String>,
pub tool_denylist: Vec<String>,
pub approval_exempt_tools: Vec<String>,
```

#### LLM-driven tool calling in `SubmitTurn`

Before each streaming response the daemon executes a brief non-streaming LLM
call (`AgentRuntime::maybe_tool_call`) that returns either a tool invocation
JSON or `{"no_action":true}`.  On zero overhead for conversational messages.

Flow:
1. `maybe_tool_call(user_message, tool_specs) → Option<LlmToolCall>`
2. If `Some`: execute → record to `Procedural` with `source="tool-use:{name}"` → emit `ToolCallStart` / `ToolCallEnd`
3. Inject `[TOOL_RESULT: …]` block into the effective user message for the main LLM call
4. `inline_reflect` receives the original user ↔ assistant exchange (not the tool-augmented prompt)

#### New CLI subcommand: `aigent tool`

```bash
aigent tool list                             # list registered tools
aigent tool call web_search query="Rust"     # call a tool directly
aigent tool call calendar_add_event title="Meeting" date="tomorrow"
```

#### `aigent memory stats` — tool execution section

```
── tool executions ──────────────────────────────────
  today (24h): 3
  all time:    17
    web_search: 10
    calendar_add_event: 5
    remind_me: 2
```

#### `default_registry` signature change

`aigent_exec::default_registry` now takes a second `agent_data_dir: PathBuf`
argument for the new data-path tools.  The only call site is in
`crates/runtime/src/server.rs` (auto-passes `.aigent/`).

#### Migration steps

1. `cargo build --workspace` — should produce 0 errors, 0 warnings.
2. Run `aigent daemon restart` to pick up the new tools and policy fields.
3. Add the new optional keys to your `config/default.toml` if desired
   (or leave them out — serde defaults apply).
4. Test: `aigent tool list` should show 7 tools.
5. Test: `aigent tool call web_search query="hello world"`.

### Verification

- [ ] `cargo build --workspace` — 0 errors, 0 warnings
- [ ] `cargo test --workspace` — all pass
- [ ] `aigent tool list` — 7 tools listed
- [ ] `aigent tool call web_search query="Rust programming"` — returns results

---

## 2026-02-24 — Phase 4: Approval Modes, Git Rollback, Brave Search & WASM Groundwork

### New features

#### 1. Tool approval modes (`[tools] approval_mode`)

A new top-level `[tools]` config section replaces the coarse
`[safety] approval_required` boolean with a three-way enum:

| Mode | Read-only tools | Write / shell | Notes |
|------|-----------------|---------------|-------|
| `safer` | prompt | prompt | max safety |
| `balanced` | auto-approved | prompt | **new default** |
| `autonomous` | auto-approved | auto-approved | workspace still sandboxed |

The legacy `approval_required = true` flag still works but is now ignored
when `approval_mode = "autonomous"`.

**Action required**: add to `config/default.toml`:

```toml
[tools]
approval_mode = "balanced"
brave_api_key = ""
git_auto_commit = false
```

Or copy from `config/default.toml.example`.

#### 2. Git auto-commit (`git_auto_commit`)

When `[tools] git_auto_commit = true`:

1. The tool executor runs `git add -A && git commit` after every successful
   `write_file` or `run_shell` invocation.
2. Commit messages are `"Aigent tool: <name> — <detail>"`.
3. The new `git_rollback` tool (8th built-in) calls `git revert HEAD --no-edit`
   to undo the last automated change.

During onboarding, if the workspace is not a git repository, `git init` is
run automatically.  Silently skipped when git is not installed.

#### 3. Brave Search API (`brave_api_key`)

`web_search` now uses the [Brave Search API](https://api.search.brave.com/)
when a key is configured:

```toml
[tools]
brave_api_key = "YOUR_KEY_HERE"
```

Or set the `BRAVE_API_KEY` environment variable (takes precedence over the
config file).  Falls back to DuckDuckGo Instant Answers when no key is set.

#### 4. `git_rollback` tool

New 8th built-in tool that can be invoked by the LLM or directly from the CLI:

```bash
aigent tool call git_rollback
```

Read-only from the approval-mode perspective — `balanced` auto-approves it.

#### 5. WASM guest source templates (`extensions/tools-src/`)

Rust source code for the three core tools compiled to `wasm32-wasip1`:

- `extensions/tools-src/read-file/`
- `extensions/tools-src/write-file/`
- `extensions/tools-src/run-shell/`

These are a **separate sub-workspace** (not compiled by `cargo build` at the
root).  To build them:

```sh
rustup target add wasm32-wasip1
cd extensions/tools-src && cargo build --release
# outputs: target/wasm32-wasip1/release/{read_file,write_file,run_shell}.wasm
```

The host-side WasmTool loader (upcoming) will pick up `.wasm` files from
`extensions/` at daemon startup, letting WASM tools coexist with or replace
native Rust implementations.

#### 6. WIT host API additions

`extensions/wit/host.wit` gains four new host-provided functions:

```wit
export git-commit:        func(message: string) -> result<string, string>;
export git-rollback-last: func() -> result<string, string>;
export git-log-last:      func() -> result<string, string>;
export secret-get:        func(name: string) -> result<string, string>;
```

These are available to WASM guest tools once the wasmtime host runtime is wired
up.

### `ExecutionPolicy` changes

Two new fields added to `ExecutionPolicy` in `aigent-exec`:

```rust
pub approval_mode: ApprovalMode,  // ← new; replaces approval_required semantics
pub git_auto_commit: bool,         // ← new
```

`build_execution_policy` in `server.rs` fills both from `config.tools`.

### `default_registry` signature change

`aigent_exec::default_registry` now takes a third argument:

```rust
// Before:
default_registry(workspace_root: PathBuf, agent_data_dir: PathBuf) -> ToolRegistry

// After:
default_registry(workspace_root: PathBuf, agent_data_dir: PathBuf, brave_api_key: Option<String>) -> ToolRegistry
```

All call sites (tests and `server.rs`) must pass `None` or the resolved key.

### Migration steps

1. `cargo build --workspace` — 0 errors, 0 warnings.
2. Add `[tools]` section to `config/default.toml` (see example above).
3. Run `aigent daemon restart` to apply the new approval mode.
4. **Optional**: set `brave_api_key` or `BRAVE_API_KEY` to enable richer search.
5. **Optional**: set `git_auto_commit = true` and run `aigent onboard` (or
   `git init <workspace>` manually) to enable rollback protection.
6. Test: `aigent tool list` — should show 8 tools.
7. Test: `aigent tool call git_rollback` — should report "workspace is not a
   git repository" or revert the last commit if git is initialised.

### Verification

- [ ] `cargo build --workspace` — 0 errors, 0 warnings
- [ ] `cargo test --workspace` — all pass
- [ ] `aigent tool list` — 8 tools listed (includes `git_rollback`)
- [ ] `aigent tool call web_search query="Rust programming"` — results returned
- [ ] Set `approval_mode = "autonomous"`, call `aigent tool call write_file path="test.txt" content="hi"` — no prompt
- [ ] Set `git_auto_commit = true`, call `write_file`, run `git log` — commit visible
- [ ] Call `git_rollback` — commit reverted


---

## 2026-02-24 — Phase 5: Full WASM Host Runtime, Platform Sandboxing & Onboarding Wizard Completion

### What changed

#### Phase 1 — Wasmtime host runtime (`wasm` feature, enabled by default)

New file `crates/exec/src/wasm.rs` implements:
- `WasmTool` — a `Tool` implementation backed by a compiled `.wasm` binary.
  Uses Wasmtime + WASIP1 with in-memory stdin/stdout pipes for the JSON
  protocol.  Module compilation happens once at load; instantiation is
  per-call (stateless, fully isolated linear memory).
- `load_wasm_tools_from_dir(extensions_dir)` — scans two layouts:
  - Direct: `<extensions_dir>/<name>.wasm`
  - Sub-workspace: `<extensions_dir>/tools-src/<crate>/target/wasm32-wasip1/release/<name>.wasm`
- `default_registry()` in `exec/src/lib.rs` now calls `load_wasm_tools_from_dir`
  after registering native tools.  WASM tools shadow native tools by name.

**Building guest tools:**

```bash
rustup target add wasm32-wasip1
cd extensions/tools-src
cargo build --release
# Daemon picks up *.wasm on next start — no daemon source change needed
```

**Native tools remain as fallback** when no `.wasm` files are present.  This
ensures all existing tests and deployments continue to work unchanged.

New workspace dependencies added:
```toml
wasmtime        = { version = "25", default-features = false, features = ["cranelift", "runtime", "component-model"] }
wasmtime-wasi   = { version = "25", default-features = false, features = ["preview1"] }
bytes           = "1"
libc            = "0.2"
```

New exec crate features:
```toml
wasm    = ["dep:wasmtime", "dep:wasmtime-wasi", "dep:bytes"]
sandbox = ["dep:libc"]
```

#### Phase 2 — Platform sandboxing (`sandbox` feature)

New file `crates/exec/src/sandbox.rs` implements:
- `apply_to_child(workspace_root: &str)` — unsafe function called in a `pre_exec`
  hook between `fork` and `exec` when spawning shell children.
- **Linux x86-64**: `PR_SET_NO_NEW_PRIVS` (no privilege escalation via setuid)
  followed by a seccomp BPF ALLOW-list covering ~80 syscalls; unrecognised
  syscalls return `ENOSYS` rather than `SIGSYS`.
- **macOS**: `sandbox_init(3)` with an inline Scheme profile allowing
  workspace file R/W, `/tmp`, standard libs, outbound TCP 80/443, and
  process management.  All other operations denied by default.
- `is_active() -> bool` — useful for daemon status reporting.

`exec/src/lib.rs` gains `ToolExecutor::run_shell_sandboxed()` which is
invoked automatically in `execute()` when `tool_name == "run_shell"` and the
`sandbox` feature is active.

**To enable sandboxing** (not the default — requires explicit opt-in):
```toml
# .cargo/config.toml or Cargo.toml
[features]
default = ["sandbox"]
```
Or build with `cargo build --workspace --features aigent-exec/sandbox`.

#### Phase 3 — Onboarding wizard completion

`crates/interfaces/tui/src/onboard.rs` gains two new wizard steps:
- `WizardStep::ApprovalMode` — choice carousel: `safer` / `balanced` / `autonomous`
- `WizardStep::ApiKeys` — masked text input for the Brave Search API key
  (skippable; falls back to DuckDuckGo when blank)

Position in the onboarding flow:
```
… → Safety → ApprovalMode → ApiKeys → Messaging → …
```

In the configuration menu the `Safety` section now covers all three steps
(`Safety` → `ApprovalMode` → `ApiKeys` → menu).

New `OnboardingDraft` fields: `approval_mode: String`, `brave_api_key: String`.
Both are persisted to `config.tools` in `apply()` and `apply_partial()`.

`prompt_safety_settings()` (prompt fallback path) now also asks for
`approval_mode` and Brave key.

#### Phase 4 — Docs

- `README.md`: Updated capabilities matrix (WASM runtime live, platform
  sandboxing live); updated `### WASM extension interface` section with build
  instructions; replaced "Future: platform sandboxing" with `sandbox` feature
  table; updated onboarding wizard entry.
- `config/default.toml.example`: unchanged (already has `[tools]` from Phase 4).
- `MIGRATION.md`: this entry.

### Migration steps

1. `cargo build --workspace` — still 0 errors, 0 warnings.
2. `cargo test --workspace` — 57/57 pass.
3. **Optional — activate WASM tools**:
   ```bash
   rustup target add wasm32-wasip1
   cd extensions/tools-src && cargo build --release
   aigent daemon restart
   aigent tool list   # read_file etc. now say "(WASM guest)" in their descriptions
   ```
4. **Optional — enable sandbox**:
   ```bash
   cargo build --workspace --features aigent-exec/sandbox
   # Rebuild the binary and install; shell children will have seccomp/macOS sandbox
   ```
5. Run `aigent onboard` or `aigent configuration` — new **Approval Mode** and
   **API Keys** wizard steps appear in the Safety section.

### Verification checklist

- [ ] `cargo build --workspace` — 0 errors, 0 warnings
- [ ] `cargo test --workspace` — all pass (57 tests)
- [ ] `aigent tool list` — 8 tools (native baseline, no WASM binaries built yet)
- [ ] Build guest tools → `aigent daemon restart` → `aigent tool list` shows
      "(WASM guest)" description suffix for `read_file`, `write_file`, `run_shell`
- [ ] `aigent onboard` → wizard shows `Approval Mode` and `API Keys` steps after `Safety`
- [ ] `approval_mode = "autonomous"` in config → `write_file` call requires no prompt
- [ ] `approval_mode = "safer"` → every tool call prompts for approval
- [ ] Build with `--features aigent-exec/sandbox` → `run_shell` child has seccomp
      filter active (use `strace -e seccomp` or `ausearch` to verify)
- [ ] macOS: build with sandbox feature → `sandbox_init` applied (check Console.app)

---

## 2026-02-25 — Phase 6: WASM-First Default, Sandbox Default-On, New CLI Commands

### Summary

Three interlocking changes that make WASM and sandboxing the production defaults
rather than opt-in features:

1. **WASM-first registry** — WASM guest tools are now registered before native
   implementations.  `ToolRegistry::get` uses `.find()` (first-match wins), so any
   compiled `.wasm` binary takes precedence over the equivalent Rust builtin.  Native
   tools are registered only for tool names that have no WASM binary present.
2. **`sandbox` and `wasm` features are now default-on** — `default = ["wasm", "sandbox"]`
   in `crates/exec/Cargo.toml`.  The platform sandbox is applied to every `run_shell`
   child without needing a special build flag.
3. **`aigent tools build` / `aigent tools status`** — two new CLI commands for
   managing guest tools without touching `rustup` or `cargo` directly.

### Breaking changes

None.  All new fields use `serde(default = …)` so existing `config/default.toml`
files continue to work.  The only behaviour change is that sandbox is now active by
default on Linux/macOS; set `sandbox_enabled = false` in `[tools]` to opt out.

### Changed files

#### `crates/exec/Cargo.toml`

```toml
[features]
default = ["wasm", "sandbox"]
sandbox = ["dep:libc"]
wasm    = ["dep:wasmtime", "dep:wasmtime-wasi", "dep:bytes"]
```

Previously `default` was empty — both features had to be opted in.

#### `crates/exec/src/lib.rs`

- `ExecutionPolicy` gains a new field:

  ```rust
  /// Apply platform sandbox to shell children (`true` by default).
  /// Set to `false` to disable without recompiling.
  pub sandbox_enabled: bool,
  ```

  `Default` impl sets `sandbox_enabled: true`.

- `default_registry()` rewritten for WASM-first:
  1. Load WASM tools → collect names into `HashSet`
  2. Register WASM tools first (they win in `.find()`)
  3. Register native tools only for names not in the WASM set
  4. Log at `info!` which tools are in WASM mode vs native fallback

- Sandbox gate guarded by `self.policy.sandbox_enabled` as well as the feature flag:
  ```rust
  #[cfg(all(feature = "sandbox", unix))]
  if tool_name == "run_shell" && self.policy.sandbox_enabled { … }
  ```

#### `crates/config/src/lib.rs`

New field on `ToolsConfig`:

```toml
# config/default.toml
[tools]
sandbox_enabled = true   # set false to disable runtime sandboxing
```

```rust
pub struct ToolsConfig {
    // … existing fields …
    #[serde(default = "default_sandbox_enabled")]
    pub sandbox_enabled: bool,
}
```

#### `crates/runtime/src/server.rs`

`build_execution_policy()` now forwards `config.tools.sandbox_enabled` into the policy.

#### `crates/interfaces/cli/src/main.rs`

Two new `ToolCommands` variants:

```text
aigent tools build   — rustup target add wasm32-wasip1 + cargo build --release
                       in extensions/tools-src/
aigent tools status  — filesystem check: WASM binary found vs native fallback
                       for each of the 8 built-in tool names, plus sandbox state
```

`aigent tools status` is a pure filesystem check — it does **not** require the
daemon to be running.

`crates/interfaces/cli/Cargo.toml` gains `aigent-exec` as a direct dependency
(zero extra build cost — already compiled transitively).

### Migration steps

1. `cargo build --workspace` — all features now compile by default, no flags needed.
2. Run `aigent tools status` — all 8 tools will show **native** until WASM guests are built.
3. `aigent tools build` — compiles guests; then `aigent daemon restart` to activate.
4. Run `aigent tools status` again — tools with a built `.wasm` now show **WASM**.
5. To disable sandboxing at runtime (without recompiling):

   ```toml
   # config/default.toml
   [tools]
   sandbox_enabled = false
   ```

### Verification checklist

- [ ] `cargo build --workspace` — 0 errors, 0 warnings
- [ ] `cargo test --workspace` — all tests pass
- [ ] `aigent tools status` before building guests → all 8 tools show "native"
- [ ] `aigent tools build` → succeeds; `aigent tools status` shows WASM for built crates
- [ ] Daemon logs show `"native Rust fallback active for N tool(s)"` on startup (no WASM)
- [ ] Daemon logs show `"wasm: N guest tool(s) active"` after building guests
- [ ] `sandbox_enabled = false` in config → shell children no longer sandboxed
- [ ] Linux: sandbox active by default → `strace -e seccomp aigent start` shows seccomp applied
- [ ] macOS: `sandbox_init` applied without `--features sandbox` build flag
