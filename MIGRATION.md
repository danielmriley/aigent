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
