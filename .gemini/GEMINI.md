# GEMINI.md

This file provides guidance to Gemini CLI when working with code in this repository.

## Build & Development Commands

```bash
# Build (dev)
cargo build --workspace

# Build (release — used by install.sh)
cargo build --release --locked

# Run all tests
cargo test --workspace

# Run a single test by name
cargo test -p aigent-memory sleep_cycle_dedup
cargo test -p aigent-exec exempt_tools_bypass_approval

# Lint (must pass with zero warnings — enforced in CI)
cargo clippy --workspace -- -D warnings

# Format
cargo fmt --all

# Build WASM guest tools (separate workspace under extensions/)
aigent tools build
# equivalent: cd extensions/tools-src && rustup target add wasm32-wasip1 && cargo build --release
```

The `rust-toolchain.toml` pins the toolchain version; `rustup` picks it up automatically. `clippy.toml` and `deny.toml` enforce additional lints.

## Workspace Crate Dependency Graph

```text
aigent-config
    └── aigent-tools          (ToolRegistry, ToolSpec, ToolParam, all built-ins)
            └── aigent-llm    (LlmClient trait, ChatMessage, streaming)
                    └── aigent-exec   (ToolExecutor, ExecutionPolicy, sandbox)
                            └── aigent-memory  (MemoryManager, 6-tier event log, MemoryIndex)
                                    └── aigent-thinker  (ext_loop, prompt, JsonStreamBuffer)
                                            └── aigent-prompt
                                                    └── aigent-agent  (AgentRuntime, run_agent_turn, sleep)
                                                            └── aigent-runtime  (daemon, UnixSocket IPC, DaemonState)
                                                                    ├── aigent-app (CLI — crates/interfaces/cli)
                                                                    ├── aigent-ui  (TUI — crates/interfaces/tui)
                                                                    └── aigent-telegram
```

The CLI binary (`aigent-app`) is the single entry point; it starts the daemon and frontends.

## Architecture Overview

### Daemon-first IPC model

The daemon (`run_unified_daemon` in `crates/runtime/src/server/mod.rs`) owns **all mutable state** behind a `Arc<Mutex<DaemonState>>`. Frontends (TUI, Telegram, CLI) are thin clients that communicate over a Unix socket (`/tmp/aigent.sock`). Each connection runs `handle_connection` from `server/connection.rs`. A `broadcast::Sender<BackendEvent>` fans out events (tokens, beliefs, tool results, proactive messages) to all subscribers.

### 6-tier memory system

All state lives in an **append-only JSONL event log** at `.aigent/memory/events.jsonl`. Writes are crash-safe: `append()` fsyncs every write; `overwrite()` uses `tmp → fsync → rename`. Corrupt lines are skipped and saved to `.corrupt` sidecar — a bad line never kills the daemon.

Tiers in priority order: `Core → UserProfile → Reflective → Semantic → Procedural → Episodic`

Key types and patterns:

- `MemoryEntry` — the stored record (UUID, tier, content, source string, confidence, valence, timestamps)
- `SourceKind::from_source(&str)` — parse the source string into a typed enum. **Do not use `from_str`** (that name is reserved for `std::str::FromStr`). Call via `entry.source_kind()`.
- `entry.source_kind().is_sleep()` — covers all sleep-cycle source variants
- `MemoryManager` (in `manager/mod.rs`) — the main façade; owns the in-memory store, event log, optional `MemoryIndex`, and `EmbedFn`
- `MemoryIndex` (redb-backed) — optional secondary index at `memory.index.redb`; O(log n) tier lookups with an LRU-256 cache. Rebuilt transparently on startup if absent or corrupt. Never scan `events.jsonl` directly; route through the index or in-memory store.

Retrieval scoring: `tier(0.35) + recency(0.20) + lexical(0.25) + embedding(0.15) + confidence(0.05)`

### Tool execution pipeline

Every tool call goes through this chain:

1. **`ToolRegistry`** (`crates/tools/src/lib.rs`) — stores `Vec<ToolEntry>` (WASM tools registered first; first-match wins). `get(name)` does a linear scan — a pending optimization is to add a `HashMap` index.
2. **`ToolExecutor`** (`crates/exec/src/lib.rs`) — applies `ExecutionPolicy` (allowlist/denylist/approval mode/security level/rate limiting) before and after every call.
3. **Sandbox** (`crates/exec/src/`) — `PR_SET_NO_NEW_PRIVS` + seccomp BPF (Linux) or `sandbox_init` (macOS) applied as a `pre_exec` hook on shell children.
4. **`run_external_thinking_loop`** (`crates/thinker/src/ext_loop.rs`) — the agentic ReAct loop: `build_ext_think_prompt → LLM stream → parse JSON → validate params → execute tool → inject observation → repeat` up to `MAX_EXT_ROUNDS = 10`.

The ext_loop uses `JsonStreamBuffer` to collect streamed JSON incrementally (brace-depth tracking). The structured step format is `{"type":"tool_call",...}` / `{"type":"final_answer",...}`. Duplicate call detection uses a `HashSet<(String, u64)>` keyed on tool name + hash of sorted args.

### Config-driven limits (do not hardcode)

All execution limits live in `AppConfig` and are wired through `default_registry()`:

- `config.agent.tool_timeout_secs` — max seconds per tool call
- `config.tools.max_shell_command_bytes` — shell command size cap
- `config.tools.max_shell_output_bytes` — shell output size cap
- `config.tools.max_file_read_bytes` — file read size cap
- `config.safety.max_calls_per_tool` — per-session call rate limit

### Sleep consolidation pipeline

Three modes, all in `crates/memory/src/`:

- **Passive** (`sleep.rs`) — heuristic promotion; no LLM
- **Agentic** (`manager/sleep_logic.rs`) — single LLM reflection pass
- **Multi-agent** (`multi_sleep.rs`) — 4 parallel specialist LLMs (Identity, Relationships, Knowledge, Reflections) + synthesis agent; rate-limited to once per 22 hours

Sleep is always append-only — results are new `MemoryEntry` records with `sleep:*` sources, never mutations.

### Vault / Obsidian projection

`sync_kv_summaries()` writes `core_summary.yaml`, `user_profile.yaml`, `reflective_opinions.yaml`, and `MEMORY.md` incrementally to `.aigent/vault/`. A `notify`-based watcher (`spawn_vault_watcher`) detects human edits and ingests them as `source = "human-edit"` entries. Always write incrementally — never rebuild the entire projection.

### Schema deduplication

`build_param_properties(&[ToolParam])` (in `crates/tools/src/lib.rs`) is the single shared helper for both `to_openai_tool_schema()` and `to_json_schema()`. Do not duplicate the property-building loop.

## Critical Invariants

- **Append-only event log** — never mutate existing events. State changes = new events. The only exception is `overwrite()` used by `wipe`/`compact`/Core retirements, which uses atomic tmp+rename.
- **WASM-first tool registration** — WASM tools registered before native builtins; first match in the `Vec` wins. Never reorder this.
- **SourceKind method name** — the parser is `SourceKind::from_source(s)`, not `from_str`. The `MemoryEntry::source_kind()` convenience method calls it.
- **No linear scans over events.jsonl** — use `MemoryManager` methods (which operate on the in-memory store + redb index). The sleep/consolidation pipeline must also use indexed retrieval only.
- **Clippy -D warnings is a hard gate** — all code must be warning-free. `#[allow(...)]` requires a justification comment.

## Mandatory Performance & Algorithmic Efficiency Protocol

For **every single task** (especially anything involving memory, lookups, loops, tools, retrieval, or data processing):

1. **Complexity Analysis in Every Planner Phase**  
   Explicitly state the time/space complexity of the current approach **and** your proposed solution (O(1), O(log n), O(n), O(n log n), O(n²), etc.).  
   If a naïve O(n) or worse solution exists, you **must** propose and justify the better alternative (or explain why O(n) is truly unavoidable here).  
   End the Planner phase with: “Complexity analysis complete. Proposed Big-O: ___(improved from___)”

2. **Project-Specific Efficiency Mandates** (Non-Negotiable)
   - **NEVER** do linear scans over events.jsonl or any tier. Always route through the redb-backed MemoryIndex + LRU cache for O(log n) tier/lookup.
   - In-memory collections: Prefer BTreeMap (ordered), HashMap (fast lookup), or Vec only for true append-only. Never use `.iter().find()` or `.contains()` on Vec when an index exists. (Note: ToolRegistry linear scan is a known pending optimization — prioritize HashMap index.)
   - WASM tools: Use zero-copy WIT interface. Minimize serialization in hot paths.
   - Daemon/async: Minimize unnecessary Tokio task spawning; prefer efficient channels.
   - Self-improvement pipeline: All specialist consolidation and distillation steps must use indexed/hybrid retrieval only — no full-history O(n) passes.
   - Obsidian vault: Always incremental writes; never rebuild the entire projection.

3. **Rust Performance Standards**
   - Prefer zero-cost abstractions, borrows over clones in hot paths.
   - Minimize allocations in performance-sensitive code.
   - For any new data structure or algorithm, document its Big-O and why it was chosen.

4. **Efficiency Reflection at Every Checkpoint**  
   After every test run or major change:  
   - State how the change preserves or improves overall system performance.  
   - Example: “Switched to redb index → O(log n) lookup instead of O(n) scan; zero impact on append-only contract.”

## Feature Flags

| Flag        | Crate            | Effect                                      |
|-------------|------------------|---------------------------------------------|
| `wasm`      | `aigent-exec`    | Wasmtime host runtime (default-on)          |
| `sandbox`   | `aigent-exec`    | seccomp/sandbox_init (default-on)           |
| `candle`    | `aigent-llm`, `aigent-agent` | Local inference via Candle |
| `marketplace` | workspace      | Extension marketplace (opt-in)              |

Disable sandbox at runtime with `[tools] sandbox_enabled = false` — no recompile needed.

## Runtime Files

| Path                              | Purpose                                              |
|-----------------------------------|------------------------------------------------------|
| `.aigent/memory/events.jsonl`     | Canonical append-only memory event log               |
| `.aigent/memory/memory.index`     | redb secondary index (rebuilt on startup if absent)  |
| `.aigent/history/YYYY-MM-DD.jsonl`| TUI conversation history (last 200 turns restored)   |
| `.aigent/vault/`                  | Obsidian-compatible markdown projection              |
| `.aigent/runtime/daemon.pid` / `.log` | Daemon lifecycle files                          |
| `/tmp/aigent.sock`                | Unix socket (configurable via `daemon.socket_path`)  |
| `config/default.toml`             | Main config (all sections: `[agent]`, `[llm]`, etc.) |

## Final Instruction

Always follow the Performance & Algorithmic Efficiency Protocol with the same priority as the Critical Invariants. Prioritize O(log n) or better whenever possible without compromising correctness or safety. This is now a permanent rule for all work on aigent in Gemini CLI.
