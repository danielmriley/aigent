# Aigent

Aigent is a persistent, self-improving AI agent written in Rust. It runs as a background daemon and connects to any frontend — a local TUI, a Telegram bot, or an external client over a Unix socket. Memory is stored in a 6-tier append-only event log that grows smarter over time through nightly sleep consolidation driven by a pipeline of parallel LLM specialist agents.

## Status

Phases 0 (Foundation), 1 (Memory), and 2 (Unified Agent Loop) are complete. The daemon-first architecture is fully implemented. An ongoing Phase 3 focuses on extension polish, systemd/launchd integration, and full channel parity.

## Building from source

**Prerequisites**

- [Rust](https://rustup.rs) stable toolchain (`rust-toolchain.toml` pins the version; `rustup` picks it up automatically)
- [Ollama](https://ollama.com) — optional, required only when using the local Ollama provider

```bash
git clone https://github.com/danielmriley/aigent
cd aigent
cargo build --release --locked
```

The binary is produced at `target/release/aigent-app`.

**One-line install to `~/.local/bin`**

```bash
./install.sh
```

Uses `cargo build --release --locked` for reproducible builds, installs atomically, prints a SHA-256 checksum, and restarts a running daemon automatically.

**First run**

```bash
# Option A — interactive setup wizard (recommended)
aigent onboard

# Option B — copy the example config and edit manually, then start
cp config/default.toml.example config/default.toml
aigent start
```

**Environment variables** (written to `.env` by the wizard automatically)

| Variable | Purpose |
| --- | --- |
| `OPENROUTER_API_KEY` | Required when provider is set to `openrouter` |
| `OLLAMA_BASE_URL` | Override Ollama endpoint (default: `http://localhost:11434`) |
| `TELEGRAM_BOT_TOKEN` | Required to enable the Telegram bot |

## Capabilities

### Memory system

Aigent uses a **6-tier memory architecture** backed by an append-only JSONL event log (`.aigent/memory/events.jsonl`) as the canonical source of truth.

| Tier | Purpose |
| --- | --- |
| `Core` | Identity, constitution, personality — consistency-firewalled; rewrite requires approval |
| `UserProfile` | Persistent user facts: preferences, goals, life context |
| `Reflective` | Agent thoughts, plans, self-critiques, proactive follow-ups |
| `Semantic` | Distilled facts and condensed knowledge promoted from episodic memory |
| `Procedural` | Learned skills, workflows, how-to knowledge |
| `Episodic` | Raw conversation turns and temporary observations |

**Retrieval** uses a hybrid weighted score:
`tier(0.35) + recency(0.20) + lexical(0.25) + embedding(0.15) + confidence(0.05)`

When Ollama is available, entries are embedded at record time via `/api/embeddings` and cosine similarity upgrades lexical matching to semantic vector search. When no embedding backend is configured, the embedding weight is redistributed proportionally across the other components. Core and UserProfile entries are always injected into the prompt regardless of score.

A **high-density relational matrix** is injected into every prompt — a compact cross-tier association table that surfaces connections between memory entries the agent might otherwise miss.

An **Obsidian-compatible vault** is auto-projected under `.aigent/vault/` with per-tier indexes, daily memory notes, topic backlinks, and wiki-style links. New entries are written incrementally so the vault stays in sync after every turn without a full rebuild.

### Storage & Performance

**Crash-safe append**: Every write to `events.jsonl` is followed by `flush()` + `sync_all()` so the entry survives a process crash or power loss immediately after `record()`. The `overwrite` path (used by `wipe`, `compact`, and Core retirements) writes to a `.tmp` sibling, fsyncs, then renames atomically — a crash at any point leaves either the old or the new file fully intact.

**Resilient JSONL loading**: Corrupt lines in `events.jsonl` are skipped with a `warn!` trace that includes the line number and error. The bad line is appended to a `events.jsonl.corrupt` sidecar file for forensic inspection. The remaining events load normally — a single bad line never takes down the daemon.

**Redb secondary index** (`aigent_memory::MemoryIndex`): An optional `redb`-backed secondary index lives alongside the JSONL log at `~/.aigent/memory/index.redb`. It stores compact entry metadata (confidence, tier, timestamp, source, content-hash) keyed by UUID and a tier lookup table. An LRU cache (256 entries) sits in front for hot-path reads. If the index file is absent or corrupt it is rebuilt transparently from the event log — zero data loss. The index is opt-in and non-critical: when unavailable, all operations fall back to the in-memory store.

### Vault Projection & Human Co-Authoring

Every sleep cycle writes four auto-generated summary artefacts to the vault root in addition to the Obsidian note/index files:

| File | Content | Tier |
|---|---|---|
| `core_summary.yaml` | Top-15 Core entries by confidence | Identity & constitution |
| `user_profile.yaml` | Top-15 UserProfile entries | User facts & preferences |
| `reflective_opinions.yaml` | Top-15 Reflective entries | Agent thoughts & opinions |
| `MEMORY.md` | Human-friendly prose consolidation linking all three | All three tiers |

**YAML format**: Each file has a `checksum: sha256:…` field and a `last_updated` timestamp so you (and the daemon) can detect real changes at a glance. Files are written incrementally — unchanged files are not touched across sleep cycles.

**Truncation policy**: Each file contains at most `KV_TIER_LIMIT` (default: 15) items, sorted by `confidence DESC → recency DESC → valence DESC`. This keeps each file well under 200 lines and prevents context-window bloat.

**Auto-injection**: On every LLM turn, the daemon reads `core_summary.yaml` and `user_profile.yaml` and prepends them as a pinned `AGENT IDENTITY` block at the very top of the prompt context (score 2.0 — always first). This guarantees the agent never forgets who it is even if retrieval ranking would otherwise demote Core entries.

**Bidirectional edits**: A background `notify`-based file watcher monitors the four summary files. When a human edits any of them directly in Obsidian (or any editor), the daemon detects the change and ingests it as a high-confidence `MemoryEntry` with `source = "human-edit"`. The appropriate tier is inferred from the filename (`core_summary.yaml` → Core, `user_profile.yaml` → UserProfile, `reflective_opinions.yaml` → Reflective). The next sleep cycle reconciles the edit with existing memory. This gives you direct, persistent control over the agent's identity — edit the YAML, shape the soul.

### Beliefs & inline reflection

Every completed conversation turn now triggers a short structured LLM call (`inline_reflect`) that extracts up to three new **beliefs** and two free-form **reflective insights** from the exchange. Beliefs are stored in Core memory with a `belief` tag and a confidence score; reflective insights are stored in the Reflective tier. Both are streamed to all subscribers as `BackendEvent::BeliefAdded` and `BackendEvent::ReflectionInsight` events in real time.

All current beliefs (up to `max_beliefs_in_prompt`, default 5, sorted by composite score: confidence × 0.6 + recency × 0.25 + valence × 0.15) are automatically injected into every LLM prompt as a `MY_BELIEFS:` block alongside the `IDENTITY:` header. This gives the agent a genuine, evolving worldview that colours every response without the user having to mention it.

### Proactive mode

An optional background task (**Task C** inside the daemon) fires every `proactive_interval_minutes` minutes and asks the LLM whether it has something genuinely worth sharing unprompted — a follow-up question, a reminder, or an insight. Enable it in `config/default.toml`:

```toml
[memory]
proactive_interval_minutes = 60   # 0 = disabled (default)
proactive_dnd_start_hour   = 22   # local time — end of active hours
proactive_dnd_end_hour     = 8    # local time — start of active hours
```

During the Do-Not-Disturb window the task runs silently and produces no output. When the agent decides it has something to say, it broadcasts a `BackendEvent::ProactiveMessage` that the TUI renders as a chat bubble and Telegram delivers as a normal message. The message is also persisted as an Episodic entry with `source = "proactive"` so future sleep cycles can reason about it.

**Sleep distillation** runs on a nightly schedule (configurable; default 22:00–06:00) and supports three modes:

- *Passive* — heuristic-only promotion of high-confidence episodic entries; no LLM required.
- *Agentic* — single-agent LLM reflection that learns about the user, reinforces personality, captures follow-ups, and resolves contradictions.
- *Multi-agent* — nightly pipeline of 4 parallel specialist LLM agents (Identity, Relationships, Knowledge, Reflections) followed by a deliberation/synthesis agent. Runs at most once per 22 hours. Falls back to single-agent agentic mode if the LLM is unavailable. Streaming progress events are relayed to the client while the cycle runs.

A **cooldown gate** (`proactive_cooldown_minutes`, default 5) prevents message bursts even when the check interval is short. Task C is aborted gracefully on daemon shutdown so it never fires mid-exit.

### LLM routing

- **Ollama** (local-first) — any locally installed model; configurable per session.
- **OpenRouter** — cloud provider supporting GPT-4o, Claude 3.x/3.5/3.7, Gemini 2.0, Llama 3.x, Mistral, Qwen, DeepSeek, and more.
- Both providers use separate model strings so Ollama model names are never forwarded to OpenRouter and vice versa.
- Streaming responses via server-sent events; both providers share a common `LlmClient` trait.
- Type `/fallback` in any message to force the cloud provider for that single turn.
- `ReloadConfig` over the daemon socket re-reads `.env` immediately — a newly set `OPENROUTER_API_KEY` takes effect without restarting the daemon.

### Interfaces

#### TUI (terminal)

- Full `ratatui`-based chat interface with collapsible sidebar, scrollable transcript, and live token streaming.
- Markdown rendering with `syntect` syntax highlighting for fenced code blocks; inline bold, italic, code, lists, blockquotes, and headings (H1–H3).
- Fuzzy-search file picker (`@` prefix) and slash-command palette.
- Clipboard copy, history mode, and keyboard-driven focus switching (sidebar / chat / input).
- Sleep cycle progress shown with an animated indicator while the nightly consolidation runs.

#### Telegram bot

- Long-polling bot with per-chat short-term context windows.
- All messages are routed through daemon IPC — memory and model state are shared with the TUI.
- 409-conflict backoff for multi-instance safety.
- Commands: `/help`, `/status`, `/context`, `/model show|list|set|provider|test`, `/think <level>`, `/memory stats|inspect-core|export-vault`, `/sleep`, `/correct`, `/pin`, `/forget`.

#### CLI subcommands

| Command | Description |
| --- | --- |
| `aigent` / `aigent start` | Open TUI (auto-starts daemon; force-onboards on first run) |
| `aigent onboard` | Interactive first-time setup wizard |
| `aigent configuration` / `aigent config` | Re-open wizard to update identity, model, Telegram, memory, or safety settings |
| `aigent telegram` | Run Telegram bot standalone (no TUI) |
| `aigent daemon start\|stop\|restart\|status` | Manage the background daemon process |
| `aigent memory stats` | Print memory tier counts and index/vault health |
| `aigent memory inspect-core [--limit N]` | Show top core memories |
| `aigent memory promotions [--limit N]` | Show recent sleep promotions |
| `aigent memory export-vault [--path DIR]` | Write Obsidian vault to disk |
| `aigent memory wipe [--layer <all\|core\|episodic\|...>] --yes` | Wipe one or all memory tiers |
| `aigent memory proactive check` | Force a proactive check right now (bypasses DND and interval) |
| `aigent memory proactive stats` | Show proactive mode activity (total sent, last sent, DND window) |
| `aigent tool list` | List all tools registered in the running daemon with descriptions |
| `aigent tool call <name> [key=val ...]` | Execute a named tool directly with key=value arguments |
| `aigent doctor` | Print current config and memory diagnostics |
| `aigent doctor --review-gate [--report FILE]` | Run phase review gate with auto-remediation |
| `aigent doctor --model-catalog [--provider <all\|ollama\|openrouter>]` | List available models |
| `aigent reset --hard --yes` | Stop daemon, wipe `.aigent` state, require re-onboarding |

### Tool execution

Built-in tools registered in the daemon and accessible via `/tools` slash commands and `aigent tool` CLI:

| Tool | Description |
| --- | --- |
| `read_file` | Read a file within the workspace (path-sandboxed, max-bytes limited) |
| `write_file` | Write or overwrite a file within the workspace |
| `run_shell` | Execute a shell command in the workspace directory (timeout-bounded) |
| `calendar_add_event` | Append an event to `.aigent/calendar.json` (local calendar store) |
| `web_search` | DuckDuckGo Instant Answer web search (no API key; timeout-bounded) |
| `draft_email` | Save an email draft to `.aigent/drafts/` as a plain-text file |
| `remind_me` | Add a reminder to `.aigent/reminders.json` for proactive surfacing |

All tools are governed by `ExecutionPolicy` (`allow_shell`, `allow_wasm`, `approval_required`, `tool_allowlist`, `tool_denylist`, `approval_exempt_tools`). An interactive approval channel gates dangerous actions before execution. The four data tools (`calendar_add_event`, `web_search`, `draft_email`, `remind_me`) are approval-exempt by default.

**LLM-driven tool calling**: before each streaming response, the daemon asks the LLM whether the user’s message requires a tool. If yes, the daemon executes the tool, records the result to Procedural memory, emits `ToolCallStart` / `ToolCallEnd` events, and injects the result into the main LLM prompt so the reply is grounded in the actual output.

### Daemon / IPC

The daemon exposes a Unix socket (`/tmp/aigent.sock` by default) and handles:

| Command | Description |
| --- | --- |
| `SubmitTurn` | Chat turn; streams `BackendEvent` tokens back to the caller |
| `GetStatus` | Live memory stats, uptime, provider, model, tool list |
| `GetMemoryPeek` | Most recent memory entries |
| `ExecuteTool` / `ListTools` | Tool invocation with safety gating |
| `RunSleepCycle` | Trigger single-agent agentic sleep on demand; streams progress |
| `RunMultiAgentSleepCycle` | Trigger the full 4-specialist sleep pipeline; streams progress |
| `TriggerProactive` | Force an immediate proactive check regardless of DND and interval |
| `GetProactiveStats` | Return proactive mode statistics (`ProactiveStatsPayload`) |
| `Subscribe` | Persistent broadcast connection for TUI and Telegram relay |
| `ReloadConfig` | Hot-reload `config/default.toml` and `.env` without restart |
| `Shutdown` / `Ping` | Graceful shutdown (flushes memory + final sleep pass) / health check |

**Broadcast events** emitted to all `Subscribe` connections:

| Event | Description |
| --- | --- |
| `Token` | Streamed LLM output chunk |
| `ReflectionInsight` | Free-form insight extracted by inline reflection after each turn |
| `BeliefAdded { claim, confidence }` | New belief persisted from inline reflection |
| `ProactiveMessage { content }` | Unprompted message from the proactive background task |
| `ExternalTurn { source, content }` | User message received from a non-TUI channel (e.g. Telegram) |
| `MemoryUpdated` / `Done` / `Error` | Turn lifecycle signals |

### WASM extension interface

A WIT host interface is defined in `extensions/wit/host.wit` for guest WASM skills:
- Workspace file I/O: `read-file`, `write-file`, `list-dir`
- Shell execution: `run-shell` (with timeout)
- Persistent key-value store: `kv-get`, `kv-set`
- HTTP: `http-get`, `http-post`
- Git: `git-commit`, `git-rollback-last`, `git-log-last`
- Secrets: `secret-get`

Guest skills implement `spec()` and `run(params)` using the stdin/stdout JSON protocol.

The **Wasmtime host runtime** (Cargo feature `wasm`, enabled by default) discovers compiled
`.wasm` binaries at daemon start and registers them as live tools:

```bash
# Build the guest sub-workspace
rustup target add wasm32-wasip1
cd extensions/tools-src && cargo build --release
# -> creates extensions/tools-src/<crate>/target/wasm32-wasip1/release/*.wasm
# The daemon picks them up automatically on next start
```

WASM tools **shadow** the native Rust baseline by name (`read_file`, `write_file`, etc.).
If no `.wasm` files are present, the native implementations remain active.

## Capabilities matrix

| Capability area | Status | Notes |
| --- | --- | --- |
| Onboarding wizard (TUI + prompt fallback) | ✅ Complete | Configures identity, model/provider, thinking level, workspace, sleep window, safety profile, **approval mode**, **Brave API key**. |
| Persistent memory (6 tiers) | ✅ Complete | Core, UserProfile, Reflective, Semantic, Procedural, Episodic — append-only event log. |
| Crash-safe JSONL (flush + fsync) | ✅ Complete | `append()` fsyncs every write; `overwrite()` uses tmp+rename+fsync; corrupt lines skipped on load. |
| Hybrid retrieval (lexical + embedding) | ✅ Complete | Ollama embeddings + cosine similarity; graceful fallback to lexical-only. |
| High-density relational matrix | ✅ Complete | Cross-tier association table injected into every prompt. |
| Obsidian vault projection | ✅ Complete | Incremental writes; per-tier indexes, daily notes, topic backlinks. |
| YAML KV summary files (3-tier) | ✅ Complete | `core_summary.yaml`, `user_profile.yaml`, `reflective_opinions.yaml`; checksum-based incremental. |
| MEMORY.md narrative | ✅ Complete | Human-friendly prose consolidation cross-referencing KV files. |
| KV auto-injection into every prompt | ✅ Complete | Core + UserProfile pinned at score 2.0 — agent always knows who it is. |
| Bidirectional vault watcher | ✅ Complete | `notify`-based watcher ingests human edits as `source="human-edit"` memories. |
| Redb secondary index + LRU cache | ✅ Complete | Opt-in fast tier/confidence lookup; transparent fallback to full scan. |
| Sleep distillation — passive | ✅ Complete | Heuristic promotion of high-confidence episodic entries; no LLM required. |
| Sleep distillation — agentic | ✅ Complete | Single-agent LLM reflection; learns about user, resolves contradictions. |
| Sleep distillation — multi-agent | ✅ Complete | 4 parallel specialists + deliberation/synthesis; 22h rate-limit; progress streaming. |
| Interactive TUI chat | ✅ Complete | Streaming, syntect markdown rendering, file picker, command palette. |
| Telegram bot runtime | ✅ Complete | Long-polling, per-chat context, shared daemon state. |
| Telegram typing indicator | ✅ Complete | `sendChatAction` refreshed every 4 s while the daemon processes a turn. |
| Daemon IPC server | ✅ Complete | Unix socket; streaming turns, broadcast events, memory peek, tool execution. |
| Daemon-first architecture | ✅ Complete | Daemon owns all state; TUI and Telegram are thin reconnectable clients. |
| Belief API | ✅ Complete | `record_belief` / `retract_belief` / `all_beliefs`; stored as tagged Core entries. |
| Inline reflection | ✅ Complete | Structured LLM call after every turn extracts beliefs + insights; streamed as events. |
| Belief injection into prompts | ✅ Complete | Up to 10 active beliefs injected as `MY_BELIEFS:` block on every turn. |
| Proactive mode (Task C) | ✅ Complete | Background task checks for proactive messages; respects DND window and configurable interval. Cooldown gate and graceful shutdown on daemon exit. |
| Tool execution system | ✅ Complete | `read_file`, `write_file`, `run_shell`, `calendar_add_event`, `web_search`, `draft_email`, `remind_me`, `git_rollback`; workspace-sandboxed, per-tool allow/deny. |
| LLM-driven tool calling | ✅ Complete | Pre-turn tool intent check via structured LLM prompt; executes tool, records to Procedural memory, injects result before streaming response. |
| Tool approval modes | ✅ Complete | Three modes: `safer` (always ask), `balanced` (read-only free, default), `autonomous` (no prompts). Configurable via `[tools] approval_mode`. |
| Git auto-commit | ✅ Complete | Automatic `git add -A && git commit` after every `write_file`/`run_shell`; `git_rollback` tool reverts last commit. |
| Brave Search integration | ✅ Complete | `web_search` uses Brave API when `brave_api_key` is set; falls back to DuckDuckGo. |
| WASM extension interface | ✅ Complete | WIT host API + Wasmtime host runtime (`wasm` feature). Guest `.wasm` binaries shadow native tools; built-in tools remain as fallback when no guests are compiled. |
| Platform sandboxing | ✅ Complete | `sandbox` feature: `PR_SET_NO_NEW_PRIVS` + seccomp BPF allow-list (x86-64 Linux); `sandbox_init` profile (macOS). Applied in child process before shell `exec`. |
| Memory CLI commands | ✅ Complete | `stats` (incl. tool exec counts), `inspect-core`, `promotions`, `export-vault`, `wipe`, `proactive check/stats`. |
| Tool CLI commands | ✅ Complete | `aigent tool list`, `aigent tool call <name> [key=value ...]`. |
| Telegram command parity | ✅ Complete | Core commands available; all memory, proactive, and tool events routed correctly. |
| Phase review gate | 🟨 In progress | `doctor --review-gate` implemented; some checks in progress. |
| Systemd/launchd unit | ⏳ Planned | Daemon auto-start on system boot. |
| Full channel parity smoke coverage | ⏳ Planned | Behaviour and memory consistency verified across TUI and Telegram. |

## Usage

### Starting the daemon

```bash
aigent daemon start          # background daemon
aigent daemon status         # check it is running
aigent start                 # foreground TUI (also starts daemon if needed)
```

### Chat (TUI)

```bash
aigent start
```

Key bindings inside the TUI:

| Key | Action |
| --- | --- |
| `Tab` / `Shift-Tab` | Cycle focus: input → chat → sidebar |
| `Enter` | Send message |
| `↑` / `↓` | Scroll transcript or navigate sidebar |
| `@<filename>` | Open fuzzy file picker |
| `/help` | Show slash-command palette |
| `/fallback` | Force cloud (OpenRouter) provider for this turn |
| `/sleep` | Trigger agentic sleep cycle now |
| `/model provider openrouter` | Switch provider |
| `/model set <name>` | Switch model |
| `Ctrl-C` | Quit (daemon keeps running) |

### Telegram bot

```bash
aigent telegram              # run bot in foreground
# or via daemon:
aigent daemon start          # bot starts automatically when telegram_enabled = true
```

### Memory commands

```bash
aigent memory stats
aigent memory inspect-core --limit 30
aigent memory promotions --limit 20
aigent memory export-vault --path ~/Documents/aigent-vault

# Proactive mode
aigent memory proactive stats    # show activity (total sent, last sent, DND window)
aigent memory proactive check    # force a check right now (bypasses DND and interval)
```

### Configuration

```bash
aigent configuration         # interactive wizard
aigent doctor                # show current config + diagnostics
aigent doctor --model-catalog --provider openrouter
```

### Daemon lifecycle

```bash
aigent daemon start
aigent daemon stop
aigent daemon restart
aigent daemon status
```

Runtime files:
- PID: `.aigent/runtime/daemon.pid`
- Log: `.aigent/runtime/daemon.log`
- Socket: `/tmp/aigent.sock` (configurable via `daemon.socket_path` in `config/default.toml`)

## Safety & Trust Model

Aigent is designed to run on your main machine with access to real files and
network.  Three interlocking layers keep it safe:

### 1 — Workspace sandbox

Every file read and write is verified to be within `agent.workspace_path`
before the syscall executes.  Path traversal (e.g. `../../etc/passwd`) is
rejected with an error.  Shell commands run with the workspace as the current
directory.

### 2 — Approval modes

Configured via `[tools] approval_mode` in `config/default.toml`:

| Mode | Read-only tools | Write / shell | Default? |
|------|-----------------|---------------|----------|
| `safer` | ❓ Requires approval | ❓ Requires approval | No |
| `balanced` | ✅ Auto-approved | ❓ Requires approval | **Yes** |
| `autonomous` | ✅ Auto-approved | ✅ Auto-approved | No |

**Read-only tools** (never affect the filesystem): `read_file`, `web_search`,
`calendar_add_event`, `remind_me`, `git_rollback`.

**Write / shell tools** (can modify the workspace): `write_file`, `run_shell`,
`draft_email`.

You can override behaviour per-tool with `tool_allowlist` / `tool_denylist` in
`[safety]` and `approval_exempt_tools` for tools that should always be
auto-approved regardless of mode.

### 3 — Git rollback

When `[tools] git_auto_commit = true`, every `write_file` and `run_shell` call
that succeeds is immediately committed to the workspace git repository with
the message `Aigent tool: <name> — <detail>`.  If the agent makes a mistake
you can:

```bash
# Via CLI:
aigent tool call git_rollback

# Or directly:
git -C <workspace> revert HEAD
```

During onboarding the wizard will `git init` the workspace if it is not already
a git repository.

### API keys & secrets

External service keys (e.g. the Brave Search API key) live in
`[tools] brave_api_key` in your config file, **or** in the `BRAVE_API_KEY`
environmental variable (env takes precedence).  The agent will also look for
keys in `<workspace>/.secrets/<name>` when the `secret-get` WASM host function
is called.

Never commit the `.secrets/` directory to version control — add it to
`.gitignore` if your workspace is a git repository.

### Future: platform sandboxing (now `sandbox` feature)

Enabled by building with `--features sandbox` in the `aigent-exec` crate:

```toml
# Cargo.toml (aigent-exec)
[features]
sandbox = ["dep:libc"]
```

| Platform | Mechanism | Scope |
|----------|-----------|-------|
| Linux x86-64 | `PR_SET_NO_NEW_PRIVS` + seccomp BPF allow-list | Shell child process (after `fork`, before `exec`) |
| macOS | `sandbox_init(3)` Scheme profile | Shell child process |
| Other | No-op (workpace sandbox still active) | — |

The seccomp filter allows ≈80 syscalls covering file I/O, networking, memory,
process management, and time.  All other syscalls return `ENOSYS` (graceful
failure) rather than `SIGSYS` (kill), so the child process degrades cleanly.

The workspace isolation and approval-mode layers remain active on all
platforms regardless of whether the `sandbox` feature is compiled in.

## Development

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

`clippy.toml` and `deny.toml` enforce additional lint and dependency audit rules.

## Phase review gate

```bash
aigent doctor --review-gate --report docs/phase-review-gate.md
```

Validates Phase 0–2 readiness checks, auto-remediates fixable items, and exits non-zero on hard failures.
