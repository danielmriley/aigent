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

**Sleep distillation** runs on a nightly schedule (configurable; default 22:00–06:00) and supports three modes:

- *Passive* — heuristic-only promotion of high-confidence episodic entries; no LLM required.
- *Agentic* — single-agent LLM reflection that learns about the user, reinforces personality, captures follow-ups, and resolves contradictions.
- *Multi-agent* — nightly pipeline of 4 parallel specialist LLM agents (Identity, Relationships, Knowledge, Reflections) followed by a deliberation/synthesis agent. Runs at most once per 22 hours. Falls back to single-agent agentic mode if the LLM is unavailable. Streaming progress events are relayed to the client while the cycle runs.

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
| `aigent memory stats` | Print memory tier counts |
| `aigent memory inspect-core [--limit N]` | Show top core memories |
| `aigent memory promotions [--limit N]` | Show recent sleep promotions |
| `aigent memory export-vault [--path DIR]` | Write Obsidian vault to disk |
| `aigent memory wipe [--layer <all\|core\|episodic\|...>] --yes` | Wipe one or all memory tiers |
| `aigent doctor` | Print current config and memory diagnostics |
| `aigent doctor --review-gate [--report FILE]` | Run phase review gate with auto-remediation |
| `aigent doctor --model-catalog [--provider <all\|ollama\|openrouter>]` | List available models |
| `aigent reset --hard --yes` | Stop daemon, wipe `.aigent` state, require re-onboarding |

### Tool execution

Built-in tools registered in the daemon and accessible via `/tools` slash commands:

| Tool | Description |
| --- | --- |
| `read_file` | Read a file within the workspace (path-sandboxed, max-bytes limited) |
| `write_file` | Write or overwrite a file within the workspace |
| `run_shell` | Execute a shell command in the workspace directory (timeout-bounded) |

All tools are governed by `ExecutionPolicy` (`allow_shell`, `allow_wasm`, `approval_required`). An interactive approval channel gates dangerous actions before execution.

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
| `Subscribe` | Persistent broadcast connection for TUI and Telegram relay |
| `ReloadConfig` | Hot-reload `config/default.toml` and `.env` without restart |
| `Shutdown` / `Ping` | Graceful shutdown (flushes memory + final sleep pass) / health check |

### WASM extension interface

A WIT host interface is defined in `extensions/wit/host.wit` for guest WASM skills:
- Workspace file I/O: `read-file`, `write-file`, `list-dir`
- Shell execution: `run-shell` (with timeout)
- Persistent key-value store: `kv-get`, `kv-set`
- HTTP: `http-get`, `http-post`

Guest skills implement `spec()` and `run(params)`.

## Capabilities matrix

| Capability area | Status | Notes |
| --- | --- | --- |
| Onboarding wizard (TUI + prompt fallback) | ✅ Complete | Configures identity, model/provider, thinking level, workspace, sleep window, safety profile. |
| Persistent memory (6 tiers) | ✅ Complete | Core, UserProfile, Reflective, Semantic, Procedural, Episodic — append-only event log. |
| Hybrid retrieval (lexical + embedding) | ✅ Complete | Ollama embeddings + cosine similarity; graceful fallback to lexical-only. |
| High-density relational matrix | ✅ Complete | Cross-tier association table injected into every prompt. |
| Obsidian vault projection | ✅ Complete | Incremental writes; per-tier indexes, daily notes, topic backlinks. |
| Sleep distillation — passive | ✅ Complete | Heuristic promotion of high-confidence episodic entries; no LLM required. |
| Sleep distillation — agentic | ✅ Complete | Single-agent LLM reflection; learns about user, resolves contradictions. |
| Sleep distillation — multi-agent | ✅ Complete | 4 parallel specialists + deliberation/synthesis; 22h rate-limit; progress streaming. |
| Interactive TUI chat | ✅ Complete | Streaming, syntect markdown rendering, file picker, command palette. |
| Telegram bot runtime | ✅ Complete | Long-polling, per-chat context, shared daemon state. |
| Daemon IPC server | ✅ Complete | Unix socket; streaming turns, broadcast events, memory peek, tool execution. |
| Daemon-first architecture | ✅ Complete | Daemon owns all state; TUI and Telegram are thin reconnectable clients. |
| Tool execution system | ✅ Complete | `read_file`, `write_file`, `run_shell`; workspace-sandboxed, approval-gated. |
| WASM extension interface | ✅ Complete | WIT host API for guest skills (file I/O, shell, KV, HTTP). |
| Memory CLI commands | ✅ Complete | `stats`, `inspect-core`, `promotions`, `export-vault`, `wipe`. |
| Telegram command parity | 🟨 In progress | Core commands available; advanced memory and tool actions pending. |
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
