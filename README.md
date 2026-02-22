# Aigent

Aigent is a persistent, self-improving Rust AI agent with local-first memory, secure execution boundaries, and dual LLM routing.

## Status

Phase 0 (Foundation) and Phase 1 (Memory) are complete. Phase 2 (Unified Agent Loop) is in progress. A daemon-first architecture refactor is planned for Phase 3.

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

Retrieval uses a weighted blend of core priority, recency, semantic relevance, and confidence. Core and UserProfile entries are always included in every prompt context.

An **Obsidian-compatible vault** is auto-projected under `.aigent/vault/` with per-tier indexes, daily memory notes, topic backlinks, and wiki-style links.

**Sleep distillation** runs on a nightly schedule (22:00–06:00 by default) and supports two modes:
- *Passive* — heuristic-only promotion of high-confidence episodic entries; no LLM required.
- *Agentic* — LLM-driven nightly reflection that learns about the user, reinforces personality, captures follow-ups, and resolves contradictions.

### LLM routing

- **Ollama** (local-first) — configurable model; falls back to OpenRouter on failure.
- **OpenRouter** — cloud fallback with support for GPT-4o, Claude, Gemini, Llama, Mistral, Qwen, DeepSeek, and others.
- Streaming responses via `chat_stream`; both providers share a common `LlmClient` trait.
- `/fallback` in any message forces the cloud provider for a single turn.

### Interfaces

#### TUI (terminal)
- Full ratatui-based chat interface with collapsible sidebar, scrollable transcript, and live streaming.
- Markdown rendering with **syntect syntax highlighting** for fenced code blocks; inline bold, italic, code, lists, blockquotes, and headings (H1–H3).
- Fuzzy-search file picker (`@` prefix) and slash-command palette.
- Clipboard copy, history mode, and keyboard-driven focus switching (sidebar / chat / input).

#### Telegram bot
- Long-polling bot with per-chat short-term context windows.
- Routes all messages through daemon IPC so memory and model state are shared with the TUI.
- 409-conflict backoff for multi-instance safety.
- Commands: `/help`, `/status`, `/context`, `/model show|list|set|provider|test`, `/think <level>`, `/memory stats|inspect-core|export-vault`, `/sleep`, `/correct`, `/pin`, `/forget`.

#### CLI subcommands

| Command | Description |
| --- | --- |
| `aigent` / `aigent start` | Open TUI (auto-starts daemon; force-onboards on first run) |
| `aigent onboard` | Interactive first-time setup wizard (TUI or prompt fallback) |
| `aigent configuration` | Re-open wizard to update identity, model, Telegram, memory, or safety settings |
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
| `read_file` | Read a file within the workspace (path-escaped, max-bytes limited) |
| `write_file` | Write or overwrite a file within the workspace |
| `run_shell` | Execute a shell command in the workspace directory (timeout-bounded) |

All tools are governed by `ExecutionPolicy` (`allow_shell`, `allow_wasm`, `approval_required`). An interactive approval channel gates dangerous actions before execution. Use `/tools run <name> {"key":"value"}` from the TUI to invoke a tool directly.

### Daemon / IPC

The daemon exposes a Unix socket (`/tmp/aigent.sock` by default) and handles:
- `SubmitTurn` — chat turns, streaming response tokens back as `BackendEvent`s.
- `GetStatus` — live memory stats, uptime, provider, model, and tool list.
- `GetMemoryPeek` — recent memory entries.
- `ExecuteTool` / `ListTools` — tool invocation with safety gating.
- `RunSleepCycle` — trigger agentic sleep consolidation on demand.
- `Subscribe` — persistent broadcast connection used by TUI and Telegram for live event relay.
- `ReloadConfig`, `Shutdown`, `Ping`.

On graceful shutdown the daemon flushes all memory and runs a final agentic sleep pass.

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
| Memory contract defaults | ✅ Complete | Onboarding enforces `eventlog` backend and nightly sleep defaults. |
| Persistent memory (6 tiers) | ✅ Complete | Core, UserProfile, Reflective, Semantic, Procedural, Episodic — append-only event log. |
| Memory CLI commands | ✅ Complete | `memory stats`, `memory inspect-core`, `memory promotions`, `memory export-vault`, `memory wipe`. |
| Obsidian vault projection | ✅ Complete | Auto-syncs to `.aigent/vault/` with tier indexes, daily notes, topic backlinks. |
| Weighted retrieval | ✅ Complete | Core priority + recency + relevance + confidence with full provenance context. |
| Sleep distillation (passive) | ✅ Complete | Heuristic promotion of high-confidence episodic entries; no LLM required. |
| Sleep distillation (agentic) | ✅ Complete | LLM-driven nightly reflection; learns about user, reinforces personality, resolves contradictions. |
| Interactive TUI chat | ✅ Complete | Scrollable transcript, syntect markdown rendering, file picker, command palette. |
| Telegram bot runtime | ✅ Complete | Long-polling bot, per-chat context, shared daemon memory and model. |
| Daemon IPC server | ✅ Complete | Unix socket; streaming turns, broadcast events, status, memory peek, tool execution. |
| Tool execution system | ✅ Complete | `read_file`, `write_file`, `run_shell`; workspace-bounded, safety-gated with approval flow. |
| WASM extension interface | ✅ Complete | WIT host API for guest skills (file I/O, shell, KV, HTTP). |
| Telegram command parity | 🟨 In progress | Core commands available; advanced memory and tool actions pending full parity. |
| Unified channel command engine | 🟨 In progress | TUI and Telegram share runtime but command handling has duplicated code paths. |
| Phase review gate | 🟨 In progress | `doctor --review-gate` implemented; some checks currently failing (see `docs/phase-review-gate.md`). |
| Daemon-first architecture | ⏳ Planned | Daemon owns Telegram task; TUI becomes a thin reconnectable client. See `docs/daemon-first-architecture.md`. |
| Systemd/launchd unit | ⏳ Planned | Daemon auto-start on system boot. |
| Full Phase 2 parity smoke coverage | ⏳ Planned | Behavior and memory consistency verified between TUI and Telegram channels. |

## Development

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## Setup and configuration

```bash
aigent onboard
aigent configuration
```

- `aigent onboard` runs first-time setup and marks onboarding as completed.
- `aigent configuration` re-opens the wizard to update bot name, provider/model, safety, sleep policy, and Telegram integration.
- `aigent start` and `aigent telegram` force onboarding when setup is not yet complete.

## Telegram + daemon lifecycle

Configure Telegram in the wizard:

```bash
aigent configuration
```

This sets `integrations.telegram_enabled = true` and stores `TELEGRAM_BOT_TOKEN` in `.env`.

Run the bot service in the background:

```bash
aigent daemon start
aigent daemon status
aigent daemon restart
aigent daemon stop
```

Run in foreground with local TUI:

```bash
aigent start
```

Runtime files:
- PID: `.aigent/runtime/daemon.pid`
- Log: `.aigent/runtime/daemon.log`
- Socket: `/tmp/aigent.sock` (configurable via `daemon.socket_path`)

## Phase review gate

Before advancing phases, run:

```bash
aigent doctor --review-gate --report docs/phase-review-gate.md
```

The command validates Phase 0–2 readiness checks and exits non-zero if any hard-fail check fails.
It auto-remediates fixable items (memory backend, nightly mode, vault projection, sleep marker)
before evaluating hard-fail checks.

## Install (local)

```bash
./install.sh
```

- Uses `cargo build --release --locked` for deterministic dependency resolution.
- Installs atomically to `${AIGENT_INSTALL_DIR:-$HOME/.local/bin}/aigent`.
- Prints a SHA-256 checksum for the installed binary when checksum tooling is available.

## Release artifacts

GitHub release workflow artifacts include:
- Platform binary (`aigent` or `aigent.exe`)
- Matching `aigent.sha256`

Verify with:

```bash
sha256sum -c aigent.sha256
```
