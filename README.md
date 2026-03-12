# Aigent

A persistent, self-improving AI agent written in Rust. Aigent runs as a long-lived background daemon with thin frontends (terminal TUI and Telegram bot) communicating over Unix socket IPC. All state lives in a 6-tier append-only event-sourced memory system that grows smarter over time through nightly sleep consolidation — a pipeline of parallel LLM specialist agents that distil, promote, and reconcile knowledge while you sleep.

## Highlights

- **6-tier memory** — Core, UserProfile, Reflective, Semantic, Procedural, Episodic — stored in an append-only JSONL event log with crash-safe `fsync` writes and a `redb` secondary index.
- **Nightly self-improvement** — multi-agent sleep cycle with four parallel specialists (Identity, Relationships, Knowledge, Reflections) + a synthesis agent. Also supports passive (no-LLM) and single-agent modes.
- **35+ tools** — filesystem, coreutils, shell, web search (6 providers), git (libgit2), scheduler, and more — all workspace-sandboxed with configurable approval modes.
- **WASM-first extensibility** — Wasmtime runtime with a WIT host interface. WASM guest binaries take precedence over native fallbacks.
- **Platform sandboxing** — seccomp BPF (Linux) and `sandbox_init` (macOS) applied to all shell children by default.
- **Hybrid LLM routing** — Ollama (local-first), OpenRouter (cloud), and Candle (local GGUF inference). A fast router model classifies messages as Chat vs Tools to skip the full reasoning loop on simple queries.
- **Obsidian integration** — auto-projection into `.aigent/vault/` with YAML summaries, daily notes, and bidirectional editing. Edit the YAML, shape the agent.
- **Beliefs & inline reflection** — after every turn the agent extracts beliefs and insights, injecting an evolving worldview into every future prompt.
- **Rich TUI** — `ratatui`-based chat with streaming tokens, syntax-highlighted markdown, file picker, slash commands, configurable themes, and expandable tool output.
- **Telegram bot** — long-polling with typing indicators, tool footnotes, and full command parity.

## Quick Start

### Prerequisites

- [Rust](https://rustup.rs) stable toolchain (pinned in `rust-toolchain.toml`; `rustup` handles it automatically)
- [Ollama](https://ollama.com) — optional; required only when using the local Ollama provider

### Build from source

```bash
git clone https://github.com/danielmriley/aigent
cd aigent
cargo build --release --locked
```

The binary is produced at `target/release/aigent-app`.

### One-line install

```bash
./install.sh
```

Builds with `--release --locked`, installs atomically to `~/.local/bin`, prints a SHA-256 checksum, and restarts a running daemon if one is detected. Supports `--candle`, `--cuda`, `--metal`, `--prefix DIR`, and `--uninstall`.

### First run

```bash
# Interactive setup wizard (recommended)
aigent onboard

# Or copy the example config and start manually
cp config/default.toml.example config/default.toml
aigent start
```

### Environment variables

Set in `.env` (the wizard writes these automatically) or export directly:

| Variable | Purpose |
| --- | --- |
| `OPENROUTER_API_KEY` | Required for the OpenRouter cloud provider |
| `OLLAMA_BASE_URL` | Override Ollama endpoint (default `http://localhost:11434`) |
| `TELEGRAM_BOT_TOKEN` | Required to enable the Telegram bot frontend |

---

## Architecture

```
aigent-config
 └─ aigent-tools          Tool registry, ToolSpec, 35+ built-ins
     └─ aigent-llm        LlmClient trait, Ollama/OpenRouter/Candle, router
         └─ aigent-exec   ToolExecutor, sandbox, WASM loader, gait (git)
             └─ aigent-memory  MemoryManager, 6-tier event log, MemoryIndex
                 └─ aigent-thinker  ReAct loop, JsonStreamBuffer
                     └─ aigent-prompt  Prompt assembly, memory injection
                         └─ aigent-agent  AgentRuntime, run_agent_turn, sleep
                             └─ aigent-runtime  Daemon, Unix socket IPC
                                 ├─ aigent-app     CLI binary (single entry point)
                                 ├─ aigent-ui      Ratatui TUI
                                 └─ aigent-telegram Telegram bot
```

The daemon (`aigent-runtime`) owns all mutable state behind `Arc<Mutex<DaemonState>>`. Frontends are thin, reconnectable clients over a Unix socket at `/tmp/aigent.sock`. A `broadcast::Sender<BackendEvent>` fans out events (tokens, beliefs, tool results, proactive messages) to all subscribers.

---

## Memory System

### 6-tier hierarchy

| Tier | Purpose |
| --- | --- |
| **Core** | Identity, constitution, personality — consistency-firewalled; rewrite requires approval |
| **UserProfile** | Persistent user facts: preferences, goals, life context |
| **Reflective** | Agent thoughts, plans, self-critiques, proactive follow-ups |
| **Semantic** | Distilled facts and condensed knowledge promoted from episodic memory |
| **Procedural** | Learned skills, workflows, how-to knowledge |
| **Episodic** | Raw conversation turns and temporary observations |

### Hybrid retrieval

Every query is scored with weighted components:

```
score = tier(0.35) + recency(0.20) + lexical(0.25) + embedding(0.15) + confidence(0.05)
```

When Ollama is available, entries are embedded at record time and cosine similarity upgrades lexical matching to semantic vector search. Without an embedding backend the weight is redistributed across the other components. Core and UserProfile entries are always injected into the prompt regardless of score.

A **relational matrix** — a compact cross-tier association table — is injected into every prompt to surface connections between memory entries.

### Storage guarantees

- **Crash-safe append** — every write to `events.jsonl` is followed by `flush()` + `sync_all()`. Overwrites (wipe, compact, Core retirements) use atomic `tmp → fsync → rename`.
- **Resilient loading** — corrupt JSONL lines are skipped and saved to a `.corrupt` sidecar. A single bad line never takes down the daemon.
- **Redb secondary index** — optional `redb`-backed index with an LRU-256 cache for O(log n) tier lookups. Rebuilt transparently if absent or corrupt.

### Obsidian vault projection

The daemon projects memory into `.aigent/vault/` as Obsidian-compatible files:

| File | Content |
| --- | --- |
| `core_summary.yaml` | Top-15 Core entries by confidence |
| `user_profile.yaml` | Top-15 UserProfile entries |
| `reflective_opinions.yaml` | Top-15 Reflective entries |
| `MEMORY.md` | Human-friendly prose rollup |

Each YAML file includes a `checksum: sha256:…` field and `last_updated` timestamp. Files are written incrementally — unchanged files are not touched.

A background `notify`-based watcher detects human edits and ingests them as `source = "human-edit"` entries in the appropriate tier. The next sleep cycle reconciles the edit. **Edit the YAML, shape the soul.**

### Beliefs & inline reflection

After every conversation turn, a structured LLM call extracts up to three **beliefs** and two **reflective insights**. Beliefs are stored in Core memory; insights in the Reflective tier. Both are streamed to all frontends in real time.

Active beliefs (ranked by confidence × 0.6 + recency × 0.25 + valence × 0.15) are injected into every prompt as a `MY_BELIEFS:` block, giving the agent a genuine, evolving worldview.

---

## Sleep & Self-Improvement

Three consolidation modes, all append-only (results are new entries, never mutations):

| Mode | Description |
| --- | --- |
| **Passive** | Heuristic promotion of high-confidence episodic entries. No LLM required. |
| **Agentic** | Single-agent LLM reflection — learns about the user, reinforces personality, resolves contradictions. |
| **Multi-agent** | 4 parallel specialist LLM agents (Identity, Relationships, Knowledge, Reflections) + a deliberation/synthesis agent. Rate-limited to once per 22 hours. Falls back to agentic mode if the LLM is unavailable. Progress is streamed to connected clients. |

Sleep runs on a nightly schedule (default 22:00–06:00, configurable) or on demand via `aigent sleep run` or the `/sleep` slash command.

### Sleep seeding

Inject synthetic episodic memories to teach the agent specific themes before a sleep cycle:

```bash
aigent sleep seed "Rust async patterns" --count 7 --valence positive --run
```

---

## Proactive Mode

An optional background task fires every `proactive_interval_minutes` and asks the LLM whether it has something worth sharing unprompted — a follow-up, a reminder, or an insight.

```toml
[memory]
proactive_interval_minutes = 60   # 0 = disabled (default)
proactive_dnd_start_hour   = 22
proactive_dnd_end_hour     = 8
```

Messages are broadcast as `ProactiveMessage` events, rendered in the TUI and delivered via Telegram. A cooldown gate prevents bursts.

---

## LLM Providers

| Provider | Description |
| --- | --- |
| **Ollama** | Local-first; any installed model. Default provider. |
| **OpenRouter** | Cloud gateway to GPT-4o, Claude, Gemini, Llama, Mistral, Qwen, DeepSeek, and more. |
| **Candle** | Local GGUF inference via the Candle framework (CPU, CUDA, or Metal). Opt-in feature flag. |

Both Ollama and OpenRouter use separate model strings; names are never cross-forwarded. Streaming via server-sent events. Type `/fallback` in any message to force the cloud provider for that turn. `ReloadConfig` over the socket hot-reloads `.env` without a restart.

### Router (fast-path classification)

A small, fast model (e.g. 0.8B parameters, ~150ms) classifies each incoming message as **Chat** or **Tools**:

- **Chat** → direct response from the primary model, skipping the full reasoning loop.
- **Tools** → triggers the external thinking loop (ReAct-style) with tool specs and structured JSON output.

Configure in `[router]` with a separate `ollama_model` or `openrouter_model`.

---

## Tools

### Built-in tools (35+)

| Category | Tools |
| --- | --- |
| **Filesystem** | `read_file`, `write_file`, `list_dir`, `mkdir`, `touch`, `rm`, `cp`, `mv`, `find`, `tree` |
| **Coreutils** | `ls`, `grep`, `head`, `tail`, `wc`, `sort`, `uniq`, `cut`, `sed`, `echo`, `seq`, `workspace_status` |
| **Shell** | `run_shell` (timeout-bounded, seccomp/sandbox_init) |
| **Web & search** | `web_search` (Brave → Tavily → Serper → Exa → SearXNG → DuckDuckGo), `browse_page` |
| **Git** | `perform_gait` (libgit2: status, log, diff, commit, checkout, branch, reset, clone, pull, push, fetch, blame, tag, stash), `git_rollback` |
| **Scheduler** | `create_cron_job`, `remove_cron_job`, `list_cron_jobs` (6-field cron expressions) |
| **Data** | `calendar_add_event`, `remind_me`, `draft_email`, `get_current_datetime` |
| **Extensions** | `list_modules` |

All coreutils support `jsonl=true` and `semantic=true` output modes for structured LLM consumption.

### WASM extensions

The Wasmtime host runtime (`wasm` feature, default-on) discovers compiled `.wasm` binaries at startup. WASM tools are registered first (first-match wins); native Rust implementations fill gaps until guests are built.

```bash
aigent tools build    # compile guests (wasm32-wasip1)
aigent tools status   # show WASM vs native per tool + sandbox state
aigent tools reload   # hot-reload without restarting the daemon
```

New tools via scaffold:

```bash
cd extensions/tools-src
./new-tool.sh my_tool "One-sentence description"
./build.sh my_tool
```

The WIT host interface (`extensions/wit/host.wit`) provides workspace file I/O, shell execution, key-value store, HTTP, git operations, and secret access.

### Tool execution pipeline

1. **Router** classifies message → Tools path
2. **External thinking loop** (ReAct, up to `max_tool_rounds` iterations) parses structured JSON from the LLM stream
3. **ExecutionPolicy** enforces allowlist/denylist, approval mode, security level, and rate limits
4. **Sandbox** applies platform-level restrictions before `exec`
5. Tool result is recorded to memory and injected back into the prompt

---

## Interfaces

### TUI

```bash
aigent start    # auto-starts daemon if needed
```

- `ratatui`-based chat with collapsible sidebar, scrollable transcript, and live token streaming
- Markdown rendering with `syntect` syntax highlighting for code blocks
- Fuzzy-search file picker (`@` prefix) and slash-command palette
- Expandable tool output — press Enter on a tool message in history mode
- Animated braille spinner throughout LLM generation, tool execution, and reflection
- Chat persistence — turns saved to `.aigent/history/YYYY-MM-DD.jsonl`, last 200 restored on open
- Configurable themes: Catppuccin Mocha, Tokyo Night, Nord (cycle with Ctrl+T)
- Context-aware keybindings bar

| Key | Action |
| --- | --- |
| `Tab` / `Shift-Tab` | Cycle focus: input → chat → sidebar |
| `Enter` | Send message (or expand tool output in history mode) |
| `↑` / `↓` | Scroll transcript or sidebar |
| `@<filename>` | Fuzzy file picker |
| `/help` | Slash command palette |
| `/fallback` | Force cloud provider for this turn |
| `/sleep` | Trigger sleep cycle |
| `Ctrl-T` | Cycle theme |
| `Ctrl-C` | Quit (daemon keeps running) |

### Telegram

```bash
aigent telegram         # standalone
aigent daemon start     # auto-starts when telegram_enabled = true
```

- Long-polling with per-chat context windows
- Typing indicator refreshed every 4s during turns
- Tool footnote appended to replies when tools are used
- Commands: `/help`, `/status`, `/context`, `/model show|list|set|provider|test`, `/think`, `/memory stats|inspect-core|export-vault`, `/sleep`, `/correct`, `/pin`, `/forget`

### CLI

| Command | Description |
| --- | --- |
| `aigent start` | Open TUI (auto-starts daemon) |
| `aigent onboard` | Interactive setup wizard |
| `aigent config` | Re-open config wizard |
| `aigent telegram` | Run Telegram bot standalone |
| `aigent daemon start\|stop\|restart\|status` | Manage the daemon |
| `aigent memory stats` | Tier counts, index health, vault status |
| `aigent memory inspect-core [--limit N]` | Show top Core memories |
| `aigent memory promotions [--limit N]` | Recent sleep promotions |
| `aigent memory beliefs [--kind K] [--limit N]` | Browse beliefs |
| `aigent memory export-vault [--path DIR]` | Write Obsidian vault |
| `aigent memory wipe [--layer TIER] --yes` | Wipe one or all tiers |
| `aigent memory proactive check\|stats` | Proactive mode controls |
| `aigent tool list` | Registered tools with descriptions |
| `aigent tool call <name> [key=val ...]` | Execute a tool directly |
| `aigent tools build\|status\|reload` | WASM guest management |
| `aigent sleep run\|status\|seed` | Sleep cycle controls |
| `aigent history clear\|export\|path` | TUI chat history |
| `aigent doctor [--review-gate] [--model-catalog]` | Config diagnostics |
| `aigent reset --hard --yes` | Full state wipe |

---

## Daemon IPC

The daemon exposes a Unix socket and handles these commands:

| Command | Description |
| --- | --- |
| `SubmitTurn` | Chat turn; streams `BackendEvent` tokens |
| `GetStatus` | Memory stats, uptime, provider, model, tool list |
| `ExecuteTool` / `ListTools` | Tool invocation with safety gating |
| `RunSleepCycle` / `RunMultiAgentSleepCycle` | On-demand sleep with progress streaming |
| `TriggerProactive` / `GetProactiveStats` | Proactive mode controls |
| `Subscribe` | Persistent broadcast connection for frontends |
| `ReloadConfig` | Hot-reload config and `.env` without restart |
| `Shutdown` / `Ping` | Graceful shutdown (flushes memory) / health check |

Broadcast events: `Token`, `ReflectionInsight`, `BeliefAdded`, `ProactiveMessage`, `ExternalTurn`, `ToolCall`, `ToolResult`, `MemoryUpdated`, `Done`, `Error`.

---

## Safety & Trust

Five interlocking layers protect your system:

### 1. Workspace sandbox

Every file read/write is verified to be within `agent.workspace_path`. Path traversal (e.g. `../../etc/passwd`) is rejected. Shell commands run with the workspace as cwd.

### 2. Approval modes

| Mode | Read-only tools | Write / shell | Default? |
| --- | --- | --- | --- |
| `safer` | Requires approval | Requires approval | No |
| `balanced` | Auto-approved | Requires approval | **Yes** |
| `autonomous` | Auto-approved | Auto-approved | No |

Per-tool overrides via `tool_allowlist`, `tool_denylist`, and `approval_exempt_tools` in `[safety]`.

### 3. Git rollback

When `git_auto_commit = true`, every write/shell success is immediately committed. Revert with `aigent tool call git_rollback` or `git revert HEAD`.

### 4. Platform sandboxing (default-on)

| Platform | Mechanism |
| --- | --- |
| Linux x86-64 | `PR_SET_NO_NEW_PRIVS` + seccomp BPF allow-list (~80 syscalls) |
| macOS | `sandbox_init(3)` profile |
| Other | No-op; workspace isolation and approval modes still active |

Disable at runtime: `[tools] sandbox_enabled = false`. Inspect with `aigent tools status`.

### 5. gait — safe native git

`perform_gait` uses libgit2 in-process. Write operations (commit, push, checkout, etc.) require the path to be inside `trusted_write_paths`. Read operations are broadly allowed when `allow_system_read = true`.

```toml
[git]
trusted_repos       = ["https://github.com/danielmriley/aigent"]
trusted_write_paths = []           # workspace + self-repo auto-added
allow_system_read   = true
```

### API keys & secrets

Keys live in config (`[tools] brave_api_key`, etc.), environment variables (take precedence), or `<workspace>/.secrets/<name>` (accessed via the `secret-get` WASM host function). Never commit `.secrets/` to version control.

---

## Configuration

All settings live in `config/default.toml`. Run the interactive wizard with `aigent config` or edit the file directly. Hot-reload without restart via `ReloadConfig` over the socket.

Key sections: `[agent]`, `[llm]`, `[memory]`, `[safety]`, `[tools]`, `[git]`, `[router]`, `[inference]`, `[ui]`, `[daemon]`, `[debug]`.

See [`config/default.toml.example`](config/default.toml.example) for the full reference with comments.

### Runtime files

| Path | Purpose |
| --- | --- |
| `.aigent/memory/events.jsonl` | Canonical append-only event log |
| `.aigent/memory/index.redb` | Redb secondary index (rebuilt if missing) |
| `.aigent/history/YYYY-MM-DD.jsonl` | TUI chat history |
| `.aigent/vault/` | Obsidian-compatible projection |
| `.aigent/runtime/daemon.pid` / `.log` | Daemon lifecycle files |
| `/tmp/aigent.sock` | Unix socket (configurable) |

---

## Feature Flags

| Flag | Crate(s) | Effect |
| --- | --- | --- |
| `wasm` | `aigent-exec` | Wasmtime host runtime (default-on) |
| `sandbox` | `aigent-exec` | seccomp / `sandbox_init` (default-on) |
| `candle` | `aigent-llm`, `aigent-agent` | Local GGUF inference via Candle |
| `qdrant` | `aigent-memory` | Qdrant vector DB integration |
| `marketplace` | workspace | Extension marketplace (opt-in) |
| `uutils` | `aigent-tools`, `aigent-exec` | Delegate to system coreutils binaries |

---

## Development

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

`clippy -D warnings` is a hard CI gate. `clippy.toml` and `deny.toml` enforce additional lint and dependency audit rules.

```bash
# Build WASM guest tools
aigent tools build

# Run phase review gate
aigent doctor --review-gate --report docs/phase-review-gate.md
```

---

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.
