# Aigent

Aigent is a persistent, self-improving Rust AI agent with local-first memory, secure execution boundaries, and dual LLM routing.

## Status

Phase 0 and Phase 1 are complete. Phase 2 is in progress.

## Capabilities matrix

| Capability area | Status | Notes |
| --- | --- | --- |
| Onboarding wizard (TUI + prompt fallback) | ✅ Available | Configures identity, model/provider, thinking level, workspace, sleep window, safety profile. |
| Memory contract defaults | ✅ Available | Onboarding enforces `eventlog` backend and nightly sleep defaults. |
| Persistent memory (4 tiers) | ✅ Available | Episodic, semantic, procedural, and core memory persisted in append-only event log. |
| Memory commands | ✅ Available | `memory stats`, `memory inspect-core`, `memory promotions`, `memory export-vault`, `memory wipe`. |
| Vault projection | ✅ Available | Exports and auto-syncs Obsidian-style vault under `.aigent/vault`. |
| Retrieval quality | ✅ Available | Weighted retrieval (core priority + recency + relevance + confidence) with provenance context. |
| Sleep/distillation loop | ✅ Available | Nightly/interval sleep cycle with promotion records and one-cycle-per-night behavior. |
| Interactive TUI chat | ✅ Available | Scrollable transcript, markdown-aware rendering, command suggestions, runtime commands. |
| Telegram bot runtime | ✅ Available | Long-polling bot, per-chat short-term context, shared model/memory runtime. |
| Telegram command parity | 🟨 In progress | Core commands are available; remaining parity for advanced memory actions is pending. |
| Unified channel command engine | 🟨 In progress | Interactive and Telegram share behavior, but command handling is still duplicated in code paths. |
| Review gate before next phase | 🟨 In progress | `doctor --review-gate` implemented; progression depends on passing environment/state checks. |
| Full daemon/channel parity completion | ⏳ Planned | Final pass for long-lived loop behavior consistency and parity smoke coverage. |

## Goals

- Memory-centric agent identity (4-tier local memory + sleep distillation)
- Dual LLM providers (Ollama local-first, OpenRouter fallback)
- Secure execution (WASM sandbox + ephemeral Docker shell)
- Persistent daemon with TUI and Telegram parity

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
- `aigent run` and `aigent telegram` force onboarding when setup is not yet marked complete.

## Telegram + daemon lifecycle

Configure Telegram in the wizard:

```bash
aigent configuration
```

This can set:
- `integrations.telegram_enabled = true`
- `TELEGRAM_BOT_TOKEN` in `.env`

Run bot service in background:

```bash
aigent daemon start
aigent daemon status
aigent daemon restart
aigent daemon stop
```

Run in foreground with local TUI + connected services:

```bash
aigent start
```

Runtime files:
- PID: `.aigent/runtime/daemon.pid`
- Log: `.aigent/runtime/daemon.log`

## Phase review gate

Before advancing phases, run:

```bash
aigent doctor --review-gate --report docs/phase-review-gate.md
```

The command validates key Phase 0–2 readiness checks and exits non-zero if any check fails.
It auto-remediates fixable items (memory backend/nightly mode/vault projection/sleep marker)
before evaluating hard-fail checks.

## Install (local)

```bash
./install.sh
```

Notes:
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
