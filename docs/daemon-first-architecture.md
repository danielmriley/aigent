# Daemon-First Architecture Proposal

**Status:** Proposal — not yet implemented  
**Date:** 2026-02-20  
**Author:** Architecture review session

---

## 1. Executive Summary

The current architecture has the daemon as one optional component among several loosely coupled pieces. The proposed architecture inverts this: the **daemon is always the single source of truth**, and every interface (TUI, Telegram, CLI tools) is a thin client that connects to it. The daemon starts once and runs indefinitely. Clients come and go freely.

---

## 2. Current Architecture (as-built)

```
aigent start
├── ensure_daemon_running()       ← spawns a child OS process if not already live
│     └── aigent [daemon process]
│           └── run_unified_daemon()
│                 owns: MemoryManager, AgentRuntime, ToolRegistry
│                 listens: /tmp/aigent.sock
│
├── tokio::spawn → Telegram bot task
│     └── each message: DaemonClient::stream_submit() → Unix socket → daemon
│
└── TUI (foreground)
      ├── tokio::spawn → DaemonClient::subscribe() → Unix socket  (broadcast events)
      └── per message:  DaemonClient::stream_submit() → Unix socket
```

### Problems with the current design

1. **Daemon lifecycle is coupled to the TUI.** `ensure_daemon_running()` is called inside `run_start_mode`, meaning the daemon is only guaranteed to be running when a TUI session is active. If the TUI is closed, the daemon process eventually orphans or is reaped.

2. **Telegram is a side-effect of TUI startup.** The Telegram bot is spawned as a background `tokio::spawn` inside `run_start_mode`. If the TUI is closed, the Telegram bot also dies (they share the same process). There is no way to have Telegram running without an open TUI session.

3. **Telegram → TUI event relay is fragile.** The path is: Telegram → IPC → daemon broadcast → IPC → subscribe task → TUI. Two socket hops, both JSON-serialized. The subscribe connection has no automatic reconnect logic — a single EOF silently severs the link, and Telegram messages stop appearing in the TUI without any indication.

4. **No daemon auto-start on system boot.** There is no systemd/launchd unit or equivalent. The daemon only exists as long as at least one active `aigent` session is running.

5. **`aigent memory stats` and similar commands load the event log directly from disk**, bypassing the daemon entirely. If the daemon is mid-write, there is a race. These commands should talk to the daemon.

6. **`aigent onboard` writes memory directly** via `seed_identity_from_config`, opening the event log from the CLI process in parallel with a potentially-running daemon. Same race risk.

---

## 3. Proposed Architecture: Daemon-First

### Core principle

> The daemon is the program. Everything else is a window into it.

```
System boot / first run
└── (user runs: aigent onboard)
      └── wizard completes → config saved → daemon started automatically

Daemon (long-lived, survives all client disconnects)
├── owns: MemoryManager (event log), AgentRuntime, ToolRegistry, Telegram bot task
├── listens: /tmp/aigent.sock   (or XDG_RUNTIME_DIR equivalent)
└── broadcast: tokio::sync::broadcast<BackendEvent>

Clients (connect/disconnect freely)
├── TUI:      aigent           → connects, subscribes, submits turns, disconnects on /exit
├── CLI chat: aigent chat      → connects, interactive line session, disconnects
├── Status:   aigent status    → connects, queries, disconnects
└── Memory:   aigent memory *  → connects, queries live daemon state, disconnects
```

### Daemon lifecycle

| Event | Effect |
|---|---|
| `aigent onboard` completes | Daemon auto-started (if not already running) |
| `aigent` (any subcommand) | Daemon auto-started if not live |
| `aigent daemon stop` | Graceful shutdown: sleep cycle, flush memory, release socket |
| TUI window closed | Daemon keeps running — Telegram, memory, sleep cycles continue |
| System reboot | Daemon not auto-started unless user installs the systemd/launchd unit |

### Telegram bot lifecycle

The Telegram bot task is owned by the daemon, not the TUI. It is started when the daemon starts (if `telegram_enabled = true` in config) and stopped when the daemon stops. A `aigent daemon reload` command re-reads config and restarts only the Telegram task if the token or enabled flag changed.

This eliminates the current race where the bot dies when the TUI is closed.

### TUI reconnect

When the TUI opens (`aigent`), it:
1. Checks if the daemon socket is live.
2. If not: starts the daemon, waits for socket to be ready.
3. Connects two persistent connections: one for `Subscribe` (broadcast events), one for request/response.
4. When the TUI is closed (`/exit` or Ctrl+C), it sends a `ClientCommand::Disconnect` to let the daemon clean up per-client state, then exits. The daemon stays up.
5. When the TUI is re-opened, it reconnects and requests a `GetRecentContext` snapshot to restore the visible conversation history.

### In-daemon Telegram → TUI path

Since Telegram runs *inside* the daemon process, the path for Telegram → TUI becomes:

```
Telegram bot task
  → daemon's internal AgentCore::submit(turn, source="telegram")
  → broadcast channel (in-process, no serialization)
  → all connected Subscribe clients
  → TUI renders the message
```

No IPC hops for Telegram events. The broadcast is a direct `tokio::sync::broadcast::send`.

---

## 4. Command Surface (proposed)

### First run

```bash
aigent onboard          # wizard; starts daemon on completion
```

If a user runs `aigent` without ever onboarding:

```
Aigent is not configured yet.
Run: aigent onboard
```

### Normal use

```bash
aigent                  # open TUI (daemon auto-starts if needed)
aigent chat             # headless line-mode chat (no TUI)
aigent status           # print daemon status and memory stats
```

### Daemon management

```bash
aigent daemon start     # start daemon manually
aigent daemon stop      # graceful shutdown
aigent daemon restart   # stop + start
aigent daemon status    # show pid, uptime, memory stats, telegram status
aigent daemon reload    # reload config (restarts Telegram task if token changed)
aigent daemon logs      # tail -f the daemon log
```

### Memory and config (all go through daemon IPC)

```bash
aigent memory stats
aigent memory inspect-core
aigent configuration
aigent onboard          # can be re-run; daemon reload follows
```

---

## 5. IPC Protocol Changes Required

The current `ClientCommand` enum needs a few additions:

```rust
pub enum ClientCommand {
    // Existing
    SubmitTurn { user: String, source: String },
    Subscribe,
    GetStatus,
    GetMemoryPeek { limit: usize },
    // ...

    // New
    GetRecentContext { limit: usize },  // fetch last N turns for TUI restore
    Disconnect,                         // client is cleanly disconnecting
    ReloadConfig,                       // already exists — used by `daemon reload`
}
```

And `ServerEvent` needs a snapshot event for TUI reconnect:

```rust
pub enum ServerEvent {
    // Existing
    Backend(BackendEvent),
    Status(DaemonStatus),
    MemoryPeek(Vec<String>),
    // ...

    // New
    RecentContext(Vec<ConversationTurn>),  // sent on reconnect
}
```

---

## 6. Daemon State: what moves inside

Currently `AgentRuntime`, `MemoryManager`, and `ToolRegistry` already live in the daemon. What needs to move in:

| Component | Current home | Proposed home |
|---|---|---|
| `MemoryManager` | Daemon ✓ | Daemon ✓ |
| `AgentRuntime` | Daemon ✓ | Daemon ✓ |
| `ToolRegistry` | Daemon ✓ | Daemon ✓ |
| Telegram bot polling loop | CLI process (tokio::spawn) | Daemon (tokio::spawn) |
| Sleep cycle scheduler | Daemon (turn-count trigger only) | Daemon (+ time-based trigger) |
| Conversation history (`VecDeque<ConversationTurn>`) | Daemon ✓ | Daemon ✓ (but now also served to reconnecting TUI) |

---

## 7. Migration Plan

The existing code is already ~80% of the way there. The changes are additive, not a rewrite.

### Phase 1 — Move Telegram bot into the daemon
- In `run_unified_daemon`, spawn the Telegram task inside the daemon process (if `telegram_enabled`).
- Add `ReloadConfig` handling that restarts the Telegram task if config changes.
- Remove Telegram spawning from `run_start_mode`.
- **Result:** Telegram survives TUI close. Telegram → TUI events require only one in-process broadcast, zero socket hops.

### Phase 2 — TUI reconnect
- Add `GetRecentContext` command and `RecentContext` server event.
- On TUI startup, after connecting to the daemon, request recent context to pre-populate the chat history.
- **Result:** Re-opening `aigent` shows the conversation history from the daemon.

### Phase 3 — Decouple daemon start from `run_start_mode`
- Make daemon auto-start happen at the top of `main()` for all interactive subcommands, not only inside `run_start_mode`.
- After `aigent onboard` completes, start the daemon automatically.
- **Result:** The daemon is always running; no command is responsible for starting it except `aigent daemon start` and first-run.

### Phase 4 — Route memory/config commands through IPC
- `aigent memory stats` → `DaemonClient::get_status()` (already has memory counts).
- `aigent memory inspect-core` → new `GetMemoryInspect` command.
- `aigent configuration` → `DaemonClient::reload_config()` after saving.
- **Result:** Eliminates direct event log reads from CLI subcommands while daemon is running.

---

## 8. What remains as-is

- The Unix socket IPC protocol — no change.
- `DaemonClient` — no change to the struct, only new command variants.
- The TUI itself (`crates/interfaces/tui/`) — no change to rendering.
- Memory storage format (`events.jsonl`) — no change.
- `aigent doctor`, `aigent reset`, `aigent memory wipe` — still operate directly on disk (safe because daemon is stopped for reset).

---

## 9. Open Questions

1. **Should the daemon auto-start on system boot?** An optional `aigent daemon install` command that writes a systemd user unit (`~/.config/systemd/user/aigent.service`) or a launchd plist would make sense as a follow-on. Not required for Phase 1–3.

2. **What happens if the user has two machines?** The event log is local. Sync is out of scope for now but the append-only event log format is well-suited for future CRDT-style merge.

3. **Should `aigent onboard` restart a running daemon?** Yes — onboarding changes the config and seeds Core memory, so the daemon needs to reload. `ReloadConfig` + `seed_core_identity` inside the daemon handles this cleanly.

4. **Per-client conversation isolation?** Currently all clients share one `VecDeque<ConversationTurn>`. If the TUI and Telegram are both active simultaneously, turns from both are interleaved in `recent_turns`. This is probably desirable (the agent should have one coherent context window), but worth making explicit.
