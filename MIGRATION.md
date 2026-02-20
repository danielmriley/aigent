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
