# Project Documentation: Aigent

## 1. Project Overview

Aigent is a modular, AI-agent framework written in Rust. It provides a comprehensive ecosystem for building, running, and interacting with AI agents. The project is structured as a Rust workspace containing multiple interdependent crates that handle different aspects of agent execution, such as memory management, LLM interactions, tool chain execution, sandboxing, and various user interfaces (CLI, Telegram, TUI).

- **Rust Edition:** 2021 (inferred standard for modern Rust crates)
- **Status:** Multi-crate Workspace
- **MSRV:** [TODO: needs clarification from root `Cargo.toml`]

## 2. Directory Structure

```text
aigent/
├── Cargo.toml
├── README.md
├── install.sh
├── config/
│   ├── default.toml
│   └── default.toml.example
├── crates/
│   ├── agent/
│   ├── config/
│   ├── exec/
│   ├── interfaces/
│   │   ├── cli/
│   │   ├── telegram/
│   │   └── tui/
│   ├── llm/
│   ├── memory/
│   ├── prompt/
│   ├── runtime/
│   ├── thinker/
│   └── tools/
├── docs/
└── extensions/
```

## 3. Workspace & Crates

### `agent`
- **Type:** `lib`
- **Purpose:** Manages the core agent loop, profile configuration, and primary event handling.
- **Key dependencies:** `tokio`, `serde` [TODO: extract exact versions]
- **Entry point:** `crates/agent/src/lib.rs`

### `config`
- **Type:** `lib`
- **Purpose:** Handles loading and parsing of workspace configuration files (e.g., `default.toml`).
- **Key dependencies:** `serde`, `toml`
- **Entry point:** `crates/config/src/lib.rs`

### `exec`
- **Type:** `lib`
- **Purpose:** Manages the execution environment, including WASM sandboxing, git operations, and hot-reloading tooling.
- **Key dependencies:** `wasmtime` (inferred), `git2` (inferred)
- **Entry point:** `crates/exec/src/lib.rs`

### `interfaces/cli`
- **Type:** `bin` / `lib`
- **Purpose:** Provides a Command Line Interface for interacting with Aigent.
- **Key dependencies:** `clap`
- **Entry point:** `crates/interfaces/cli/src/main.rs` (or `lib.rs`)

### `interfaces/telegram`
- **Type:** `bin` / `lib`
- **Purpose:** Telegram bot integration acting as a conversational interface for the agent.
- **Key dependencies:** `teloxide` (inferred)
- **Entry point:** `crates/interfaces/telegram/src/lib.rs`

### `interfaces/tui`
- **Type:** `bin` / `lib`
- **Purpose:** Terminal User Interface for richer console-based interaction.
- **Key dependencies:** `ratatui` or `crossterm` (inferred)
- **Entry point:** `crates/interfaces/tui/src/lib.rs`

### `llm`
- **Type:** `lib`
- **Purpose:** Interfaces with Large Language Models, including Candle-based local models and external APIs.
- **Key dependencies:** `candle-core`, `reqwest`
- **Entry point:** `crates/llm/src/lib.rs`

### `memory`
- **Type:** `lib`
- **Purpose:** Implements vector stores, event logs, sentiment tracking, and retrieval mechanisms for the agent's memory.
- **Key dependencies:** `serde`, vector database clients
- **Entry point:** `crates/memory/src/lib.rs`

### `prompt`
- **Type:** `lib`
- **Purpose:** Manages prompt templates and generation logic.
- **Entry point:** `crates/prompt/src/lib.rs`

### `runtime`
- **Type:** `lib`
- **Purpose:** Coordinates the execution context and lifecycle of agents.
- **Entry point:** `crates/runtime/src/lib.rs`

### `thinker`
- **Type:** `lib`
- **Purpose:** Handles the reasoning, planning, and decision-making logic of the AI.
- **Entry point:** `crates/thinker/src/lib.rs`

### `tools`
- **Type:** `lib`
- **Purpose:** Provides the suite of tools (function calling) that the agent can execute.
- **Entry point:** `crates/tools/src/lib.rs`

## 4. File-by-File Mapping

*(Note: Below maps the commonly identified structure from the provided workspace tree.)*

### `crates/agent/src/agent_loop.rs`
**Purpose:** Defines the core operational loop where the agent processes events and acts.
**Contains:** `AgentLoop` struct, async iteration logic.
**Key types/functions:** `run_loop()`

### `crates/agent/src/error.rs`
**Purpose:** Centralizes error types for the `agent` crate.
**Contains:** Custom `Error` enum implementing standard error traits.
**Key types/functions:** `AgentError`

### `crates/agent/src/events.rs`
**Purpose:** Defines the event bus and event typings the agent reacts to.
**Contains:** Event enums and publisher/subscriber traits.
**Key types/functions:** `Event`

### `crates/agent/src/micro_profile.rs`
**Purpose:** Manages lightweight, scoped profiles for sub-tasks or specialized agent personas.
**Contains:** Profile configuration structs.
**Key types/functions:** `MicroProfile`

### `crates/exec/src/sandbox.rs`
**Purpose:** Provides a restricted environment for the agent to safely execute code or perform OS actions.
**Contains:** Isolation configurations and process management.
**Key types/functions:** `Sandbox`, `apply_macos()`

### `crates/exec/src/wasm.rs`
**Purpose:** Handles WebAssembly module loading and execution.
**Contains:** Wasmtime integration logic.

### `crates/llm/src/candle_backend.rs`
**Purpose:** Implements local LLM inference using the Hugging Face Candle framework.
**Contains:** Model loading and tensor operations for inference.
**Key types/functions:** `CandleModel`

### `crates/llm/src/embedding.rs`
**Purpose:** Generates text embeddings for memory and retrieval.
**Contains:** Embedding abstractions and API calls.

### `crates/memory/src/vector_store.rs`
**Purpose:** Interfaces with vector databases for semantic search.
**Contains:** Storage, querying, and indexing traits.
**Key types/functions:** `VectorStore` trait, `insert()`, `search()`

### `crates/memory/src/event_log.rs`
**Purpose:** Maintains an append-only log of agent actions and observations.
**Contains:** History tracking logic.

### `config/default.toml`
**Purpose:** The default configuration file template for setting up an Aigent instance.
**Contains:** API keys placeholders, default model parameters, and database paths.

### `install.sh`
**Purpose:** Bootstrap script to setup the environment and compile the application.
**Contains:** Shell commands for installing Rust, dependencies, and building `aigent`.

*(More files exist across crates and can be expanded upon source code sync).*

## 5. Module Hierarchy & Architecture

- **Core Loop:** `runtime` -> `agent`
- **Intelligence:** `agent` invokes `thinker` to decide actions.
- **I/O & Senses:** `interfaces/*` feed events to `agent/src/events.rs`.
- **Knowledge:** `memory` (RAG, `vector_store.rs`, `event_log.rs`) provides context.
- **Action Execution:** `tools` and `exec` (sandbox, CLI actions, WASM) perform requested actions.
- **LLM Engine:** `llm` crate handles the actual abstraction of LLM backends (Ollama, Candle, etc.).

All components are glued together by the `runtime` crate and configured via `config`.

## 6. Key Dependencies & Features

| Dependency | Typical Version | Purpose / Use Case |
| --- | --- | --- |
| `tokio` | `1.x` | Async execution runtime for the agent loop and IO. |
| `serde` & `serde_json`| `1.0` | Serialization and deserialization of API responses, memory records, and configs. |
| `candle-core` | `0.x` | Local LLM inference engine. |
| `wasmtime` | `1x.x` | (Inferred from `wasm.rs`) Executing WebAssembly tools securely. |
| `clap` | `4.x` | Command-line argument parsing for the CLI interface. |

*[TODO: Needs verification against actual `Cargo.toml` dependencies.]*

## 7. Build & Run Instructions

**Automatic Installation:**
```bash
./install.sh
```

**Common Cargo Commands:**
```bash
# Build the entire workspace
cargo build

# Run tests across all crates
cargo test --workspace

# Build for release
cargo build --release

# Run the primary CLI interface (assuming ainent-app or cli bin)
cargo run --bin aigent-app
```
