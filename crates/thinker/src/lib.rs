//! The thinker crate: external thinking loop and prompt generation.
//!
//! Extracted from `aigent-runtime` so the core LLM ↔ tool loop is
//! independently testable and has a minimal dependency footprint.
//!
//! # Architecture
//!
//! ```text
//!   aigent_agent::run_agent_turn()
//!         │
//!         ├─ build_external_thinking_block() ← thinker::prompt
//!         └─ run_external_thinking_loop()    ← thinker::ext_loop
//!               │
//!               ├─ LlmRouter::chat_messages_stream()  (aigent-llm)
//!               ├─ ToolExecutor::execute()             (aigent-exec)
//!               └─ emits ThinkerEvent via EventSink   (thinker::events)
//! ```

pub mod events;
pub mod ext_loop;
pub mod json_stream;
pub mod prompt;
pub mod tool_loop;

pub use events::{ThinkerEvent, ToolCallInfo, ToolResult};
pub use ext_loop::run_external_thinking_loop;
pub use json_stream::{AgentStep, JsonStreamBuffer};
pub use prompt::build_external_thinking_block;
pub use tool_loop::{EventSink, ToolExecution, ToolLoopResult, build_tools_json};
