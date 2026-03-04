//! The thinker crate: structured tool calling loop and external thinking
//! prompt generation.
//!
//! Extracted from `aigent-runtime` so the core LLM ↔ tool loop is
//! independently testable and has a minimal dependency footprint.
//!
//! # Architecture
//!
//! ```text
//!   connection.rs (runtime)
//!         │
//!         ├─ build_chat_prompt()          ← runtime's prompt_builder
//!         ├─ build_external_thinking_block() ← thinker::prompt
//!         └─ run_tool_loop()              ← thinker::tool_loop
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
pub use tool_loop::{
    EventSink, ToolExecution, ToolLoopResult, build_tools_json,
    execute_tool_calls_public, run_tool_loop,
};
