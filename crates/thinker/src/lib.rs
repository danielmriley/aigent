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

/// A typed chunk delivered over `token_tx` from the external thinking loop
/// to the caller (connection/sleep/subagents).
///
/// Using a single ordered channel for both variants guarantees that a
/// `Thought` emitted in the `final_answer` step always arrives before the
/// corresponding `Token` answer — eliminating the race that previously caused
/// the 💭 thought bubble to appear *after* the response in the TUI.
#[derive(Debug)]
pub enum TurnChunk {
    /// A clean response token to stream to the user.
    Token(String),
    /// The model's chain-of-thought, to be surfaced as an `AgentThought` event.
    Thought(String),
}
