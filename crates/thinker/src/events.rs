//! Progress events emitted by the tool loop.
//!
//! These types are intentionally decoupled from `aigent_runtime::BackendEvent`
//! so that the thinker crate has no dependency on the runtime crate.  The
//! runtime maps `ThinkerEvent` → `BackendEvent` at the call site.

use serde::{Deserialize, Serialize};

/// Information about a tool call that is about to be executed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallInfo {
    pub name: String,
    pub args: String,
}

/// Result of a completed tool invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub name: String,
    pub success: bool,
    pub output: String,
    /// Wall-clock duration of the tool invocation in milliseconds.
    #[serde(default)]
    pub duration_ms: u64,
}

/// Events emitted by the thinker tool loop for progress reporting.
#[derive(Debug, Clone)]
pub enum ThinkerEvent {
    ToolCallStart(ToolCallInfo),
    ToolCallEnd(ToolResult),
    /// The model's chain-of-thought from a JSON agent step.
    /// Emitted by the external thinking loop so the UI can display
    /// a "💭 thinking…" line without exposing raw JSON.
    AgentThought(String),
}
