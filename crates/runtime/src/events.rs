use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallInfo {
    pub name: String,
    pub args: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub name: String,
    pub success: bool,
    pub output: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackendEvent {
    Token(String),
    ToolCallStart(ToolCallInfo),
    ToolCallEnd(ToolResult),
    Thinking,
    Done,
    Error(String),
    MemoryUpdated,
    /// A turn submitted from an external source (e.g. Telegram). Contains
    /// the source label and the original user message so the TUI can display
    /// a user bubble before the streaming assistant response arrives.
    ExternalTurn { source: String, content: String },
}
