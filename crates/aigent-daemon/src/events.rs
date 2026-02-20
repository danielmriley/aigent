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
}
