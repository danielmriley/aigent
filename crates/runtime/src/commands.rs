use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::BackendEvent;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonStatus {
    pub bot_name: String,
    pub provider: String,
    pub model: String,
    pub thinking_level: String,
    pub memory_total: usize,
    pub memory_core: usize,
    pub memory_user_profile: usize,
    pub memory_reflective: usize,
    pub memory_semantic: usize,
    pub memory_episodic: usize,
    pub uptime_secs: u64,
    pub available_tools: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClientCommand {
    SubmitTurn { user: String, source: String },
    GetStatus,
    GetMemoryPeek { limit: usize },
    ExecuteTool { name: String, args: HashMap<String, String> },
    ListTools,
    ReloadConfig,
    Shutdown,
    Ping,
    /// Open a persistent connection that receives broadcast events from all turns.
    Subscribe,
    /// Manually trigger an agentic sleep cycle for memory consolidation.
    RunSleepCycle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServerEvent {
    Backend(BackendEvent),
    Status(DaemonStatus),
    MemoryPeek(Vec<String>),
    ToolList(Vec<aigent_tools::ToolSpec>),
    ToolResult { success: bool, output: String },
    Ack(String),
}
