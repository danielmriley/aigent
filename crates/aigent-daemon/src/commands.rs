use serde::{Deserialize, Serialize};

use crate::BackendEvent;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonStatus {
    pub bot_name: String,
    pub provider: String,
    pub model: String,
    pub thinking_level: String,
    pub memory_total: usize,
    pub uptime_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClientCommand {
    SubmitTurn { user: String },
    GetStatus,
    GetMemoryPeek { limit: usize },
    Shutdown,
    Ping,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServerEvent {
    Backend(BackendEvent),
    Status(DaemonStatus),
    MemoryPeek(Vec<String>),
    Ack(String),
}
