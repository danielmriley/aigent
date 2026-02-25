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
    /// Manually trigger the full nightly multi-agent sleep cycle.
    /// Falls back to single-agent if the LLM is unavailable.
    RunMultiAgentSleepCycle,
    /// Immediately run the proactive check irrespective of the configured
    /// interval and DND window.
    TriggerProactive,
    /// Return statistics about proactive mode activity.
    GetProactiveStats,
    /// Return sleep cycle status: last passive/nightly times, schedule info.
    GetSleepStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SleepStatusPayload {
    pub auto_sleep_mode: String,
    pub passive_interval_hours: u64,
    pub last_passive_sleep_at: Option<String>,
    pub last_nightly_sleep_at: Option<String>,
    pub quiet_window_start: u8,
    pub quiet_window_end: u8,
    pub timezone: String,
    pub in_quiet_window: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProactiveStatsPayload {
    pub total_sent: u64,
    pub last_proactive_at: Option<String>,
    pub interval_minutes: u64,
    pub dnd_start_hour: u8,
    pub dnd_end_hour: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServerEvent {
    Backend(BackendEvent),
    Status(DaemonStatus),
    MemoryPeek(Vec<String>),
    ToolList(Vec<aigent_tools::ToolSpec>),
    ToolResult { success: bool, output: String },
    Ack(String),
    /// Incremental status line emitted during a long-running sleep cycle.
    /// This is NOT a terminal event — the client continues reading until
    /// `Ack` arrives.
    StatusLine(String),
    /// Response to `GetProactiveStats`.
    ProactiveStats(ProactiveStatsPayload),
    /// Response to `GetSleepStatus`.
    SleepStatus(SleepStatusPayload),
}
