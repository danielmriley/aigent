use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::BackendEvent;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonStatus {
    pub bot_name: String,
    pub provider: String,
    pub model: String,
    pub thinking_level: String,
    pub external_thinking: bool,
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
    /// Hot-reload dynamic modules from the modules directory without touching
    /// the main config.
    ReloadTools,
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
    /// Run content-level deduplication across all memory tiers.
    DeduplicateMemory,
    /// Inject a batch of synthetic memory entries directly into the daemon's
    /// MemoryManager.  Designed for accelerated sleep-cycle testing: seed
    /// episodic observations with preference language, then immediately call
    /// `RunMultiAgentSleepCycle` to observe opinion formation.
    SeedMemories { entries: Vec<SeedEntry> },
    /// Reload the daemon's in-memory store from the event log on disk.
    /// Called by `aigent memory wipe` after writing tombstones so the daemon's
    /// live state stays consistent with the persisted event log.
    ReloadMemory,
}

/// A single synthetic memory entry used by [`ClientCommand::SeedMemories`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedEntry {
    /// The memory content text.
    pub content: String,
    /// Target tier: "episodic" (default), "semantic", "reflective",
    /// "procedural", "user_profile", or "core".
    pub tier: String,
    /// Source label recorded on the entry.  Defaults to `"test-seed"` when empty.
    pub source: String,
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
    /// Detailed tool listing with source metadata.
    ToolInfoList(Vec<aigent_tools::ToolInfo>),
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
