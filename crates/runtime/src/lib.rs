pub mod agent_loop;
mod client;
mod commands;
mod events;
pub mod history;
pub mod micro_profile;
// Re-export prompt crate types for backward compatibility with downstream crates.
pub use aigent_prompt::{build_chat_prompt, truncate_for_prompt, PromptInputs};
pub mod schedule_store;
pub mod scheduler;
mod runtime;
mod server;

pub use agent_loop::{
    AgentLoop, EvalScore, LlmToolCall, ProactiveOutput, ReactPhase, ReactSnapshot,
    ReflectionBelief, ReflectionOutput, SwarmRole, TurnSource,
};
pub use client::DaemonClient;
pub use commands::{ClientCommand, DaemonStatus, ProactiveStatsPayload, SleepStatusPayload, ServerEvent};
pub use events::BackendEvent;
pub use scheduler::{HeartbeatFn, ScheduledTask, SchedulerState, TaskSchedule, spawn_scheduler};
pub use runtime::{AgentRuntime, ConversationTurn, SleepGenerationResult};
pub use server::run_unified_daemon;

// Re-export thinker types for backward compatibility with downstream crates.
pub use aigent_thinker::{
    ToolCallInfo, ToolResult, ToolLoopResult, ToolExecution,
    build_tools_json, run_tool_loop,
};
