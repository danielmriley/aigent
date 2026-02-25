pub mod agent_loop;
mod client;
mod commands;
mod events;
pub mod history;
pub mod micro_profile;
pub mod prompt_builder;
mod runtime;
mod server;

pub use agent_loop::{LlmToolCall, ProactiveOutput, ReflectionBelief, ReflectionOutput, TurnSource};
pub use client::DaemonClient;
pub use commands::{ClientCommand, DaemonStatus, ProactiveStatsPayload, SleepStatusPayload, ServerEvent};
pub use events::{BackendEvent, ToolCallInfo, ToolResult};
pub use runtime::{AgentRuntime, ConversationTurn};
pub use server::run_unified_daemon;
