pub mod agent_loop;
mod client;
mod commands;
mod events;
pub mod history;
pub mod micro_profile;
pub mod prompt_builder;
pub mod react_loop;
mod runtime;
mod server;
pub mod tool_loop;

pub use agent_loop::{
    AgentLoop, EvalScore, LlmToolCall, ProactiveOutput, ReactPhase, ReactSnapshot,
    ReflectionBelief, ReflectionOutput, SwarmRole, TurnSource,
};
pub use client::DaemonClient;
pub use commands::{ClientCommand, DaemonStatus, ProactiveStatsPayload, SleepStatusPayload, ServerEvent};
pub use events::{BackendEvent, ToolCallInfo, ToolResult};
pub use react_loop::{ReactConfig, ReactLoopResult, SubAgentResult};
pub use runtime::{AgentRuntime, ConversationTurn, SleepGenerationResult};
pub use server::run_unified_daemon;
pub use tool_loop::{ToolLoopResult, ToolExecution, run_tool_loop, build_tools_json};
