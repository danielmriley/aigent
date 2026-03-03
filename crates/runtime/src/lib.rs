//! `aigent-runtime` — daemon/server layer that hosts the agent over Unix sockets.
//!
//! Agent logic (reasoning, reflection, sleep) lives in [`aigent_agent`].
//! This crate provides the daemon orchestration, IPC protocol, scheduler,
//! and chat history persistence.

mod client;
mod commands;
mod error;
pub mod history;
pub mod schedule_store;
pub mod scheduler;
mod server;

// ── Local types ───────────────────────────────────────────────────────────────

pub use client::DaemonClient;
pub use commands::{ClientCommand, DaemonStatus, ProactiveStatsPayload, SleepStatusPayload, ServerEvent};
pub use error::ServerError;
pub use scheduler::{HeartbeatFn, ScheduledTask, SchedulerState, TaskSchedule, spawn_scheduler};
pub use server::run_unified_daemon;

// ── Re-exports from aigent-agent ──────────────────────────────────────────────
// Downstream crates (TUI, CLI, Telegram) have historically imported these from
// `aigent_runtime`.  The re-exports preserve backward compatibility.

pub use aigent_agent::{
    AgentLoop, AgentRuntime, BackendEvent, ConversationTurn, EvalScore,
    LlmToolCall, ProactiveOutput, ReactPhase, ReactSnapshot,
    ReflectionBelief, ReflectionOutput, SleepGenerationResult, SwarmRole,
    AgentError, AgentResult, ToolCallInfo, ToolResult, TurnSource,
};

pub use aigent_agent::agent_loop;
pub use aigent_agent::micro_profile;

// ── Re-exports from aigent-prompt ─────────────────────────────────────────────

pub use aigent_prompt::{build_chat_prompt, truncate_for_prompt, PromptInputs};

// ── Re-exports from aigent-thinker ────────────────────────────────────────────

pub use aigent_thinker::{ToolLoopResult, ToolExecution, build_tools_json, run_tool_loop};
