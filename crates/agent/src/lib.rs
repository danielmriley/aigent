//! `aigent-agent` — pure agent logic: reasoning, reflection, memory interaction,
//! sleep cycles, and the ReAct state machine.
//!
//! This crate has **zero** server, daemon, or IPC concerns.  [`AgentRuntime`]
//! receives a `&mut MemoryManager` from the caller so it never owns mutable
//! global state.

pub mod agent_loop;
mod error;
mod events;
pub mod micro_profile;
mod runtime;
pub mod subagents;
pub mod turn;

// ── Error types ───────────────────────────────────────────────────────────────

pub use error::{AgentError, AgentResult};

// ── Agent loop types ──────────────────────────────────────────────────────────

pub use agent_loop::{
    AgentLoop, EvalScore, LlmToolCall, ProactiveOutput, ReactPhase, ReactSnapshot,
    ReflectionBelief, ReflectionOutput, SwarmRole, TurnSource,
};

// ── Event types ───────────────────────────────────────────────────────────────

pub use events::BackendEvent;

// Re-export thinker event types so consumers have a single import path.
pub use aigent_thinker::{ToolCallInfo, ToolResult};

// ── Runtime types ─────────────────────────────────────────────────────────────

pub use runtime::{AgentRuntime, ConversationTurn, SleepGenerationResult, SUMMARIZE_THRESHOLD};

// ── Unified agent turn ────────────────────────────────────────────────────────

pub use turn::{AgentTurnInput, run_agent_turn};
