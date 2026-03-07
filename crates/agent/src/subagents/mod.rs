//! Parallel subagent reasoning pipeline for chat interactions.
//!
//! Mirrors the multi-agent sleep consolidation pattern in
//! `crates/memory/src/multi_sleep.rs` and `crates/agent/src/runtime/sleep.rs`,
//! adapted for live conversation turns.
//!
//! Pipeline:
//!   1. Three specialist agents (Researcher, Planner, Critic) evaluate the
//!      current query + context in parallel via `tokio::join!`.
//!   2. Their structured outputs are parsed into [`SubagentAnalysis`] structs.
//!   3. The results are formatted into a `<subagent_debate>` context block
//!      that the Captain (main agent loop) consumes for its final response.
//!
//! Subagents are **read-only**: they never execute tools, mutate memory,
//! or hold locks.  Only the Captain writes state.

pub mod manager;
pub mod prompts;
pub mod types;

pub use manager::SubagentManager;
pub use types::{SubagentAnalysis, SubagentRole};
