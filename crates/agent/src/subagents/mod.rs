//! Parallel subagent reasoning pipeline for chat interactions.
//!
//! Each specialist (Researcher, Planner, Critic) runs a bounded external
//! thinking loop over a read-only tool subset in parallel via `tokio::join!`.
//! Their `final_answer` texts are assembled into a `<subagent_debate>` block
//! that the Captain (main agent loop) consumes for its final response.
//!
//! Subagents are **read-only**: they may call tools flagged `read_only = true`
//! (web search, file reads, datetime) but never write files, mutate memory,
//! or produce any side effects visible to the user.

pub mod manager;
pub mod prompts;
pub mod router;
pub mod types;

pub use manager::SubagentManager;
pub use router::needs_specialists;
pub use types::SubagentRole;
