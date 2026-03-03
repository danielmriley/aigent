//! Centralized prompt assembly for the main LLM conversation call.
//!
//! This crate owns the prompt layout, grounding rules, truth-seeking
//! directives, and all text formatting for the agent's system prompt.
//!
//! The main entry point is [`build_chat_prompt`] which takes a
//! [`PromptInputs`] struct of pre-computed data and returns a monolithic
//! system prompt string.  The function is purely synchronous — all async
//! work (embeddings, memory retrieval) is done by the caller beforehand.
//!
//! # Crate layout
//!
//! - [`builder`] — `build_chat_prompt` + private block helpers
//! - [`truncate`] — `truncate_for_prompt` utility (used widely)

mod builder;
pub mod truncate;

pub use builder::build_chat_prompt;
pub use truncate::truncate_for_prompt;

use aigent_memory::{MemoryStats, retrieval::RankedMemoryContext};
use uuid::Uuid;

// ─── types ───────────────────────────────────────────────────────────────────

/// A single user↔assistant exchange in conversation history.
#[derive(Debug, Clone)]
pub struct ConversationTurn {
    pub user: String,
    pub assistant: String,
}

/// All pre-computed data needed to assemble the final LLM prompt.
///
/// Callers build this struct (doing async work like embeddings beforehand)
/// and then pass it to [`build_chat_prompt`] which is purely synchronous.
///
/// Memory data is pre-extracted from `MemoryManager` — the builder never
/// touches the manager directly, making it a pure function of its inputs.
pub struct PromptInputs<'a> {
    pub config: &'a aigent_config::AppConfig,
    pub user_message: &'a str,
    pub recent_turns: &'a [ConversationTurn],
    pub tool_specs: &'a [aigent_tools::ToolSpec],
    pub pending_follow_ups: &'a [(Uuid, String)],
    /// Ranked memory context items (pre-computed with optional embeddings).
    pub context_items: &'a [RankedMemoryContext],
    /// Memory statistics snapshot (taken once before prompt assembly).
    pub stats: MemoryStats,

    // ── pre-computed memory blocks ───────────────────────────────────────
    /// Output of `MemoryManager::cached_identity_block()`.
    pub identity_block: String,
    /// Output of `MemoryManager::cached_beliefs_block(max)`.
    pub beliefs_block: String,
    /// Output of `MemoryManager::user_name_from_core()`.
    pub user_name: Option<String>,
    /// Output of `MemoryManager::relational_state_block()`.
    pub relational_block: Option<String>,
}
