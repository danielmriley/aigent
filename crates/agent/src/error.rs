//! Typed error types for the agent crate.

use thiserror::Error;

/// Errors produced by [`crate::AgentRuntime`] methods.
#[derive(Debug, Error)]
pub enum AgentError {
    /// An LLM call failed (network, timeout, model unavailable).
    #[error("LLM: {0}")]
    Llm(#[source] anyhow::Error),

    /// Memory recording or retrieval failed.
    #[error("memory: {0}")]
    Memory(#[source] anyhow::Error),

    /// Sleep cycle generation or application failed.
    #[error("sleep cycle: {0}")]
    Sleep(#[source] anyhow::Error),

    /// Catch-all for errors that don't fit the above categories.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Convenience alias used throughout the agent crate.
pub type AgentResult<T> = std::result::Result<T, AgentError>;
