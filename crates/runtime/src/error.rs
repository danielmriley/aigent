//! Typed error types for the server / daemon crate.

use thiserror::Error;

/// Errors produced by the daemon server layer.
#[derive(Debug, Error)]
pub enum ServerError {
    /// Unix socket I/O failure.
    #[error("socket I/O: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization / deserialization failure.
    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),

    /// Schedule persistence error (invalid task definitions, etc.).
    #[error("schedule: {0}")]
    Schedule(String),

    /// Chat history I/O error.
    #[error("history: {0}")]
    History(String),

    /// Catch-all for errors that don't fit the above categories.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
