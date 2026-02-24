//! Types shared across the unified agent loop (reflection, proactive, turn source).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ─── reflection ──────────────────────────────────────────────────────────────

/// Structured result produced by [`crate::AgentRuntime::inline_reflect`].
///
/// Reflects on a single completed exchange and yields zero or more new
/// beliefs and free-form insight strings.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReflectionOutput {
    pub beliefs: Vec<ReflectionBelief>,
    pub reflections: Vec<String>,
}

/// A single belief extracted from an exchange.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionBelief {
    pub claim: String,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

fn default_confidence() -> f32 {
    0.65
}

// ─── proactive ───────────────────────────────────────────────────────────────

/// Structured result produced by [`crate::AgentRuntime::run_proactive_check`].
///
/// `action` is `None` when the daemon decides not to send anything; otherwise
/// it is a short tag such as `"follow_up"` or `"reminder"`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProactiveOutput {
    pub action: Option<String>,
    pub message: Option<String>,
    pub urgency: Option<f32>,
}

// ─── turn source ─────────────────────────────────────────────────────────────

/// Where a turn originated.  Passed through the server as a metadata hint so
/// that post-processing (e.g. proactive recording) can tag entries correctly.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TurnSource {
    Tui,
    Telegram { chat_id: i64 },
    Cli,
    Proactive,
}

// ─── tool calling ─────────────────────────────────────────────────────────────

/// A structured tool call produced by [`crate::AgentRuntime::maybe_tool_call`].
///
/// When the LLM decides that a tool should be invoked in order to answer the
/// user's message, it returns one of these.  The daemon executes the named tool
/// with the supplied `args`, records the result, and passes it back to the LLM
/// as additional context for the final streaming response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmToolCall {
    /// Name of the tool to invoke (must match a `ToolSpec::name` in the registry).
    pub tool: String,
    /// Key-value arguments to pass to the tool.
    #[serde(default)]
    pub args: HashMap<String, String>,
}
