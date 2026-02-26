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
    ///
    /// Values are `serde_json::Value` because the LLM may emit integers,
    /// booleans, or strings depending on the parameter type.  Use
    /// [`stringify_args`] to coerce them to the `HashMap<String, String>`
    /// that [`ToolExecutor::execute`] expects.
    #[serde(default)]
    pub args: HashMap<String, serde_json::Value>,
}

impl LlmToolCall {
    /// Coerce all argument values to strings for passing to the tool executor.
    ///
    /// Integers become `"5"`, booleans become `"true"`, strings stay as-is,
    /// and other JSON types are serialized to their compact JSON representation.
    pub fn stringify_args(&self) -> HashMap<String, String> {
        self.args
            .iter()
            .map(|(k, v)| {
                let s = match v {
                    serde_json::Value::String(s) => s.clone(),
                    serde_json::Value::Number(n) => n.to_string(),
                    serde_json::Value::Bool(b) => b.to_string(),
                    serde_json::Value::Null => String::new(),
                    other => other.to_string(),
                };
                (k.clone(), s)
            })
            .collect()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn stringify_string_value() {
        let call = LlmToolCall {
            tool: "test".into(),
            args: HashMap::from([("key".into(), json!("value"))]),
        };
        let args = call.stringify_args();
        assert_eq!(args["key"], "value");
    }

    #[test]
    fn stringify_number_value() {
        let call = LlmToolCall {
            tool: "test".into(),
            args: HashMap::from([("count".into(), json!(42))]),
        };
        let args = call.stringify_args();
        assert_eq!(args["count"], "42");
    }

    #[test]
    fn stringify_float_value() {
        let call = LlmToolCall {
            tool: "test".into(),
            args: HashMap::from([("score".into(), json!(2.72))]),
        };
        let args = call.stringify_args();
        assert_eq!(args["score"], "2.72");
    }

    #[test]
    fn stringify_bool_value() {
        let call = LlmToolCall {
            tool: "test".into(),
            args: HashMap::from([("flag".into(), json!(true))]),
        };
        let args = call.stringify_args();
        assert_eq!(args["flag"], "true");
    }

    #[test]
    fn stringify_null_value() {
        let call = LlmToolCall {
            tool: "test".into(),
            args: HashMap::from([("empty".into(), json!(null))]),
        };
        let args = call.stringify_args();
        assert_eq!(args["empty"], "");
    }

    #[test]
    fn stringify_array_value_uses_json() {
        let call = LlmToolCall {
            tool: "test".into(),
            args: HashMap::from([("items".into(), json!(["a", "b"]))]),
        };
        let args = call.stringify_args();
        assert_eq!(args["items"], "[\"a\",\"b\"]");
    }

    #[test]
    fn stringify_object_value_uses_json() {
        let call = LlmToolCall {
            tool: "test".into(),
            args: HashMap::from([("nested".into(), json!({"x": 1}))]),
        };
        let args = call.stringify_args();
        assert_eq!(args["nested"], "{\"x\":1}");
    }

    #[test]
    fn stringify_empty_args() {
        let call = LlmToolCall {
            tool: "test".into(),
            args: HashMap::new(),
        };
        let args = call.stringify_args();
        assert!(args.is_empty());
    }

    #[test]
    fn stringify_mixed_types() {
        let call = LlmToolCall {
            tool: "mixed".into(),
            args: HashMap::from([
                ("name".into(), json!("Alice")),
                ("age".into(), json!(30)),
                ("active".into(), json!(true)),
                ("note".into(), json!(null)),
            ]),
        };
        let args = call.stringify_args();
        assert_eq!(args.len(), 4);
        assert_eq!(args["name"], "Alice");
        assert_eq!(args["age"], "30");
        assert_eq!(args["active"], "true");
        assert_eq!(args["note"], "");
    }

    // ── ReflectionOutput / ProactiveOutput serde ───────────────────────────

    #[test]
    fn reflection_output_default() {
        let r = ReflectionOutput::default();
        assert!(r.beliefs.is_empty());
        assert!(r.reflections.is_empty());
    }

    #[test]
    fn proactive_output_default() {
        let p = ProactiveOutput::default();
        assert!(p.action.is_none());
        assert!(p.message.is_none());
        assert!(p.urgency.is_none());
    }

    #[test]
    fn reflection_belief_default_confidence() {
        let json = r#"{"claim":"test"}"#;
        let belief: ReflectionBelief = serde_json::from_str(json).unwrap();
        assert_eq!(belief.claim, "test");
        assert!((belief.confidence - 0.65).abs() < f32::EPSILON);
    }

    #[test]
    fn llm_tool_call_serde_roundtrip() {
        let call = LlmToolCall {
            tool: "read_file".into(),
            args: HashMap::from([("path".into(), json!("test.txt"))]),
        };
        let json = serde_json::to_string(&call).unwrap();
        let back: LlmToolCall = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tool, "read_file");
        assert_eq!(back.args["path"], json!("test.txt"));
    }

    #[test]
    fn llm_tool_call_missing_args_defaults_to_empty() {
        let json = r#"{"tool":"test"}"#;
        let call: LlmToolCall = serde_json::from_str(json).unwrap();
        assert_eq!(call.tool, "test");
        assert!(call.args.is_empty());
    }
}
