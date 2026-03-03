//! Types shared across the unified agent loop (reflection, proactive, turn source).

use std::collections::HashMap;

use std::fmt;
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



// ── ReAct state machine ──────────────────────────────────────────────────────

/// Phase within a single ReAct iteration.
///
/// The canonical loop is: **Think → Act → Observe → Critique → (iterate or finish)**.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReactPhase {
    /// The agent reasons about the current context and decides next steps.
    Think,
    /// The agent executes one or more tool calls (or sends a sub-agent task).
    Act,
    /// The agent consumes the results of the action.
    Observe,
    /// The agent evaluates result quality and decides whether to iterate.
    Critique,
    /// Terminal state — the loop has produced a final answer.
    Done,
}

impl fmt::Display for ReactPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReactPhase::Think => write!(f, "think"),
            ReactPhase::Act => write!(f, "act"),
            ReactPhase::Observe => write!(f, "observe"),
            ReactPhase::Critique => write!(f, "critique"),
            ReactPhase::Done => write!(f, "done"),
        }
    }
}

/// Role of a sub-agent within a swarm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SwarmRole {
    /// Decomposes complex goals into sub-tasks and assigns them.
    Planner,
    /// Executes tool calls and manages workspace mutations.
    Executor,
    /// Validates outputs against acceptance criteria.
    Verifier,
    /// Gathers information from memory, web, or code search.
    Researcher,
    /// Top-level coordinator that routes between sub-agents.
    Supervisor,
}

impl fmt::Display for SwarmRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SwarmRole::Planner => write!(f, "planner"),
            SwarmRole::Executor => write!(f, "executor"),
            SwarmRole::Verifier => write!(f, "verifier"),
            SwarmRole::Researcher => write!(f, "researcher"),
            SwarmRole::Supervisor => write!(f, "supervisor"),
        }
    }
}

/// A scored evaluation of a completed task or sub-task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalScore {
    /// Numeric quality score (0.0 = failure, 1.0 = perfect).
    pub score: f32,
    /// Short explanation of the score.
    pub rationale: String,
    /// Whether the result is good enough to accept without human review.
    pub accepted: bool,
}

/// Snapshot of the ReAct loop's internal state, useful for TUI rendering
/// and for persisting to the Procedural memory tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactSnapshot {
    pub phase: ReactPhase,
    pub round: usize,
    pub max_rounds: usize,
    /// The current "thought" text (populated in Think phase).
    pub thought: Option<String>,
    /// Tool calls planned or just executed (populated in Act phase).
    pub actions: Vec<LlmToolCall>,
    /// Observation summary (populated in Observe phase).
    pub observation: Option<String>,
    /// Critique / self-evaluation (populated in Critique phase).
    pub critique: Option<String>,
    /// Final answer when Done.
    pub answer: Option<String>,
}

/// The `AgentLoop` hosts the ReAct state machine and multi-agent swarm
/// orchestration.  It is instantiated per-turn (or per-task) by the daemon
/// connection handler and driven to completion.
pub struct AgentLoop {
    /// Current ReAct phase.
    pub phase: ReactPhase,
    /// Current iteration within the ReAct loop (0-indexed).
    pub round: usize,
    /// Maximum rounds before forcing a final answer.
    pub max_rounds: usize,
    /// Accumulated thoughts, actions, observations, and critiques.
    pub history: Vec<ReactSnapshot>,
    /// Role when this loop is running as a sub-agent inside a swarm.
    pub role: Option<SwarmRole>,
}

impl Default for AgentLoop {
    fn default() -> Self {
        Self::new(8)
    }
}

impl AgentLoop {
    pub fn new(max_rounds: usize) -> Self {
        Self {
            phase: ReactPhase::Think,
            round: 0,
            max_rounds,
            history: Vec::new(),
            role: None,
        }
    }

    /// Create a sub-agent loop with a specific swarm role.
    pub fn with_role(max_rounds: usize, role: SwarmRole) -> Self {
        Self {
            phase: ReactPhase::Think,
            round: 0,
            max_rounds,
            history: Vec::new(),
            role: Some(role),
        }
    }

    /// Advance the phase to the next step in the ReAct cycle.
    pub fn advance(&mut self) -> ReactPhase {
        self.phase = match self.phase {
            ReactPhase::Think => ReactPhase::Act,
            ReactPhase::Act => ReactPhase::Observe,
            ReactPhase::Observe => ReactPhase::Critique,
            ReactPhase::Critique => {
                self.round += 1;
                if self.round >= self.max_rounds {
                    ReactPhase::Done
                } else {
                    ReactPhase::Think
                }
            }
            ReactPhase::Done => ReactPhase::Done,
        };
        self.phase
    }

    /// Force the loop to completion.
    pub fn finish(&mut self) {
        self.phase = ReactPhase::Done;
    }

    /// Whether the loop has terminated.
    pub fn is_done(&self) -> bool {
        self.phase == ReactPhase::Done
    }

    /// Record a snapshot of the current round state.
    pub fn record_snapshot(&mut self, snapshot: ReactSnapshot) {
        self.history.push(snapshot);
    }

    /// Score the overall outcome of the agent loop based on its history.
    ///
    /// Heuristic scoring (no LLM call):
    /// - Penalises loops that hit max_rounds without finishing
    /// - Rewards loops that produced a final answer
    /// - Penalises high error rates in tool calls
    /// - Returns an `EvalScore` with the computed score and rationale
    pub fn score_outcome(&self) -> EvalScore {
        let mut score: f32 = 0.5; // base
        let mut reasons = Vec::new();

        // Completed before max rounds?
        let finished_cleanly = self.phase == ReactPhase::Done && self.round < self.max_rounds;
        if finished_cleanly {
            score += 0.2;
            reasons.push("completed before max rounds".to_string());
        } else if self.phase == ReactPhase::Done {
            score += 0.1;
            reasons.push("completed at max rounds".to_string());
        } else {
            score -= 0.2;
            reasons.push("did not reach Done phase".to_string());
        }

        // Has a final answer?
        let has_answer = self.history.last().is_some_and(|s| s.answer.is_some());
        if has_answer {
            score += 0.15;
            reasons.push("produced final answer".to_string());
        }

        // Critique quality (if any critiques exist)
        let critique_count = self.history.iter().filter(|s| s.critique.is_some()).count();
        if critique_count > 0 {
            score += 0.05;
            reasons.push(format!("{critique_count} self-critiques performed"));
        }

        // Round efficiency: fewer rounds = better
        if self.max_rounds > 0 {
            let efficiency = 1.0 - (self.round as f32 / self.max_rounds as f32);
            let bonus = efficiency * 0.1;
            score += bonus;
            reasons.push(format!("efficiency bonus: {bonus:.2}"));
        }

        score = score.clamp(0.0, 1.0);
        let accepted = score >= 0.6;

        EvalScore {
            score,
            rationale: reasons.join("; "),
            accepted,
        }
    }
}

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
