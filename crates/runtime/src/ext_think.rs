//! Externalized JSON reasoning loop.
//!
//! When `config.agent.external_thinking` is `true`, this module replaces the
//! normal streaming path.  Instead of letting the LLM think internally (which
//! causes long monologues and timeouts with large local models like Qwen 3.5
//! 35B MoE), we force the model to output short JSON steps and the *Rust code*
//! becomes the thinker — parsing, executing tools, feeding observations back,
//! and deciding when to stop.
//!
//! The JSON schema is:
//!
//! ```json
//! {
//!   "type": "tool_call" | "final_answer",
//!   "thought": "brief 1-sentence reasoning",
//!   "tool_call": { "name": "tool_name", "args": { ... } } | null,
//!   "final_answer": "message to user" | null
//! }
//! ```
//!
//! This is Phase 1 of Externalized Reasoning.  It is entirely backward-
//! compatible — when `external_thinking = false` (the default), no code in
//! this module is reached.

use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use tokio::sync::{broadcast, mpsc};
use tracing::{debug, info, warn};

use aigent_exec::ToolExecutor;
use aigent_llm::{ChatMessage, LlmRouter, Provider};
use aigent_tools::ToolRegistry;

use crate::agent_loop::LlmToolCall;
use crate::tool_loop::ToolExecution;
use crate::events::{BackendEvent, ToolCallInfo, ToolResult as ToolResultEvent};

// ── AgentStep: the structured JSON schema the model must produce ─────────────

/// A single reasoning step produced by the LLM in external thinking mode.
///
/// Deserialized from the model's JSON output.  The `type` field is used as
/// the serde tag to distinguish variants.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentStep {
    /// The model wants to invoke a tool before answering.
    ToolCall {
        thought: String,
        tool_call: ExtToolCall,
    },
    /// The model is ready to deliver its response to the user.
    FinalAnswer {
        thought: String,
        final_answer: String,
    },
}

/// Tool call payload inside an [`AgentStep::ToolCall`].
///
/// Intentionally mirrors [`LlmToolCall`] but with its own serde derivation
/// so we can deserialize directly from the model's JSON.
#[derive(Debug, Clone, Deserialize)]
pub struct ExtToolCall {
    pub name: String,
    #[serde(default)]
    pub args: HashMap<String, serde_json::Value>,
}

impl From<ExtToolCall> for LlmToolCall {
    fn from(tc: ExtToolCall) -> Self {
        Self {
            tool: tc.name,
            args: tc.args,
        }
    }
}

// ── Result type ──────────────────────────────────────────────────────────────

/// Outcome of the external thinking loop.
#[derive(Debug, Clone)]
pub struct ExtThinkResult {
    /// The final answer text to show the user.
    pub content: String,
    /// Total reasoning steps consumed.
    pub steps: usize,
    /// Tool executions performed during reasoning (for procedural memory).
    pub tool_executions: Vec<ToolExecution>,
}

// ── Configuration (read from AppConfig at call site) ─────────────────────────

/// Runtime parameters for the external thinking loop, extracted from config
/// so we don't need a reference to AppConfig inside the loop.
#[derive(Debug, Clone)]
pub struct ExtThinkConfig {
    pub step_timeout: Duration,
    pub max_steps: usize,
}

// ── Core loop ────────────────────────────────────────────────────────────────

/// Maximum retries per step on timeout or parse error.
const MAX_RETRIES_PER_STEP: usize = 2;

/// Run the externalized JSON reasoning loop.
///
/// This function drives a Think→Act→Observe cycle entirely in Rust:
///
/// 1. Send the current message history to the LLM (without streaming — we
///    need the full JSON output before we can parse it).
/// 2. Parse the JSON into an [`AgentStep`].
/// 3. If `ToolCall`: execute the tool, append the observation, loop.
/// 4. If `FinalAnswer`: stream the answer to the user and return.
/// 5. On timeout or parse error: inject an error observation and retry.
///
/// The `thought` field from each step is emitted via `BackendEvent::AgentThought`
/// so the TUI can display it in real-time.
pub async fn run_external_thinking_loop(
    config: &ExtThinkConfig,
    llm: &LlmRouter,
    primary: Provider,
    ollama_model: &str,
    openrouter_model: &str,
    messages: &mut Vec<ChatMessage>,
    tool_registry: &ToolRegistry,
    tool_executor: &ToolExecutor,
    token_tx: mpsc::Sender<String>,
    event_tx: Option<&broadcast::Sender<BackendEvent>>,
) -> Result<ExtThinkResult> {
    let mut steps = 0usize;
    let mut tool_executions: Vec<ToolExecution> = Vec::new();

    loop {
        if steps >= config.max_steps {
            warn!(steps, max = config.max_steps, "external thinking: hit step limit, forcing answer");
            // Ask the model to produce a final answer now.
            messages.push(ChatMessage::user(
                "You have reached the maximum number of reasoning steps. \
                 You MUST respond with a final_answer JSON now.",
            ));
        }

        // ── 1. Call the LLM with timeout ─────────────────────────────────
        let raw_output = call_llm_with_timeout(
            config,
            llm,
            primary,
            ollama_model,
            openrouter_model,
            messages,
        )
        .await;

        let raw = match raw_output {
            Ok(text) => text,
            Err(err) => {
                warn!(?err, step = steps, "external thinking: LLM call failed");
                // Inject error observation and retry.
                let retry_msg = format!(
                    "Step timed out or LLM error: {err}. Try again with a valid JSON response.",
                );
                messages.push(ChatMessage::user(&retry_msg));
                steps += 1;
                if steps >= config.max_steps + MAX_RETRIES_PER_STEP {
                    bail!("external thinking: exhausted all retries after {steps} steps");
                }
                continue;
            }
        };

        debug!(step = steps, raw_len = raw.len(), "external thinking: got LLM output");

        // ── 2. Parse the JSON ────────────────────────────────────────────
        let step = match parse_agent_step(&raw) {
            Ok(s) => s,
            Err(err) => {
                warn!(?err, step = steps, "external thinking: JSON parse failed");
                let retry_msg = format!(
                    "Invalid JSON response: {err}. Respond with ONLY valid JSON matching the schema.",
                );
                messages.push(ChatMessage::user(&retry_msg));
                steps += 1;
                if steps >= config.max_steps + MAX_RETRIES_PER_STEP {
                    bail!("external thinking: exhausted retries after {steps} parse failures");
                }
                continue;
            }
        };

        steps += 1;

        match step {
            // ── 3. Tool call ─────────────────────────────────────────────
            AgentStep::ToolCall { thought, tool_call } => {
                // Emit thought to UI.
                emit_thought(event_tx, &thought);

                info!(
                    step = steps,
                    thought = %thought,
                    tool = %tool_call.name,
                    "external thinking: tool_call"
                );

                // Emit tool start.
                if let Some(tx) = event_tx {
                    let _ = tx.send(BackendEvent::ToolCallStart(ToolCallInfo {
                        name: tool_call.name.clone(),
                        args: serde_json::to_string(&tool_call.args).unwrap_or_default(),
                    }));
                }

                // Execute the tool.
                let llm_call: LlmToolCall = tool_call.clone().into();
                let str_args = llm_call.stringify_args();
                let start = std::time::Instant::now();
                let result = tool_executor
                    .execute(tool_registry, &tool_call.name, &str_args)
                    .await;
                let duration_ms = start.elapsed().as_millis() as u64;

                let (success, output) = match result {
                    Ok(out) => (out.success, out.output),
                    Err(err) => (false, format!("tool error: {err}")),
                };

                // Track for procedural memory.
                tool_executions.push(ToolExecution {
                    tool_name: tool_call.name.clone(),
                    args: tool_call.args.clone(),
                    success,
                    output: output.clone(),
                    duration_ms,
                });

                // Emit tool end.
                if let Some(tx) = event_tx {
                    let _ = tx.send(BackendEvent::ToolCallEnd(ToolResultEvent {
                        name: tool_call.name.clone(),
                        success,
                        output: crate::prompt_builder::truncate_for_prompt(&output, 500),
                        duration_ms,
                    }));
                }

                // Record assistant message (the JSON we got).
                messages.push(ChatMessage::assistant(&raw));

                // Build observation and feed it back.
                let obs = format!(
                    "Tool `{}` returned (success={success}):\n{output}",
                    tool_call.name,
                );
                let truncated_obs = crate::prompt_builder::truncate_for_prompt(&obs, 2048);
                messages.push(ChatMessage::user(&truncated_obs));
            }

            // ── 4. Final answer ──────────────────────────────────────────
            AgentStep::FinalAnswer { thought, final_answer } => {
                emit_thought(event_tx, &thought);

                info!(
                    step = steps,
                    thought = %thought,
                    answer_len = final_answer.len(),
                    "external thinking: final_answer"
                );

                // Stream the answer to the UI.
                let _ = token_tx.send(final_answer.clone()).await;

                return Ok(ExtThinkResult {
                    content: final_answer,
                    steps,
                    tool_executions,
                });
            }
        }

        // Safety check after max_steps + retries.
        if steps > config.max_steps + MAX_RETRIES_PER_STEP {
            break;
        }
    }

    // If we fell through, produce a fallback.
    bail!("external thinking: loop terminated without final answer after {steps} steps")
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Call the LLM and collect the full (non-streaming) response with a timeout.
async fn call_llm_with_timeout(
    config: &ExtThinkConfig,
    llm: &LlmRouter,
    primary: Provider,
    ollama_model: &str,
    openrouter_model: &str,
    messages: &[ChatMessage],
) -> Result<String> {
    // We use a non-streaming call since we need the complete JSON before parsing.
    // Create a dummy channel to collect tokens into a buffer.
    let (tx, mut rx) = mpsc::channel::<String>(256);
    let collect_handle = tokio::spawn(async move {
        let mut buf = String::new();
        while let Some(chunk) = rx.recv().await {
            buf.push_str(&chunk);
        }
        buf
    });

    let mut msgs = messages.to_vec();
    let fut = llm.chat_messages_stream(
        primary,
        ollama_model,
        openrouter_model,
        &mut msgs,
        None, // No tools schema — we use our JSON schema instead.
        tx,
    );

    let response = tokio::time::timeout(config.step_timeout, fut)
        .await
        .context("step timed out")?
        .context("LLM call failed")?;

    // Always await the collector task to avoid a dangling spawn.
    let collected = collect_handle.await.unwrap_or_default();

    // Prefer the response content; fall back to collected stream tokens.
    let text = if !response.content.is_empty() {
        response.content
    } else {
        collected
    };

    if text.trim().is_empty() {
        bail!("LLM returned empty response");
    }

    Ok(text)
}

/// Parse an [`AgentStep`] from raw LLM output.
///
/// Handles common model quirks:
/// - Leading/trailing whitespace
/// - ```json wrapper blocks
/// - Stray text before/after the JSON object
pub fn parse_agent_step(raw: &str) -> Result<AgentStep> {
    let trimmed = raw.trim();

    // Strip markdown code fences.
    let json_str = if trimmed.starts_with("```") {
        let inner = trimmed
            .strip_prefix("```json")
            .or_else(|| trimmed.strip_prefix("```"))
            .unwrap_or(trimmed);
        inner
            .strip_suffix("```")
            .unwrap_or(inner)
            .trim()
    } else {
        trimmed
    };

    // Try to extract the JSON object if there's surrounding text.
    let json_str = extract_json_object(json_str).unwrap_or(json_str);

    serde_json::from_str(json_str)
        .with_context(|| format!("failed to parse AgentStep from: {}", &json_str[..json_str.len().min(200)]))
}

/// Extract the first `{...}` JSON object from a string, handling nested braces.
fn extract_json_object(s: &str) -> Option<&str> {
    let start = s.find('{')?;
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;

    for (i, ch) in s[start..].char_indices() {
        if escape {
            escape = false;
            continue;
        }
        match ch {
            '\\' if in_string => escape = true,
            '"' => in_string = !in_string,
            '{' if !in_string => depth += 1,
            '}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return Some(&s[start..start + i + 1]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Emit an `AgentThought` backend event.
fn emit_thought(
    event_tx: Option<&broadcast::Sender<BackendEvent>>,
    thought: &str,
) {
    if let Some(tx) = event_tx {
        let _ = tx.send(BackendEvent::AgentThought(thought.to_string()));
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_tool_call_step() {
        let json = r#"{
            "type": "tool_call",
            "thought": "I need to read the config file",
            "tool_call": { "name": "read_file", "args": { "path": "/etc/hosts" } },
            "final_answer": null
        }"#;
        let step = parse_agent_step(json).unwrap();
        match step {
            AgentStep::ToolCall { thought, tool_call } => {
                assert_eq!(thought, "I need to read the config file");
                assert_eq!(tool_call.name, "read_file");
                assert_eq!(
                    tool_call.args.get("path").and_then(|v| v.as_str()),
                    Some("/etc/hosts")
                );
            }
            _ => panic!("expected ToolCall variant"),
        }
    }

    #[test]
    fn parse_final_answer_step() {
        let json = r#"{
            "type": "final_answer",
            "thought": "I have the answer now",
            "tool_call": null,
            "final_answer": "The file contains localhost entries."
        }"#;
        let step = parse_agent_step(json).unwrap();
        match step {
            AgentStep::FinalAnswer { thought, final_answer } => {
                assert_eq!(thought, "I have the answer now");
                assert_eq!(final_answer, "The file contains localhost entries.");
            }
            _ => panic!("expected FinalAnswer variant"),
        }
    }

    #[test]
    fn parse_with_markdown_fence() {
        let json = "```json\n{\"type\":\"final_answer\",\"thought\":\"done\",\"final_answer\":\"hello\"}\n```";
        let step = parse_agent_step(json).unwrap();
        assert!(matches!(step, AgentStep::FinalAnswer { .. }));
    }

    #[test]
    fn parse_with_surrounding_text() {
        let json = "Sure! Here is my response:\n{\"type\":\"final_answer\",\"thought\":\"ok\",\"final_answer\":\"hi\"}\nDone.";
        let step = parse_agent_step(json).unwrap();
        assert!(matches!(step, AgentStep::FinalAnswer { .. }));
    }

    #[test]
    fn parse_rejects_invalid_type() {
        let json = r#"{"type": "unknown", "thought": "hmm"}"#;
        assert!(parse_agent_step(json).is_err());
    }

    #[test]
    fn parse_rejects_garbage() {
        assert!(parse_agent_step("not json at all").is_err());
    }

    #[test]
    fn extract_json_object_nested() {
        let s = r#"text before {"key": {"inner": 1}} text after"#;
        assert_eq!(
            extract_json_object(s),
            Some(r#"{"key": {"inner": 1}}"#)
        );
    }

    #[test]
    fn extract_json_object_with_string_braces() {
        let s = r#"{"key": "value with { braces }"}"#;
        assert_eq!(extract_json_object(s), Some(s));
    }

    #[test]
    fn ext_tool_call_to_llm_tool_call() {
        let tc = ExtToolCall {
            name: "read_file".to_string(),
            args: HashMap::from([("path".to_string(), serde_json::json!("/tmp/test"))]),
        };
        let llm_tc: LlmToolCall = tc.into();
        assert_eq!(llm_tc.tool, "read_file");
        assert_eq!(
            llm_tc.args.get("path").and_then(|v| v.as_str()),
            Some("/tmp/test")
        );
    }
}
