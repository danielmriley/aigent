//! External thinking loop.
//!
//! When `config.agent.external_thinking` is enabled, the model outputs
//! structured JSON steps (`{"type":"tool_call", ...}` or
//! `{"type":"final_answer", ...}`) as plain text — not via native tool
//! calling schemas.
//!
//! This module intercepts the token stream, buffers JSON objects using
//! the brace-counting state machine in [`crate::json_stream`], and:
//!
//! - Emits `ThinkerEvent::AgentThought` for the `thought` field.
//! - For `final_answer`: forwards only the clean answer text to the user.
//! - For `tool_call`: executes the tool and loops back for another step.
//!
//! The token stream to the user (`user_token_tx`) receives **only** clean
//! text — never raw JSON.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use aigent_exec::ToolExecutor;
use aigent_llm::{ChatMessage, LlmRouter, Provider};
use aigent_tools::ToolRegistry;

use crate::events::{ThinkerEvent, ToolCallInfo, ToolResult as ToolResultEvent};
use crate::json_stream::{AgentStep, JsonStreamBuffer};
use crate::tool_loop::{EventSink, ToolExecution, ToolLoopResult};

/// Maximum number of tool-call → observation → re-prompt rounds.
const MAX_EXT_ROUNDS: usize = 10;

/// Run the external thinking loop.
///
/// This replaces `run_tool_loop` when `external_thinking` is active.
/// The model is called in plain-text mode (no native tool schemas) and
/// its JSON output is intercepted, parsed, and routed cleanly.
#[allow(clippy::too_many_arguments)]
pub async fn run_external_thinking_loop(
    llm: &LlmRouter,
    primary: Provider,
    ollama_model: &str,
    openrouter_model: &str,
    messages: &mut Vec<ChatMessage>,
    tool_registry: &ToolRegistry,
    tool_executor: &ToolExecutor,
    user_token_tx: mpsc::Sender<String>,
    event_sink: Option<&EventSink>,
    step_timeout: Duration,
) -> Result<ToolLoopResult> {
    let mut all_executions: Vec<ToolExecution> = Vec::new();
    let final_provider = primary;

    for round in 0..MAX_EXT_ROUNDS {
        debug!(round, "external thinking loop iteration");

        // ── Call the LLM (streaming, no native tools) ────────────────────
        // We create a private channel to intercept the raw JSON tokens
        // before they reach the user.
        let (intercept_tx, mut intercept_rx) = mpsc::channel::<String>(256);

        let llm_fut = llm.chat_messages_stream(
            primary,
            ollama_model,
            openrouter_model,
            messages,
            None,   // no native tool schemas — the JSON prompt handles dispatch
            intercept_tx,
            false,
            false,
        );

        // Accumulate the raw JSON from the intercept channel.
        let accumulator = tokio::spawn(async move {
            let mut jsb = JsonStreamBuffer::new();
            let mut raw_all = String::new();
            while let Some(chunk) = intercept_rx.recv().await {
                raw_all.push_str(&chunk);
                jsb.feed(&chunk);
                // As soon as we have a complete object, stop consuming.
                // (The LLM may continue sending whitespace/newlines, but
                // we already have what we need.)
                if jsb.has_complete() {
                    // Drain remaining tokens so the sender doesn't block.
                    while intercept_rx.try_recv().is_ok() {}
                    break;
                }
            }
            (jsb, raw_all)
        });

        // Wait for LLM to finish (or timeout).
        let response = match tokio::time::timeout(step_timeout, llm_fut).await {
            Ok(result) => result?,
            Err(_) => {
                warn!(round, timeout_secs = step_timeout.as_secs(),
                      "external thinking LLM call timed out");
                bail!("external thinking LLM call timed out after {}s", step_timeout.as_secs());
            }
        };

        // Wait for the accumulator task.
        let (mut jsb, raw_all) = accumulator.await
            .map_err(|e| anyhow::anyhow!("accumulator task panicked: {e}"))?;

        // Also feed the final `response.content` in case the LLM returned
        // content outside the streaming path (some providers do this).
        if !response.content.is_empty() && raw_all.is_empty() {
            jsb.feed(&response.content);
        }

        // ── Parse the accumulated JSON step ──────────────────────────────
        let step = match jsb.take_parsed() {
            Some(Ok(s)) => s,
            Some(Err(e)) => {
                warn!(round, error = %e, "failed to parse agent step, treating as final answer");
                // Fallback: send the raw content as-is so the user sees something.
                let fallback = if raw_all.is_empty() { response.content.clone() } else { raw_all };
                let _ = user_token_tx.send(fallback.clone()).await;
                return Ok(ToolLoopResult {
                    provider: final_provider,
                    content: fallback,
                    tool_executions: all_executions,
                });
            }
            None => {
                // No complete JSON object — treat whatever we got as plain text.
                warn!(round, "no complete JSON object in response, treating as plain text");
                let fallback = if raw_all.is_empty() { response.content.clone() } else { raw_all };
                let _ = user_token_tx.send(fallback.clone()).await;
                return Ok(ToolLoopResult {
                    provider: final_provider,
                    content: fallback,
                    tool_executions: all_executions,
                });
            }
        };

        // ── Dispatch based on step type ──────────────────────────────────
        match step {
            AgentStep::FinalAnswer { thought, answer } => {
                // Emit thought for UI display.
                if !thought.is_empty() {
                    if let Some(sink) = event_sink {
                        sink(ThinkerEvent::AgentThought(thought));
                    }
                }
                // Stream the clean answer to the user.
                let _ = user_token_tx.send(answer.clone()).await;

                // Record the assistant's JSON reply so the conversation
                // history stays consistent for the LLM.
                messages.push(ChatMessage::assistant(&raw_all));

                return Ok(ToolLoopResult {
                    provider: final_provider,
                    content: answer,
                    tool_executions: all_executions,
                });
            }

            AgentStep::ToolCall { thought, tool_name, args } => {
                info!(round, tool = %tool_name, "external thinking: tool call");

                // Emit thought.
                if !thought.is_empty() {
                    if let Some(sink) = event_sink {
                        sink(ThinkerEvent::AgentThought(thought));
                    }
                }

                // Emit ToolCallStart event.
                let call_info = ToolCallInfo {
                    name: tool_name.clone(),
                    args: serde_json::Value::Object(args.clone()).to_string(),
                };
                if let Some(sink) = event_sink {
                    sink(ThinkerEvent::ToolCallStart(call_info));
                }

                // Convert args to string map for the executor.
                let string_args: HashMap<String, String> = args
                    .iter()
                    .map(|(k, v)| {
                        let s = match v {
                            serde_json::Value::String(s) => s.clone(),
                            other => other.to_string(),
                        };
                        (k.clone(), s)
                    })
                    .collect();

                // Execute the tool.
                let start = Instant::now();
                let result = tool_executor.execute(tool_registry, &tool_name, &string_args).await;
                let duration_ms = start.elapsed().as_millis() as u64;
                let (success, output) = match result {
                    Ok(ref o) => (o.success, o.output.clone()),
                    Err(ref e) => (false, e.to_string()),
                };

                // Emit ToolCallEnd event.
                let result_event = ToolResultEvent {
                    name: tool_name.clone(),
                    success,
                    output: output.clone(),
                    duration_ms,
                };
                if let Some(sink) = event_sink {
                    sink(ThinkerEvent::ToolCallEnd(result_event));
                }

                // Record execution.
                all_executions.push(ToolExecution {
                    tool_name: tool_name.clone(),
                    args: args.into_iter().map(|(k, v)| (k, v)).collect(),
                    success,
                    output: output.clone(),
                    duration_ms,
                });

                // Append the assistant JSON reply + tool observation to
                // the conversation so the model can reason about the result.
                messages.push(ChatMessage::assistant(&raw_all));

                // Truncate very long tool outputs to avoid blowing the
                // context window.
                let obs = if output.len() > 4000 {
                    format!("{}... (truncated, {} total chars)", &output[..4000], output.len())
                } else {
                    output
                };
                let observation_msg = format!(
                    "TOOL RESULT for {tool_name}:\n{obs}"
                );
                messages.push(ChatMessage::user(&observation_msg));
            }
        }
    }

    warn!("external thinking loop exhausted {MAX_EXT_ROUNDS} rounds");

    // Fallback: compose something from tool results.
    let fallback = if all_executions.is_empty() {
        "I was unable to complete the request within the allowed number of steps.".to_string()
    } else {
        all_executions
            .iter()
            .map(|e| format!("[{}]: {}", e.tool_name, &e.output[..e.output.len().min(500)]))
            .collect::<Vec<_>>()
            .join("\n\n")
    };
    let _ = user_token_tx.send(fallback.clone()).await;

    Ok(ToolLoopResult {
        provider: final_provider,
        content: fallback,
        tool_executions: all_executions,
    })
}
