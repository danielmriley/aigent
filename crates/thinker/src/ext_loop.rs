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
//!
//! ## Multi-round (ReAct) chaining
//!
//! The loop supports chained tool calls (e.g. `web_search` → `browse_page`
//! → `final_answer`).  Each iteration:
//!
//! 1. Calls the LLM with a **drain channel** (tokens are absorbed, not shown).
//! 2. Parses from `response.content` — the authoritative complete output
//!    accumulated by the LLM client — avoiding any streaming race conditions.
//! 3. Dispatches the parsed step (emit thought, execute tool, or return answer).
//! 4. Appends the assistant reply + tool observation to `messages` and loops.

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
        //
        // We supply a *drain* channel that silently absorbs the raw streamed
        // tokens.  We do NOT parse from the stream — instead we parse from
        // `response.content`, which the LLM client accumulates internally
        // and returns as one complete string.
        //
        // This avoids all race conditions between the spawned drain task and
        // the LLM future: there is no early-break, no partial `raw_all`,
        // and no risk of discarding buffered tokens.
        let (drain_tx, mut drain_rx) = mpsc::channel::<String>(512);

        // Spawn a lightweight task to consume tokens so `tx.send()` inside
        // the LLM never blocks.  The task ends naturally once the sender is
        // dropped (i.e. after llm_fut completes).
        let drain_handle = tokio::spawn(async move {
            while drain_rx.recv().await.is_some() {}
        });

        let llm_fut = llm.chat_messages_stream(
            primary,
            ollama_model,
            openrouter_model,
            messages,
            None,   // no native tool schemas — the JSON prompt handles dispatch
            drain_tx,
            false,
            false,
        );

        // Wait for the LLM to finish (or timeout).
        let response = match tokio::time::timeout(step_timeout, llm_fut).await {
            Ok(result) => result?,
            Err(_) => {
                warn!(round, timeout_secs = step_timeout.as_secs(),
                      "external thinking LLM call timed out");
                bail!("external thinking LLM call timed out after {}s", step_timeout.as_secs());
            }
        };

        // Ensure the drain task finishes (the sender was dropped when
        // llm_fut completed, so this is nearly instantaneous).
        let _ = drain_handle.await;

        // ── Parse from the authoritative response.content ────────────────
        //
        // `response.content` is the full text accumulated by the LLM client
        // during streaming.  Parsing from this single complete string avoids
        // every partial-data and race-condition bug that plagued the old
        // approach of parsing from the intercepted stream.
        let full_text = response.content.clone();
        debug!(round, len = full_text.len(), "ext_loop: LLM returned content");

        if full_text.is_empty() {
            warn!(round, "ext_loop: empty response from LLM");
            let fallback = "(no response from model)".to_string();
            let _ = user_token_tx.send(fallback.clone()).await;
            return Ok(ToolLoopResult {
                provider: final_provider,
                content: fallback,
                tool_executions: all_executions,
            });
        }

        // Feed the complete text into a fresh JsonStreamBuffer.
        let mut jsb = JsonStreamBuffer::new();
        jsb.feed(&full_text);

        let step = match jsb.take_parsed() {
            Some(Ok(s)) => s,
            Some(Err(e)) => {
                warn!(round, error = %e, "failed to parse agent step, treating as final answer");
                // Fallback: strip obvious JSON wrapper and send as-is.
                let fallback = strip_json_wrapper(&full_text);
                let _ = user_token_tx.send(fallback.clone()).await;
                return Ok(ToolLoopResult {
                    provider: final_provider,
                    content: fallback,
                    tool_executions: all_executions,
                });
            }
            None => {
                // No complete JSON object — the model produced plain text
                // (sometimes happens on easy questions).  Send it directly.
                warn!(round, "no complete JSON object in response, treating as plain text");
                let _ = user_token_tx.send(full_text.clone()).await;
                return Ok(ToolLoopResult {
                    provider: final_provider,
                    content: full_text,
                    tool_executions: all_executions,
                });
            }
        };

        // ── Dispatch based on step type ──────────────────────────────────
        match step {
            AgentStep::FinalAnswer { thought, answer } => {
                // Emit the model's chain-of-thought for UI display.
                if !thought.is_empty() {
                    if let Some(sink) = event_sink {
                        sink(ThinkerEvent::AgentThought(thought));
                    }
                }

                // Stream the clean answer to the user (no raw JSON).
                let _ = user_token_tx.send(answer.clone()).await;

                // Record the assistant's full reply for conversation history.
                messages.push(ChatMessage::assistant(&full_text));

                return Ok(ToolLoopResult {
                    provider: final_provider,
                    content: answer,
                    tool_executions: all_executions,
                });
            }

            AgentStep::ToolCall { thought, tool_name, args } => {
                info!(round, tool = %tool_name, "external thinking: tool call");

                // Emit the model's chain-of-thought.
                if !thought.is_empty() {
                    if let Some(sink) = event_sink {
                        sink(ThinkerEvent::AgentThought(thought));
                    }
                }

                // Emit ToolCallStart event for UI.
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

                // Emit ToolCallEnd event for UI.
                let result_event = ToolResultEvent {
                    name: tool_name.clone(),
                    success,
                    output: output.clone(),
                    duration_ms,
                };
                if let Some(sink) = event_sink {
                    sink(ThinkerEvent::ToolCallEnd(result_event));
                }

                // Record execution for the final result.
                all_executions.push(ToolExecution {
                    tool_name: tool_name.clone(),
                    args: args.into_iter().collect(),
                    success,
                    output: output.clone(),
                    duration_ms,
                });

                // ── Append to conversation history for the next round ────
                //
                // Use `full_text` (the complete LLM output) as the assistant
                // message — never a partial/truncated string.  This ensures
                // the model sees its own complete JSON on re-entry and can
                // reason coherently across multiple tool-call rounds.
                messages.push(ChatMessage::assistant(&full_text));

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

                // Continue to the next round — the model will see the tool
                // result and decide whether to use another tool or produce
                // a final answer.
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

/// Best-effort extraction of readable text from a malformed JSON response.
///
/// If the model produced valid JSON with a `final_answer` or `thought`
/// field but the step-type was unrecognised, pull those out.  Otherwise
/// return the input unchanged.
fn strip_json_wrapper(raw: &str) -> String {
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(raw.trim()) {
        if let Some(ans) = val.get("final_answer").and_then(|v| v.as_str()) {
            return ans.to_string();
        }
        if let Some(thought) = val.get("thought").and_then(|v| v.as_str()) {
            return thought.to_string();
        }
    }
    raw.to_string()
}
