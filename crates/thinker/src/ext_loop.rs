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

        if full_text.trim().is_empty() {
            warn!(round, "ext_loop: LLM returned empty/whitespace-only response");
            let fallback = "I experienced a sudden processing failure and returned an empty response. Please try again.".to_string();
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
                // Guard against blank/whitespace-only results that would
                // render as an empty response in the TUI.
                let stripped = strip_json_wrapper(&full_text);
                let fallback = if stripped.trim().is_empty() {
                    "I encountered an internal formatting error while thinking.".to_string()
                } else {
                    stripped
                };
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
                let fallback = if full_text.trim().is_empty() {
                    "I wasn't able to produce a response. Please try again.".to_string()
                } else {
                    full_text.clone()
                };
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

                // ── Hallucinated tool guard ─────────────────────────────
                //
                // If the model invents a tool name that doesn't exist in the
                // registry, we feed it an error message and let it self-correct
                // on the next round (e.g. switch to web_search).  We do NOT
                // force a final_answer — the model may recover productively.
                let tool_handle = match tool_registry.get(&tool_name) {
                    Some(t) => t,
                    None => {
                        warn!(round, tool = %tool_name, "model hallucinated non-existent tool");

                        if let Some(sink) = event_sink {
                            sink(ThinkerEvent::AgentThought(
                                format!("I tried to use a tool that doesn't exist: {tool_name}"),
                            ));
                        }

                        // Record the assistant's attempt in conversation history.
                        messages.push(ChatMessage::assistant(&full_text));

                        // Feed a clear error so the model knows to pick a real tool.
                        let error_obs = format!(
                            "ERROR: Tool '{tool_name}' does not exist. \
                             Available tools are listed in the system prompt. \
                             Use only those tools (e.g. web_search, browse_page, run_shell). \
                             Respond with EXACTLY ONE JSON object."
                        );
                        messages.push(ChatMessage::user(&error_obs));

                        // Continue to next round — let the model self-correct.
                        continue;
                    }
                };

                // ── Required-parameter validation ───────────────────────
                //
                // Check that all required params are present in the args map.
                // Missing required args cause runtime failures that the model
                // often cannot recover from (the tool executor errors out with
                // a cryptic message).  By catching this early, we feed a clear
                // error that lets the model self-correct on the next round.
                {
                    let spec = tool_handle.spec();
                    let missing: Vec<&str> = spec.params.iter()
                        .filter(|p| p.required && !args.contains_key(&p.name))
                        .map(|p| p.name.as_str())
                        .collect();

                    if !missing.is_empty() {
                        warn!(round, tool = %tool_name, ?missing,
                              "model omitted required tool parameters");

                        if let Some(sink) = event_sink {
                            sink(ThinkerEvent::AgentThought(format!(
                                "Tool '{tool_name}' called without required args: {}",
                                missing.join(", "),
                            )));
                        }

                        messages.push(ChatMessage::assistant(&full_text));

                        let params_help: Vec<String> = spec.params.iter()
                            .filter(|p| p.required)
                            .map(|p| format!("\"{}\" ({})", p.name, p.description))
                            .collect();
                        let error_obs = format!(
                            "ERROR: Tool '{tool_name}' is missing required parameter(s): {missing}. \
                             Required params for {tool_name}: {params_help}. \
                             Re-call the tool with ALL required parameters populated. \
                             Respond with EXACTLY ONE JSON object.",
                            missing = missing.join(", "),
                            params_help = params_help.join(", "),
                        );
                        messages.push(ChatMessage::user(&error_obs));

                        continue;
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
                // context window.  Use a generous limit so the model can
                // see past large navigation headers on heavy sites.
                let obs = if output.len() > 12_000 {
                    format!("{}... (truncated, {} total chars)", &output[..12_000], output.len())
                } else {
                    output
                };
                let observation_msg = format!(
                    "TOOL RESULT for {tool_name}:\n\
                     {obs}\n\n\
                     Respond with EXACTLY ONE JSON object. \
                     Either {{\"type\":\"final_answer\",...}} or {{\"type\":\"tool_call\",...}}. \
                     No prose outside JSON."
                );
                messages.push(ChatMessage::user(&observation_msg));

                // Continue to the next round — the model will see the tool
                // result and decide whether to use another tool or produce
                // a final answer.
            }
        }
    }

    warn!("external thinking loop exhausted {MAX_EXT_ROUNDS} rounds");

    // Fallback: always send a clean, human-readable message.
    // Never dump raw tool output (HTML, JSON fragments, etc.) to the user —
    // it often renders as blank or garbled in the TUI.
    let fallback = if all_executions.is_empty() {
        "I was unable to complete the request within the allowed number of steps.".to_string()
    } else {
        "I was unable to formulate a complete answer within the allowed steps,          but I have gathered some tool data in the background. Please try again          or rephrase your question.".to_string()
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
