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

/// Maximum consecutive parse/timeout retries before bailing.
const MAX_CONSECUTIVE_RETRIES: usize = 3;

/// Exponential backoff durations for retries (1s, 2s, 3s).
const RETRY_BACKOFF_MS: [u64; 3] = [1000, 2000, 3000];

/// Run the externalized JSON reasoning loop.
///
/// This function drives a Think→Act→Observe cycle entirely in Rust:
///
/// 1. Send the current message history to the LLM (with Ollama `format: "json"`
///    when the provider is Ollama so the model is forced into JSON mode at the
///    API level).
/// 2. Parse the JSON into an [`AgentStep`], with multi-layer fallback parsing.
/// 3. If `ToolCall`: execute the tool, append the observation, loop.
/// 4. If `FinalAnswer`: stream the answer to the user and return.
/// 5. On timeout or parse error: inject an error observation, back off, retry.
///
/// The `thought` field from each step is emitted via `BackendEvent::AgentThought`
/// so the TUI can display it in real-time.
#[allow(clippy::too_many_arguments)]
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
    let mut consecutive_retries = 0usize;
    let mut has_used_tool = false;
    let mut challenge_count = 0usize;

    loop {
        if steps >= config.max_steps {
            warn!(steps, max = config.max_steps, "external thinking: hit step limit, forcing answer");
            messages.push(ChatMessage::user(
                "You have reached the maximum number of reasoning steps. \
                 You MUST respond with a final_answer JSON now. Output ONLY: \
                 {\"type\":\"final_answer\",\"thought\":\"...\",\"final_answer\":\"...\"}",
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
            Ok(text) => {
                // Reset retry counter on successful LLM response.
                consecutive_retries = 0;
                text
            }
            Err(err) => {
                warn!(?err, step = steps, "external thinking: LLM call failed");
                consecutive_retries += 1;
                if consecutive_retries > MAX_CONSECUTIVE_RETRIES {
                    bail!(
                        "external thinking: exhausted {MAX_CONSECUTIVE_RETRIES} consecutive retries \
                         after {steps} steps (last error: {err})"
                    );
                }
                // Exponential backoff.
                let delay = RETRY_BACKOFF_MS
                    .get(consecutive_retries.saturating_sub(1))
                    .copied()
                    .unwrap_or(3000);
                tokio::time::sleep(Duration::from_millis(delay)).await;

                let retry_msg = format!(
                    "Step timed out or LLM error: {err}. \
                     Respond with ONLY the JSON object — no markdown, no extra text. Example: \
                     {{\"type\":\"final_answer\",\"thought\":\"...\",\"final_answer\":\"...\"}}"
                );
                messages.push(ChatMessage::user(&retry_msg));
                steps += 1;
                continue;
            }
        };

        debug!(step = steps, raw_len = raw.len(), "external thinking: got LLM output");

        // ── 2. Parse the JSON (multi-layer fallback) ─────────────────────
        let step = match parse_agent_step(&raw) {
            Ok(s) => {
                consecutive_retries = 0;
                s
            }
            Err(err) => {
                warn!(?err, step = steps, raw = %raw.chars().take(300).collect::<String>(),
                      "external thinking: JSON parse failed");
                consecutive_retries += 1;
                if consecutive_retries > MAX_CONSECUTIVE_RETRIES {
                    bail!(
                        "external thinking: exhausted {MAX_CONSECUTIVE_RETRIES} consecutive \
                         parse retries after {steps} steps (last error: {err})"
                    );
                }
                // Exponential backoff.
                let delay = RETRY_BACKOFF_MS
                    .get(consecutive_retries.saturating_sub(1))
                    .copied()
                    .unwrap_or(3000);
                tokio::time::sleep(Duration::from_millis(delay)).await;

                // Inject a very explicit correction message.
                let retry_msg = format!(
                    "Your last response was not valid JSON. Parse error: {err}\n\n\
                     You MUST respond with ONLY the JSON object below — no markdown fences, \
                     no explanation, no extra text:\n\
                     {{\"type\":\"final_answer\",\"thought\":\"brief reasoning\",\
                     \"final_answer\":\"your response to the user\"}}\n\n\
                     Or for a tool call:\n\
                     {{\"type\":\"tool_call\",\"thought\":\"brief reasoning\",\
                     \"tool_call\":{{\"name\":\"TOOL_NAME\",\"args\":{{...}}}}}}"
                );
                messages.push(ChatMessage::user(&retry_msg));
                steps += 1;
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

                has_used_tool = true;

                // Build observation and feed it back, with a nudge toward
                // final_answer if the tool succeeded — this cuts prefill
                // time because the model doesn't deliberate about next steps.
                let obs = if success {
                    format!(
                        "Tool `{}` returned (success=true):\n{output}\n\nNow give a final_answer with this result.",
                        tool_call.name,
                    )
                } else {
                    format!(
                        "Tool `{}` returned (success=false):\n{output}",
                        tool_call.name,
                    )
                };
                let truncated_obs = crate::prompt_builder::truncate_for_prompt(&obs, 4096);
                messages.push(ChatMessage::user(&truncated_obs));
            }

            // ── 4. Final answer ──────────────────────────────────────────
            AgentStep::FinalAnswer { thought, final_answer } => {
                // ── Self-check: force tool use before accepting final_answer ──
                // Three independent triggers can challenge the model:
                //  (a) Passive-resignation language ("I don't have access", etc.)
                //  (b) Hallucinated factual claims (dates, file paths) without
                //      having called any tool first
                //  (c) Suspiciously short thought with no tool used
                // We challenge up to MAX_CHALLENGES times to keep the loop
                // from cycling indefinitely.
                const MAX_CHALLENGES: usize = 3;

                let thought_lc = thought.to_lowercase();
                let answer_lc = final_answer.to_lowercase();

                // (a) Passive-resignation detector
                let passive_resignation = [
                    "do not have", "don't have", "i don't have", "i do not have",
                    "cannot access", "can't access", "cannot provide", "can't provide",
                    "i cannot provide", "unable to provide", "unable to access",
                    "i am ready", "ready to answer", "no access",
                    "no real-time", "real-time access", "real time access",
                    "as an ai", "as a language model", "placeholder",
                    "lack of real", "training data", "knowledge cutoff",
                    // Catch phrases the model invents to avoid tool use
                    "i lack", "lack a tool", "no tool", "no way to",
                    "not able to", "not possible to", "no means to",
                    "no capability", "no function", "don't know",
                    "do not know", "not equipped", "not designed to",
                ].iter().any(|p| thought_lc.contains(p));

                // (b) Hallucinated-fact detector — does the answer contain
                //     file paths that the model should have verified with a
                //     tool?  Date/time patterns are NOT flagged because
                //     CURRENT_DATETIME is now injected into the prompt.
                let has_file_path = answer_lc.contains("/home/")
                    || answer_lc.contains("/usr/")
                    || answer_lc.contains("/tmp/")
                    || answer_lc.contains("/etc/");
                let answer_has_factual_claims = has_file_path;

                // Trigger challenge if:
                //  - passive resignation detected (any step), OR
                //  - model hasn't used ANY tool and answer has factual claims, OR
                //  - model hasn't used ANY tool and thought is suspiciously short
                let should_challenge = challenge_count < MAX_CHALLENGES
                    && (passive_resignation
                        || (!has_used_tool && answer_has_factual_claims)
                        || (!has_used_tool && thought.len() < 40));

                if should_challenge {
                    let reason = if passive_resignation {
                        "passive resignation language"
                    } else if answer_has_factual_claims {
                        "factual claims without tool verification"
                    } else {
                        "suspiciously short reasoning without tool use"
                    };

                    warn!(
                        step = steps,
                        thought = %thought,
                        reason = reason,
                        "external thinking: challenging final_answer — {reason}"
                    );
                    emit_thought(event_tx, &thought);
                    messages.push(ChatMessage::assistant(&raw));

                    let override_msg = format!(
                        "SYSTEM OVERRIDE — TOOL MANDATORY (reason: {reason}):\n\
                         Your answer was rejected because: {reason}.\n\
                         You have NOT used any tool yet. That is WRONG.\n\n\
                         RULES:\n\
                         - If a tool can fetch/verify facts, you MUST call it.\n\
                         - Saying you lack real-time access when run_shell exists is a FAILURE.\n\
                         - File contents, system info, web lookups → ALWAYS use a tool.\n\
                         - For date/time, use CURRENT_DATETIME from the prompt.\n\n\
                         For file listing:\n\
                         {{\"type\":\"tool_call\",\"thought\":\"I will list files.\",\
                         \"tool_call\":{{\"name\":\"run_shell\",\"args\":{{\"command\":\"ls -la\"}}}}}}\n\n\
                         Output a tool_call JSON now — NOT a final_answer."
                    );
                    messages.push(ChatMessage::user(&override_msg));

                    challenge_count += 1;
                    steps += 1;
                    consecutive_retries = 0;
                    continue;
                }

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
                });
            }
        }

        // Safety check: hard cap at 2x max_steps.
        if steps > config.max_steps * 2 {
            break;
        }
    }

    // If we fell through, produce a fallback.
    bail!("external thinking: loop terminated without final answer after {steps} steps")
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Call the LLM and collect the full response with an **activity-based** timeout.
///
/// Instead of applying a hard cap on total response time (which fails for
/// large models with slow time-to-first-token), the timeout resets every time
/// the model produces a new token.  This accommodates:
///
/// - Slow model loading on first request (Ollama loads into VRAM lazily).
/// - Long prefill / TTFT for large models (35B+) with big system prompts.
/// - JSON-constrained decoding which can be slower than free-form generation.
///
/// The timeout only fires when the model goes **completely silent** for
/// `config.step_timeout` (default 60 s), which indicates a genuine hang.
///
/// When the provider is Ollama, enables `format: "json"` at the API level so
/// the model is constrained to valid JSON output.
async fn call_llm_with_timeout(
    config: &ExtThinkConfig,
    llm: &LlmRouter,
    primary: Provider,
    ollama_model: &str,
    openrouter_model: &str,
    messages: &[ChatMessage],
) -> Result<String> {
    let (tx, mut rx) = mpsc::channel::<String>(256);

    // Enable JSON mode for the request — this tells Ollama to add `format: "json"`
    // which constrains the model's output to valid JSON at the token-sampling level.
    let json_mode = true;

    let msgs = messages.to_vec();
    let llm_fut = llm.chat_messages_stream(
        primary,
        ollama_model,
        openrouter_model,
        &msgs,
        None, // No tools schema — we use our JSON schema instead.
        tx,
        json_mode,
        true, // Suppress native <think> reasoning — we drive thinking externally.
    );
    tokio::pin!(llm_fut);

    let mut buf = String::new();
    let mut llm_result: Option<Result<_>> = None;

    // Activity-based timeout loop: the countdown resets every time the model
    // produces a token.  When the LLM future completes we drain any residual
    // buffered tokens before breaking.
    loop {
        if llm_result.is_some() {
            // LLM future already completed — drain remaining buffered tokens.
            match rx.try_recv() {
                Ok(chunk) => {
                    buf.push_str(&chunk);
                    continue;
                }
                Err(_) => break,
            }
        }

        tokio::select! {
            // Prefer detecting LLM completion over reading tokens to avoid an
            // unnecessary extra loop iteration.
            biased;

            result = &mut llm_fut, if llm_result.is_none() => {
                llm_result = Some(result);
                // Don't break yet — drain remaining buffered tokens on next iteration.
            }

            maybe = tokio::time::timeout(config.step_timeout, rx.recv()) => {
                match maybe {
                    Ok(Some(chunk)) => buf.push_str(&chunk),
                    Ok(None) => break, // Channel closed — all tokens drained.
                    Err(_) => bail!(
                        "step timed out (no tokens received for {}s)",
                        config.step_timeout.as_secs(),
                    ),
                }
            }
        }
    }

    let response = llm_result
        .context("LLM stream ended without completing")?
        .context("LLM call failed")?;

    // Prefer the assembled response from the provider; fall back to the token
    // buffer which contains the same data collected piecemeal.
    let text = if !response.content.is_empty() {
        response.content
    } else if !buf.is_empty() {
        buf
    } else {
        bail!("LLM returned empty response");
    };

    Ok(text)
}

/// Parse an [`AgentStep`] from raw LLM output.
///
/// Multi-layer fallback parsing to handle common model quirks:
///
/// 1. **Direct**: try `serde_json::from_str` on the trimmed input.
/// 2. **Strip fences**: remove ````json ... ```` wrappers.
/// 3. **Extract object**: find the first `{...}` balanced JSON object.
/// 4. **Heuristic repair**: if the JSON has the right fields but wrong/missing
///    `"type"`, infer it from the presence of `"tool_call"` vs `"final_answer"`.
pub fn parse_agent_step(raw: &str) -> Result<AgentStep> {
    let trimmed = raw.trim();

    // ── Layer 1: direct parse ────────────────────────────────────────────
    if let Ok(step) = serde_json::from_str::<AgentStep>(trimmed) {
        return Ok(step);
    }

    // ── Layer 2: strip markdown code fences ──────────────────────────────
    let defenced = strip_code_fences(trimmed);
    if let Ok(step) = serde_json::from_str::<AgentStep>(defenced) {
        return Ok(step);
    }

    // ── Layer 3: extract first JSON object ───────────────────────────────
    if let Some(json_str) = extract_json_object(defenced) {
        if let Ok(step) = serde_json::from_str::<AgentStep>(json_str) {
            return Ok(step);
        }

        // ── Layer 4: heuristic type repair ───────────────────────────────
        if let Ok(step) = repair_and_parse(json_str) {
            return Ok(step);
        }
    }

    // Also try extracting from the original (pre-defence) text.
    if let Some(json_str) = extract_json_object(trimmed) {
        if let Ok(step) = serde_json::from_str::<AgentStep>(json_str) {
            return Ok(step);
        }
        if let Ok(step) = repair_and_parse(json_str) {
            return Ok(step);
        }
    }

    anyhow::bail!(
        "failed to parse AgentStep from: {}",
        &trimmed[..trimmed.len().min(300)]
    )
}

/// Strip markdown code fences: ````json ... ```` or ```` ... ````
fn strip_code_fences(s: &str) -> &str {
    let s = s.trim();
    if !s.starts_with("```") {
        return s;
    }
    let inner = s
        .strip_prefix("```json")
        .or_else(|| s.strip_prefix("```JSON"))
        .or_else(|| s.strip_prefix("```"))
        .unwrap_or(s);
    inner.strip_suffix("```").unwrap_or(inner).trim()
}

/// Attempt to repair a JSON object that has the right fields but is missing
/// or has a wrong `"type"` discriminator, then parse it.
fn repair_and_parse(json_str: &str) -> Result<AgentStep> {
    let mut obj: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(json_str).context("not a JSON object")?;

    // Infer or fix the "type" field.
    let has_tool_call = obj.get("tool_call").is_some_and(|v| !v.is_null());
    let has_final_answer = obj
        .get("final_answer")
        .is_some_and(|v| v.is_string() && !v.as_str().unwrap_or("").is_empty());

    let inferred_type = if has_tool_call && !has_final_answer {
        "tool_call"
    } else if has_final_answer {
        "final_answer"
    } else {
        anyhow::bail!("cannot infer type: no tool_call or final_answer field found");
    };

    obj.insert(
        "type".to_string(),
        serde_json::Value::String(inferred_type.to_string()),
    );

    // Ensure "thought" exists (some models omit it).
    obj.entry("thought".to_string())
        .or_insert_with(|| serde_json::Value::String("(no thought)".to_string()));

    let repaired = serde_json::Value::Object(obj);
    serde_json::from_value(repaired).context("repaired JSON still failed to parse as AgentStep")
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
