//! Small-model router: classifies messages as CHAT or TOOLS and provides a
//! lightweight single-shot chat path that bypasses the thinker loop entirely.
//!
//! In the future we can add semantic routing or confidence thresholds here.

use std::time::Duration;

use anyhow::Result;
use tokio::sync::mpsc;
use tracing::info;

use aigent_llm::{ChatMessage, LlmClient, LlmRouter, ModelProvider, Provider, RouterDecision};

// ── Classification ───────────────────────────────────────────────────────────

/// Ask the router model to classify `user_message` as [`RouterDecision::Chat`]
/// or [`RouterDecision::Tools`].
///
/// On any error or timeout the function returns `Tools` — always safe to fall
/// through to the full thinker.
///
/// When `verbose` is `true`, the full classify prompt and raw model response
/// are logged at `info!` level for debugging.
pub async fn route_query(
    llm: &LlmRouter,
    provider: Provider,
    model: &str,
    classify_system_prompt: &str,
    classify_timeout: Duration,
    user_message: &str,
    verbose: bool,
) -> RouterDecision {
    let start = std::time::Instant::now();

    let messages = vec![
        ChatMessage::system(classify_system_prompt),
        ChatMessage::user(user_message),
    ];

    if verbose {
        info!(
            "[ROUTER:VERBOSE] classify call: model={} timeout={}s prompt_len={} user_msg={:?}",
            model,
            classify_timeout.as_secs(),
            classify_system_prompt.len(),
            &user_message[..user_message.len().min(200)],
        );
    }

    // Try up to 2 attempts.  The first timeout is usually a cold-start
    // (model loading into VRAM).  A single retry is enough because the
    // model is warm after the first attempt even if it timed out.
    let max_attempts = 2;
    let mut decision = RouterDecision::Tools;
    for attempt in 0..max_attempts {
        let result = tokio::time::timeout(
            classify_timeout,
            llm.chat(
                ModelProvider::from(provider),
                model,
                &messages,
                None,
            ),
        )
        .await;

        match result {
            Ok(Ok(ref resp)) => {
                let text = resp.content.trim().to_ascii_lowercase();
                if verbose {
                    info!("[ROUTER:VERBOSE] raw response: {:?}", resp.content.trim());
                }
                decision = if text.contains("chat") {
                    RouterDecision::Chat
                } else {
                    RouterDecision::Tools
                };
                break; // success — no retry needed
            }
            Ok(Err(ref e)) => {
                info!("router LLM error, falling back to TOOLS: {}", e);
                decision = RouterDecision::Tools;
                break; // hard error — retry won't help
            }
            Err(_) => {
                if attempt + 1 < max_attempts {
                    info!(
                        "router classify timed out after {}s (attempt {}/{}) — retrying (model likely cold-loading)",
                        classify_timeout.as_secs(), attempt + 1, max_attempts,
                    );
                    // Continue to retry — model should be warm now.
                } else {
                    info!(
                        "router classify timed out after {}s (attempt {}/{}) — falling back to TOOLS",
                        classify_timeout.as_secs(), attempt + 1, max_attempts,
                    );
                    decision = RouterDecision::Tools;
                }
            }
        }
    }

    let elapsed = start.elapsed().as_millis();
    info!("router decision: {:?} in {}ms (model: {})", decision, elapsed, model);

    decision
}

// ── Simple chat (no tool loop) ───────────────────────────────────────────────

/// Single-shot streaming chat that bypasses the thinker/tool loop entirely.
///
/// Used when the router decides the small model can handle the turn on its own.
/// The caller provides the **full** system prompt (same as the primary model
/// path) so the agent's personality, memory, and context are intact.
pub async fn run_simple_chat(
    llm: &LlmRouter,
    provider: Provider,
    model: &str,
    messages: &[ChatMessage],
    chunk_tx: mpsc::Sender<String>,
    timeout: Duration,
) -> Result<String> {
    let result = tokio::time::timeout(
        timeout,
        llm.chat_stream(
            ModelProvider::from(provider),
            model,
            messages,
            None, // no tools
            chunk_tx,
        ),
    )
    .await;

    match result {
        Ok(Ok(resp)) => Ok(resp.content),
        Ok(Err(e)) => Err(e),
        Err(_) => Err(anyhow::anyhow!(
            "router chat timed out after {}s",
            timeout.as_secs()
        )),
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_debug_display() {
        assert_eq!(format!("{:?}", RouterDecision::Chat), "Chat");
        assert_eq!(format!("{:?}", RouterDecision::Tools), "Tools");
    }

    #[test]
    fn decision_equality() {
        assert_eq!(RouterDecision::Chat, RouterDecision::Chat);
        assert_ne!(RouterDecision::Chat, RouterDecision::Tools);
    }
}
