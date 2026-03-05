//! Small-model router: classifies messages as CHAT or TOOLS and provides a
//! lightweight single-shot chat path that bypasses the thinker loop entirely.
//!
//! In the future we can add semantic routing or confidence thresholds here.

use std::time::Duration;

use anyhow::Result;
use tokio::sync::mpsc;
use tracing::debug;

use aigent_llm::{ChatMessage, LlmClient, LlmRouter, ModelProvider, Provider, RouterDecision};

// ── Classification ───────────────────────────────────────────────────────────

/// Ask the router model to classify `user_message` as [`RouterDecision::Chat`]
/// or [`RouterDecision::Tools`].
///
/// On any error or timeout the function returns `Tools` — always safe to fall
/// through to the full thinker.
pub async fn route_query(
    llm: &LlmRouter,
    provider: Provider,
    model: &str,
    classify_system_prompt: &str,
    classify_timeout: Duration,
    user_message: &str,
) -> RouterDecision {
    let start = std::time::Instant::now();

    let messages = vec![
        ChatMessage::system(classify_system_prompt),
        ChatMessage::user(user_message),
    ];

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

    let decision = match result {
        Ok(Ok(ref resp)) => {
            let text = resp.content.trim().to_ascii_lowercase();
            if text.contains("chat") {
                RouterDecision::Chat
            } else {
                // Includes "tools", timeout, error, ambiguous replies.
                RouterDecision::Tools
            }
        }
        // Timeout or LLM error → safe fallback.
        _ => RouterDecision::Tools,
    };

    let elapsed = start.elapsed().as_millis();
    debug!("router decision: {:?} in {}ms", decision, elapsed);

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
