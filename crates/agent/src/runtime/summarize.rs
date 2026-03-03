//! Conversation summarization — condense evicted turns into a compact paragraph.

use aigent_llm::Provider;
use aigent_prompt::ConversationTurn;
use tracing::{debug, instrument, warn};

use super::AgentRuntime;
use crate::AgentResult;

/// Maximum number of recent turns to keep after summarization.
const KEEP_RECENT: usize = 3;

/// Turn threshold that triggers summarization.
pub const SUMMARIZE_THRESHOLD: usize = 8;

impl AgentRuntime {
    /// Summarize the oldest conversation turns into a compact paragraph.
    ///
    /// Returns `(new_summary, kept_turns)`:
    /// - `new_summary` is the updated cumulative summary (incorporating any
    ///    existing prior summary).
    /// - `kept_turns` is the slice of most-recent turns that should remain
    ///    in the live buffer.
    ///
    /// If `recent_turns.len() < SUMMARIZE_THRESHOLD`, returns `None` — no
    /// summarization is needed.
    #[instrument(skip(self, existing_summary, recent_turns), fields(
        turn_count = recent_turns.len(),
        has_prior = existing_summary.is_some(),
    ))]
    pub async fn summarize_conversation(
        &self,
        existing_summary: Option<&str>,
        recent_turns: &[ConversationTurn],
    ) -> AgentResult<Option<(String, Vec<ConversationTurn>)>> {
        if recent_turns.len() < SUMMARIZE_THRESHOLD {
            return Ok(None);
        }

        let split = recent_turns.len().saturating_sub(KEEP_RECENT);
        let to_summarize = &recent_turns[..split];
        let to_keep = recent_turns[split..].to_vec();

        // Format the turns to summarize into a readable transcript.
        let mut transcript = String::with_capacity(to_summarize.len() * 200);
        for turn in to_summarize {
            transcript.push_str("User: ");
            transcript.push_str(&turn.user);
            transcript.push_str("\nAssistant: ");
            transcript.push_str(&turn.assistant);
            transcript.push('\n');
        }

        // Build the summarization prompt.
        let prior_ctx = match existing_summary {
            Some(s) => format!(
                "There is an existing summary of even earlier conversation:\n\
                 ---\n{s}\n---\n\n\
                 Incorporate it with the new turns below into a single cohesive summary.\n\n"
            ),
            None => String::new(),
        };

        let prompt = format!(
            "You are a precise summarizer. Condense the following conversation \
             turns into a single compact paragraph (3-5 sentences) that captures \
             the key topics, decisions, and any unresolved questions. \
             Preserve important names, numbers, and action items.\n\n\
             {prior_ctx}\
             CONVERSATION TURNS TO SUMMARIZE:\n{transcript}\n\n\
             SUMMARY:"
        );

        debug!(evicting = split, keeping = to_keep.len(), "summarizing conversation turns");

        let primary = Provider::from(self.config.llm.provider.as_str());
        let result = self
            .llm
            .chat_with_fallback(
                primary,
                &self.config.llm.ollama_model,
                &self.config.llm.openrouter_model,
                &prompt,
            )
            .await;

        match result {
            Ok((_provider, summary)) => {
                let summary = summary.trim().to_string();
                debug!(summary_len = summary.len(), "conversation summary generated");
                Ok(Some((summary, to_keep)))
            }
            Err(err) => {
                // Summarization is best-effort — if it fails, just keep all
                // turns and log the error.  The caller will fall back to the
                // existing hard-trim behaviour.
                warn!(?err, "conversation summarization failed; keeping all turns");
                Ok(None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aigent_config::AppConfig;

    fn make_runtime() -> AgentRuntime {
        AgentRuntime::new(AppConfig::default())
    }

    fn make_turns(n: usize) -> Vec<ConversationTurn> {
        (0..n)
            .map(|i| ConversationTurn {
                user: format!("user message {i}"),
                assistant: format!("assistant reply {i}"),
            })
            .collect()
    }

    #[tokio::test]
    async fn below_threshold_returns_none() {
        let rt = make_runtime();
        let turns = make_turns(5);
        let result = rt.summarize_conversation(None, &turns).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn at_threshold_minus_one_returns_none() {
        let rt = make_runtime();
        let turns = make_turns(SUMMARIZE_THRESHOLD - 1);
        let result = rt.summarize_conversation(None, &turns).await.unwrap();
        assert!(result.is_none());
    }


}
