//! Proactive messaging — check if the agent should reach out.

use tracing::{debug, info};
use aigent_llm::{Provider, extract_json_output};
use crate::agent_loop::ProactiveOutput;

use super::AgentRuntime;

impl AgentRuntime {
    /// Proactive check that operates on pre-built summary strings.
    ///
    /// The caller is responsible for building the `beliefs_summary` and
    /// `reflections_summary` while holding the lock, then releasing the lock
    /// before calling this method.  This avoids holding the mutex across the
    /// LLM network call.
    pub async fn run_proactive_check_from_summaries(
        &self,
        beliefs_summary: &str,
        reflections_summary: &str,
    ) -> Option<ProactiveOutput> {
        let primary = if self.config.llm.provider.to_lowercase() == "openrouter" {
            Provider::OpenRouter
        } else {
            Provider::Ollama
        };

        let prompt = format!(
            "You are {name}, an AI companion.  Based on your current beliefs and recent \
             reflections, decide whether there is something genuinely useful, timely, or \
             caring you could proactively share with the user right now \
             — a check-in, a reminder, an insight, or a follow-up concern.\n\
             Only return an action if it adds real value; default to silence.\n\
             Respond only with valid JSON:\n\
             {{\"action\":\"follow_up\"|\"reminder\"|\"insight\"|null,\
              \"message\":\"...\",\
              \"urgency\":0.5}}\n\
             Set action to null and omit message when staying silent.\n\n\
             CURRENT_BELIEFS:\n{beliefs}\n\nRECENT_REFLECTIONS:\n{reflections}",
            name = self.config.agent.name,
            beliefs = beliefs_summary,
            reflections = reflections_summary,
        );

        let Ok((_provider, raw)) = self
            .llm
            .chat_with_fallback(
                primary,
                &self.config.llm.ollama_model,
                &self.config.llm.openrouter_model,
                &prompt,
            )
            .await
        else {
            debug!("run_proactive_check: LLM unavailable");
            return None;
        };

        let output: ProactiveOutput = extract_json_output(&raw)?;

        if output.action.is_none() || output.message.as_deref().map(|m| m.is_empty()).unwrap_or(true) {
            debug!("run_proactive_check: decided to stay silent");
            return None;
        }

        info!(
            action = ?output.action,
            urgency = ?output.urgency,
            "run_proactive_check: proactive message will be sent"
        );
        Some(output)
    }

}
