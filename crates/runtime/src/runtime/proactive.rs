//! Proactive messaging — check if the agent should reach out.

use tracing::{debug, info};
use aigent_llm::{Provider, extract_json_output};
use aigent_memory::{MemoryManager, MemoryTier};
use crate::agent_loop::ProactiveOutput;
use crate::prompt_builder::truncate_for_prompt;

use super::AgentRuntime;

impl AgentRuntime {
    /// Proactive check run on a background timer.
    ///
    /// Looks at the agent's current beliefs, recent reflections, and current
    /// context to decide whether there is something worth proactively sharing
    /// with the user.  Returns `None` when the agent decides to stay silent,
    /// or `Some(ProactiveOutput)` with the message to deliver.
    pub async fn run_proactive_check(
        &self,
        memory: &mut MemoryManager,
    ) -> Option<ProactiveOutput> {
        let primary = if self.config.llm.provider.to_lowercase() == "openrouter" {
            Provider::OpenRouter
        } else {
            Provider::Ollama
        };

        // Build a brief summary of current beliefs and recent reflections.
        let beliefs_summary = {
            let beliefs = memory.all_beliefs();
            if beliefs.is_empty() {
                "(none yet)".to_string()
            } else {
                beliefs
                    .iter()
                    .take(8)
                    .map(|e| format!("- {}", e.content))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        };

        let recent_reflections = {
            let ctx = memory.context_for_prompt_ranked_with_embed("recent", 5, None);
            let reflections: Vec<String> = ctx
                .iter()
                .filter(|item| item.entry.tier == MemoryTier::Reflective)
                .map(|item| format!("- {}", truncate_for_prompt(&item.entry.content, 200)))
                .collect();
            if reflections.is_empty() {
                "(none yet)".to_string()
            } else {
                reflections.join("\n")
            }
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
            reflections = recent_reflections,
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
