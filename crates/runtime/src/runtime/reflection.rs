//! Inline post-turn reflection — extract beliefs, sentiments, and follow-ups.

use anyhow::Result;
use tracing::{debug, warn};
use aigent_llm::{Provider, extract_json_output};
use aigent_memory::{MemoryManager, MemoryTier};
use crate::agent_loop::ReflectionOutput;
use crate::prompt_builder::truncate_for_prompt;
use crate::{BackendEvent};

use super::AgentRuntime;

impl AgentRuntime {
    /// Inline reflection pass run after every turn.
    ///
    /// Sends a compact prompt to the LLM asking it to extract up to 3 new
    /// beliefs and free-form insights from the exchange that just completed.
    /// Any extracted beliefs are persisted via `memory.record_belief()`.
    /// Reflective observations are persisted via `memory.record(Reflective, …)`.
    ///
    /// Returns the [`BackendEvent`] variants (`ReflectionInsight` /
    /// `BeliefAdded`) that the caller should broadcast to subscribers.
    pub async fn inline_reflect(
        &self,
        memory: &mut MemoryManager,
        user_message: &str,
        assistant_reply: &str,
    ) -> Result<Vec<BackendEvent>> {
        let primary = if self.config.llm.provider.to_lowercase() == "openrouter" {
            Provider::OpenRouter
        } else {
            Provider::Ollama
        };

        let prompt = format!(
            "You are a silent memory analyst.  Given the exchange below, extract up to 3 \
             *new* lasting beliefs or observations ({name} has about the user or the world) \
             that are worth remembering.  \
             Also list up to 2 short free-form reflective insights.\n\
             Respond only with valid JSON matching this schema:\n\
             {{\"beliefs\":[{{\"claim\":\"...\",\"confidence\":0.7}},...],\"reflections\":[\"...\",...]}}\n\
             If there is nothing worth remembering, return {{\"beliefs\":[],\"reflections\":[]}}.\n\n\
             EXCHANGE:\nUser: {user}\nAssistant: {asst}",
            name = self.config.agent.name,
            user = truncate_for_prompt(user_message, 400),
            asst = truncate_for_prompt(assistant_reply, 400),
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
            debug!("inline_reflect: LLM unavailable — skipping");
            return Ok(Vec::new());
        };

        let output: ReflectionOutput = match extract_json_output(&raw) {
            Some(v) => v,
            None => {
                debug!("inline_reflect: could not parse JSON — skipping");
                return Ok(Vec::new());
            }
        };

        let mut events = Vec::new();

        for belief in &output.beliefs {
            if let Err(err) = memory.record_belief(&belief.claim, belief.confidence).await {
                warn!(?err, claim = %belief.claim, "inline_reflect: failed to record belief");
            } else {
                events.push(BackendEvent::BeliefAdded {
                    claim: belief.claim.clone(),
                    confidence: belief.confidence,
                });
            }
        }

        for insight in &output.reflections {
            if let Err(err) = memory
                .record(MemoryTier::Reflective, insight.clone(), "inline-reflect")
                .await
            {
                warn!(?err, "inline_reflect: failed to record reflective insight");
            } else {
                events.push(BackendEvent::ReflectionInsight(insight.clone()));
            }
        }

        debug!(
            beliefs = output.beliefs.len(),
            reflections = output.reflections.len(),
            "inline_reflect: complete"
        );

        Ok(events)
    }

}
