//! Sleep cycle orchestration — agentic and multi-agent consolidation.

use anyhow::Result;
use tracing::{info, instrument, warn};
use tokio::sync::mpsc;
use aigent_llm::{Provider};
use aigent_memory::{MemoryManager, MemoryTier, SleepSummary, parse_agentic_insights};
use aigent_memory::multi_sleep::{
    SpecialistRole, batch_memories, build_identity_context, deliberation_prompt,
    merge_insights, specialist_prompt,
};

use super::AgentRuntime;

impl AgentRuntime {
    /// Run an agentic sleep cycle: build a reflection prompt, call the LLM,
    /// parse the insights, and apply them to memory.
    ///
    /// Falls back to passive-only distillation if the LLM call fails.
    #[instrument(skip(self, memory))]
    pub async fn run_agentic_sleep_cycle(
        &self,
        memory: &mut MemoryManager,
        progress: &mpsc::UnboundedSender<String>,
    ) -> Result<SleepSummary> {
        let primary = if self.config.llm.provider.to_lowercase() == "openrouter" {
            Provider::OpenRouter
        } else {
            Provider::Ollama
        };

        let _ = progress.send("Reflecting on today's memories…".into());
        let prompt = memory.agentic_sleep_prompt();
        info!(prompt_len = prompt.len(), "agentic sleep: sending reflection prompt to LLM");

        match self
            .llm
            .chat_with_fallback(
                primary,
                &self.config.llm.ollama_model,
                &self.config.llm.openrouter_model,
                &prompt,
            )
            .await
        {
            Ok((_provider, reply)) => {
                info!(reply_len = reply.len(), "agentic sleep: LLM reply received");
                let insights = parse_agentic_insights(&reply);
                let summary_text = Some(format!(
                    "Agentic sleep: {} learned, {} follow-ups, {} reflections, {} profile updates",
                    insights.learned_about_user.len(),
                    insights.follow_ups.len(),
                    insights.reflective_thoughts.len(),
                    insights.user_profile_updates.len(),
                ));
                let _ = progress.send("Applying insights to memory…".into());
                memory.apply_agentic_sleep_insights(insights, summary_text).await
            }
            Err(err) => {
                warn!(?err, "agentic sleep: LLM unavailable, falling back to passive distillation");
                memory.run_sleep_cycle().await
            }
        }
    }

    /// Call the LLM with logging. Returns `None` on failure so callers can
    /// degrade gracefully to a single-agent fallback.
    async fn sleep_llm_call(
        &self,
        primary: Provider,
        prompt: &str,
        role_label: &str,
    ) -> Option<String> {
        match self
            .llm
            .chat_with_fallback(
                primary,
                &self.config.llm.ollama_model,
                &self.config.llm.openrouter_model,
                prompt,
            )
            .await
        {
            Ok((_provider, reply)) => {
                info!(
                    role = role_label,
                    reply_len = reply.len(),
                    "multi-agent sleep: specialist reply received"
                );
                Some(reply)
            }
            Err(err) => {
                warn!(?err, role = role_label, "multi-agent sleep: LLM call failed");
                None
            }
        }
    }

    /// Full nightly multi-agent memory consolidation.
    ///
    /// Pipeline per batch:
    ///   1. Run 4 specialist agents in parallel (tokio::join!)
    ///   2. Detect Core ID conflicts between specialist replies
    ///   3. Run synthesis agent with deliberation prompt
    ///   4. Parse synthesis reply into AgenticSleepInsights
    ///   5. Accumulate into running merger
    ///
    /// After all batches:
    ///   6. Merge all batch AgenticSleepInsights into one
    ///   7. Apply via memory.apply_agentic_sleep_insights()
    ///
    /// Falls back to run_agentic_sleep_cycle() (single-agent) if the LLM is
    /// unavailable or any batch fails completely.

    /// Full nightly multi-agent memory consolidation.
    ///
    /// Pipeline per batch:
    ///   1. Run 4 specialist agents in parallel (tokio::join!)
    ///   2. Detect Core ID conflicts between specialist replies
    ///   3. Run synthesis agent with deliberation prompt
    ///   4. Parse synthesis reply into AgenticSleepInsights
    ///   5. Accumulate into running merger
    ///
    /// After all batches:
    ///   6. Merge all batch AgenticSleepInsights into one
    ///   7. Apply via memory.apply_agentic_sleep_insights()
    ///
    /// Falls back to run_agentic_sleep_cycle() (single-agent) if the LLM is
    /// unavailable or any batch fails completely.
    #[instrument(skip(self, memory))]
    pub async fn run_multi_agent_sleep_cycle(
        &self,
        memory: &mut MemoryManager,
        progress: &mpsc::UnboundedSender<String>,
    ) -> Result<SleepSummary> {
        let primary = if self.config.llm.provider.to_lowercase() == "openrouter" {
            Provider::OpenRouter
        } else {
            Provider::Ollama
        };

        let batch_size = self.config.memory.multi_agent_sleep_batch_size;
        let batches = batch_memories(memory.all(), batch_size);

        info!(
            batch_count = batches.len(),
            batch_size,
            "multi-agent sleep: starting consolidation"
        );
        let _ = progress.send(format!(
            "Starting multi-agent memory consolidation ({} batch{})…",
            batches.len(),
            if batches.len() == 1 { "" } else { "es" }
        ));

        let bot_name = &self.config.agent.name;
        let user_name = &self.config.agent.user_name;

        let mut all_batch_insights = Vec::new();
        let mut any_batch_succeeded = false;

        for (batch_idx, batch) in batches.iter().enumerate() {
            info!(
                batch = batch_idx + 1,
                total = batches.len(),
                entries = batch.len(),
                "multi-agent sleep: processing batch"
            );
            let _ = progress.send(format!(
                "Batch {}/{}: consulting 4 specialist agents in parallel…",
                batch_idx + 1,
                batches.len()
            ));

            let identity = memory.identity.clone();

            let arch_prompt =
                specialist_prompt(SpecialistRole::Archivist, batch, &identity, bot_name, user_name);
            let psych_prompt = specialist_prompt(
                SpecialistRole::Psychologist,
                batch,
                &identity,
                bot_name,
                user_name,
            );
            let strat_prompt =
                specialist_prompt(SpecialistRole::Strategist, batch, &identity, bot_name, user_name);
            let critic_prompt =
                specialist_prompt(SpecialistRole::Critic, batch, &identity, bot_name, user_name);

            // Run all 4 specialists in parallel.
            let (arch_reply, psych_reply, strat_reply, critic_reply) = tokio::join!(
                self.sleep_llm_call(primary, &arch_prompt, "Archivist"),
                self.sleep_llm_call(primary, &psych_prompt, "Psychologist"),
                self.sleep_llm_call(primary, &strat_prompt, "Strategist"),
                self.sleep_llm_call(primary, &critic_prompt, "Critic"),
            );

            // If any specialist failed, fall back to single-agent for this batch.
            let (arch_reply, psych_reply, strat_reply, critic_reply) =
                match (arch_reply, psych_reply, strat_reply, critic_reply) {
                    (Some(a), Some(p), Some(s), Some(c)) => (a, p, s, c),
                    _ => {
                        warn!(
                            batch = batch_idx + 1,
                            "multi-agent sleep: specialist call failed — using single-agent fallback for this batch"
                        );
                        let _ = progress.send(format!(
                            "Batch {}/{}: specialist unavailable — using single-agent fallback…",
                            batch_idx + 1,
                            batches.len()
                        ));
                        // Build the standard single-agent prompt for this batch's entries.
                        let fallback_prompt = aigent_memory::sleep::agentic_sleep_prompt(
                            batch,
                            bot_name,
                            user_name,
                            &identity.trait_scores,
                        );
                        match self
                            .sleep_llm_call(primary, &fallback_prompt, "fallback-single")
                            .await
                        {
                            Some(reply) => {
                                let insights = parse_agentic_insights(&reply);
                                all_batch_insights.push(insights);
                                any_batch_succeeded = true;
                            }
                            None => {
                                warn!(batch = batch_idx + 1, "multi-agent sleep: fallback also failed — skipping batch");
                            }
                        }
                        continue;
                    }
                };

            // Parse specialist replies to detect Core ID conflicts.
            let arch_insights = parse_agentic_insights(&arch_reply);
            let psych_insights = parse_agentic_insights(&psych_reply);
            let strat_insights = parse_agentic_insights(&strat_reply);
            let critic_insights = parse_agentic_insights(&critic_reply);

            // Conflicting IDs: mentioned in retire by one specialist AND in
            // rewrite/consolidate by another.
            let all_retire: std::collections::HashSet<String> = arch_insights
                .retire_core_ids
                .iter()
                .chain(&psych_insights.retire_core_ids)
                .chain(&strat_insights.retire_core_ids)
                .chain(&critic_insights.retire_core_ids)
                .cloned()
                .collect();

            let all_rewrite_or_consolidate: std::collections::HashSet<String> = arch_insights
                .rewrite_core
                .iter()
                .chain(&psych_insights.rewrite_core)
                .chain(&strat_insights.rewrite_core)
                .chain(&critic_insights.rewrite_core)
                .map(|(id, _)| id.clone())
                .chain(
                    arch_insights
                        .consolidate_core
                        .iter()
                        .chain(&psych_insights.consolidate_core)
                        .chain(&strat_insights.consolidate_core)
                        .chain(&critic_insights.consolidate_core)
                        .flat_map(|(ids_csv, _)| {
                            ids_csv.split(',').map(|s| s.trim().to_string()).collect::<Vec<_>>()
                        }),
                )
                .collect();

            let conflicting_ids: Vec<String> = all_retire
                .intersection(&all_rewrite_or_consolidate)
                .cloned()
                .collect();

            info!(
                batch = batch_idx + 1,
                conflicts = conflicting_ids.len(),
                "multi-agent sleep: running synthesis deliberation"
            );
            let _ = progress.send(format!(
                "Batch {}/{}: synthesising specialist reports{}…",
                batch_idx + 1,
                batches.len(),
                if conflicting_ids.is_empty() {
                    String::new()
                } else {
                    format!(" ({} conflict{})", conflicting_ids.len(), if conflicting_ids.len() == 1 { "" } else { "s" })
                }
            ));

            let specialist_reports = vec![
                (SpecialistRole::Archivist, arch_reply),
                (SpecialistRole::Psychologist, psych_reply),
                (SpecialistRole::Strategist, strat_reply),
                (SpecialistRole::Critic, critic_reply),
            ];

            let identity_ctx =
                build_identity_context(batch, &identity, bot_name, user_name);
            let delib_prompt = deliberation_prompt(
                &specialist_reports,
                &conflicting_ids,
                &identity_ctx,
                bot_name,
                user_name,
            );

            match self
                .sleep_llm_call(primary, &delib_prompt, "synthesis")
                .await
            {
                Some(synthesis_reply) => {
                    let insights = parse_agentic_insights(&synthesis_reply);
                    info!(
                        batch = batch_idx + 1,
                        learned = insights.learned_about_user.len(),
                        follow_ups = insights.follow_ups.len(),
                        "multi-agent sleep: batch synthesis complete"
                    );
                    all_batch_insights.push(insights);
                    any_batch_succeeded = true;
                }
                None => {
                    warn!(
                        batch = batch_idx + 1,
                        "multi-agent sleep: synthesis call failed — merging specialist insights directly"
                    );
                    // Merge the 4 specialist insights as a degraded fallback.
                    let batch_merged = merge_insights(vec![
                        arch_insights,
                        psych_insights,
                        strat_insights,
                        critic_insights,
                    ]);
                    all_batch_insights.push(batch_merged);
                    any_batch_succeeded = true;
                }
            }
        }

        if !any_batch_succeeded {
            warn!("multi-agent sleep: all batches failed — falling back to single-agent sleep");
            let _ = progress.send("All batches failed — falling back to single-agent sleep…".into());
            return self.run_agentic_sleep_cycle(memory, progress).await;
        }

        let final_insights = merge_insights(all_batch_insights);
        info!(
            learned = final_insights.learned_about_user.len(),
            follow_ups = final_insights.follow_ups.len(),
            reflections = final_insights.reflective_thoughts.len(),
            profile_updates = final_insights.user_profile_updates.len(),
            "multi-agent sleep: applying merged insights"
        );
        let _ = progress.send(format!(
            "Applying merged insights — {} learned, {} reflections…",
            final_insights.learned_about_user.len(),
            final_insights.reflective_thoughts.len(),
        ));

        let summary_text = Some(format!(
            "Multi-agent sleep: {} learned, {} follow-ups, {} reflections, {} profile updates ({} batches)",
            final_insights.learned_about_user.len(),
            final_insights.follow_ups.len(),
            final_insights.reflective_thoughts.len(),
            final_insights.user_profile_updates.len(),
            batches.len(),
        ));

        let result = memory.apply_agentic_sleep_insights(final_insights, summary_text).await?;
        // Write a sentinel entry so the 22-hour rate-limit survives daemon restarts.
        let _ = memory.record(
            MemoryTier::Semantic,
            "multi-agent sleep cycle completed",
            "sleep:multi-agent-cycle",
        ).await;
        Ok(result)
    }

}
