//! Sleep cycle orchestration — agentic and multi-agent consolidation.
//!
//! The generation functions (`generate_agentic_sleep_insights`,
//! `generate_multi_agent_sleep_insights`) are pure — they take read-only
//! snapshots of memories and identity and return structured insights without
//! touching `MemoryManager`.  The caller is responsible for applying the
//! insights via `memory.apply_agentic_sleep_insights(...)` once it
//! re-acquires mutable access.

use crate::AgentResult;
use tracing::{info, instrument, warn};
use tokio::sync::mpsc;
use aigent_llm::{Provider};
use aigent_memory::{
    AgenticSleepInsights, IdentityKernel, MemoryEntry, SleepSummary,
    parse_agentic_insights,
};
use aigent_memory::multi_sleep::{
    SpecialistRole, batch_memories, build_identity_context, deliberation_prompt,
    merge_insights, specialist_prompt,
};

use super::AgentRuntime;

/// The result of a sleep-cycle *generation* pass.
///
/// The caller inspects the variant to decide how to apply the result:
///   - `Insights` → call `memory.apply_agentic_sleep_insights(insights, …)`
///   - `PassiveFallback` → the LLM was unavailable and we already fell back to
///     passive distillation inside the MemoryManager, so the summary is final.
pub enum SleepGenerationResult {
    /// Structured insights produced by the LLM (single-agent or multi-agent).
    Insights(Box<AgenticSleepInsights>),
    /// LLM was unavailable — passive heuristic distillation was applied
    /// directly and this is the resulting summary.
    PassiveFallback(SleepSummary),
}

impl AgentRuntime {
    /// Generate agentic sleep insights from a read-only memory snapshot.
    ///
    /// Does **not** mutate `MemoryManager`.  Returns `SleepGenerationResult`
    /// so the caller can apply insights after re-acquiring mutable access.
    ///
    /// Falls back to `PassiveFallback` if the LLM call fails — in that case
    /// the caller should run `memory.run_sleep_cycle()` itself.
    #[instrument(skip(self, memories, identity))]
    pub async fn generate_agentic_sleep_insights(
        &self,
        memories: &[MemoryEntry],
        identity: &IdentityKernel,
        progress: &mpsc::UnboundedSender<String>,
    ) -> AgentResult<SleepGenerationResult> {
        let primary = Provider::from(self.config.llm.provider.as_str());

        let _ = progress.send("Reflecting on today's memories…".into());
        let (bot_name, user_name) = (&self.config.agent.name, &self.config.agent.user_name);
        let prompt = aigent_memory::sleep::agentic_sleep_prompt(
            memories, bot_name, user_name, &identity.trait_scores,
        );
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
                let _ = progress.send("Insights generated — ready to apply to memory.".into());
                Ok(SleepGenerationResult::Insights(Box::new(insights)))
            }
            Err(err) => {
                warn!(?err, "agentic sleep: LLM unavailable — caller should run passive distillation");
                // Signal to caller that no LLM insights were produced.
                Ok(SleepGenerationResult::PassiveFallback(SleepSummary {
                    distilled: String::new(),
                    promoted_ids: Vec::new(),
                    promotions: Vec::new(),
                }))
            }
        }
    }

    /// Call the LLM with logging and per-call timing.
    /// Returns `None` on failure so callers can degrade gracefully to a
    /// single-agent fallback.
    async fn sleep_llm_call(
        &self,
        primary: Provider,
        prompt: &str,
        role_label: &str,
    ) -> Option<String> {
        let t0 = std::time::Instant::now();
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
                    elapsed_secs = t0.elapsed().as_secs(),
                    "multi-agent sleep: specialist reply received"
                );
                Some(reply)
            }
            Err(err) => {
                warn!(
                    ?err,
                    role = role_label,
                    elapsed_secs = t0.elapsed().as_secs(),
                    "multi-agent sleep: LLM call failed"
                );
                None
            }
        }
    }

    /// Full nightly multi-agent memory consolidation (generation only).
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
    ///   7. Return `SleepGenerationResult::Insights(merged)`
    ///
    /// Does **not** mutate `MemoryManager`.  Operates on read-only snapshots
    /// so the caller can hold the lock only for snapshot-taking and
    /// insight-application, never across LLM network calls.
    ///
    /// Falls back to `generate_agentic_sleep_insights()` (single-agent) if
    /// the LLM is unavailable or all batches fail.
    #[instrument(skip(self, memories, identity))]
    pub async fn generate_multi_agent_sleep_insights(
        &self,
        memories: &[MemoryEntry],
        identity: &IdentityKernel,
        progress: &mpsc::UnboundedSender<String>,
    ) -> AgentResult<SleepGenerationResult> {
        let primary = Provider::from(self.config.llm.provider.as_str());

        let batch_size = self.config.memory.multi_agent_sleep_batch_size;
        let batches = batch_memories(memories, batch_size);

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
        let pipeline_start = std::time::Instant::now();

        for (batch_idx, batch) in batches.iter().enumerate() {
            let batch_start = std::time::Instant::now();
            info!(
                batch = batch_idx + 1,
                total = batches.len(),
                entries = batch.len(),
                "multi-agent sleep: processing batch"
            );
            let _ = progress.send(format!(
                "Batch {}/{}: consulting 5 specialist agents in parallel…",
                batch_idx + 1,
                batches.len()
            ));

            let identity_snap = identity.clone();

            let arch_prompt =
                specialist_prompt(SpecialistRole::Archivist, batch, &identity_snap, bot_name, user_name);
            let psych_prompt = specialist_prompt(
                SpecialistRole::Psychologist,
                batch,
                &identity_snap,
                bot_name,
                user_name,
            );
            let strat_prompt =
                specialist_prompt(SpecialistRole::Strategist, batch, &identity_snap, bot_name, user_name);
            let critic_prompt =
                specialist_prompt(SpecialistRole::Critic, batch, &identity_snap, bot_name, user_name);
            let identity_prompt =
                specialist_prompt(SpecialistRole::Identity, batch, &identity_snap, bot_name, user_name);

            // Run all 5 specialists in parallel.
            let specialists_start = std::time::Instant::now();
            let (arch_reply, psych_reply, strat_reply, critic_reply, identity_reply) = tokio::join!(
                self.sleep_llm_call(primary, &arch_prompt, "Archivist"),
                self.sleep_llm_call(primary, &psych_prompt, "Psychologist"),
                self.sleep_llm_call(primary, &strat_prompt, "Strategist"),
                self.sleep_llm_call(primary, &critic_prompt, "Critic"),
                self.sleep_llm_call(primary, &identity_prompt, "Identity"),
            );
            info!(
                batch = batch_idx + 1,
                specialists_elapsed_secs = specialists_start.elapsed().as_secs(),
                "multi-agent sleep: parallel specialists phase complete"
            );

            // Collect available replies — degrade gracefully to a partial team
            // rather than discarding all successful results when one specialist
            // fails.  Only fall back to single-agent when ALL five fail.
            let mut specialist_reports: Vec<(SpecialistRole, String)> = Vec::with_capacity(5);
            if let Some(r) = arch_reply     { specialist_reports.push((SpecialistRole::Archivist,    r)); }
            if let Some(r) = psych_reply    { specialist_reports.push((SpecialistRole::Psychologist, r)); }
            if let Some(r) = strat_reply    { specialist_reports.push((SpecialistRole::Strategist,   r)); }
            if let Some(r) = critic_reply   { specialist_reports.push((SpecialistRole::Critic,        r)); }
            if let Some(r) = identity_reply { specialist_reports.push((SpecialistRole::Identity,      r)); }

            if specialist_reports.is_empty() {
                // All 5 failed — single-agent fallback for this batch.
                warn!(
                    batch = batch_idx + 1,
                    "multi-agent sleep: all specialists failed — using single-agent fallback for this batch"
                );
                let _ = progress.send(format!(
                    "Batch {}/{}: all specialists unavailable — using single-agent fallback…",
                    batch_idx + 1,
                    batches.len()
                ));
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
                        all_batch_insights.push(parse_agentic_insights(&reply));
                        any_batch_succeeded = true;
                    }
                    None => {
                        warn!(batch = batch_idx + 1, "multi-agent sleep: fallback also failed — skipping batch");
                    }
                }
                continue;
            }

            if specialist_reports.len() < 5 {
                warn!(
                    batch = batch_idx + 1,
                    succeeded = specialist_reports.len(),
                    "multi-agent sleep: partial specialist team — proceeding with available reports"
                );
            }

            // Parse available specialist replies to detect Core ID conflicts.
            let parsed_insights: Vec<AgenticSleepInsights> = specialist_reports
                .iter()
                .map(|(_, reply)| parse_agentic_insights(reply))
                .collect();

            // Conflicting IDs: mentioned in retire by one specialist AND in
            // rewrite/consolidate by another.
            let all_retire: std::collections::HashSet<String> = parsed_insights
                .iter()
                .flat_map(|i| i.retire_core_ids.iter().cloned())
                .collect();

            let all_rewrite_or_consolidate: std::collections::HashSet<String> = parsed_insights
                .iter()
                .flat_map(|i| i.rewrite_core.iter().map(|(id, _)| id.clone()))
                .chain(parsed_insights.iter().flat_map(|i| {
                    i.consolidate_core
                        .iter()
                        .flat_map(|(ids_csv, _)| {
                            ids_csv.split(',').map(|s| s.trim().to_string()).collect::<Vec<_>>()
                        })
                }))
                .collect();

            let conflicting_ids: Vec<String> = all_retire
                .intersection(&all_rewrite_or_consolidate)
                .cloned()
                .collect();

            info!(
                batch = batch_idx + 1,
                specialists = specialist_reports.len(),
                conflicts = conflicting_ids.len(),
                "multi-agent sleep: running synthesis deliberation"
            );
            let _ = progress.send(format!(
                "Batch {}/{}: synthesising {} specialist report{}{}…",
                batch_idx + 1,
                batches.len(),
                specialist_reports.len(),
                if specialist_reports.len() == 1 { "" } else { "s" },
                if conflicting_ids.is_empty() {
                    String::new()
                } else {
                    format!(" ({} conflict{})", conflicting_ids.len(), if conflicting_ids.len() == 1 { "" } else { "s" })
                }
            ));

            let identity_ctx =
                build_identity_context(batch, &identity_snap, bot_name, user_name);
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
                        batch_elapsed_secs = batch_start.elapsed().as_secs(),
                        "multi-agent sleep: batch synthesis complete"
                    );
                    all_batch_insights.push(insights);
                    any_batch_succeeded = true;
                }
                None => {
                    warn!(
                        batch = batch_idx + 1,
                        batch_elapsed_secs = batch_start.elapsed().as_secs(),
                        "multi-agent sleep: synthesis call failed — merging available specialist insights directly"
                    );
                    // Merge whatever specialist insights we have as a degraded fallback.
                    let batch_merged = merge_insights(parsed_insights);
                    all_batch_insights.push(batch_merged);
                    any_batch_succeeded = true;
                }
            }
        }

        if !any_batch_succeeded {
            warn!("multi-agent sleep: all batches failed — falling back to single-agent sleep");
            let _ = progress.send("All batches failed — falling back to single-agent sleep…".into());
            return self.generate_agentic_sleep_insights(memories, identity, progress).await;
        }

        let final_insights = merge_insights(all_batch_insights);
        info!(
            learned = final_insights.learned_about_user.len(),
            follow_ups = final_insights.follow_ups.len(),
            reflections = final_insights.reflective_thoughts.len(),
            profile_updates = final_insights.user_profile_updates.len(),
            pipeline_elapsed_secs = pipeline_start.elapsed().as_secs(),
            "multi-agent sleep: all batches complete — insights ready to apply"
        );
        let _ = progress.send(format!(
            "Insights generated — {} learned, {} reflections — ready to apply.",
            final_insights.learned_about_user.len(),
            final_insights.reflective_thoughts.len(),
        ));

        Ok(SleepGenerationResult::Insights(Box::new(final_insights)))
    }

}
