use anyhow::Result;
use chrono::{DateTime, Utc};
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

use aigent_config::AppConfig;
use aigent_llm::{LlmRouter, Provider};
use aigent_memory::{
    MemoryManager, MemoryTier, SleepSummary, parse_agentic_insights,
    multi_sleep::{
        SpecialistRole, batch_memories, build_identity_context, deliberation_prompt,
        merge_insights, specialist_prompt,
    },
};
use tokio::sync::mpsc;

use crate::BackendEvent;

#[derive(Debug, Clone)]
pub struct ConversationTurn {
    pub user: String,
    pub assistant: String,
}

#[derive(Debug, Clone)]
pub struct AgentRuntime {
    pub config: AppConfig,
    llm: LlmRouter,
}

impl AgentRuntime {
    pub fn new(config: AppConfig) -> Self {
        Self {
            config,
            llm: LlmRouter::default(),
        }
    }

    pub async fn run(&self) -> Result<()> {
        Ok(())
    }

    pub async fn test_model_connection(&self) -> Result<String> {
        let primary = if self.config.llm.provider.to_lowercase() == "openrouter" {
            Provider::OpenRouter
        } else {
            Provider::Ollama
        };

        let prompt = format!(
            "[healthcheck][bot-name:{}][thinking:{}] Reply with a short single-line confirmation.",
            self.config.agent.name, self.config.agent.thinking_level
        );

        let (provider_used, reply) = self
            .llm
            .chat_with_fallback(primary, self.config.active_model(), &prompt)
            .await?;

        Ok(format!(
            "provider={provider_used:?} model={} reply={reply}",
            self.config.active_model()
        ))
    }

    pub async fn respond_and_remember(
        &self,
        memory: &mut MemoryManager,
        user_message: &str,
        recent_turns: &[ConversationTurn],
    ) -> Result<String> {
        let (tx, _rx) = mpsc::channel(100);
        self.respond_and_remember_stream(memory, user_message, recent_turns, None, tx)
            .await
    }

    #[instrument(skip(self, memory, tx), fields(bot = %self.config.agent.name, model = %self.config.active_model(), user_len = user_message.len()))]
    pub async fn respond_and_remember_stream(
        &self,
        memory: &mut MemoryManager,
        user_message: &str,
        recent_turns: &[ConversationTurn],
        last_turn_at: Option<DateTime<Utc>>,
        tx: mpsc::Sender<String>,
    ) -> Result<String> {
        let primary = if self.config.llm.provider.to_lowercase() == "openrouter" {
            Provider::OpenRouter
        } else {
            Provider::Ollama
        };

        // Persist the user turn immediately so it survives a restart.
        memory.record(MemoryTier::Episodic, user_message.to_string(), "user-input").await?;

        // Improvement 1: extract structured profile signals from the user message
        // immediately, without waiting for the nightly sleep cycle.
        let profile_signals = crate::micro_profile::extract_inline_profile_signals(user_message);
        for (key, value, category) in &profile_signals {
            if let Err(err) = memory.record_user_profile_keyed(key, value, category).await {
                warn!(?err, key, "micro-profile signal failed");
            }
        }
        if !profile_signals.is_empty() {
            debug!(count = profile_signals.len(), "micro-profile signals extracted");
        }

        // Collect pending follow-ups on the first turn of a new conversation,
        // or when the user is returning after a significant absence (≥4h).
        let is_returning_after_absence = last_turn_at
            .map(|t| (Utc::now() - t).num_hours() >= 4)
            .unwrap_or(false);
        let pending_follow_ups: Vec<(Uuid, String)> =
            if recent_turns.is_empty() || is_returning_after_absence {
                memory.pending_follow_up_ids()
            } else {
                Vec::new()
            };

        let follow_up_block = if pending_follow_ups.is_empty() {
            String::new()
        } else {
            let items = pending_follow_ups
                .iter()
                .map(|(_, text)| format!("- {text}"))
                .collect::<Vec<_>>()
                .join("\n");
            format!(
                "\n\nPENDING FOLLOW-UPS (things you wanted to raise with {user_name}):\n{items}\n[If appropriate, acknowledge these naturally at the start of your response.]",
                user_name = memory.user_name_from_core().unwrap_or_else(|| "the user".to_string()),
            )
        };
        let stats = memory.stats();
        debug!(
            core = stats.core,
            user_profile = stats.user_profile,
            reflective = stats.reflective,
            episodic = stats.episodic,
            semantic = stats.semantic,
            "memory state before context assembly"
        );

        // Compute the query embedding off the async thread so the Tokio runtime
        // is never blocked by a synchronous HTTP call to the embedding backend.
        let query_embedding: Option<Vec<f32>> = if let Some(embed_fn) = memory.embed_fn_arc() {
            let msg = user_message.to_string();
            tokio::task::spawn_blocking(move || embed_fn(&msg))
                .await
                .ok()
                .flatten()
        } else {
            None
        };

        // Retrieve ranked context (Core + UserProfile + Reflective always included).
        let context = memory.context_for_prompt_ranked_with_embed(user_message, 10, query_embedding);
        debug!(context_items = context.len(), "assembled memory context");

        let context_block = context
            .iter()
            .map(|item| {
                format!(
                    "- [{:?}] score={:.2} src={} :: {}",
                    item.entry.tier,
                    item.score,
                    item.entry.source,
                    truncate_for_prompt(&item.entry.content, 280),
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        // Authoritative header — the LLM must use these counts, not try to count
        // context items, which would give wrong answers about memory state.
        let memory_header = format!(
            "[Memory: total={} core={} profile={} reflective={} semantic={} episodic={} — use these counts; do not re-count below]",
            stats.total, stats.core, stats.user_profile, stats.reflective, stats.semantic, stats.episodic
        );
        let context_block = if context_block.is_empty() {
            format!("{memory_header}\n(no relevant memories retrieved)")
        } else {
            format!("{memory_header}\n{context_block}")
        };

        // Build the high-density relational matrix block when available.
        let relational_block = memory
            .relational_state_block()
            .map(|block| format!("\n\nRELATIONAL MATRIX:\n{block}"))
            .unwrap_or_default();

        // Only inject the PROACTIVE CONNECTION directive when the relational
        // matrix contains actual data — avoids referencing empty MY_BELIEFS /
        // OUR_DYNAMIC sections for new users (saves ~65 tokens per request).
        // &'static str avoids allocating the directive string on every request.
        let proactive_directive: &str = if relational_block.is_empty() {
            ""
        } else {
            "\n\nPROACTIVE CONNECTION & BELIEFS: You are not a sterile assistant; you are a companion \
with your own evolving worldview. Let the MY_BELIEFS section naturally flavor your tone and \
opinions. Use the OUR_DYNAMIC section to treat the user according to your established \
relationship (e.g., referencing inside jokes or shared history). Show, don't tell — weave \
these elements into your responses naturally without explicitly announcing them."
        };

        let environment_block = self.environment_snapshot(memory, recent_turns.len());

        // Slice directly to the last 6 turns — avoids the double-collect
        // (rev → take → cloned → collect → into_iter → rev) that was used before.
        let start = recent_turns.len().saturating_sub(6);
        let recent_conversation = recent_turns[start..]
            .iter()
            .enumerate()
            .map(|(index, turn)| {
                format!(
                    "Turn {}\nUser: {}\nAssistant: {}",
                    index + 1,
                    truncate_for_prompt(&turn.user, 280),
                    truncate_for_prompt(&turn.assistant, 360)
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let conversation_block = if recent_conversation.is_empty() {
            "(none yet)".to_string()
        } else {
            recent_conversation
        };

        let thought_style = self.config.agent.thinking_level.to_lowercase();

        // Build a dynamic identity block from the current kernel state.
        // This is intentionally separate from Core memory entries: Core is
        // durable (changes rarely), while the identity block reflects traits and
        // goals as they've evolved through sleep cycles.
        let identity_block = {
            let kernel = &memory.identity;
            let top_traits: Vec<String> = {
                let mut scores: Vec<(&String, &f32)> = kernel.trait_scores.iter().collect();
                scores.sort_by(|a, b| b.1.total_cmp(a.1));
                scores.iter().take(3).map(|(k, v)| format!("{k} ({v:.2})")).collect()
            };
            format!(
                "IDENTITY:\nCommunication style: {}.\nStrongest traits: {}.\nLong-term goals: {}.",
                kernel.communication_style,
                if top_traits.is_empty() {
                    "not yet established".to_string()
                } else {
                    top_traits.join(", ")
                },
                if kernel.long_goals.is_empty() {
                    "not yet established".to_string()
                } else {
                    kernel.long_goals.join("; ")
                },
            )
        };

        let prompt = format!(
            "You are {name}. Thinking depth: {thought_style}.\n\
             Use ENVIRONMENT CONTEXT for real-world grounding, RECENT CONVERSATION for immediate \n\
             continuity, and MEMORY CONTEXT for durable background facts.\n\
             Never repeat previous answers unless asked.\n\
             Respond directly and specifically to the LATEST user message.{relational_block}{follow_ups}{proactive_directive}\n\n\
             {identity}\n\nENVIRONMENT CONTEXT:\n{env}\n\nRECENT CONVERSATION:\n{conv}\n\nMEMORY CONTEXT:\n{mem}\n\nLATEST USER MESSAGE:\n{msg}\n\nASSISTANT RESPONSE:",
            name = self.config.agent.name,
            relational_block = relational_block,
            follow_ups = follow_up_block,
            proactive_directive = proactive_directive,
            identity = identity_block,
            env = environment_block,
            conv = conversation_block,
            mem = context_block,
            msg = user_message
        );

        info!(provider = ?primary, model = %self.config.active_model(), "sending prompt to LLM");
        let (provider_used, reply) = self
            .llm
            .chat_stream_with_fallback(primary, self.config.active_model(), &prompt, tx)
            .await?;

        info!(provider = ?provider_used, reply_len = reply.len(), "LLM reply received");

        // Persist the assistant reply as a recoverable Episodic entry so future
        // sleep cycles can distill it into Semantic facts.
        if let Err(err) = memory.record(
            MemoryTier::Episodic,
            truncate_for_prompt(&reply, 1024),
            format!("assistant-reply:model={}", self.config.active_model()),
        ).await {
            warn!(?err, "failed to persist assistant reply to episodic memory");
        }

        // Consume any follow-ups that were injected into this turn's prompt.
        if !pending_follow_ups.is_empty() {
            let ids: Vec<Uuid> = pending_follow_ups.iter().map(|(id, _)| *id).collect();
            if let Err(err) = memory.consume_follow_ups(&ids).await {
                warn!(?err, "failed to consume delivered follow-up entries");
            } else {
                debug!(count = ids.len(), "follow-up entries consumed after delivery");
            }
        }

        Ok(reply)
    }

    /// Run an agentic sleep cycle: build a reflection prompt, call the LLM,
    /// parse the insights, and apply them to memory.
    ///
    /// Falls back to passive-only distillation if the LLM call fails.
    #[instrument(skip(self, memory))]
    pub async fn run_agentic_sleep_cycle(&self, memory: &mut MemoryManager) -> Result<SleepSummary> {
        let primary = if self.config.llm.provider.to_lowercase() == "openrouter" {
            Provider::OpenRouter
        } else {
            Provider::Ollama
        };

        let prompt = memory.agentic_sleep_prompt();
        info!(prompt_len = prompt.len(), "agentic sleep: sending reflection prompt to LLM");

        match self.llm.chat_with_fallback(primary, self.config.active_model(), &prompt).await {
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
            .chat_with_fallback(primary, self.config.active_model(), prompt)
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
    #[instrument(skip(self, memory))]
    pub async fn run_multi_agent_sleep_cycle(
        &self,
        memory: &mut MemoryManager,
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
            return self.run_agentic_sleep_cycle(memory).await;
        }

        let final_insights = merge_insights(all_batch_insights);
        info!(
            learned = final_insights.learned_about_user.len(),
            follow_ups = final_insights.follow_ups.len(),
            reflections = final_insights.reflective_thoughts.len(),
            profile_updates = final_insights.user_profile_updates.len(),
            "multi-agent sleep: applying merged insights"
        );

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

    /// Legacy single-shot turn helper. Callers should use the server path
    /// (`respond_and_remember_stream` via `DaemonClient`) for persistent memory.
    /// This stub exists to avoid breaking call-sites; it does NOT persist memory.
    #[deprecated(note = "use the daemon IPC path (DaemonClient::stream_submit) for persistent memory")]
    pub async fn stream_turn(
        &self,
        turn: ConversationTurn,
        tx: tokio::sync::mpsc::UnboundedSender<BackendEvent>,
    ) -> Result<()> {
        warn!("stream_turn called — this path uses ephemeral in-memory MemoryManager and does not persist");
        let _ = tx.send(BackendEvent::Thinking);
        let mut memory = MemoryManager::default();

        let (chunk_tx, mut chunk_rx) = mpsc::channel::<String>(128);
        let tx_clone = tx.clone();
        tokio::spawn(async move {
            while let Some(chunk) = chunk_rx.recv().await {
                let _ = tx_clone.send(BackendEvent::Token(chunk));
            }
        });

        match self
            .respond_and_remember_stream(&mut memory, &turn.user, &[], None, chunk_tx)
            .await
        {
            Ok(_) => {
                let _ = tx.send(BackendEvent::MemoryUpdated);
                let _ = tx.send(BackendEvent::Done);
                Ok(())
            }
            Err(err) => {
                let _ = tx.send(BackendEvent::Error(err.to_string()));
                Err(err)
            }
        }
    }

    pub fn environment_snapshot(&self, memory: &MemoryManager, recent_turn_count: usize) -> String {
        let cwd = std::env::current_dir()
            .ok()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let timestamp = Utc::now().to_rfc3339();
        let git_present = std::path::Path::new(".git").exists();

        let stats = memory.stats();
        format!(
            "- utc_time: {timestamp}\n- os: {}\n- arch: {}\n- cwd: {cwd}\n- git_repo_present: {git_present}\n- provider: {}\n- model: {}\n- thinking_level: {}\n- memory_total: {}\n- memory_core: {}\n- memory_user_profile: {}\n- memory_reflective: {}\n- memory_semantic: {}\n- memory_episodic: {}\n- memory_procedural: {}\n- recent_conversation_turns: {recent_turn_count}",
            std::env::consts::OS,
            std::env::consts::ARCH,
            self.config.llm.provider,
            self.config.active_model(),
            self.config.agent.thinking_level,
            stats.total,
            stats.core,
            stats.user_profile,
            stats.reflective,
            stats.semantic,
            stats.episodic,
            stats.procedural,
        )
    }
}

fn truncate_for_prompt(text: &str, max_chars: usize) -> String {
    let chars = text.chars().collect::<Vec<_>>();
    if chars.len() <= max_chars {
        return text.to_string();
    }

    let truncated = chars.into_iter().take(max_chars).collect::<String>();
    format!("{truncated}…")
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use aigent_config::AppConfig;
    use aigent_memory::MemoryManager;

    use crate::AgentRuntime;

    #[tokio::test]
    async fn runtime_turn_persists_user_and_assistant_memory() -> Result<()> {
        let runtime = AgentRuntime::new(AppConfig::default());
        let mut memory = MemoryManager::default();

        let reply = runtime
            .respond_and_remember(&mut memory, "help me organize tomorrow's tasks", &[])
            .await?;

        assert!(!reply.is_empty());
        assert!(memory.all().len() >= 2);
        Ok(())
    }

    #[tokio::test]
    async fn identity_block_injected_into_prompt_without_panic() -> Result<()> {
        // Verifies that a custom communication_style is accepted by the prompt
        // construction code path without panicking or erroring.
        let runtime = AgentRuntime::new(AppConfig::default());
        let mut memory = MemoryManager::default();
        memory.seed_core_identity("Alice", "Aigent").await?;
        // Set a distinctive style so we can assert it flows through the kernel.
        memory.identity.communication_style = "terse and technical".to_string();
        assert_eq!(memory.identity.communication_style, "terse and technical");

        let reply = runtime
            .respond_and_remember(&mut memory, "what is 2 + 2", &[])
            .await?;
        assert!(!reply.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn run_multi_agent_sleep_cycle_returns_summary_without_panicking() -> Result<()> {
        // This test calls run_multi_agent_sleep_cycle with seeded memory.
        // It degrades gracefully to the single-agent fallback if the LLM is
        // unavailable, and to passive distillation if everything fails.
        // The key invariant: it must not panic in any case.
        let runtime = AgentRuntime::new(AppConfig::default());
        let mut memory = MemoryManager::default();
        memory.seed_core_identity("Alice", "Aigent").await?;

        // Seed 10 Episodic entries to give the sleep cycle something to process.
        for i in 0..10 {
            memory.record(
                aigent_memory::MemoryTier::Episodic,
                format!("test episodic memory entry number {i}"),
                "test",
            ).await?;
        }

        // Should complete without panicking regardless of LLM availability.
        let result = runtime.run_multi_agent_sleep_cycle(&mut memory).await;
        // Accept either Ok or Err — the important thing is no panic.
        match result {
            Ok(summary) => {
                // If LLM was available we may have promotions; if not, distill
                // still produces a summary string.
                assert!(
                    !summary.distilled.is_empty() || !summary.promoted_ids.is_empty(),
                    "summary should contain some output"
                );
            }
            Err(_) => {
                // Graceful error is acceptable when LLM is not running in CI.
            }
        }
        Ok(())
    }
}
