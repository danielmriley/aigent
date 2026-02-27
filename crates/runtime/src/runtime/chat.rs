//! Conversation turn handling — respond, remember, and stream.

use uuid::Uuid;
use anyhow::Result;
use chrono::{DateTime, Utc};
use tracing::{debug, info, instrument, warn};
use tokio::sync::mpsc;
use aigent_llm::{Provider};
use aigent_memory::{MemoryEntry, MemoryManager, MemoryTier};
use crate::prompt_builder::truncate_for_prompt;

use super::AgentRuntime;
use super::ConversationTurn;

impl AgentRuntime {

    pub async fn respond_and_remember(
        &self,
        memory: &mut MemoryManager,
        user_message: &str,
        recent_turns: &[ConversationTurn],
    ) -> Result<String> {
        let (tx, _rx) = mpsc::channel(100);
        self.respond_and_remember_stream(memory, user_message, recent_turns, None, tx, &[])
            .await
    }

    #[instrument(skip(self, memory, tx, tool_specs), fields(bot = %self.config.agent.name, model = %self.config.active_model(), user_len = user_message.len()))]
    pub async fn respond_and_remember_stream(
        &self,
        memory: &mut MemoryManager,
        user_message: &str,
        recent_turns: &[ConversationTurn],
        last_turn_at: Option<DateTime<Utc>>,
        tx: mpsc::Sender<String>,
        tool_specs: &[aigent_tools::ToolSpec],
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

        // Compute the query embedding asynchronously via the embedding backend.
        let query_embedding: Option<Vec<f32>> = if let Some(embed_fn) = memory.embed_fn_arc() {
            embed_fn(user_message.to_string()).await
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

        // Inject current beliefs so the LLM can express a genuine worldview.
        // Cap to `max_beliefs_in_prompt` (sorted by confidence desc) to prevent
        // context-window bloat as beliefs accumulate over time.
        let beliefs_block = {
            let max_n = self.config.memory.max_beliefs_in_prompt;
            let mut beliefs = memory.all_beliefs();
            // Sort by composite score: confidence × 0.6 + recency × 0.25 + valence × 0.15
            // Recency factor decays as 1/(1+days) so today's beliefs score 1.0 and a
            // 30-day-old belief scores ~0.03.  The most relevant + recent beliefs always
            // appear first regardless of raw confidence.
            let now = Utc::now();
            beliefs.sort_by(|a, b| {
                let belief_score = |e: &&MemoryEntry| {
                    let days = (now - e.created_at).num_days().max(0) as f32;
                    let recency = 1.0_f32 / (1.0 + days);
                    e.confidence * 0.6 + recency * 0.25 + e.valence.clamp(0.0, 1.0) * 0.15
                };
                belief_score(b).total_cmp(&belief_score(a))
            });
            let take_n = if max_n == 0 { beliefs.len() } else { max_n.min(beliefs.len()) };
            if take_n == 0 {
                String::new()
            } else {
                let items = beliefs[..take_n]
                    .iter()
                    .map(|e| format!("- {}", e.content))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!("\n\nMY_BELIEFS:\n{items}")
            }
        };

        // Build a concise "Available tools" section so the LLM knows it can
        // autonomously request tool invocations.  Empty when the daemon was
        // started without a registry (e.g. in tests or legacy callers).
        let tools_section = if tool_specs.is_empty() {
            String::new()
        } else {
            let list = tool_specs
                .iter()
                .map(|s| {
                    if s.params.is_empty() {
                        format!("  • {}: {}", s.name, s.description)
                    } else {
                        let params = s
                            .params
                            .iter()
                            .map(|p| {
                                format!(
                                    "\"{}\" ({}){}",
                                    p.name,
                                    p.description,
                                    if p.required { " *required" } else { "" }
                                )
                            })
                            .collect::<Vec<_>>()
                            .join(", ");
                        format!("  • {}: {} — params: {}", s.name, s.description, params)
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            format!(
                "\n\nAVAILABLE TOOLS (use them autonomously when they help answer the request):\n{list}\n\
                 To invoke a tool, output a JSON block on its own line: \
                 {{\"tool\":\"name\",\"args\":{{\"key\":\"value\"}}}}"
            )
        };

        let identity = format!("{identity_block}{beliefs_block}");

        let prompt = format!(
            "You are {name}. Thinking depth: {thought_style}.\n\
             Use ENVIRONMENT CONTEXT for real-world grounding, RECENT CONVERSATION for immediate \n\
             continuity, and MEMORY CONTEXT for durable background facts.\n\
             Never repeat previous answers unless asked.\n\
             Respond directly and specifically to the LATEST user message.{relational_block}{follow_ups}{proactive_directive}\n\n\
             {identity}{tools_section}\n\nENVIRONMENT CONTEXT:\n{env}\n\nRECENT CONVERSATION:\n{conv}\n\nMEMORY CONTEXT:\n{mem}\n\nLATEST USER MESSAGE:\n{msg}\n\nASSISTANT RESPONSE:",
            name = self.config.agent.name,
            relational_block = relational_block,
            follow_ups = follow_up_block,
            proactive_directive = proactive_directive,
            identity = identity,
            tools_section = tools_section,
            env = environment_block,
            conv = conversation_block,
            mem = context_block,
            msg = user_message
        );

        info!(provider = ?primary, model = %self.config.active_model(), "sending prompt to LLM");
        let (provider_used, reply) = self
            .llm
            .chat_stream_with_fallback(
                primary,
                &self.config.llm.ollama_model,
                &self.config.llm.openrouter_model,
                &prompt,
                tx,
            )
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

}
