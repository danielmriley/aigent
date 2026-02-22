use anyhow::Result;
use chrono::{DateTime, Utc};
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

use aigent_config::AppConfig;
use aigent_llm::{LlmRouter, Provider};
use aigent_memory::{MemoryManager, MemoryTier, SleepSummary, parse_agentic_insights};
use tokio::sync::mpsc;

use crate::BackendEvent;

#[derive(Debug, Clone)]
pub struct ConversationTurn {
    pub user: String,
    pub assistant: String,
}

#[derive(Debug)]
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
        memory.record(MemoryTier::Episodic, user_message.to_string(), "user-input")?;

        // Improvement 1: extract structured profile signals from the user message
        // immediately, without waiting for the nightly sleep cycle.
        let profile_signals = crate::micro_profile::extract_inline_profile_signals(user_message);
        for (key, value, category) in &profile_signals {
            if let Err(err) = memory.record_user_profile_keyed(key, value, category) {
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

        // Retrieve ranked context (Core + UserProfile + Reflective always included).
        let context = memory.context_for_prompt_ranked(user_message, 10);
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

        // Build a dedicated user-profile section when available.
        let user_profile_block = memory
            .user_profile_block()
            .map(|block| format!("\n\nUSER PROFILE:\n{block}"))
            .unwrap_or_default();

        let environment_block = self.environment_snapshot(memory, recent_turns.len());

        let recent_conversation = recent_turns
            .iter()
            .rev()
            .take(6)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
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
        let prompt = format!(
            "You are {name}. Thinking depth: {thought_style}.\n\
             Use ENVIRONMENT CONTEXT for real-world grounding, RECENT CONVERSATION for immediate \n\
             continuity, and MEMORY CONTEXT for durable background facts.\n\
             Never repeat previous answers unless asked.\n\
             Respond directly and specifically to the LATEST user message.{profile_block}{follow_ups}\n\n\
             ENVIRONMENT CONTEXT:\n{env}\n\nRECENT CONVERSATION:\n{conv}\n\nMEMORY CONTEXT:\n{mem}\n\nLATEST USER MESSAGE:\n{msg}\n\nASSISTANT RESPONSE:",
            name = self.config.agent.name,
            profile_block = user_profile_block,
            follow_ups = follow_up_block,
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
        ) {
            warn!(?err, "failed to persist assistant reply to episodic memory");
        }

        // Consume any follow-ups that were injected into this turn's prompt.
        if !pending_follow_ups.is_empty() {
            let ids: Vec<Uuid> = pending_follow_ups.iter().map(|(id, _)| *id).collect();
            if let Err(err) = memory.consume_follow_ups(&ids) {
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
                memory.apply_agentic_sleep_insights(insights, summary_text)
            }
            Err(err) => {
                warn!(?err, "agentic sleep: LLM unavailable, falling back to passive distillation");
                memory.run_sleep_cycle()
            }
        }
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
}
