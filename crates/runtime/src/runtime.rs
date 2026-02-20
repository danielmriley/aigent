use anyhow::Result;
use chrono::Utc;
use tracing::{debug, info, instrument, warn};

use aigent_config::AppConfig;
use aigent_llm::{LlmRouter, Provider};
use aigent_memory::{MemoryManager, MemoryTier};
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
        self.respond_and_remember_stream(memory, user_message, recent_turns, tx)
            .await
    }

    #[instrument(skip(self, memory, tx), fields(bot = %self.config.agent.name, model = %self.config.active_model(), user_len = user_message.len()))]
    pub async fn respond_and_remember_stream(
        &self,
        memory: &mut MemoryManager,
        user_message: &str,
        recent_turns: &[ConversationTurn],
        tx: mpsc::Sender<String>,
    ) -> Result<String> {
        let primary = if self.config.llm.provider.to_lowercase() == "openrouter" {
            Provider::OpenRouter
        } else {
            Provider::Ollama
        };

        // Persist the user turn immediately so it survives a restart.
        memory.record(MemoryTier::Episodic, user_message.to_string(), "user-input")?;
        let stats = memory.stats();
        debug!(core = stats.core, episodic = stats.episodic, semantic = stats.semantic, "memory state before context assembly");

        let context = memory.context_for_prompt_ranked(user_message, 8);
        debug!(context_items = context.len(), "assembled memory context");

        let context_block = context
            .iter()
            .map(|item| {
                format!(
                    "- [{:?}] score={:.2} source={} :: {}",
                    item.entry.tier,
                    item.score,
                    item.entry.source,
                    truncate_for_prompt(&item.entry.content, 300),
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        // Authoritative header — the LLM must use these counts, not try to count
        // context items, which would give wrong answers about memory state.
        let memory_header = format!(
            "[Memory state: total={} core={} semantic={} episodic={} — do not count items below to determine these totals]",
            stats.total, stats.core, stats.semantic, stats.episodic
        );
        let context_block = if context_block.is_empty() {
            format!("{memory_header}\n(no relevant memories retrieved)")
        } else {
            format!("{memory_header}\n{context_block}")
        };
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
            "You are {}. Thinking depth: {}.\nUse ENVIRONMENT CONTEXT for real-world grounding, RECENT CONVERSATION for immediate continuity, and MEMORY CONTEXT for durable background facts.\nNever repeat previous answers unless asked.\nRespond directly and specifically to the LATEST user message.\n\nENVIRONMENT CONTEXT:\n{}\n\nRECENT CONVERSATION:\n{}\n\nMEMORY CONTEXT:\n{}\n\nLATEST USER MESSAGE:\n{}\n\nASSISTANT RESPONSE:",
            self.config.agent.name,
            thought_style,
            environment_block,
            conversation_block,
            context_block,
            user_message
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

        Ok(reply)
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
            .respond_and_remember_stream(&mut memory, &turn.user, &[], chunk_tx)
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

        format!(
            "- utc_time: {timestamp}\n- os: {}\n- arch: {}\n- cwd: {cwd}\n- git_repo_present: {git_present}\n- provider: {}\n- model: {}\n- thinking_level: {}\n- memory_total: {}\n- memory_core: {}\n- memory_semantic: {}\n- memory_episodic: {}\n- memory_procedural: {}\n- recent_conversation_turns: {recent_turn_count}",
            std::env::consts::OS,
            std::env::consts::ARCH,
            self.config.llm.provider,
            self.config.active_model(),
            self.config.agent.thinking_level,
            memory.all().len(),
            memory.entries_by_tier(MemoryTier::Core).len(),
            memory.entries_by_tier(MemoryTier::Semantic).len(),
            memory.entries_by_tier(MemoryTier::Episodic).len(),
            memory.entries_by_tier(MemoryTier::Procedural).len(),
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
