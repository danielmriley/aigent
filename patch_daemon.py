import re

with open('crates/aigent-daemon/src/lib.rs', 'r') as f:
    content = f.read()

# Add tokio::sync::mpsc to imports
content = content.replace('use aigent_memory::{MemoryManager, MemoryTier};', 'use aigent_memory::{MemoryManager, MemoryTier};\nuse tokio::sync::mpsc;')

# Add respond_and_remember_stream to AgentRuntime
daemon_old = """    pub async fn respond_and_remember(
        &self,
        memory: &mut MemoryManager,
        user_message: &str,
        recent_turns: &[ConversationTurn],
    ) -> Result<String> {
        let primary = if self.config.llm.provider.to_lowercase() == "openrouter" {
            Provider::OpenRouter
        } else {
            Provider::Ollama
        };

        memory.record(MemoryTier::Episodic, user_message.to_string(), "user-input")?;

        let context = memory.context_for_prompt_ranked(user_message, 8);
        let context_block = context
            .iter()
            .map(|item| {
                format!(
                    "- [{:?}] score={:.2} source={} created={} hash={} why={} :: {}",
                    item.entry.tier,
                    item.score,
                    item.entry.source,
                    item.entry.created_at.to_rfc3339(),
                    item.entry.provenance_hash,
                    truncate_for_prompt(&item.rationale, 160),
                    truncate_for_prompt(&item.entry.content, 280),
                )
            })
            .collect::<Vec<_>>()
            .join("\\n");
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
                    "Turn {}\\nUser: {}\\nAssistant: {}",
                    index + 1,
                    truncate_for_prompt(&turn.user, 280),
                    truncate_for_prompt(&turn.assistant, 360)
                )
            })
            .collect::<Vec<_>>()
            .join("\\n\\n");

        let conversation_block = if recent_conversation.is_empty() {
            "(none yet)".to_string()
        } else {
            recent_conversation
        };

        let thought_style = self.config.agent.thinking_level.to_lowercase();
        let prompt = format!(
            "You are {}. Thinking depth: {}.\\nUse ENVIRONMENT CONTEXT for real-world grounding, RECENT CONVERSATION for immediate continuity, and MEMORY CONTEXT for durable background facts.\\nNever repeat previous answers unless asked.\\nRespond directly and specifically to the LATEST user message.\\n\\nENVIRONMENT CONTEXT:\\n{}\\n\\nRECENT CONVERSATION:\\n{}\\n\\nMEMORY CONTEXT:\\n{}\\n\\nLATEST USER MESSAGE:\\n{}\\n\\nASSISTANT RESPONSE:",
            self.config.agent.name,
            thought_style,
            environment_block,
            conversation_block,
            context_block,
            user_message
        );

        let (provider_used, reply) = self
            .llm
            .chat_with_fallback(primary, self.config.active_model(), &prompt)
            .await?;
        memory.record(
            MemoryTier::Semantic,
            format!(
                "assistant replied via {:?} for model {}",
                provider_used,
                self.config.active_model()
            ),
            format!("assistant-turn:model={}", self.config.active_model()),
        )?;

        Ok(reply)
    }"""

daemon_new = """    pub async fn respond_and_remember(
        &self,
        memory: &mut MemoryManager,
        user_message: &str,
        recent_turns: &[ConversationTurn],
    ) -> Result<String> {
        let (tx, _rx) = mpsc::channel(100);
        self.respond_and_remember_stream(memory, user_message, recent_turns, tx).await
    }

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

        memory.record(MemoryTier::Episodic, user_message.to_string(), "user-input")?;

        let context = memory.context_for_prompt_ranked(user_message, 8);
        let context_block = context
            .iter()
            .map(|item| {
                format!(
                    "- [{:?}] score={:.2} source={} created={} hash={} why={} :: {}",
                    item.entry.tier,
                    item.score,
                    item.entry.source,
                    item.entry.created_at.to_rfc3339(),
                    item.entry.provenance_hash,
                    truncate_for_prompt(&item.rationale, 160),
                    truncate_for_prompt(&item.entry.content, 280),
                )
            })
            .collect::<Vec<_>>()
            .join("\\n");
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
                    "Turn {}\\nUser: {}\\nAssistant: {}",
                    index + 1,
                    truncate_for_prompt(&turn.user, 280),
                    truncate_for_prompt(&turn.assistant, 360)
                )
            })
            .collect::<Vec<_>>()
            .join("\\n\\n");

        let conversation_block = if recent_conversation.is_empty() {
            "(none yet)".to_string()
        } else {
            recent_conversation
        };

        let thought_style = self.config.agent.thinking_level.to_lowercase();
        let prompt = format!(
            "You are {}. Thinking depth: {}.\\nUse ENVIRONMENT CONTEXT for real-world grounding, RECENT CONVERSATION for immediate continuity, and MEMORY CONTEXT for durable background facts.\\nNever repeat previous answers unless asked.\\nRespond directly and specifically to the LATEST user message.\\n\\nENVIRONMENT CONTEXT:\\n{}\\n\\nRECENT CONVERSATION:\\n{}\\n\\nMEMORY CONTEXT:\\n{}\\n\\nLATEST USER MESSAGE:\\n{}\\n\\nASSISTANT RESPONSE:",
            self.config.agent.name,
            thought_style,
            environment_block,
            conversation_block,
            context_block,
            user_message
        );

        let (provider_used, reply) = self
            .llm
            .chat_stream_with_fallback(primary, self.config.active_model(), &prompt, tx)
            .await?;
        memory.record(
            MemoryTier::Semantic,
            format!(
                "assistant replied via {:?} for model {}",
                provider_used,
                self.config.active_model()
            ),
            format!("assistant-turn:model={}", self.config.active_model()),
        )?;

        Ok(reply)
    }"""

content = content.replace(daemon_old, daemon_new)

with open('crates/aigent-daemon/src/lib.rs', 'w') as f:
    f.write(content)
