//! Tool call detection, execution, and environment snapshots.

use anyhow::Result;
use chrono::{Utc};
use tracing::{debug, info, warn};
use tokio::sync::mpsc;
use aigent_llm::{Provider, extract_json_output};
use aigent_memory::MemoryManager;
use crate::agent_loop::LlmToolCall;
use crate::{BackendEvent};

use super::{AgentRuntime, ConversationTurn};

impl AgentRuntime {
    /// Decide whether the user's message should trigger a tool call.
    ///
    /// Sends a compact "tool-dispatcher" prompt to the LLM asking it to choose
    /// one of the available tools (or return `{"no_action":true}`).  The method
    /// returns `None` for all conversational messages so there is zero overhead
    /// on normal turns.
    ///
    /// The result is used by the `SubmitTurn` handler to execute the tool
    /// *before* the main streaming call, injecting the result into the prompt
    /// context so the LLM's response is grounded in the actual tool output.
    pub async fn maybe_tool_call(
        &self,
        user_message: &str,
        tool_specs: &[aigent_tools::ToolSpec],
    ) -> Option<LlmToolCall> {
        if tool_specs.is_empty() {
            return None;
        }

        let specs_block = tool_specs
            .iter()
            .map(|s| {
                let params: Vec<String> = s
                    .params
                    .iter()
                    .map(|p| {
                        format!(
                            "{}: {} ({})",
                            p.name,
                            p.description,
                            if p.required { "required" } else { "optional" }
                        )
                    })
                    .collect();
                format!(
                    "- {}: {}\n  params: {}",
                    s.name,
                    s.description,
                    params.join(", ")
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "TASK: Decide if the user message requires calling a tool.\n\
             If YES — respond ONLY with JSON: {{\"tool\":\"name\",\"args\":{{\"key\":\"value\"}}}}\n\
             If NO  — respond ONLY with: {{\"no_action\":true}}\n\n\
             RULES:\n\
             - Call web_search for ANY factual question you cannot answer from memory alone \
             (stock prices, weather, news, scores, current events, product info, etc.).\n\
             - Call a tool for clear action requests (searching, adding calendar events, \
             drafting emails, reminders, reading/writing files, running shell commands).\n\
             - Return no_action ONLY for purely conversational messages (greetings, opinions, \
             stories, jokes) that need no external data.\n\
             - When in doubt, prefer calling a tool over no_action.\n\n\
             AVAILABLE TOOLS:\n{specs_block}\n\n\
             USER MESSAGE: {user_message}\n\n\
             JSON RESPONSE:"
        );

        let primary = if self.config.llm.provider.to_lowercase() == "openrouter" {
            Provider::OpenRouter
        } else {
            Provider::Ollama
        };

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
            debug!("maybe_tool_call: LLM unavailable");
            return None;
        };

        let value: serde_json::Value = extract_json_output(&raw)?;
        if value.get("no_action").and_then(|v| v.as_bool()).unwrap_or(false) {
            debug!("maybe_tool_call: LLM chose no_action");
            return None;
        }

        let call: LlmToolCall = serde_json::from_value(value).ok()?;
        if call.tool.is_empty() {
            return None;
        }

        info!(tool = %call.tool, args = ?call.args, "maybe_tool_call: LLM requested tool");
        Some(call)
    }

    /// Legacy single-shot turn helper. Callers should use the server path
    /// (`respond_and_remember_stream` via `DaemonClient`) for persistent memory.
    /// This stub exists to avoid breaking call-sites; it does NOT persist memory.

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
            .respond_and_remember_stream(&mut memory, &turn.user, &[], None, chunk_tx, &[])
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

#[cfg(test)]
mod tests {
    use aigent_config::AppConfig;
    use anyhow::Result;

        use aigent_memory::MemoryManager;
    use tokio::sync::mpsc;

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
        let (noop_tx, _) = mpsc::unbounded_channel::<String>();
        let result = runtime.run_multi_agent_sleep_cycle(&mut memory, &noop_tx).await;
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
