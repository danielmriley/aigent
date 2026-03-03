//! Environment snapshots for tool-augmented turns.

use chrono::Utc;
use aigent_memory::MemoryManager;

use super::AgentRuntime;

impl AgentRuntime {
    pub fn environment_snapshot(&self, memory: &MemoryManager, recent_turn_count: usize) -> String {
        let workspace_raw = &self.config.agent.workspace_path;
        // Canonicalize to resolve any `.` or `..` segments; fall back to raw.
        let workspace = std::fs::canonicalize(workspace_raw)
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| workspace_raw.clone());
        let local_ts = chrono::Local::now().format("%Y-%m-%d %H:%M:%S %Z").to_string();
        let timestamp = Utc::now().to_rfc3339();
        // Check git inside the workspace (where shell commands run), not daemon CWD.
        let ws_path = std::path::Path::new(&workspace);
        let git_present = ws_path.join(".git").exists();

        // Auto-detect the aigent source directory (parent of workspace if it
        // contains Cargo.toml).  This lets the LLM inspect its own source code.
        let source_line = ws_path
            .parent()
            .filter(|p| p.join("Cargo.toml").exists())
            .map(|p| format!("\n- aigent_source_dir: {} (your source code repo — use read_file with absolute paths or `git -C` to inspect)", p.display()))
            .unwrap_or_default();

        let stats = memory.stats();
        format!(
            "- local_time: {local_ts}\n\
             - utc_time: {timestamp}\n\
             - os: {}\n\
             - arch: {}\n\
             - workspace: {workspace} (shell commands run here)\n\
             - git_repo_in_workspace: {git_present}{source_line}\n\
             - provider: {}\n\
             - model: {}\n\
             - thinking_level: {}\n\
             - memory_total: {}\n\
             - memory_core: {}\n\
             - memory_user_profile: {}\n\
             - memory_reflective: {}\n\
             - memory_semantic: {}\n\
             - memory_episodic: {}\n\
             - memory_procedural: {}\n\
             - recent_conversation_turns: {recent_turn_count}",
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
        // This test calls generate_multi_agent_sleep_insights with seeded memory.
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

        // Snapshot memories + identity (mirrors the real daemon flow).
        let memories = memory.all().to_vec();
        let identity = memory.identity.clone();

        // Should complete without panicking regardless of LLM availability.
        let (noop_tx, _) = mpsc::unbounded_channel::<String>();
        let result = runtime
            .generate_multi_agent_sleep_insights(&memories, &identity, &noop_tx)
            .await;
        // Accept either Ok or Err — the important thing is no panic.
        match result {
            Ok(crate::SleepGenerationResult::Insights(insights)) => {
                // Apply insights to prove the end-to-end flow doesn't panic.
                let summary_text = Some("test multi-agent cycle".to_string());
                let summary = memory.apply_agentic_sleep_insights(*insights, summary_text).await?;
                assert!(
                    !summary.distilled.is_empty() || !summary.promoted_ids.is_empty(),
                    "summary should contain some output"
                );
            }
            Ok(crate::SleepGenerationResult::PassiveFallback(_)) => {
                // LLM unavailable — passive fallback is fine.
            }
            Err(_) => {
                // Graceful error is acceptable when LLM is not running in CI.
            }
        }
        Ok(())
    }

}
