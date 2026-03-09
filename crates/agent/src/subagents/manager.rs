//! Subagent manager — orchestrates parallel specialist thinking loops.
//!
//! Each specialist runs a **bounded external thinking loop** (max 3 rounds by
//! default) over a **read-only subset of the tool registry**.  This lets them
//! ground their analysis in live data (current date, web search, file reads)
//! before their `final_answer` is assembled into the captain's debate block.
//!
//! Safety guarantee: the read-only sub-registry is built from tools whose
//! `metadata.read_only == true`, so specialists can never write files, run
//! shell commands, or mutate any persistent state.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use aigent_config::SubagentsConfig;
use aigent_exec::ToolExecutor;
use aigent_llm::{ChatMessage, LlmRouter, Provider};
use aigent_thinker::{build_external_thinking_block, run_external_thinking_loop};
use aigent_tools::ToolRegistry;

use super::prompts::truncate_context;
use super::types::SubagentRole;

/// Maximum characters of the captain's system prompt to snapshot into each
/// specialist's context.  Keeps parallel calls fast on smaller context windows.
const SUBAGENT_CONTEXT_LIMIT: usize = 4_000;

/// Orchestrates parallel specialist thinking loops and collects their
/// free-form analysis texts.
pub struct SubagentManager<'a> {
    llm: &'a LlmRouter,
    config: &'a SubagentsConfig,
    ollama_model: &'a str,
    openrouter_model: &'a str,
    /// Full daemon tool registry — will be filtered to read-only subset.
    registry: Arc<ToolRegistry>,
    /// Shared executor (applies `ExecutionPolicy` before every tool call).
    executor: Arc<ToolExecutor>,
    /// Per-LLM-call timeout (mirrors captain's `step_timeout`).
    step_timeout: Duration,
    /// Per-tool-call timeout (mirrors captain's `tool_timeout`).
    tool_timeout: Duration,
}

impl<'a> SubagentManager<'a> {
    #[allow(clippy::too_many_arguments)] // all args are logically distinct; grouping would add noise
    pub fn new(
        llm: &'a LlmRouter,
        config: &'a SubagentsConfig,
        ollama_model: &'a str,
        openrouter_model: &'a str,
        registry: Arc<ToolRegistry>,
        executor: Arc<ToolExecutor>,
        step_timeout: Duration,
        tool_timeout: Duration,
    ) -> Self {
        Self {
            llm,
            config,
            ollama_model,
            openrouter_model,
            registry,
            executor,
            step_timeout,
            tool_timeout,
        }
    }

    /// Run all three specialist thinking loops in parallel and return their
    /// `final_answer` texts.
    ///
    /// The `system_prompt` is the already-built captain system prompt
    /// (memory context, identity, etc.).  Specialists receive a truncated
    /// snapshot of it as read-only context alongside a read-only tool subset.
    pub async fn run_parallel_team(
        &self,
        primary: Provider,
        user_message: &str,
        system_prompt: &str,
    ) -> Vec<(SubagentRole, String)> {
        // Build a read-only sub-registry once and share it across all three
        // specialists via Arc — no clones of the underlying tool state.
        let ro_registry = Arc::new(aigent_exec::read_only_registry(&self.registry));
        let tool_specs = ro_registry.list_specs();

        let researcher_sys = self.build_specialist_system_prompt(
            SubagentRole::Researcher,
            system_prompt,
            &tool_specs,
        );
        let planner_sys = self.build_specialist_system_prompt(
            SubagentRole::Planner,
            system_prompt,
            &tool_specs,
        );
        let critic_sys = self.build_specialist_system_prompt(
            SubagentRole::Critic,
            system_prompt,
            &tool_specs,
        );

        let max_rounds = if self.config.max_rounds == 0 {
            3
        } else {
            self.config.max_rounds
        };

        info!("subagents: launching 3 specialist thinking loops in parallel");

        let (researcher_result, planner_result, critic_result) = tokio::join!(
            self.run_specialist_loop(primary, researcher_sys, user_message, &ro_registry, "Researcher", max_rounds),
            self.run_specialist_loop(primary, planner_sys,   user_message, &ro_registry, "Planner",    max_rounds),
            self.run_specialist_loop(primary, critic_sys,    user_message, &ro_registry, "Critic",     max_rounds),
        );

        let mut results = Vec::with_capacity(3);
        if let Some(text) = researcher_result {
            results.push((SubagentRole::Researcher, text));
        }
        if let Some(text) = planner_result {
            results.push((SubagentRole::Planner, text));
        }
        if let Some(text) = critic_result {
            results.push((SubagentRole::Critic, text));
        }

        info!(succeeded = results.len(), "subagents: parallel team complete");
        results
    }

    /// Format the team's free-form analyses into the `<subagent_debate>` block
    /// that the captain consumes as pre-turn context.
    pub fn format_debate_block(results: &[(SubagentRole, String)]) -> String {
        if results.is_empty() {
            return String::new();
        }
        let mut block = String::from("<subagent_debate>\n");
        for (role, analysis) in results {
            block.push_str(&format!("=== {} ===\n{}\n\n", role.label(), analysis.trim()));
        }
        block.push_str("</subagent_debate>");
        debug!(content = %block, "subagents: debate block sent to captain");
        block
    }

    /// Per-specialist outer timeout — 90 seconds covers multiple inner rounds
    /// on a small local model.  The team degrades gracefully if one specialist
    /// times out (same pattern as the multi-agent sleep pipeline).
    const CALL_TIMEOUT: Duration = Duration::from_secs(90);

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Build the system prompt for a single specialist.
    ///
    /// Includes: role directive, truncated captain context snapshot, and the
    /// external-thinking JSON format block so the specialist can make tool
    /// calls and produce a `final_answer`.
    fn build_specialist_system_prompt(
        &self,
        role: SubagentRole,
        captain_system: &str,
        tool_specs: &[aigent_tools::ToolSpec],
    ) -> String {
        let context = truncate_context(captain_system, SUBAGENT_CONTEXT_LIMIT);
        let ext_think_block = build_external_thinking_block(tool_specs);
        let role_directive = match role {
            SubagentRole::Researcher => &self.config.researcher_prompt,
            SubagentRole::Planner => &self.config.planner_prompt,
            SubagentRole::Critic => &self.config.critic_prompt,
        };
        format!(
            "=== {role} SPECIALIST ===\n\
{role_directive}\n\n\
You have access to tools. Use them to ground your analysis in real data before \
producing your final_answer. Your final_answer will be shown to the captain agent \
as expert pre-turn research — be concise and factual.\n\n\
CAPTAIN CONTEXT (truncated):\n{context}\n\n\
{ext_think_block}"
        )
    }

    /// Run one specialist's bounded thinking loop.
    ///
    /// Returns the `final_answer` content, or `None` on LLM failure / timeout.
    /// `tool_executions` are silently dropped — sub-agents have no side effects.
    async fn run_specialist_loop(
        &self,
        primary: Provider,
        system_prompt: String,
        user_message: &str,
        ro_registry: &Arc<ToolRegistry>,
        role_label: &str,
        max_rounds: usize,
    ) -> Option<String> {
        let mut messages = vec![
            ChatMessage::system(system_prompt),
            ChatMessage::user(user_message.to_string()),
        ];

        // Drain channel: sub-agents stream to nowhere — we only want final_answer.
        let (drain_tx, mut drain_rx) = mpsc::channel::<String>(64);
        tokio::spawn(async move {
            while drain_rx.recv().await.is_some() {}
        });

        let fut = run_external_thinking_loop(
            self.llm,
            primary,
            self.ollama_model,
            self.openrouter_model,
            &mut messages,
            ro_registry,
            &self.executor,
            drain_tx,
            None, // no event_sink — sub-agents are silent to the TUI
            self.step_timeout,
            self.tool_timeout,
            max_rounds,
        );

        match tokio::time::timeout(Self::CALL_TIMEOUT, fut).await {
            Ok(Ok(result)) => {
                let content = result.content.trim().to_string();
                debug!(
                    role = role_label,
                    len = content.len(),
                    "subagent: thinking loop complete"
                );
                debug!(role = role_label, content = %content, "subagent: final answer");
                if content.is_empty() {
                    None
                } else {
                    Some(content)
                }
            }
            Ok(Err(err)) => {
                warn!(?err, role = role_label, "subagent: thinking loop failed");
                None
            }
            Err(_) => {
                warn!(role = role_label, "subagent: thinking loop timed out");
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_debate_block_empty_returns_empty() {
        assert_eq!(SubagentManager::format_debate_block(&[]), "");
    }

    #[test]
    fn format_debate_block_contains_role_and_content() {
        let results = vec![
            (SubagentRole::Researcher, "Today is 2026-03-08.".to_string()),
            (SubagentRole::Critic, "No pitfalls identified.".to_string()),
        ];
        let block = SubagentManager::format_debate_block(&results);
        assert!(block.contains("<subagent_debate>"));
        assert!(block.contains("</subagent_debate>"));
        assert!(block.contains("=== Researcher ==="));
        assert!(block.contains("Today is 2026-03-08."));
        assert!(block.contains("=== Critic ==="));
        assert!(block.contains("No pitfalls identified."));
    }
}
