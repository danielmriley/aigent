//! Unified agent turn execution.
//!
//! [`run_agent_turn`] is the single entry point for all agent reasoning,
//! whether triggered by a user message, a proactive wake-up, or a
//! scheduled cron task.  It always uses the external thinking loop —
//! the one consistent pipeline regardless of provider or interface.
//!
//! ## Internal flow
//!
//! 1. **Sub-agent debate** (optional, gated by `config.subagents.enabled`):
//!    Researcher / Planner / Critic run in parallel on a smaller model.
//!    Their findings are injected as a system message before the captain
//!    receives the user turn.
//!
//! 2. **External thinking loop**: the captain model reasons in structured
//!    JSON steps, calling tools as needed, until it produces a `final_answer`.
//!
//! All callers — `connection.rs` (user turns), `sleep.rs` (proactive
//! wake-ups), the cron scheduler (background tasks), and future interfaces
//! such as Telegram — use this function unchanged.  The only difference
//! between callers is *what messages they pass in* and *what they do with
//! the result*.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tokio::sync::mpsc;
use tracing::info;

use aigent_config::AppConfig;
use aigent_exec::ToolExecutor;
use aigent_llm::{ChatMessage, LlmRouter, Provider};
use aigent_tools::ToolRegistry;
use aigent_thinker::{EventSink, ToolLoopResult, ThinkerEvent, run_external_thinking_loop};

use crate::subagents::{SubagentManager, needs_specialists};

/// All inputs required to execute a single agent turn.
///
/// Build this struct, call [`run_agent_turn`], and handle the result.
/// Do not branch on `config.agent.external_thinking` — this function
/// handles dispatch internally and always selects the external thinking
/// loop.
pub struct AgentTurnInput<'a> {
    /// LLM router (Ollama / OpenRouter backends).
    pub llm: &'a LlmRouter,
    /// Full application config (models, timeouts, feature flags).
    pub config: &'a AppConfig,
    /// Assembled message history: `[system, …history, user_message]`.
    ///
    /// When sub-agents are enabled, the debate block is inserted just
    /// before the final user message.  The slice is mutated in-place by
    /// the external thinking loop as tool results are appended.
    pub messages: &'a mut Vec<ChatMessage>,
    /// Tool registry shared with the daemon.
    pub registry: Arc<ToolRegistry>,
    /// Tool executor shared with the daemon.
    pub executor: Arc<ToolExecutor>,
    /// Channel that receives streaming text tokens.
    ///
    /// For user-facing turns, wire this to the client write-half.
    /// For silent background turns (proactive, cron), pass a draining
    /// channel that discards the tokens.
    pub token_tx: mpsc::Sender<String>,
    /// Optional sink for structured thinker events (thoughts, tool calls).
    ///
    /// The runtime bridges these into [`crate::BackendEvent`] for the TUI.
    /// Background turns typically pass `None`.
    pub event_sink: Option<Box<dyn Fn(ThinkerEvent) + Send + Sync>>,
}

/// Run one complete agent turn through the external thinking loop.
///
/// Returns a [`ToolLoopResult`] containing the final assistant text and
/// all tool executions that occurred.  Memory recording and event
/// broadcasting are the caller's responsibility.
pub async fn run_agent_turn(input: AgentTurnInput<'_>) -> Result<ToolLoopResult> {
    let primary = Provider::from(input.config.llm.provider.as_str());
    let step_timeout = Duration::from_secs(input.config.agent.step_timeout_seconds);
    let tool_timeout = Duration::from_secs(input.config.agent.tool_timeout_secs);

    // ── Optional sub-agent debate ────────────────────────────────────────
    //
    // Specialist sub-agents (Researcher, Planner, Critic) run in parallel
    // on a smaller/faster model.  Their structured analyses are injected
    // as a system-level context block so the captain can leverage
    // multi-perspective reasoning without doing all the legwork itself.
    //
    // The captain still runs the full external thinking loop afterwards —
    // sub-agents are read-only advisors, not decision makers.
    //
    // Pure social exchanges (greetings, acks, closings) skip the pipeline
    // so that "Hello!" gets an instant captain response.  The router errs
    // toward running specialists: false positives cost a few extra seconds;
    // false negatives produce shallower answers on questions that deserved
    // deeper analysis.
    let subagent_user_msg = input.messages.last()
        .and_then(|m| m.content.as_deref())
        .unwrap_or("");

    if input.config.subagents.enabled
        && input.messages.len() >= 2
        && needs_specialists(subagent_user_msg)
    {
        let user_message = subagent_user_msg;
        let system_text = input.messages.first()
            .and_then(|m| m.content.as_deref())
            .unwrap_or("");

        // Resolve subagent model: use the subagent-specific model if set,
        // otherwise fall back to the captain's model.  This allows a smaller
        // fast model (e.g. qwen3:8b) to feed the captain (e.g. qwen3:35b).
        let subagent_ollama = if input.config.subagents.ollama_model.is_empty() {
            input.config.llm.ollama_model.as_str()
        } else {
            input.config.subagents.ollama_model.as_str()
        };
        let subagent_openrouter = if input.config.subagents.openrouter_model.is_empty() {
            input.config.llm.openrouter_model.as_str()
        } else {
            input.config.subagents.openrouter_model.as_str()
        };

        // Notify the TUI that specialists are being spawned.
        if let Some(sink) = &input.event_sink {
            sink(ThinkerEvent::SubAgentProgress(
                "Spawning specialist agents (Researcher, Planner, Critic)\u{2026}".to_string(),
            ));
        }

        let subagent_start = std::time::Instant::now();
        let manager = SubagentManager::new(
            input.llm,
            &input.config.subagents,
            subagent_ollama,
            subagent_openrouter,
            Arc::clone(&input.registry),
            Arc::clone(&input.executor),
            step_timeout,
            tool_timeout,
        );

        let team_results = manager
            .run_parallel_team(primary, user_message, system_text)
            .await;

        let debate_block = SubagentManager::format_debate_block(&team_results);

        if !debate_block.is_empty() {
            // Notify the TUI that specialists are ready.
            if let Some(sink) = &input.event_sink {
                sink(ThinkerEvent::SubAgentProgress(format!(
                    "Specialists ready ({}/3 succeeded)",
                    team_results.len()
                )));
            }
            // Insert just before the final user message so the captain
            // sees the debate as pre-turn context.
            let insert_pos = input.messages.len().saturating_sub(1);
            input.messages.insert(
                insert_pos,
                ChatMessage::system(format!(
                    "SUBAGENT ANALYSIS (use these specialist perspectives \
                     to inform your reasoning):\n{debate_block}"
                )),
            );
            info!(
                specialists = team_results.len(),
                elapsed_ms = subagent_start.elapsed().as_millis() as u64,
                "subagents: debate block injected into captain context"
            );
        } else {
            if let Some(sink) = &input.event_sink {
                sink(ThinkerEvent::SubAgentProgress(
                    "Specialists returned no usable analysis — proceeding without debate."
                        .to_string(),
                ));
            }
            info!("subagents: all specialists failed or returned empty — proceeding without debate");
        }
    }

    // ── External thinking loop ───────────────────────────────────────────
    //
    // This is THE agent loop — the one and only execution path.
    // Works identically for Ollama (local) and OpenRouter (cloud).
    let event_sink_ref: Option<&EventSink> = input.event_sink.as_deref();

    run_external_thinking_loop(
        input.llm,
        primary,
        &input.config.llm.ollama_model,
        &input.config.llm.openrouter_model,
        input.messages,
        &input.registry,
        &input.executor,
        input.token_tx,
        event_sink_ref,
        step_timeout,
        tool_timeout,
        input.config.agent.max_steps_per_turn,
    )
    .await
}
