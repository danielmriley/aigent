//! ReAct-aware tool loop — wraps structured tool calling with explicit
//! Think → Act → Observe → Critique phases.
//!
//! This module adds a thin orchestration layer on top of [`crate::tool_loop`]
//! that:
//!
//! 1. Drives an [`AgentLoop`] state machine through ReAct phases.
//! 2. Emits [`BackendEvent::ReactPhaseChanged`] so the TUI can render
//!    phase transitions in real-time.
//! 3. Records [`ReactSnapshot`]s for procedural memory / debugging.
//! 4. Supports an optional critique pass where the LLM self-evaluates
//!    before deciding to iterate or finish.
//!
//! When critique is disabled (default for fast mode), the loop behaves
//! identically to [`run_tool_loop`] but with phase tracking overlaid.

use anyhow::Result;
use tokio::sync::mpsc;
use tracing::debug;

use aigent_exec::ToolExecutor;
use aigent_llm::{ChatMessage, LlmRouter, Provider};
use aigent_tools::ToolRegistry;

use crate::agent_loop::{AgentLoop, ReactPhase, ReactSnapshot, SwarmRole};
use crate::events::BackendEvent;
use crate::tool_loop::{ToolExecution, ToolLoopResult};

// ── Configuration ────────────────────────────────────────────────────────────

/// Options for the ReAct loop.
#[derive(Debug, Clone)]
pub struct ReactConfig {
    /// Maximum ReAct rounds (Think→Act→Observe→Critique = 1 round).
    pub max_rounds: usize,
    /// Whether to run a critique phase after each observe step.
    /// When false, the loop only transitions Think → Act → Observe → Think.
    pub enable_critique: bool,
    /// Swarm role for this agent (None = solo agent).
    pub role: Option<SwarmRole>,
}

impl Default for ReactConfig {
    fn default() -> Self {
        Self {
            max_rounds: 5,
            enable_critique: false,
            role: None,
        }
    }
}

/// Result of the ReAct loop.
#[derive(Debug, Clone)]
pub struct ReactLoopResult {
    /// Same as ToolLoopResult.
    pub tool_result: ToolLoopResult,
    /// All ReAct snapshots recorded during the loop.
    pub snapshots: Vec<ReactSnapshot>,
    /// Total ReAct rounds completed.
    pub rounds_completed: usize,
}

// ── Main loop ────────────────────────────────────────────────────────────────

/// Run the ReAct-aware tool loop.
///
/// This is the primary entry point for agent turns when `use_native_calling`
/// is enabled.  It wraps the structured tool loop with explicit ReAct phase
/// tracking and optional self-critique.
#[allow(clippy::too_many_arguments)]
pub async fn run_react_loop(
    config: &ReactConfig,
    llm: &LlmRouter,
    primary: Provider,
    ollama_model: &str,
    openrouter_model: &str,
    messages: &mut Vec<ChatMessage>,
    tools_json: Option<&serde_json::Value>,
    tool_registry: &ToolRegistry,
    tool_executor: &ToolExecutor,
    token_tx: mpsc::Sender<String>,
    event_tx: Option<&tokio::sync::broadcast::Sender<BackendEvent>>,
) -> Result<ReactLoopResult> {
    let mut agent_loop = if let Some(role) = config.role {
        AgentLoop::with_role(config.max_rounds, role)
    } else {
        AgentLoop::new(config.max_rounds)
    };

    let mut all_executions: Vec<ToolExecution> = Vec::new();
    let mut final_content = String::new();
    let mut final_provider = primary;

    // Emit initial phase.
    emit_phase_change(&agent_loop, event_tx);

    while !agent_loop.is_done() {
        match agent_loop.phase {
            ReactPhase::Think => {
                debug!(
                    round = agent_loop.round,
                    "react: Think phase — sending to LLM"
                );

                // The "think" phase is the LLM call.  On the last allowed
                // round, omit tools to force a text answer.
                let effective_tools =
                    if agent_loop.round < agent_loop.max_rounds - 1 {
                        tools_json
                    } else {
                        None
                    };

                let response = llm
                    .chat_messages_stream(
                        primary,
                        ollama_model,
                        openrouter_model,
                        messages,
                        effective_tools,
                        token_tx.clone(),
                        false,
                    )
                    .await?;

                final_provider = response.provider;

                if response.tool_calls.is_empty() {
                    // No tools requested — record snapshot and finish.
                    final_content = response.content.clone();
                    agent_loop.record_snapshot(ReactSnapshot {
                        phase: ReactPhase::Think,
                        round: agent_loop.round,
                        max_rounds: agent_loop.max_rounds,
                        thought: Some(response.content),
                        actions: vec![],
                        observation: None,
                        critique: None,
                        answer: Some(final_content.clone()),
                    });
                    agent_loop.finish();
                    emit_phase_change(&agent_loop, event_tx);
                    break;
                }

                // Record think snapshot with planned actions.
                let planned_actions: Vec<_> = response
                    .tool_calls
                    .iter()
                    .map(|tc| crate::agent_loop::LlmToolCall {
                        tool: tc.function.name.clone(),
                        args: tc
                            .function
                            .arguments
                            .as_object()
                            .map(|o| {
                                o.iter()
                                    .map(|(k, v)| (k.clone(), v.clone()))
                                    .collect()
                            })
                            .unwrap_or_default(),
                    })
                    .collect();

                agent_loop.record_snapshot(ReactSnapshot {
                    phase: ReactPhase::Think,
                    round: agent_loop.round,
                    max_rounds: agent_loop.max_rounds,
                    thought: if response.content.is_empty() {
                        None
                    } else {
                        Some(response.content.clone())
                    },
                    actions: planned_actions,
                    observation: None,
                    critique: None,
                    answer: None,
                });

                // Append assistant tool-call message.
                messages.push(ChatMessage::assistant_tool_calls(
                    response.tool_calls.clone(),
                ));

                // Advance to Act phase.
                agent_loop.advance();
                emit_phase_change(&agent_loop, event_tx);

                // Execute tool calls immediately (Act phase).
                let executions = crate::tool_loop::execute_tool_calls_public(
                    &response.tool_calls,
                    tool_registry,
                    tool_executor,
                    event_tx,
                )
                .await;

                // Append tool results to messages.
                for (call, exec) in
                    response.tool_calls.iter().zip(executions.iter())
                {
                    messages.push(ChatMessage::tool_result(
                        &call.id,
                        &exec.output,
                    ));
                }

                all_executions.extend(executions.clone());

                // Advance to Observe phase.
                agent_loop.advance();
                emit_phase_change(&agent_loop, event_tx);

                // Record observation (summary of tool results).
                let obs_summary: String = executions
                    .iter()
                    .map(|e| {
                        let status = if e.success { "ok" } else { "fail" };
                        let snip = if e.output.len() > 200 {
                            format!("{}…", &e.output[..200])
                        } else {
                            e.output.clone()
                        };
                        format!("[{} {}] {}", e.tool_name, status, snip)
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                agent_loop.record_snapshot(ReactSnapshot {
                    phase: ReactPhase::Observe,
                    round: agent_loop.round,
                    max_rounds: agent_loop.max_rounds,
                    thought: None,
                    actions: vec![],
                    observation: Some(obs_summary),
                    critique: None,
                    answer: None,
                });

                // Advance to Critique (or back to Think).
                if config.enable_critique {
                    agent_loop.advance(); // → Critique
                    emit_phase_change(&agent_loop, event_tx);

                    // TODO: Full critique phase would do another LLM call
                    // asking "Is this sufficient? Score the result."
                    // For now, auto-advance.
                    agent_loop.advance(); // Critique → Think (increments round)
                    emit_phase_change(&agent_loop, event_tx);
                } else {
                    // Skip Critique: Observe → Think directly.
                    // Manually advance: Observe → Critique → Think.
                    agent_loop.advance(); // → Critique
                    agent_loop.advance(); // Critique → Think (increments round)
                    emit_phase_change(&agent_loop, event_tx);
                }
            }
            ReactPhase::Act
            | ReactPhase::Observe
            | ReactPhase::Critique => {
                // These are handled inline above after Think.
                // If we somehow land here, just advance.
                agent_loop.advance();
                emit_phase_change(&agent_loop, event_tx);
            }
            ReactPhase::Done => break,
        }
    }

    // If exhausted without final text, build a summary.
    if final_content.is_empty() && !all_executions.is_empty() {
        final_content = all_executions
            .iter()
            .map(|e| {
                format!(
                    "[{}]: {}",
                    e.tool_name,
                    &e.output[..e.output.len().min(500)]
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");
    }

    let snapshots = agent_loop.history.clone();
    let rounds = agent_loop.round;

    Ok(ReactLoopResult {
        tool_result: ToolLoopResult {
            provider: final_provider,
            content: final_content,
            tool_executions: all_executions,
        },
        snapshots,
        rounds_completed: rounds,
    })
}

// ── Sub-agent spawning ───────────────────────────────────────────────────────

/// Result from a sub-agent task.
#[derive(Debug, Clone)]
pub struct SubAgentResult {
    pub role: SwarmRole,
    pub success: bool,
    pub output: String,
    pub snapshots: Vec<ReactSnapshot>,
}

/// Spawn a sub-agent with its own ReAct loop running concurrently.
///
/// The sub-agent gets its own message history (starting from `messages`)
/// and runs independently.  Returns a `JoinHandle` that resolves to
/// the sub-agent's result.
///
/// This is the building block for swarm patterns:
/// - Supervisor spawns Researcher + Executor sub-agents
/// - Each sub-agent runs its own ReAct loop
/// - Results are collected and merged by the supervisor
#[allow(clippy::too_many_arguments)]
pub fn spawn_sub_agent(
    role: SwarmRole,
    task_description: String,
    llm: LlmRouter,
    primary: Provider,
    ollama_model: String,
    openrouter_model: String,
    mut messages: Vec<ChatMessage>,
    tools_json: Option<serde_json::Value>,
    tool_registry: &'static ToolRegistry,
    tool_executor: &'static ToolExecutor,
    event_tx: Option<tokio::sync::broadcast::Sender<BackendEvent>>,
) -> tokio::task::JoinHandle<Result<SubAgentResult>> {
    // Emit spawn event.
    if let Some(ref tx) = event_tx {
        let _ = tx.send(BackendEvent::SubAgentSpawned {
            role,
            task: task_description.clone(),
        });
    }

    // Add the task as a user message for the sub-agent.
    messages.push(ChatMessage::user(format!(
        "[Sub-agent role: {}] Task: {}",
        role, task_description
    )));

    let config = ReactConfig {
        max_rounds: 3,
        enable_critique: false,
        role: Some(role),
    };

    tokio::spawn(async move {
        let (token_tx, mut token_rx) = mpsc::channel(256);

        // Drain tokens (sub-agents don't stream to the user).
        tokio::spawn(async move {
            while token_rx.recv().await.is_some() {}
        });

        let tools_ref = tools_json.as_ref();
        let event_ref = event_tx.as_ref();

        let result = run_react_loop(
            &config,
            &llm,
            primary,
            &ollama_model,
            &openrouter_model,
            &mut messages,
            tools_ref,
            tool_registry,
            tool_executor,
            token_tx,
            event_ref,
        )
        .await?;

        // Emit completion event.
        if let Some(ref tx) = event_tx {
            let _ = tx.send(BackendEvent::SubAgentCompleted {
                role,
                success: !result.tool_result.content.is_empty(),
                summary: if result.tool_result.content.len() > 200 {
                    format!("{}…", &result.tool_result.content[..200])
                } else {
                    result.tool_result.content.clone()
                },
            });
        }

        Ok(SubAgentResult {
            role,
            success: !result.tool_result.content.is_empty(),
            output: result.tool_result.content,
            snapshots: result.snapshots,
        })
    })
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn emit_phase_change(
    agent_loop: &AgentLoop,
    event_tx: Option<&tokio::sync::broadcast::Sender<BackendEvent>>,
) {
    if let Some(tx) = event_tx {
        let _ = tx.send(BackendEvent::ReactPhaseChanged {
            phase: agent_loop.phase,
            round: agent_loop.round as u32,
            max_rounds: agent_loop.max_rounds as u32,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn react_config_defaults() {
        let cfg = ReactConfig::default();
        assert_eq!(cfg.max_rounds, 5);
        assert!(!cfg.enable_critique);
        assert!(cfg.role.is_none());
    }

    #[test]
    fn agent_loop_phase_tracking() {
        let mut aloop = AgentLoop::new(2);
        assert_eq!(aloop.phase, ReactPhase::Think);

        aloop.advance(); // → Act
        assert_eq!(aloop.phase, ReactPhase::Act);

        aloop.advance(); // → Observe
        assert_eq!(aloop.phase, ReactPhase::Observe);

        aloop.advance(); // → Critique
        assert_eq!(aloop.phase, ReactPhase::Critique);

        aloop.advance(); // → Think (round 1)
        assert_eq!(aloop.phase, ReactPhase::Think);
        assert_eq!(aloop.round, 1);

        // Run through again.
        aloop.advance(); // → Act
        aloop.advance(); // → Observe
        aloop.advance(); // → Critique
        aloop.advance(); // → Done (round 2 = max)
        assert_eq!(aloop.phase, ReactPhase::Done);
        assert!(aloop.is_done());
    }

    #[test]
    fn agent_loop_finish_early() {
        let mut aloop = AgentLoop::new(5);
        aloop.finish();
        assert!(aloop.is_done());
        assert_eq!(aloop.round, 0);
    }

    #[test]
    fn agent_loop_with_role() {
        let aloop = AgentLoop::with_role(3, SwarmRole::Researcher);
        assert_eq!(aloop.role, Some(SwarmRole::Researcher));
        assert_eq!(aloop.max_rounds, 3);
    }
}
