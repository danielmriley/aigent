//! Unix domain socket connection handling and command dispatch.

use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::sync::{Mutex, broadcast, mpsc, watch};
use tracing::{info, warn};

use aigent_config::AppConfig;
use aigent_memory::MemoryTier;

use crate::{BackendEvent, ClientCommand, ConversationTurn, ServerEvent};

use super::{DaemonState, is_in_window, safe_truncate};

pub(super) async fn handle_connection(
    stream: UnixStream,
    state: Arc<Mutex<DaemonState>>,
    shutdown_tx: watch::Sender<bool>,
) -> Result<()> {
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);
    let mut line = String::new();
    if reader.read_line(&mut line).await? == 0 {
        return Ok(());
    }

    let command: ClientCommand = serde_json::from_str(line.trim())?;

    // Clone broadcast sender cheaply before any long operation.
    let event_tx = state.lock().await.event_tx.clone();

    match command {
        // ── Persistent subscription — keeps connection alive and forwards
        //    all externally-sourced BackendEvents to the connected client.
        ClientCommand::Subscribe => {
            let mut rx = event_tx.subscribe();
            loop {
                match rx.recv().await {
                    Ok(event) => {
                        if send_event(&mut write_half, ServerEvent::Backend(event))
                            .await
                            .is_err()
                        {
                            break; // client disconnected
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!(n, "subscribe client lagged; {n} events dropped");
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
            return Ok(());
        }
        ClientCommand::SubmitTurn { user, source } => {
            let is_external = source != "tui";
            let turn_start = std::time::Instant::now();
            // Broadcast the user message so TUI subscribers can show the bubble.
            if is_external {
                let _ = event_tx.send(BackendEvent::ExternalTurn {
                    source: source.clone(),
                    content: user.clone(),
                });
            }
            send_event(
                &mut write_half,
                ServerEvent::Backend(BackendEvent::Thinking),
            )
            .await?;

            let (chunk_tx, mut chunk_rx) = mpsc::channel::<String>(128);
            let writer = Arc::new(Mutex::new(write_half));
            let writer_clone = writer.clone();
            let event_tx_stream = if is_external { Some(event_tx.clone()) } else { None };
            let streamer = tokio::spawn(async move {
                while let Some(chunk) = chunk_rx.recv().await {
                    {
                        let mut guard = writer_clone.lock().await;
                        let _ = send_event(
                            &mut guard,
                            ServerEvent::Backend(BackendEvent::Token(chunk.clone())),
                        )
                        .await;
                    }
                    if let Some(ref tx) = event_tx_stream {
                        let _ = tx.send(BackendEvent::Token(chunk));
                    }
                }
            });

            // Phase 1: take what we need from state and RELEASE THE LOCK before
            // any LLM call.  This keeps GetStatus / GetMemoryPeek responsive
            // while the turn is being processed.
            //

                // ═══════════════════════════════════════════════════════════════
                // NEW PATH: Native structured tool calling via chat messages API
                // ═══════════════════════════════════════════════════════════════
                //
                // Snapshot-merge pattern: do all memory prep under the state
                // lock, extract owned values, release the lock, then run the
                // LLM call without holding any lock.  The canonical
                // MemoryManager stays in DaemonState the entire time so
                // background tasks never see an empty shell.
                let (rt_clone, recent, tool_specs, registry, executor, conv_summary,
                     timing,
                     context, stats, identity_block, beliefs_block, user_name, relational_block,
                     memory_prep_elapsed) = {
                    let mut s = state.lock().await;
                    let rt = s.runtime.clone();

                    // Record user input while we hold the lock.
                    let _: Result<_, _> = s.memory.record(aigent_memory::MemoryTier::Episodic, user.clone(), "user-input").await;

                    let t = rt.config.debug.timing;
                    let phase_start = std::time::Instant::now();

                    // Build the system prompt using the existing prompt builder.
                    let query_embedding: Option<Vec<f32>> = if let Some(embed_fn) = s.memory.embed_fn_arc() {
                        embed_fn(user.to_string()).await
                    } else {
                        None
                    };
                    let context = s.memory.context_for_prompt_ranked_with_embed(&user, 10, query_embedding);
                    let stats = s.memory.stats();

                    // Pre-compute memory blocks for the prompt builder (pure function).
                    let identity_block = s.memory.cached_identity_block().to_string();
                    let beliefs_block = s.memory.cached_beliefs_block(rt.config.memory.max_beliefs_in_prompt).to_string();
                    let user_name = s.memory.user_name_from_core();
                    let relational_block = s.memory.relational_state_block(
                        rt.config.memory.max_relational_in_prompt,
                    );
                    let mpe = phase_start.elapsed();

                    let recent = s.recent_turns.iter().cloned().collect::<Vec<_>>();
                    let specs = s.tool_registry.list_specs();
                    let reg = Arc::clone(&s.tool_registry);
                    let exe = Arc::clone(&s.tool_executor);
                    let csumm = s.conversation_summary.clone();
                    (rt, recent, specs, reg, exe, csumm,
                     t,
                     context, stats, identity_block, beliefs_block, user_name, relational_block,
                     mpe)
                };
                // Lock released — memory stays in DaemonState, accessible to background tasks.

                // Tools always go as text in the external thinking system prompt.
                let prompt_inputs = aigent_prompt::PromptInputs {
                    config: &rt_clone.config,
                    user_message: &user,
                    recent_turns: &recent,
                    tool_specs: &tool_specs,
                    pending_follow_ups: &[],
                    context_items: &context,
                    stats,
                    identity_block,
                    beliefs_block,
                    user_name,
                    relational_block,
                    conversation_summary: conv_summary,
                    chat_only: false,
                };

                let build_start = std::time::Instant::now();
                let system_prompt = aigent_prompt::build_chat_prompt(&prompt_inputs);
                let build_elapsed = build_start.elapsed();
                if timing {
                    info!("[TIMING] memory_prep={}ms prompt_build={}ms chars={} (~{} tokens)",
                        memory_prep_elapsed.as_millis(),
                        build_elapsed.as_millis(),
                        system_prompt.len(),
                        system_prompt.len() / 4);
                }

                // Build the messages array: system + history + current user turn.
                let mut messages = vec![aigent_llm::ChatMessage::system(&system_prompt)];
                for turn in &recent {
                    messages.push(aigent_llm::ChatMessage::user(&turn.user));
                    messages.push(aigent_llm::ChatMessage::assistant(&turn.assistant));
                }
                messages.push(aigent_llm::ChatMessage::user(&user));

                // ── Unified agent turn ───────────────────────────────────────
                // All reasoning goes through run_agent_turn — one pipeline,
                // external thinking loop, optional sub-agent debate.
                let llm_start = std::time::Instant::now();

                // Bridge thinker events into the runtime BackendEvent system.
                let event_tx_clone = event_tx.clone();
                let event_sink = move |evt: aigent_thinker::ThinkerEvent| {
                    use aigent_thinker::ThinkerEvent;
                    let backend_evt = match evt {
                        ThinkerEvent::ToolCallStart(info) => crate::BackendEvent::ToolCallStart(info),
                        ThinkerEvent::ToolCallEnd(result) => crate::BackendEvent::ToolCallEnd(result),
                        ThinkerEvent::AgentThought(thought) => crate::BackendEvent::AgentThought(thought),
                    };
                    let _ = event_tx_clone.send(backend_evt);
                };

                let loop_result = aigent_agent::run_agent_turn(aigent_agent::AgentTurnInput {
                    llm: &rt_clone.llm,
                    config: &rt_clone.config,
                    messages: &mut messages,
                    registry: Arc::clone(&registry),
                    executor: Arc::clone(&executor),
                    token_tx: chunk_tx,
                    event_sink: Some(Box::new(event_sink)),
                }).await;

                let llm_elapsed = llm_start.elapsed();
                if timing {
                    info!("[TIMING] llm_response={}ms", llm_elapsed.as_millis());
                }

                let _ = streamer.await;

                // ── Post-LLM phase: re-acquire lock for memory recording ─────
                // The canonical MemoryManager was never displaced from state,
                // so any concurrent writes during the LLM call are preserved.
                let memory_start = std::time::Instant::now();
                // External thinking always handles its own internal reasoning;
                // a separate reflection pass is not needed.
                let skip_reflection = true;

                let outcome: Result<Vec<BackendEvent>, anyhow::Error> = {
                    let mut s = state.lock().await;

                    let reply_result: Result<String, anyhow::Error> = match loop_result {
                        Ok(ref result) => {
                            // Record tool invocations to Episodic memory so
                            // the sleep cycle can distill procedural patterns.
                            // Raw tool logs belong in Episodic (transient);
                            // Procedural is reserved for learned patterns.
                            //
                            // "Pain hook": tool failures also write a direct
                            // Procedural memory so the agent has immediate
                            // awareness of failure patterns even before the
                            // next sleep cycle runs.
                            for exec in &result.tool_executions {
                                let status = if exec.success { "succeeded" } else { "failed" };
                                let episodic_text = format!(
                                    "Tool '{}' {} ({}ms). Output: {}",
                                    exec.tool_name,
                                    status,
                                    exec.duration_ms,
                                    safe_truncate(&exec.output, 400),
                                );
                                let _: Result<_, _> = s.memory.record(
                                    MemoryTier::Episodic,
                                    episodic_text,
                                    format!("tool-log:{}", exec.tool_name),
                                ).await;

                                // Pain hook: failed tools → immediate Procedural signal.
                                if !exec.success {
                                    let pain_text = format!(
                                        "CAUTION: Tool '{}' failed. Error: {}",
                                        exec.tool_name,
                                        safe_truncate(&exec.output, 200),
                                    );
                                    let _: Result<_, _> = s.memory.record(
                                        MemoryTier::Procedural,
                                        pain_text,
                                        format!("tool-failure:{}", exec.tool_name),
                                    ).await;
                                }
                            }
                            // Record assistant reply
                            if !result.content.is_empty() {
                                let model_tag = rt_clone.config.active_model().to_string();
                                let _: Result<_, _> = s.memory.record(
                                    aigent_memory::MemoryTier::Episodic,
                                    aigent_prompt::truncate_for_prompt(&result.content, 1024),
                                    format!("assistant-reply:model={}", model_tag),
                                ).await;
                            }
                            // Persist reasoning traces when enabled.
                            if rt_clone.config.memory.store_reasoning_traces {
                                for trace in &result.reasoning_traces {
                                    if !trace.is_empty() {
                                        let _: Result<_, _> = s.memory.record(
                                            aigent_memory::MemoryTier::Reflective,
                                            aigent_prompt::truncate_for_prompt(trace, 500),
                                            "agent-reasoning".to_string(),
                                        ).await;
                                    }
                                }
                            }
                            Ok(result.content.clone())
                        }
                        Err(e) => Err(e),
                    };

                    // Reflect on the exchange — skip when:
                    //  • external thinking mode is active (has its own loop)
                    //  • the router took the CHAT fast-path
                    //  • `memory.reflection_enabled` is false (saves LLM tokens)
                    let skip_reflection = skip_reflection
                        || !rt_clone.config.memory.reflection_enabled;
                    let reflect_events: Vec<BackendEvent> = if skip_reflection {
                        vec![]
                    } else {
                        match reply_result {
                            Ok(ref reply) => rt_clone
                                .inline_reflect(&mut s.memory, &user, reply)
                                .await
                                .unwrap_or_default(),
                            Err(_) => vec![],
                        }
                    };

                    let _ = s.memory.flush_all();

                    match reply_result {
                        Ok(reply) => {
                            s.last_turn_at = Some(Utc::now());
                            s.recent_turns.push_back(ConversationTurn {
                                user: user.clone(),
                                assistant: reply,
                            });
                            s.turn_count += 1;

                            // Spawn background conversation summarization when
                            // the turn buffer crosses the threshold.
                            if s.recent_turns.len() >= aigent_agent::SUMMARIZE_THRESHOLD {
                                let state3 = state.clone();
                                let rt3 = s.runtime.clone();
                                tokio::spawn(async move {
                                    let (summary_opt, turns_snapshot) = {
                                        let s = state3.lock().await;
                                        let turns: Vec<ConversationTurn> =
                                            s.recent_turns.iter().cloned().collect();
                                        (s.conversation_summary.clone(), turns)
                                    };
                                    match rt3.summarize_conversation(
                                        summary_opt.as_deref(),
                                        &turns_snapshot,
                                    ).await {
                                        Ok(Some((summary, kept))) => {
                                            let mut s = state3.lock().await;
                                            s.conversation_summary = Some(summary);
                                            s.recent_turns = kept.into_iter().collect();
                                            tracing::info!(
                                                kept = s.recent_turns.len(),
                                                "conversation summarized"
                                            );
                                        }
                                        Ok(None) => {} // below threshold after re-check
                                        Err(e) => {
                                            tracing::warn!(?e, "background summarization failed");
                                        }
                                    }
                                });
                            }

                            if s.runtime.config.memory.auto_sleep_turn_interval > 0
                                && s.turn_count % s.runtime.config.memory.auto_sleep_turn_interval == 0
                            {
                                let state2 = state.clone();
                                let event_tx2 = s.event_tx.clone();
                                tokio::spawn(async move {
                                    let (rt, memories, identity) = {
                                        let s = state2.lock().await;
                                        (s.runtime.clone(), s.memory.all().to_vec(), s.memory.identity.clone())
                                    };
                                    // NOTE: do NOT broadcast SleepCycleRunning or Done here.
                                    // Auto-sleep runs in the background after the turn is
                                    // already complete.  Broadcasting lifecycle events would
                                    // interfere with the TUI's spinner state if the user
                                    // starts a new turn before auto-sleep finishes.
                                    let (noop_tx, _) = mpsc::unbounded_channel::<String>();
                                    let gen_result = rt.generate_agentic_sleep_insights(&memories, &identity, &noop_tx).await;
                                    let mut s = state2.lock().await;
                                    match gen_result {
                                        Ok(crate::SleepGenerationResult::Insights(insights)) => {
                                            match s.memory.apply_agentic_sleep_insights(*insights, Some("auto-turn sleep".into())).await {
                                                Ok(ref sum) => info!(summary = %sum.distilled, "auto-turn sleep complete"),
                                                Err(ref err) => warn!(?err, "auto-turn sleep apply failed"),
                                            }
                                        }
                                        Ok(crate::SleepGenerationResult::PassiveFallback(_)) => {
                                            match s.memory.run_sleep_cycle().await {
                                                Ok(ref sum) => info!(summary = %sum.distilled, "auto-turn passive sleep complete"),
                                                Err(ref err) => warn!(?err, "auto-turn passive sleep failed"),
                                            }
                                        }
                                        Err(ref err) => warn!(?err, "auto-turn sleep failed"),
                                    }
                                    let _ = s.memory.flush_all();
                                    let _ = event_tx2.send(BackendEvent::MemoryUpdated);
                                });
                            }
                            Ok(reflect_events)
                        }
                        Err(err) => Err(err),
                    }
                };

                let memory_elapsed = memory_start.elapsed();
                if timing {
                    let total = turn_start.elapsed();
                    info!(
                        "[TIMING] turn_total={}ms mem_prep={}ms build={}ms llm={}ms memory={}ms model={}",
                        total.as_millis(),
                        memory_prep_elapsed.as_millis(),
                        build_elapsed.as_millis(),
                        llm_elapsed.as_millis(),
                        memory_elapsed.as_millis(),
                        rt_clone.config.active_model(),
                    );
                }

                let mut writer = writer.lock().await;
                match outcome {
                    Ok(reflect_events) => {
                        send_event(&mut writer, ServerEvent::Backend(BackendEvent::MemoryUpdated)).await?;
                        for ev in reflect_events {
                            let _ = send_event(&mut writer, ServerEvent::Backend(ev.clone())).await;
                            let _ = event_tx.send(ev);
                        }
                        send_event(&mut writer, ServerEvent::Backend(BackendEvent::Done)).await?;
                        if is_external {
                            let _ = event_tx.send(BackendEvent::Done);
                        }
                    }
                    Err(err) => {
                        send_event(&mut writer, ServerEvent::Backend(BackendEvent::Error(err.to_string()))).await?;
                        send_event(&mut writer, ServerEvent::Backend(BackendEvent::Done)).await?;
                        if is_external {
                            let _ = event_tx.send(BackendEvent::Error(err.to_string()));
                            let _ = event_tx.send(BackendEvent::Done);
                        }
                    }
                }
        }
        ClientCommand::GetStatus => {
            let state = state.lock().await;
            send_event(&mut write_half, ServerEvent::Status(state.status())).await?;
        }
        ClientCommand::GetMemoryPeek { limit } => {
            let state = state.lock().await;
            let peek = state
                .memory
                .recent(limit.max(1))
                .into_iter()
                .map(|entry| entry.content.clone())
                .collect::<Vec<_>>();
            send_event(&mut write_half, ServerEvent::MemoryPeek(peek)).await?;
        }
        ClientCommand::ExecuteTool { name, args } => {
            let mut state = state.lock().await;
            let result = state
                .tool_executor
                .execute(&state.tool_registry, &name, &args)
                .await;
            match result {
                Ok(output) => {
                    // Record tool outcome to Episodic so the sleep cycle can
                    // reason about tool-use patterns. Raw logs belong in Episodic
                    // (transient); Procedural is for distilled patterns the sleep
                    // cycle extracts from these Episodic entries.
                    let outcome_text = format!(
                        "Tool '{}' {}: {}",
                        name,
                        if output.success { "succeeded" } else { "failed" },
                        safe_truncate(&output.output, 200),
                    );
                    if let Err(err) = state.memory.record(
                        MemoryTier::Episodic,
                        outcome_text,
                        format!("tool-log:{name}"),
                    ).await {
                        warn!(?err, tool = %name, "failed to record tool outcome to episodic memory");
                    }
                    // Pain hook: tool ran but returned failure → Procedural signal.
                    if !output.success {
                        let _ = state.memory.record(
                            MemoryTier::Procedural,
                            format!("CAUTION: Tool '{}' failed. Error: {}", name,
                                safe_truncate(&output.output, 200)),
                            format!("tool-failure:{name}"),
                        ).await;
                    }
                    send_event(
                        &mut write_half,
                        ServerEvent::ToolResult {
                            success: output.success,
                            output: output.output,
                        },
                    )
                    .await?;
                }
                Err(err) => {
                    // Record executor-level failures (policy denial, unknown tool,
                    // rate limit) to Episodic so the sleep cycle can surface them.
                    let _ = state.memory.record(
                        MemoryTier::Episodic,
                        format!("Tool '{}' rejected by executor: {}", name, err),
                        format!("tool-error:{name}"),
                    ).await;
                    send_event(
                        &mut write_half,
                        ServerEvent::ToolResult {
                            success: false,
                            output: err.to_string(),
                        },
                    )
                    .await?;
                }
            }
        }
        ClientCommand::ListTools => {
            let state = state.lock().await;
            let specs = state.tool_registry.list_specs();
            send_event(&mut write_half, ServerEvent::ToolList(specs)).await?;
        }
        ClientCommand::ReloadConfig => {
            let mut state = state.lock().await;
            // Re-read .env so a newly written OPENROUTER_API_KEY (or other
            // secrets saved via onboarding) takes effect immediately without
            // requiring a daemon restart.
            let _ = dotenvy::from_path_override(std::path::Path::new(".env"));
            let updated = AppConfig::load_from("config/default.toml")?;

            // Hot-reload dynamic modules when the modules subsystem is enabled
            // and auto_reload is on.
            let modules_msg = if updated.tools.modules.enabled && updated.tools.modules.auto_reload {
                reload_dynamic_modules(&state.tool_registry, &updated)
            } else {
                String::new()
            };

            state.runtime.config = updated;
            let msg = if modules_msg.is_empty() {
                "config reloaded".to_string()
            } else {
                format!("config reloaded; {modules_msg}")
            };
            send_event(
                &mut write_half,
                ServerEvent::Ack(msg),
            )
            .await?;
        }
        ClientCommand::ReloadTools => {
            let state = state.lock().await;
            let msg = if state.runtime.config.tools.modules.enabled {
                reload_dynamic_modules(&state.tool_registry, &state.runtime.config)
            } else {
                "modules subsystem is disabled".to_string()
            };
            send_event(
                &mut write_half,
                ServerEvent::Ack(msg),
            )
            .await?;
        }
        ClientCommand::Shutdown => {
            let _ = shutdown_tx.send(true);
            send_event(
                &mut write_half,
                ServerEvent::Ack("shutdown requested".to_string()),
            )
            .await?;
        }
        ClientCommand::Ping => {
            send_event(&mut write_half, ServerEvent::Ack("pong".to_string())).await?;
        }
        ClientCommand::RunSleepCycle => {
            // Snapshot → generate in background → stream progress → apply.
            // Uses the full multi-agent pipeline (4 specialists + synthesis)
            // which already falls back to single-agent if any LLM call fails.
            let (runtime, memories, identity) = {
                let s = state.lock().await;
                (s.runtime.clone(), s.memory.all().to_vec(), s.memory.identity.clone())
            };
            let (progress_tx, mut progress_rx) = mpsc::unbounded_channel::<String>();
            let (done_tx, mut done_rx) =
                tokio::sync::oneshot::channel::<crate::AgentResult<crate::SleepGenerationResult>>();
            tokio::spawn(async move {
                let result = runtime
                    .generate_multi_agent_sleep_insights(&memories, &identity, &progress_tx)
                    .await;
                let _ = done_tx.send(result);
            });
            loop {
                tokio::select! {
                    msg = progress_rx.recv() => {
                        if let Some(msg) = msg {
                            send_event(&mut write_half, ServerEvent::StatusLine(msg)).await?;
                        }
                    }
                    result = &mut done_rx => {
                        let gen_result = result.expect("sleep task panicked");
                        while let Ok(m) = progress_rx.try_recv() {
                            send_event(&mut write_half, ServerEvent::StatusLine(m)).await?;
                        }
                        let msg = {
                            let mut s = state.lock().await;
                            match gen_result {
                                Ok(crate::SleepGenerationResult::Insights(insights)) => {
                                    let summary_text = Some("User-triggered sleep cycle".into());
                                    match s.memory.apply_agentic_sleep_insights(*insights, summary_text).await {
                                        Ok(summary) => {
                                            let _ = s.memory.record(
                                                aigent_memory::MemoryTier::Semantic,
                                                "multi-agent sleep cycle completed",
                                                "sleep:multi-agent-cycle",
                                            ).await;
                                            let _ = s.memory.flush_all();
                                            format!("sleep cycle complete: {}", summary.distilled)
                                        }
                                        Err(err) => format!("sleep cycle apply failed: {err}"),
                                    }
                                }
                                Ok(crate::SleepGenerationResult::PassiveFallback(_)) => {
                                    match s.memory.run_sleep_cycle().await {
                                        Ok(summary) => {
                                            let _ = s.memory.flush_all();
                                            format!("sleep cycle complete (passive fallback): {}", summary.distilled)
                                        }
                                        Err(err) => format!("sleep cycle passive fallback failed: {err}"),
                                    }
                                }
                                Err(err) => format!("sleep cycle failed: {err}"),
                            }
                        };
                        send_event(&mut write_half, ServerEvent::Ack(msg)).await?;
                        break;
                    }
                }
            }
        }
        ClientCommand::RunMultiAgentSleepCycle => {
            let (runtime, memories, identity) = {
                let s = state.lock().await;
                (s.runtime.clone(), s.memory.all().to_vec(), s.memory.identity.clone())
            };
            let (progress_tx, mut progress_rx) = mpsc::unbounded_channel::<String>();
            let (done_tx, mut done_rx) =
                tokio::sync::oneshot::channel::<crate::AgentResult<crate::SleepGenerationResult>>();
            tokio::spawn(async move {
                let result = runtime
                    .generate_multi_agent_sleep_insights(&memories, &identity, &progress_tx)
                    .await;
                let _ = done_tx.send(result);
            });
            loop {
                tokio::select! {
                    msg = progress_rx.recv() => {
                        if let Some(msg) = msg {
                            send_event(&mut write_half, ServerEvent::StatusLine(msg)).await?;
                        }
                    }
                    result = &mut done_rx => {
                        let gen_result = result.expect("sleep task panicked");
                        while let Ok(m) = progress_rx.try_recv() {
                            send_event(&mut write_half, ServerEvent::StatusLine(m)).await?;
                        }
                        let msg = {
                            let mut s = state.lock().await;
                            match gen_result {
                                Ok(crate::SleepGenerationResult::Insights(insights)) => {
                                    let summary_text = Some("Multi-agent sleep cycle".into());
                                    match s.memory.apply_agentic_sleep_insights(*insights, summary_text).await {
                                        Ok(summary) => {
                                            let _ = s.memory.record(
                                                aigent_memory::MemoryTier::Semantic,
                                                "multi-agent sleep cycle completed",
                                                "sleep:multi-agent-cycle",
                                            ).await;
                                            let _ = s.memory.flush_all();
                                            format!("multi-agent sleep cycle complete: {}", summary.distilled)
                                        }
                                        Err(err) => format!("multi-agent sleep cycle apply failed: {err}"),
                                    }
                                }
                                Ok(crate::SleepGenerationResult::PassiveFallback(_)) => {
                                    match s.memory.run_sleep_cycle().await {
                                        Ok(summary) => {
                                            let _ = s.memory.flush_all();
                                            format!("multi-agent sleep cycle (passive fallback): {}", summary.distilled)
                                        }
                                        Err(err) => format!("multi-agent sleep cycle passive fallback failed: {err}"),
                                    }
                                }
                                Err(err) => format!("multi-agent sleep cycle failed: {err}"),
                            }
                        };
                        send_event(&mut write_half, ServerEvent::Ack(msg)).await?;
                        break;
                    }
                }
            }
        }
        ClientCommand::TriggerProactive => {
            // Snapshot beliefs/reflections, release lock, call LLM, re-acquire lock.
            let (rt_clone, beliefs_summary, reflections_summary, event_tx_clone) = {
                let s = state.lock().await;
                let rt = s.runtime.clone();
                let tx = s.event_tx.clone();

                let beliefs = s.memory.all_beliefs();
                let b_summary = if beliefs.is_empty() {
                    "(none yet)".to_string()
                } else {
                    beliefs
                        .iter()
                        .take(8)
                        .map(|e| format!("- {}", e.content))
                        .collect::<Vec<_>>()
                        .join("\n")
                };
                let ctx = s.memory.context_for_prompt_ranked_with_embed("recent", 5, None);
                let r_summary = {
                    let reflections: Vec<String> = ctx
                        .iter()
                        .filter(|item| item.entry.tier == MemoryTier::Reflective)
                        .map(|item| {
                            format!("- {}", aigent_prompt::truncate_for_prompt(&item.entry.content, 200))
                        })
                        .collect();
                    if reflections.is_empty() {
                        "(none yet)".to_string()
                    } else {
                        reflections.join("\n")
                    }
                };
                (rt, b_summary, r_summary, tx)
            };
            let outcome = rt_clone
                .run_proactive_check_from_summaries(&beliefs_summary, &reflections_summary)
                .await;
            let msg = {
                let mut s = state.lock().await;
                let _ = s.memory.flush_all();
                if let Some(out) = outcome {
                    if let Some(ref m) = out.message {
                        let ev = BackendEvent::ProactiveMessage { content: m.clone() };
                        let _ = event_tx_clone.send(ev);
                        let _ = s
                            .memory
                            .record(MemoryTier::Episodic, format!("[proactive] {m}"), "proactive")
                            .await;
                        s.proactive_total_sent += 1;
                        s.last_proactive_at = Some(Utc::now());
                        format!("proactive message sent: {m}")
                    } else {
                        "proactive check ran — no message to send".to_string()
                    }
                } else {
                    "proactive check ran — decided to stay silent".to_string()
                }
            };
            send_event(&mut write_half, ServerEvent::Ack(msg)).await?;
        }
        ClientCommand::GetProactiveStats => {
            use crate::commands::ProactiveStatsPayload;
            let s = state.lock().await;
            let payload = ProactiveStatsPayload {
                total_sent: s.proactive_total_sent,
                last_proactive_at: s.last_proactive_at.map(|t| t.to_rfc3339()),
                interval_minutes: s.runtime.config.memory.proactive_interval_minutes,
                dnd_start_hour: s.runtime.config.memory.proactive_dnd_start_hour,
                dnd_end_hour: s.runtime.config.memory.proactive_dnd_end_hour,
            };
            send_event(&mut write_half, ServerEvent::ProactiveStats(payload)).await?;
        }
        ClientCommand::GetSleepStatus => {
            use crate::commands::SleepStatusPayload;
            let s = state.lock().await;
            let cfg = &s.runtime.config.memory;
            let tz: chrono_tz::Tz = cfg.timezone.parse().unwrap_or(chrono_tz::UTC);
            let start_h = cfg.night_sleep_start_hour as u32;
            let end_h = cfg.night_sleep_end_hour as u32;
            let payload = SleepStatusPayload {
                auto_sleep_mode: cfg.auto_sleep_mode.clone(),
                passive_interval_hours: cfg.auto_sleep_turn_interval as u64,
                last_passive_sleep_at: s.last_passive_sleep_at
                    .map(|t| format!("{}s ago", t.elapsed().as_secs())),
                last_nightly_sleep_at: s.last_multi_agent_sleep_at
                    .map(|t| format!("{}s ago", t.elapsed().as_secs())),
                quiet_window_start: start_h as u8,
                quiet_window_end: end_h as u8,
                timezone: cfg.timezone.clone(),
                in_quiet_window: is_in_window(Utc::now(), tz, start_h, end_h),
            };
            send_event(&mut write_half, ServerEvent::SleepStatus(payload)).await?;
        }
        ClientCommand::DeduplicateMemory => {
            let mut s = state.lock().await;
            match s.memory.deduplicate_by_content().await {
                Ok(removed) => {
                    let msg = if removed > 0 {
                        let _ = s.memory.flush_all();
                        format!("deduplication complete: removed {removed} duplicate entries")
                    } else {
                        "deduplication complete: no duplicates found".to_string()
                    };
                    send_event(&mut write_half, ServerEvent::Ack(msg)).await?;
                }
                Err(err) => {
                    send_event(
                        &mut write_half,
                        ServerEvent::Ack(format!("deduplication failed: {err}")),
                    )
                    .await?;
                }
            }
        }
    }

    Ok(())
}

async fn send_event(
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    event: ServerEvent,
) -> Result<()> {
    let encoded = serde_json::to_string(&event)?;
    writer.write_all(encoded.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;
    Ok(())
}

// ── Dynamic module reload ─────────────────────────────────────────────────────

/// Scan the configured modules directory for `.wasm` files and register them
/// as dynamic tools (replacing any previously loaded dynamic modules).
///
/// This is a *synchronous* registry mutation — safe to call while holding
/// the `DaemonState` mutex because the RwLock inside `ToolRegistry` is a
/// `std::sync::RwLock`, not a tokio one.
fn reload_dynamic_modules(
    registry: &aigent_tools::ToolRegistry,
    config: &aigent_config::AppConfig,
) -> String {
    use std::path::Path;

    let modules_cfg = &config.tools.modules;
    let workspace = Path::new(&config.agent.workspace_path);
    let modules_dir = workspace.join(&modules_cfg.modules_dir);

    // Remove all previously loaded dynamic tools.
    let old_names = registry.dynamic_tool_names();
    for name in &old_names {
        registry.unregister(name);
    }

    if !modules_dir.is_dir() {
        if old_names.is_empty() {
            return "modules dir not found; nothing to load".to_string();
        }
        return format!(
            "modules dir not found; unloaded {} previous dynamic tool(s)",
            old_names.len()
        );
    }

    // Discover .wasm files in the modules directory.
    let mut loaded: Vec<String> = Vec::new();
    let max = if modules_cfg.max_modules == 0 {
        usize::MAX
    } else {
        modules_cfg.max_modules
    };

    if let Ok(entries) = std::fs::read_dir(&modules_dir) {
        for entry in entries.flatten() {
            if loaded.len() >= max {
                warn!(
                    max_modules = modules_cfg.max_modules,
                    "hit max_modules limit — ignoring remaining modules"
                );
                break;
            }
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("wasm") {
                // For now, log the discovered module.  Actual WASM instantiation
                // will be wired up in Phase 2 (sandboxed compilation pipeline).
                // This ensures the discovery + registration plumbing is tested.
                let stem = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                info!(module = %stem, path = %path.display(), "discovered dynamic module");
                loaded.push(stem);
            }
        }
    }

    let msg = format!(
        "modules reload: unloaded {} old, discovered {} new dynamic tool(s)",
        old_names.len(),
        loaded.len()
    );
    info!("{}", msg);
    msg
}
