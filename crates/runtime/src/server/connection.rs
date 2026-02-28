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
            // ── Path selection: native structured tool calling vs legacy text-based ──
            let use_native_calling = {
                let s = state.lock().await;
                s.runtime.config.tools.use_native_calling
            };

            if use_native_calling {
                // ═══════════════════════════════════════════════════════════════
                // NEW PATH: Native structured tool calling via chat messages API
                // ═══════════════════════════════════════════════════════════════
                let (rt_clone, mut memory, recent, _last_turn_at, tool_specs, registry, executor) = {
                    let mut s = state.lock().await;
                    let rt = s.runtime.clone();
                    let recent = s.recent_turns.iter().cloned().collect::<Vec<_>>();
                    let lta = s.last_turn_at;
                    let mem = std::mem::take(&mut s.memory);
                    let specs = s.tool_registry.list_specs();
                    let reg = Arc::clone(&s.tool_registry);
                    let exe = Arc::clone(&s.tool_executor);
                    (rt, mem, recent, lta, specs, reg, exe)
                };

                // Persist the user turn and do memory work before calling the LLM.
                let _: Result<_, _> = memory.record(aigent_memory::MemoryTier::Episodic, user.clone(), "user-input").await;

                // Build the system prompt using the existing prompt builder.
                let query_embedding: Option<Vec<f32>> = if let Some(embed_fn) = memory.embed_fn_arc() {
                    embed_fn(user.to_string()).await
                } else {
                    None
                };
                let context = memory.context_for_prompt_ranked_with_embed(&user, 10, query_embedding);
                let stats = memory.stats();

                let mut prompt_inputs = crate::prompt_builder::PromptInputs {
                    config: &rt_clone.config,
                    memory: &mut memory,
                    user_message: &user,
                    recent_turns: &recent,
                    tool_specs: &[],  // Don't inject tools into the text prompt — they go via API
                    pending_follow_ups: &[],
                    context_items: &context,
                    stats,
                };
                let system_prompt = crate::prompt_builder::build_chat_prompt(&mut prompt_inputs);

                // Build the chat messages array.
                // Include recent conversation turns as proper user/assistant
                // pairs so the model has multi-turn context — not just the
                // flattened text summary baked into the system prompt.
                let mut messages = vec![aigent_llm::ChatMessage::system(&system_prompt)];
                for turn in &recent {
                    messages.push(aigent_llm::ChatMessage::user(&turn.user));
                    messages.push(aigent_llm::ChatMessage::assistant(&turn.assistant));
                }
                messages.push(aigent_llm::ChatMessage::user(&user));

                // Build tools JSON from specs
                let tools_json = if tool_specs.is_empty() {
                    None
                } else {
                    Some(crate::tool_loop::build_tools_json(&tool_specs))
                };

                let primary = if rt_clone.config.llm.provider.to_lowercase() == "openrouter" {
                    aigent_llm::Provider::OpenRouter
                } else {
                    aigent_llm::Provider::Ollama
                };

                // Run the structured tool loop — registry/executor are Arc-cloned
                // so we do NOT hold the DaemonState lock during LLM calls.
                let loop_result = {
                    crate::tool_loop::run_tool_loop(
                        &rt_clone.llm,
                        primary,
                        &rt_clone.config.llm.ollama_model,
                        &rt_clone.config.llm.openrouter_model,
                        &mut messages,
                        tools_json.as_ref(),
                        &registry,
                        &executor,
                        chunk_tx,
                        Some(&event_tx),
                    ).await
                };

                let reply_result: Result<String, anyhow::Error> = match loop_result {
                    Ok(ref result) => {
                        // Record tool results to procedural memory
                        for exec in &result.tool_executions {
                            let outcome = format!(
                                "Tool '{}' executed. Output (first 400 chars): {}",
                                exec.tool_name,
                                safe_truncate(&exec.output, 400),
                            );
                            let _: Result<_, _> = memory.record(
                                MemoryTier::Procedural,
                                outcome,
                                format!("tool-use:{}", exec.tool_name),
                            ).await;
                        }
                        // Record assistant reply
                        if !result.content.is_empty() {
                            let _: Result<_, _> = memory.record(
                                aigent_memory::MemoryTier::Episodic,
                                crate::prompt_builder::truncate_for_prompt(&result.content, 1024),
                                format!("assistant-reply:model={}", rt_clone.config.active_model()),
                            ).await;
                        }
                        Ok(result.content.clone())
                    }
                    Err(e) => Err(e),
                };

                // Reflect on the exchange
                let reflect_events: Vec<BackendEvent> = match reply_result {
                    Ok(ref reply) => rt_clone
                        .inline_reflect(&mut memory, &user, reply)
                        .await
                        .unwrap_or_default(),
                    Err(_) => vec![],
                };

                let _ = streamer.await;

                // Re-acquire lock to restore state
                let outcome: Result<Vec<BackendEvent>, anyhow::Error> = {
                    let mut s = state.lock().await;
                    s.memory = memory;
                    let _ = s.memory.flush_all();

                    match reply_result {
                        Ok(reply) => {
                            s.last_turn_at = Some(Utc::now());
                            s.recent_turns.push_back(ConversationTurn {
                                user: user.clone(),
                                assistant: reply,
                            });
                            while s.recent_turns.len() > 8 {
                                let _ = s.recent_turns.pop_front();
                            }
                            s.turn_count += 1;

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
                                    let _ = event_tx2.send(BackendEvent::SleepCycleRunning);
                                    let (noop_tx, _) = mpsc::unbounded_channel::<String>();
                                    let gen_result = rt.generate_agentic_sleep_insights(&memories, &identity, &noop_tx).await;
                                    let mut s = state2.lock().await;
                                    match gen_result {
                                        Ok(crate::SleepGenerationResult::Insights(insights)) => {
                                            match s.memory.apply_agentic_sleep_insights(insights, Some("auto-turn sleep".into())).await {
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
                                    let _ = event_tx2.send(BackendEvent::Done);
                                });
                            }
                            Ok(reflect_events)
                        }
                        Err(err) => Err(err),
                    }
                };

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
            } else {
            // ═══════════════════════════════════════════════════════════════════
            // LEGACY PATH: text-based maybe_tool_call + single execution
            // ═══════════════════════════════════════════════════════════════════
            let (rt_early, tool_specs) = {
                let s = state.lock().await;
                (s.runtime.clone(), s.tool_registry.list_specs())
            };
            let tool_call_intent = rt_early.maybe_tool_call(&user, &tool_specs).await;

            // Execute the tool (if intent detected) and collect the result.
            let tool_result_text: Option<(String, String)> = if let Some(ref call) = tool_call_intent {
                // Emit ToolCallStart to broadcast subscribers.
                let info = crate::events::ToolCallInfo {
                    name: call.tool.clone(),
                    args: serde_json::to_string(&call.args).unwrap_or_default(),
                };
                let _ = event_tx.send(BackendEvent::ToolCallStart(info));

                // Execute — holds lock only for the duration of the tool run.
                let exec_result = {
                    let s = state.lock().await;
                    s.tool_executor
                        .execute(&s.tool_registry, &call.tool, &call.stringify_args())
                        .await
                };

                let (success, output) = match exec_result {
                    Ok(ref o) => (o.success, o.output.clone()),
                    Err(ref e) => (false, e.to_string()),
                };
                let _ = event_tx.send(BackendEvent::ToolCallEnd(crate::events::ToolResult {
                    name: call.tool.clone(),
                    success,
                    output: output.clone(),
                }));
                info!(tool = %call.tool, success, output_len = output.len(), "tool call executed");
                Some((call.tool.clone(), output))
            } else {
                None
            };

            let (rt_clone, mut memory, mut recent, last_turn_at) = {
                let mut s = state.lock().await;
                let rt = s.runtime.clone();
                let recent = s.recent_turns.iter().cloned().collect::<Vec<_>>();
                let lta = s.last_turn_at;
                let mem = std::mem::take(&mut s.memory);
                (rt, mem, recent, lta)
                // MutexGuard dropped — lock released here
            };

            // LLM respond + inline reflect, both WITHOUT holding state lock.
            //
            // If a tool was called above, (a) record the result to Procedural
            // memory for long-term recall, (b) inject a synthetic conversation
            // turn so the RECENT CONVERSATION block in the prompt contains an
            // explicit tool-result exchange, and (c) build an effective user
            // message that presents the result as ground truth.
            if let Some((ref tool_name, ref tool_output)) = tool_result_text {
                let outcome_text = format!(
                    "Tool '{}' executed in response to user turn. Output (first 400 chars): {}",
                    tool_name,
                    safe_truncate(tool_output, 400),
                );
                if let Err(err) = memory.record(
                    MemoryTier::Procedural,
                    outcome_text,
                    format!("tool-use:{tool_name}"),
                ).await {
                    warn!(?err, tool = %tool_name, "failed to record tool result to procedural memory");
                }

                // (b) Inject the tool result as an explicit conversation turn.
                recent.push(ConversationTurn {
                    user: format!("[system: tool '{}' was invoked for this request]", tool_name),
                    assistant: format!("[TOOL RESULT from '{}']: {}", tool_name, tool_output),
                });
            }

            // Build the effective user message.  When a tool was executed, the
            // user question is stated first (so the LLM knows what to answer),
            // followed by the full tool output in a clearly demarcated block.
            let effective_user: std::borrow::Cow<str> = if let Some((ref tool_name, ref output)) = tool_result_text {
                let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC");
                std::borrow::Cow::Owned(format!(
                    "{user}\n\n\
                     ===== TOOL RESULT (retrieved {now}) from '{tool_name}' =====\n\
                     {output}\n\
                     ===== END TOOL RESULT =====\n\n\
                     CRITICAL INSTRUCTION — you MUST follow every point:\n\
                     1. The TOOL RESULT embedded above is in your context RIGHT NOW between the ===== markers.\n\
                     2. Synthesise a complete, natural, helpful final answer from it immediately.\n\
                     3. Quote numbers, dates, names, and facts verbatim from the TOOL RESULT.\n\
                     4. Speak conversationally as yourself — never use phrases like \"according to the tool\" \
                        or \"based on the data retrieved\".\n\
                     5. Do NOT claim the result is unavailable, pending, missing, or \"not yet in your memory\".\n\
                     6. Do NOT output raw JSON or attempt to call a tool again.\n\
                     7. If the tool returned an error, state the error honestly and suggest alternatives.",
                    user = user,
                    now = now,
                    tool_name = tool_name,
                    output = output,
                ))
            } else {
                std::borrow::Cow::Borrowed(&user)
            };

            // When a tool was already executed, suppress the "AVAILABLE TOOLS"
            // section from the streaming prompt.
            let effective_specs: &[aigent_tools::ToolSpec] = if tool_result_text.is_some() {
                &[]  // tool already ran — LLM should just answer
            } else {
                &tool_specs
            };
            let mut reply_result = rt_clone
                .respond_and_remember_stream(&mut memory, &effective_user, &recent, last_turn_at, chunk_tx, effective_specs)
                .await;

            // ── Fallback: post-stream tool detection ──────────────────────
            // If maybe_tool_call missed the intent (e.g. model returned bare
            // JSON on the main path), detect tool JSON in the streamed reply,
            // execute the tool, and do a second LLM call with the result.
            if tool_result_text.is_none() {
                if let Ok(ref reply) = reply_result {
                    let trimmed = reply.trim();
                    if let Ok(call) = serde_json::from_str::<crate::agent_loop::LlmToolCall>(trimmed) {
                        if !call.tool.is_empty() {
                            warn!(tool = %call.tool, "fallback: streamed reply was tool JSON — executing post-hoc");
                            // Tell clients to discard the raw tool-JSON tokens.
                            {
                                let mut guard = writer.lock().await;
                                let _ = send_event(
                                    &mut guard,
                                    ServerEvent::Backend(BackendEvent::ClearStream),
                                ).await;
                            }
                            let _ = event_tx.send(BackendEvent::ClearStream);
                            let info = crate::events::ToolCallInfo {
                                name: call.tool.clone(),
                                args: serde_json::to_string(&call.args).unwrap_or_default(),
                            };
                            let _ = event_tx.send(BackendEvent::ToolCallStart(info));

                            let exec_result = {
                                let s = state.lock().await;
                                s.tool_executor
                                    .execute(&s.tool_registry, &call.tool, &call.stringify_args())
                                    .await
                            };
                            let (success, output) = match exec_result {
                                Ok(ref o) => (o.success, o.output.clone()),
                                Err(ref e) => (false, e.to_string()),
                            };
                            let _ = event_tx.send(BackendEvent::ToolCallEnd(crate::events::ToolResult {
                                name: call.tool.clone(),
                                success,
                                output: output.clone(),
                            }));

                            // Build a grounded follow-up prompt and re-stream.
                            let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC");
                            let followup = format!(
                                "{user}\n\n\
                                 ===== TOOL RESULT (retrieved {now}) from '{tool_name}' =====\n\
                                 {output}\n\
                                 ===== END TOOL RESULT =====\n\n\
                                 CRITICAL INSTRUCTION — you MUST follow every point:\n\
                                 1. The TOOL RESULT embedded above is in your context RIGHT NOW between the ===== markers.\n\
                                 2. Synthesise a complete, natural, helpful final answer from it immediately.\n\
                                 3. Quote numbers, dates, names, and facts verbatim from the TOOL RESULT.\n\
                                 4. Speak conversationally as yourself — never use phrases like \"according to the tool\" \
                                    or \"based on the data retrieved\".\n\
                                 5. Do NOT claim the result is unavailable, pending, missing, or \"not yet in your memory\".\n\
                                 6. Do NOT output raw JSON or attempt to call a tool again.\n\
                                 7. If the tool returned an error, state the error honestly and suggest alternatives.",
                                user = user,
                                now = now,
                                tool_name = call.tool,
                                output = output,
                            );

                            let (followup_tx, mut followup_rx) = mpsc::channel::<String>(128);
                            let writer2 = writer.clone();
                            let event_tx2 = if is_external { Some(event_tx.clone()) } else { None };
                            let followup_streamer = tokio::spawn(async move {
                                while let Some(chunk) = followup_rx.recv().await {
                                    {
                                        let mut guard = writer2.lock().await;
                                        let _ = send_event(
                                            &mut guard,
                                            ServerEvent::Backend(BackendEvent::Token(chunk.clone())),
                                        ).await;
                                    }
                                    if let Some(ref tx) = event_tx2 {
                                        let _ = tx.send(BackendEvent::Token(chunk));
                                    }
                                }
                            });

                            reply_result = rt_clone
                                .respond_and_remember_stream(
                                    &mut memory, &followup, &recent, last_turn_at, followup_tx, &tool_specs,
                                )
                                .await;
                            let _ = followup_streamer.await;
                        }
                    }
                }
            }

            // Reflect on the original user ↔ assistant exchange (not the tool-augmented prompt).
            let reflect_events: Vec<BackendEvent> = match reply_result {
                Ok(ref reply) => rt_clone
                    .inline_reflect(&mut memory, &user, reply)
                    .await
                    .unwrap_or_default(),
                Err(_) => vec![],
            };

            // Wait for the streaming task to finish flushing all tokens.
            let _ = streamer.await;

            // Re-acquire lock to restore state and do bookkeeping.
            let outcome: Result<Vec<BackendEvent>, anyhow::Error> = {
                let mut s = state.lock().await;
                // Always restore memory, even on error.
                s.memory = memory;
                let _ = s.memory.flush_all();

                match reply_result {
                    Ok(reply) => {
                        s.last_turn_at = Some(Utc::now());
                        s.recent_turns.push_back(ConversationTurn {
                            user: user.clone(),
                            assistant: reply,
                        });
                        while s.recent_turns.len() > 8 {
                            let _ = s.recent_turns.pop_front();
                        }
                        s.turn_count += 1;

                        // Spawn auto-sleep as a background task so it doesn't
                        // add latency to the turn response.
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
                                let _ = event_tx2.send(BackendEvent::SleepCycleRunning);
                                let (noop_tx, _) = mpsc::unbounded_channel::<String>();
                                let gen_result = rt.generate_agentic_sleep_insights(&memories, &identity, &noop_tx).await;
                                let mut s = state2.lock().await;
                                match gen_result {
                                    Ok(crate::SleepGenerationResult::Insights(insights)) => {
                                        match s.memory.apply_agentic_sleep_insights(insights, Some("auto-turn sleep".into())).await {
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
                                let _ = event_tx2.send(BackendEvent::Done);
                            });
                        }

                        Ok(reflect_events)
                    }
                    Err(err) => Err(err),
                }
            };

            let mut writer = writer.lock().await;
            match outcome {
                Ok(reflect_events) => {
                    send_event(
                        &mut writer,
                        ServerEvent::Backend(BackendEvent::MemoryUpdated),
                    )
                    .await?;
                    // Stream reflection events to the current client and all subscribers.
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
                    send_event(
                        &mut writer,
                        ServerEvent::Backend(BackendEvent::Error(err.to_string())),
                    )
                    .await?;
                    send_event(&mut writer, ServerEvent::Backend(BackendEvent::Done)).await?;
                    if is_external {
                        let _ = event_tx.send(BackendEvent::Error(err.to_string()));
                        let _ = event_tx.send(BackendEvent::Done);
                    }
                }
            }
            } // end of legacy path else block
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
                    // Record tool outcome so the sleep cycle can reason about
                    // tool-use patterns and add TOOL_INSIGHT entries.
                    let outcome_text = format!(
                        "Tool '{}' {}: {}",
                        name,
                        if output.success { "succeeded" } else { "failed" },
                        safe_truncate(&output.output, 200),
                    );
                    if let Err(err) = state.memory.record(
                        MemoryTier::Procedural,
                        outcome_text,
                        format!("tool-execution:{name}"),
                    ).await {
                        warn!(?err, tool = %name, "failed to record tool outcome to procedural memory");
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
            state.runtime.config = updated;
            send_event(
                &mut write_half,
                ServerEvent::Ack("config reloaded".to_string()),
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
                tokio::sync::oneshot::channel::<Result<crate::SleepGenerationResult, anyhow::Error>>();
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
                                    match s.memory.apply_agentic_sleep_insights(insights, summary_text).await {
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
                tokio::sync::oneshot::channel::<Result<crate::SleepGenerationResult, anyhow::Error>>();
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
                                    match s.memory.apply_agentic_sleep_insights(insights, summary_text).await {
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
                            format!("- {}", crate::prompt_builder::truncate_for_prompt(&item.entry.content, 200))
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
