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

            let outcome = {
                let mut state = state.lock().await;
                let recent = state.recent_turns.iter().cloned().collect::<Vec<_>>();
                let last_turn_at = state.last_turn_at;
                let mut memory = std::mem::take(&mut state.memory);
                let reply_result = state
                    .runtime
                    .respond_and_remember_stream(&mut memory, &user, &recent, last_turn_at, chunk_tx)
                    .await;

                let outcome = match reply_result {
                    Ok(reply) => {
                        state.last_turn_at = Some(Utc::now());
                        state.recent_turns.push_back(ConversationTurn {
                            user,
                            assistant: reply,
                        });
                        while state.recent_turns.len() > 8 {
                            let _ = state.recent_turns.pop_front();
                        }
                        state.turn_count += 1;

                        let mut extras = Vec::new();
                        if state.runtime.config.memory.auto_sleep_turn_interval > 0
                            && state.turn_count
                                % state.runtime.config.memory.auto_sleep_turn_interval
                                == 0
                        {
                            // Use agentic sleep for richer consolidation.
                            let (noop_tx, _) = mpsc::unbounded_channel::<String>();
                        let sleep_result = state
                                .runtime
                                .run_agentic_sleep_cycle(&mut memory, &noop_tx)
                                .await;
                            match sleep_result {
                                Ok(summary) => extras.push(format!("sleep cycle: {}", summary.distilled)),
                                Err(err) => warn!(?err, "auto agentic sleep cycle failed"),
                            }
                        }

                        Ok(extras)
                    }
                    Err(err) => Err(err),
                };

                state.memory = memory;
                // Sync the vault once per turn (deferred from per-record calls).
                let _ = state.memory.flush_all();
                outcome
            };

            let _ = streamer.await;
            let mut writer = writer.lock().await;
            match outcome {
                Ok(extras) => {
                    send_event(
                        &mut writer,
                        ServerEvent::Backend(BackendEvent::MemoryUpdated),
                    )
                    .await?;
                    for extra in extras {
                        send_event(
                            &mut writer,
                            ServerEvent::Backend(BackendEvent::Token(extra)),
                        )
                        .await?;
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
                        &output.output[..output.output.len().min(200)],
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
            // Take memory out of state and release the lock before the LLM call.
            // Spawn the cycle in a separate task so we can stream StatusLine
            // progress events back to the client while it runs.
            //
            // Uses the full multi-agent pipeline (4 specialists + synthesis)
            // which already falls back to single-agent if any LLM call fails.
            let (runtime, memory) = {
                let mut s = state.lock().await;
                let runtime = s.runtime.clone();
                let memory = std::mem::take(&mut s.memory);
                (runtime, memory)
            };
            let (progress_tx, mut progress_rx) = mpsc::unbounded_channel::<String>();
            let (done_tx, mut done_rx) =
                tokio::sync::oneshot::channel::<(String, aigent_memory::MemoryManager)>();
            tokio::spawn(async move {
                let mut mem = memory;
                let msg = match runtime.run_multi_agent_sleep_cycle(&mut mem, &progress_tx).await {
                    Ok(summary) => format!("sleep cycle complete: {}", summary.distilled),
                    Err(err) => format!("sleep cycle failed: {err}"),
                };
                let _ = done_tx.send((msg, mem));
            });
            loop {
                tokio::select! {
                    msg = progress_rx.recv() => {
                        if let Some(msg) = msg {
                            send_event(&mut write_half, ServerEvent::StatusLine(msg)).await?;
                        }
                    }
                    result = &mut done_rx => {
                        let (msg, memory_back) = result.expect("sleep task panicked");
                        while let Ok(m) = progress_rx.try_recv() {
                            send_event(&mut write_half, ServerEvent::StatusLine(m)).await?;
                        }
                        {
                            let mut s = state.lock().await;
                            s.memory = memory_back;
                            let _ = s.memory.flush_all();
                        }
                        send_event(&mut write_half, ServerEvent::Ack(msg)).await?;
                        break;
                    }
                }
            }
        }
        ClientCommand::RunMultiAgentSleepCycle => {
            let (runtime, memory) = {
                let mut s = state.lock().await;
                let runtime = s.runtime.clone();
                let memory = std::mem::take(&mut s.memory);
                (runtime, memory)
            };
            let (progress_tx, mut progress_rx) = mpsc::unbounded_channel::<String>();
            let (done_tx, mut done_rx) =
                tokio::sync::oneshot::channel::<(String, aigent_memory::MemoryManager)>();
            tokio::spawn(async move {
                let mut mem = memory;
                let msg = match runtime.run_multi_agent_sleep_cycle(&mut mem, &progress_tx).await {
                    Ok(summary) => format!("multi-agent sleep cycle complete: {}", summary.distilled),
                    Err(err) => format!("multi-agent sleep cycle failed: {err}"),
                };
                let _ = done_tx.send((msg, mem));
            });
            loop {
                tokio::select! {
                    msg = progress_rx.recv() => {
                        if let Some(msg) = msg {
                            send_event(&mut write_half, ServerEvent::StatusLine(msg)).await?;
                        }
                    }
                    result = &mut done_rx => {
                        let (msg, memory_back) = result.expect("sleep task panicked");
                        while let Ok(m) = progress_rx.try_recv() {
                            send_event(&mut write_half, ServerEvent::StatusLine(m)).await?;
                        }
                        {
                            let mut s = state.lock().await;
                            s.memory = memory_back;
                            let _ = s.memory.flush_all();
                        }
                        send_event(&mut write_half, ServerEvent::Ack(msg)).await?;
                        break;
                    }
                }
            }
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
