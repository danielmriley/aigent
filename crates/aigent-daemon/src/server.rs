use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{Mutex, mpsc, watch};
use tracing::{error, info};

use aigent_config::AppConfig;
use aigent_memory::MemoryManager;

use crate::{
    AgentRuntime, BackendEvent, ClientCommand, ConversationTurn, DaemonStatus, ServerEvent,
};

struct DaemonState {
    runtime: AgentRuntime,
    memory: MemoryManager,
    recent_turns: VecDeque<ConversationTurn>,
    turn_count: usize,
    started_at: Instant,
}

impl DaemonState {
    fn status(&self) -> DaemonStatus {
        DaemonStatus {
            bot_name: self.runtime.config.agent.name.clone(),
            provider: self.runtime.config.llm.provider.clone(),
            model: self.runtime.config.active_model().to_string(),
            thinking_level: self.runtime.config.agent.thinking_level.clone(),
            memory_total: self.memory.all().len(),
            uptime_secs: self.started_at.elapsed().as_secs(),
        }
    }
}

pub async fn run_unified_daemon(
    config: AppConfig,
    memory_log_path: impl AsRef<Path>,
    socket_path: impl AsRef<Path>,
) -> Result<()> {
    let socket_path = socket_path.as_ref().to_path_buf();
    if socket_path.exists() {
        let _ = std::fs::remove_file(&socket_path);
    }

    let memory = MemoryManager::with_event_log(memory_log_path)?;
    let runtime = AgentRuntime::new(config);
    runtime.run().await?;

    let state = Arc::new(Mutex::new(DaemonState {
        runtime,
        memory,
        recent_turns: VecDeque::new(),
        turn_count: 0,
        started_at: Instant::now(),
    }));

    let listener = UnixListener::bind(&socket_path)?;
    let (shutdown_tx, mut shutdown_rx) = watch::channel(false);
    info!(path = %socket_path.display(), "unified daemon listening");

    loop {
        tokio::select! {
            changed = shutdown_rx.changed() => {
                if changed.is_ok() && *shutdown_rx.borrow() {
                    break;
                }
            }
            accept = listener.accept() => {
                let (stream, _) = accept?;
                let state = state.clone();
                let shutdown_tx = shutdown_tx.clone();
                tokio::spawn(async move {
                    if let Err(err) = handle_connection(stream, state, shutdown_tx).await {
                        error!(?err, "daemon connection handler failed");
                    }
                });
            }
        }
    }

    info!("daemon shutting down gracefully");
    {
        let mut state = state.lock().await;
        let _ = state.memory.flush_all();
        let _ = state.memory.run_sleep_cycle();
        let _ = state.memory.flush_all();
    }
    let _ = std::fs::remove_file(&socket_path);
    Ok(())
}

async fn handle_connection(
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
    match command {
        ClientCommand::SubmitTurn { user } => {
            send_event(
                &mut write_half,
                ServerEvent::Backend(BackendEvent::Thinking),
            )
            .await?;

            let (chunk_tx, mut chunk_rx) = mpsc::channel::<String>(128);
            let writer = Arc::new(Mutex::new(write_half));
            let writer_clone = writer.clone();
            let streamer = tokio::spawn(async move {
                while let Some(chunk) = chunk_rx.recv().await {
                    let mut guard = writer_clone.lock().await;
                    let _ =
                        send_event(&mut guard, ServerEvent::Backend(BackendEvent::Token(chunk)))
                            .await;
                }
            });

            let outcome = {
                let mut state = state.lock().await;
                let recent = state.recent_turns.iter().cloned().collect::<Vec<_>>();
                let mut memory = std::mem::take(&mut state.memory);
                let reply = state
                    .runtime
                    .respond_and_remember_stream(&mut memory, &user, &recent, chunk_tx)
                    .await?;

                state.recent_turns.push_back(ConversationTurn {
                    user,
                    assistant: reply.clone(),
                });
                while state.recent_turns.len() > 8 {
                    let _ = state.recent_turns.pop_front();
                }
                state.turn_count += 1;

                let mut extras = Vec::new();
                if state.runtime.config.memory.auto_sleep_turn_interval > 0
                    && state.turn_count % state.runtime.config.memory.auto_sleep_turn_interval == 0
                {
                    if let Ok(summary) = memory.run_sleep_cycle() {
                        extras.push(format!("internal sleep cycle: {}", summary.distilled));
                    }
                }

                state.memory = memory;

                (reply, extras)
            };

            let _ = streamer.await;
            let mut writer = writer.lock().await;
            send_event(
                &mut writer,
                ServerEvent::Backend(BackendEvent::MemoryUpdated),
            )
            .await?;
            for extra in outcome.1 {
                send_event(
                    &mut writer,
                    ServerEvent::Backend(BackendEvent::Token(extra)),
                )
                .await?;
                send_event(&mut writer, ServerEvent::Backend(BackendEvent::Done)).await?;
            }
            send_event(&mut writer, ServerEvent::Backend(BackendEvent::Done)).await?;
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
