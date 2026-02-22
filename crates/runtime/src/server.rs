use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use chrono::{DateTime, Utc};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{Mutex, broadcast, mpsc, watch};
use tracing::{error, info, warn};

use aigent_config::AppConfig;
use aigent_exec::{ExecutionPolicy, ToolExecutor};
use aigent_memory::{EmbedFn, MemoryManager, MemoryTier};
use aigent_tools::ToolRegistry;

use crate::{
    AgentRuntime, BackendEvent, ClientCommand, ConversationTurn, DaemonStatus, ServerEvent,
};

/// Broadcast channel capacity. Old events are dropped when subscribers lag.
const BROADCAST_CAP: usize = 256;

struct DaemonState {
    runtime: AgentRuntime,
    memory: MemoryManager,
    tool_registry: ToolRegistry,
    tool_executor: ToolExecutor,
    recent_turns: VecDeque<ConversationTurn>,
    turn_count: usize,
    started_at: Instant,
    /// Timestamp of the last completed conversation turn (persisted in-process).
    last_turn_at: Option<DateTime<Utc>>,
    /// Broadcast sender — fans out external-source BackendEvents to all subscribers.
    event_tx: broadcast::Sender<BackendEvent>,
}

impl DaemonState {
    fn status(&self) -> DaemonStatus {
        let stats = self.memory.stats();
        DaemonStatus {
            bot_name: self.runtime.config.agent.name.clone(),
            provider: self.runtime.config.llm.provider.clone(),
            model: self.runtime.config.active_model().to_string(),
            thinking_level: self.runtime.config.agent.thinking_level.clone(),
            memory_total: stats.total,
            memory_core: stats.core,
            memory_user_profile: stats.user_profile,
            memory_reflective: stats.reflective,
            memory_semantic: stats.semantic,
            memory_episodic: stats.episodic,
            uptime_secs: self.started_at.elapsed().as_secs(),
            available_tools: self.tool_registry.list_specs().iter().map(|s| s.name.clone()).collect(),
        }
    }
}

fn build_execution_policy(config: &AppConfig) -> ExecutionPolicy {
    let workspace_root = PathBuf::from(&config.agent.workspace_path);
    ExecutionPolicy {
        approval_required: config.safety.approval_required,
        allow_shell: config.safety.allow_shell,
        allow_wasm: config.safety.allow_wasm,
        workspace_root,
    }
}

/// Build a synchronous embedding function that calls the Ollama `/api/embeddings`
/// endpoint.  Falls back to `None` silently so the system continues to work
/// when Ollama is unavailable.
fn make_ollama_embed_fn(model: &str) -> EmbedFn {
    let model = model.to_string();
    let base_url = std::env::var("OLLAMA_BASE_URL")
        .unwrap_or_else(|_| "http://localhost:11434".to_string());
    let url = format!("{}/api/embeddings", base_url.trim_end_matches('/'));

    Arc::new(move |text: &str| {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .ok()?;
        let body = serde_json::json!({ "model": model, "prompt": text });
        let resp = client.post(&url).json(&body).send().ok()?;
        let json: serde_json::Value = resp.json().ok()?;
        let embedding = json["embedding"]
            .as_array()?
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect::<Vec<f32>>();
        if embedding.is_empty() { None } else { Some(embedding) }
    })
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

    let mut memory = MemoryManager::with_event_log(memory_log_path)?;

    // Wire in the Ollama embedding backend so all new entries are automatically
    // embedded and retrieval uses hybrid lexical+vector scoring.
    {
        let embed_model = config.llm.ollama_model.clone();
        memory.set_embed_fn(make_ollama_embed_fn(&embed_model));
        info!(model = %embed_model, "embedding backend configured");
    }

    // Auto-seed Core identity if it's missing (safety net for upgrades or
    // first-run cases where onboarding seeded the config but not the event log).
    {
        let stats = memory.stats();
        info!(
            core = stats.core, episodic = stats.episodic,
            semantic = stats.semantic, "daemon memory state at startup"
        );
        if stats.core == 0 {
            let user_name = config.agent.user_name.trim().to_string();
            let bot_name = config.agent.name.trim().to_string();
            if !user_name.is_empty() && !bot_name.is_empty() {
                if let Err(err) = memory.seed_core_identity(&user_name, &bot_name) {
                    warn!(?err, "daemon startup: failed to auto-seed core identity");
                }
            } else {
                warn!("daemon startup: core memory is empty and config has no user/bot names — run `aigent onboard`");
            }
        }
    }

    let policy = build_execution_policy(&config);
    let workspace_root = policy.workspace_root.clone();
    let tool_registry = aigent_exec::default_registry(workspace_root);
    let tool_executor = ToolExecutor::new(policy);

    let runtime = AgentRuntime::new(config);
    runtime.run().await?;

    let (event_tx, _) = broadcast::channel::<BackendEvent>(BROADCAST_CAP);

    let state = Arc::new(Mutex::new(DaemonState {
        runtime,
        memory,
        tool_registry,
        tool_executor,
        recent_turns: VecDeque::new(),
        turn_count: 0,
        started_at: Instant::now(),
        last_turn_at: None,
        event_tx,
    }));

    let listener = UnixListener::bind(&socket_path)?;
    let (shutdown_tx, mut shutdown_rx) = watch::channel(false);
    info!(path = %socket_path.display(), "unified daemon listening");

    // Background compaction: remove old episodic entries every 24 hours so the
    // event log stays bounded even when the main sleep cycle doesn't fire.
    {
        let compaction_state = state.clone();
        let mut compaction_rx = shutdown_tx.subscribe();
        tokio::spawn(async move {
            let interval = std::time::Duration::from_secs(24 * 60 * 60);
            loop {
                tokio::select! {
                    _ = tokio::time::sleep(interval) => {
                        let mut s = compaction_state.lock().await;
                        match s.memory.compact_episodic(7) {
                            Ok(removed) if removed > 0 => {
                                info!(removed, "background episodic compaction complete");
                            }
                            Ok(_) => {}
                            Err(err) => warn!(?err, "background episodic compaction failed"),
                        }
                    }
                    changed = compaction_rx.changed() => {
                        if changed.is_ok() && *compaction_rx.borrow() {
                            break;
                        }
                    }
                }
            }
        });
    }

    // Background agentic sleep: consolidates memory and evolves personality
    // every N hours (default 8) without requiring an explicit CLI command.
    // Offset by half the interval so it doesn't co-fire with compaction.
    {
        let sleep_state = state.clone();
        let mut sleep_rx = shutdown_tx.subscribe();
        tokio::spawn(async move {
            let hours = std::env::var("AIGENT_SLEEP_INTERVAL_HOURS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(8);
            let interval = std::time::Duration::from_secs(hours * 60 * 60);
            // Initial offset: half the interval.
            tokio::select! {
                _ = tokio::time::sleep(interval / 2) => {}
                changed = sleep_rx.changed() => {
                    if changed.is_ok() && *sleep_rx.borrow() { return; }
                }
            }
            loop {
                tokio::select! {
                    _ = tokio::time::sleep(interval) => {
                        let mut s = sleep_state.lock().await;
                        let mut memory = std::mem::take(&mut s.memory);
                        match s.runtime.run_agentic_sleep_cycle(&mut memory).await {
                            Ok(ref summary) if !summary.distilled.is_empty() => {
                                info!(summary = %summary.distilled, "background sleep cycle complete");
                            }
                            Ok(_) => {}
                            Err(ref err) => warn!(?err, "background sleep cycle failed"),
                        }
                        s.memory = memory;
                    }
                    changed = sleep_rx.changed() => {
                        if changed.is_ok() && *sleep_rx.borrow() { break; }
                    }
                }
            }
        });
    }

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
        // Run agentic sleep on shutdown for a final consolidation pass.
        // Use std::mem::take to satisfy the borrow checker (runtime + memory
        // cannot be borrowed simultaneously through the same struct).
        let mut memory = std::mem::take(&mut state.memory);
        let _ = state.runtime.run_agentic_sleep_cycle(&mut memory).await;
        state.memory = memory;
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
                            let sleep_result = state
                                .runtime
                                .run_agentic_sleep_cycle(&mut memory)
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
                    ) {
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
            let mut state = state.lock().await;
            // Use mem::take to satisfy the borrow checker (runtime + memory
            // cannot be borrowed simultaneously through the same struct).
            let mut memory = std::mem::take(&mut state.memory);
            let result = state.runtime.run_agentic_sleep_cycle(&mut memory).await;
            state.memory = memory;
            let _ = state.memory.flush_all();
            let msg = match result {
                Ok(summary) => format!("sleep cycle complete: {}", summary.distilled),
                Err(err) => format!("sleep cycle failed: {err}"),
            };
            send_event(&mut write_half, ServerEvent::Ack(msg)).await?;
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
