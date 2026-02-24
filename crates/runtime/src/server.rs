use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use chrono::{DateTime, Timelike, Utc};
use chrono_tz::Tz;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{Mutex, broadcast, mpsc, watch};
use tracing::{error, info, warn};

use aigent_config::AppConfig;
use aigent_exec::{ExecutionPolicy, ToolExecutor};
use aigent_memory::{EmbedFn, MemoryManager, MemoryTier, VaultEditEvent, spawn_vault_watcher};
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
    /// Instant of the last successful multi-agent sleep cycle run.
    last_multi_agent_sleep_at: Option<std::time::Instant>,
    /// Total proactive messages sent since daemon start.
    proactive_total_sent: u64,
    /// Timestamp of the last proactive message sent.
    last_proactive_at: Option<DateTime<Utc>>,
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
fn make_ollama_embed_fn(model: &str, base_url: &str) -> EmbedFn {
    let model = model.to_string();
    // Allow the environment variable to override the config value at runtime.
    let base_url = std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| base_url.to_string());
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
        let embed_base_url = config.llm.ollama_base_url.clone();
        memory.set_embed_fn(make_ollama_embed_fn(&embed_model, &embed_base_url));
        info!(model = %embed_model, "embedding backend configured");
    }

    // Apply config-driven memory tuning.
    memory.set_kv_tier_limit(config.memory.kv_tier_limit);

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
                if let Err(err) = memory.seed_core_identity(&user_name, &bot_name).await {
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

    // Extract sleep scheduling config before `config` is moved into the runtime.
    let sleep_quiet_start = config.memory.night_sleep_start_hour as u32;
    let sleep_quiet_end = config.memory.night_sleep_end_hour as u32;
    let sleep_interval_hours: u64 = std::env::var("AIGENT_SLEEP_INTERVAL_HOURS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    // IANA timezone for the nightly quiet window (default UTC).
    let sleep_tz: Tz = config.memory.timezone.parse().unwrap_or_else(|_| {
        tracing::warn!(tz = %config.memory.timezone, "unrecognised timezone — falling back to UTC");
        chrono_tz::UTC
    });
    // Lightweight forgetting parameters.
    let forget_after_days = config.memory.forget_episodic_after_days;
    let forget_min_confidence = config.memory.forget_min_confidence;
    // Proactive mode parameters.
    let proactive_interval_minutes = config.memory.proactive_interval_minutes;
    let proactive_dnd_start = config.memory.proactive_dnd_start_hour as u32;
    let proactive_dnd_end = config.memory.proactive_dnd_end_hour as u32;

    let runtime = AgentRuntime::new(config);
    runtime.run().await?;

    let (event_tx, _) = broadcast::channel::<BackendEvent>(BROADCAST_CAP);

    // Recover the last multi-agent sleep timestamp from persisted memory so
    // the 22-hour rate-limit survives daemon restarts.
    let last_multi_agent_sleep_at = memory
        .all()
        .iter()
        .filter(|e| e.source == "sleep:multi-agent-cycle")
        .max_by_key(|e| e.created_at)
        .and_then(|e| {
            let age = (Utc::now() - e.created_at).to_std().ok()?;
            Instant::now().checked_sub(age)
        });

    // Extract vault path before memory is moved into DaemonState so the
    // bidirectional watcher can be started with an owned PathBuf.
    let vault_path_for_watcher: Option<std::path::PathBuf> =
        memory.vault_path().map(|p| p.to_path_buf());

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
        last_multi_agent_sleep_at,        proactive_total_sent: 0,
        last_proactive_at: None,    }));

    // ── Bidirectional vault watcher ─────────────────────────────────────────────────
    // Watch the four summary files in the vault for human edits and ingest
    // any changes as high-confidence memories with source="human-edit".
    if let Some(vault_path) = vault_path_for_watcher {
        let (vault_edit_tx, mut vault_edit_rx) =
            tokio::sync::mpsc::unbounded_channel::<VaultEditEvent>();
        let _watcher_handle = spawn_vault_watcher(vault_path, vault_edit_tx);

        let watcher_state = state.clone();
        tokio::spawn(async move {
            while let Some(ev) = vault_edit_rx.recv().await {
                let mut s = watcher_state.lock().await;
                // Route to the appropriate tier based on the filename.
                let tier = if ev.filename == aigent_memory::KV_CORE {
                    MemoryTier::Core
                } else if ev.filename == aigent_memory::KV_USER_PROFILE {
                    MemoryTier::UserProfile
                } else {
                    MemoryTier::Reflective
                };
                // Record the raw edit so the next sleep cycle can reconcile it.
                // Truncate to 800 chars to avoid enormous prompt tokens.
                let note = format!(
                    "[human-edit] {} was updated in the vault:\n{}",
                    ev.filename,
                    &ev.content[..ev.content.len().min(800)]
                );
                if let Err(err) = s.memory.record(tier, note, "human-edit").await {
                    warn!(?err, file = %ev.filename, "vault watcher: failed to record human edit");
                } else {
                    info!(file = %ev.filename, tier = ?tier, "vault watcher: human edit ingested");
                }
            }
        });
    }

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
                        match s.memory.compact_episodic(7).await {
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

    // Task A — Frequent passive distillation (every 8h, no quiet window required).
    // Runs memory.run_sleep_cycle() (heuristic only, no LLM) so it is safe to run
    // at any time. Skip only if a conversation happened in the last 5 minutes.
    {
        let sleep_state = state.clone();
        let mut sleep_rx = shutdown_tx.subscribe();
        tokio::spawn(async move {
            let passive_interval = std::time::Duration::from_secs(sleep_interval_hours * 60 * 60);
            let poll_interval = std::time::Duration::from_secs(5 * 60);
            let mut last_passive_at = std::time::Instant::now()
                .checked_sub(passive_interval / 2)
                .unwrap_or_else(std::time::Instant::now);

            loop {
                tokio::select! {
                    _ = tokio::time::sleep(poll_interval) => {}
                    changed = sleep_rx.changed() => {
                        if changed.is_ok() && *sleep_rx.borrow() { break; }
                        continue;
                    }
                }

                if last_passive_at.elapsed() < passive_interval {
                    continue;
                }

                // Skip if conversation is actively ongoing.
                let recently_active = {
                    let s = sleep_state.lock().await;
                    s.last_turn_at
                        .map(|t| (Utc::now() - t).num_minutes() < 5)
                        .unwrap_or(false)
                };
                if recently_active {
                    continue;
                }

                last_passive_at = std::time::Instant::now();
                let mut s = sleep_state.lock().await;
                let mut memory = std::mem::take(&mut s.memory);
                match memory.run_sleep_cycle().await {
                    Ok(ref summary) if !summary.promoted_ids.is_empty() => {
                        info!(
                            promoted = summary.promoted_ids.len(),
                            "background passive distillation complete"
                        );
                    }
                    Ok(_) => {}
                    Err(ref err) => warn!(?err, "background passive distillation failed"),
                }
                // Lightweight forgetting: prune old low-confidence episodic entries.
                if forget_after_days > 0 {
                    let pruned = memory.run_forgetting_pass(forget_after_days, forget_min_confidence);
                    if pruned > 0 {
                        info!(pruned, forget_after_days, "lightweight forgetting applied during sleep");
                    }
                }
                s.memory = memory;
            }
        });
    }

    // Task B — Nightly multi-agent consolidation (once per night, in quiet window).
    // Runs runtime.run_multi_agent_sleep_cycle() which calls 4 LLM specialists
    // plus a synthesis agent. Gated by:
    //   1. Must be within the quiet window (night_sleep_start_hour..night_sleep_end_hour)
    //   2. At least 22 hours since the last multi-agent cycle
    //   3. No conversation in the last 15 minutes
    {
        let sleep_state = state.clone();
        let mut sleep_rx = shutdown_tx.subscribe();
        tokio::spawn(async move {
            let min_gap = std::time::Duration::from_secs(22 * 60 * 60);
            let poll_interval = std::time::Duration::from_secs(5 * 60);

            loop {
                tokio::select! {
                    _ = tokio::time::sleep(poll_interval) => {}
                    changed = sleep_rx.changed() => {
                        if changed.is_ok() && *sleep_rx.borrow() { break; }
                        continue;
                    }
                }

                // Time-of-day guard: only consolidate in the quiet window.
                let hour = Utc::now().with_timezone(&sleep_tz).hour();
                let in_quiet_window = if sleep_quiet_start <= sleep_quiet_end {
                    hour >= sleep_quiet_start && hour < sleep_quiet_end
                } else {
                    hour >= sleep_quiet_start || hour < sleep_quiet_end
                };
                if !in_quiet_window {
                    continue;
                }

                // Rate-limit: at least 22h since last multi-agent cycle.
                let already_ran = {
                    let s = sleep_state.lock().await;
                    s.last_multi_agent_sleep_at
                        .map(|t| t.elapsed() < min_gap)
                        .unwrap_or(false)
                };
                if already_ran {
                    continue;
                }

                // Conversation recency guard.
                let recently_active = {
                    let s = sleep_state.lock().await;
                    s.last_turn_at
                        .map(|t| (Utc::now() - t).num_minutes() < 15)
                        .unwrap_or(false)
                };
                if recently_active {
                    continue;
                }

                // All guards passed — run the nightly multi-agent cycle.
                // Clone runtime and take memory, then release lock before the
                // LLM call so incoming connections are never blocked here.
                let (rt_clone, mut memory) = {
                    let mut s = sleep_state.lock().await;
                    let rt = s.runtime.clone();
                    let mem = std::mem::take(&mut s.memory);
                    (rt, mem)
                };
                let (noop_tx, _) = mpsc::unbounded_channel::<String>();
                let result = rt_clone.run_multi_agent_sleep_cycle(&mut memory, &noop_tx).await;
                {
                    let mut s = sleep_state.lock().await;
                    s.memory = memory;
                    match result {
                        Ok(ref summary) if !summary.distilled.is_empty() => {
                            s.last_multi_agent_sleep_at = Some(std::time::Instant::now());
                            info!(summary = %summary.distilled, "background multi-agent sleep cycle complete");
                        }
                        Ok(_) => {
                            s.last_multi_agent_sleep_at = Some(std::time::Instant::now());
                        }
                        Err(ref err) => warn!(?err, "background multi-agent sleep cycle failed"),
                    }
                }
            }
        });
    }

    // Task C — Proactive mode: fire periodically and optionally send an
    // unprompted message.  Disabled when proactive_interval_minutes == 0.
    if proactive_interval_minutes > 0 {
        let proactive_state = state.clone();
        let mut proactive_shutdown_rx = shutdown_tx.subscribe();
        tokio::spawn(async move {
            let interval = std::time::Duration::from_secs(proactive_interval_minutes * 60);
            let poll = std::time::Duration::from_secs(60);
            let mut last_check = std::time::Instant::now()
                .checked_sub(interval / 2)
                .unwrap_or_else(std::time::Instant::now);

            loop {
                tokio::select! {
                    _ = tokio::time::sleep(poll) => {}
                    changed = proactive_shutdown_rx.changed() => {
                        if changed.is_ok() && *proactive_shutdown_rx.borrow() { break; }
                        continue;
                    }
                }

                if last_check.elapsed() < interval {
                    continue;
                }

                // DND window guard.
                let hour = Utc::now().with_timezone(&sleep_tz).hour();
                let in_dnd = if proactive_dnd_start <= proactive_dnd_end {
                    hour >= proactive_dnd_start && hour < proactive_dnd_end
                } else {
                    hour >= proactive_dnd_start || hour < proactive_dnd_end
                };
                if in_dnd {
                    continue;
                }

                last_check = std::time::Instant::now();

                let (rt_clone, mut memory) = {
                    let mut s = proactive_state.lock().await;
                    let rt = s.runtime.clone();
                    let mem = std::mem::take(&mut s.memory);
                    (rt, mem)
                };

                let outcome = rt_clone.run_proactive_check(&mut memory).await;

                {
                    let mut s = proactive_state.lock().await;
                    s.memory = memory;
                    let _ = s.memory.flush_all();
                    if let Some(out) = outcome {
                        if let Some(ref msg) = out.message {
                            let event = BackendEvent::ProactiveMessage { content: msg.clone() };
                            let _ = s.event_tx.send(event);
                            let _ = s
                                .memory
                                .record(
                                    MemoryTier::Episodic,
                                    format!("[proactive] {msg}"),
                                    "proactive",
                                )
                                .await;
                            s.proactive_total_sent += 1;
                            s.last_proactive_at = Some(Utc::now());
                            info!(message_len = msg.len(), "Task C: proactive message sent");
                        }
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
        // Clone runtime and take memory while holding the lock, then release
        // before the LLM call so incoming connections (e.g. from daemon restart)
        // are not blocked indefinitely.
        let (rt_clone, mut memory) = {
            let mut s = state.lock().await;
            let _ = s.memory.flush_all();
            let rt = s.runtime.clone();
            let mem = std::mem::take(&mut s.memory);
            (rt, mem)
        };
        let (noop_tx, _) = mpsc::unbounded_channel::<String>();
        let _ = rt_clone.run_agentic_sleep_cycle(&mut memory, &noop_tx).await;
        let mut s = state.lock().await;
        s.memory = memory;
        let _ = s.memory.flush_all();
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
                        // Inline reflection: extract beliefs and insights from the exchange.
                        let reflect_events = state.runtime
                            .inline_reflect(&mut memory, &user, &reply)
                            .await
                            .unwrap_or_default();

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

                        Ok((extras, reflect_events))
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
                Ok((extras, reflect_events)) => {
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
            // Take memory out of state and release the lock before the LLM call.
            // Spawn the cycle in a separate task so we can stream StatusLine
            // progress events back to the client while it runs.
            let (runtime, memory) = {
                let mut s = state.lock().await;
                let runtime = s.runtime.clone();
                let memory = std::mem::take(&mut s.memory);
                (runtime, memory)
            };
            let (progress_tx, mut progress_rx) = mpsc::unbounded_channel::<String>();
            let (done_tx, mut done_rx) =
                tokio::sync::oneshot::channel::<(String, MemoryManager)>();
            tokio::spawn(async move {
                let mut mem = memory;
                let msg = match runtime.run_agentic_sleep_cycle(&mut mem, &progress_tx).await {
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
                tokio::sync::oneshot::channel::<(String, MemoryManager)>();
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
        ClientCommand::TriggerProactive => {
            let (rt_clone, mut memory) = {
                let mut s = state.lock().await;
                let rt = s.runtime.clone();
                let mem = std::mem::take(&mut s.memory);
                (rt, mem)
            };
            let event_tx_clone = event_tx.clone();
            let outcome = rt_clone.run_proactive_check(&mut memory).await;
            let msg = {
                let mut s = state.lock().await;
                s.memory = memory;
                let _ = s.memory.flush_all();
                if let Some(out) = outcome {
                    if let Some(ref m) = out.message {
                        let ev = BackendEvent::ProactiveMessage { content: m.clone() };
                        let _ = event_tx_clone.send(ev);
                        let _ = s
                            .memory
                            .record(
                                MemoryTier::Episodic,
                                format!("[proactive] {m}"),
                                "proactive",
                            )
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
