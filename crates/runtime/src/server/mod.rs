//! Unified daemon server — orchestrates memory, tools, and the agent runtime.

mod connection;

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use chrono::{DateTime, Timelike, Utc};
use chrono_tz::Tz;

use tokio::net::UnixListener;
use tokio::sync::{Mutex, broadcast, mpsc, watch};
use tracing::{error, info, warn};

use aigent_config::AppConfig;
use aigent_exec::{ExecutionPolicy, ToolExecutor};
use aigent_memory::{EmbedFn, MemoryManager, MemoryTier, VaultEditEvent, spawn_vault_watcher};
use aigent_tools::ToolRegistry;

use crate::{
    AgentRuntime, BackendEvent, ConversationTurn, DaemonStatus,
};

/// Broadcast channel capacity. Old events are dropped when subscribers lag.
const BROADCAST_CAP: usize = 256;


/// Return `&text[..limit]` rounded down to a UTF-8 char boundary.
/// Avoids panics when the target byte index falls inside a multi-byte character.
pub(super) fn safe_truncate(text: &str, limit: usize) -> &str {
    if limit >= text.len() {
        return text;
    }
    let mut end = limit;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}
/// Returns `true` when `now` (UTC) falls within the `[start_hour, end_hour)` window
/// expressed in the given timezone.  Handles midnight-wrap correctly
/// (e.g. 22 00 06 spans midnight and wraps around 0).
pub(super) fn is_in_window(now: DateTime<Utc>, tz: Tz, start_hour: u32, end_hour: u32) -> bool {
    let hour = now.with_timezone(&tz).hour();
    if start_hour <= end_hour {
        hour >= start_hour && hour < end_hour
    } else {
        // Window wraps midnight (e.g. 22:00 – 06:00)
        hour >= start_hour || hour < end_hour
    }
}

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
    /// Instant of the last successful passive distillation run.
    last_passive_sleep_at: Option<std::time::Instant>,
    /// Total proactive messages sent since daemon start.
    proactive_total_sent: u64,
    /// Timestamp of the last proactive message sent.
    last_proactive_at: Option<DateTime<Utc>>,
    /// Abort handle for Task C (proactive mode).  `None` when proactive mode is
    /// disabled.  Aborted gracefully during daemon shutdown so the task does not
    /// outlive the daemon process.
    proactive_handle: Option<tokio::task::AbortHandle>,
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
        approval_mode: config.tools.approval_mode.clone(),
        approval_required: config.safety.approval_required,
        allow_shell: config.safety.allow_shell,
        allow_wasm: config.safety.allow_wasm,
        workspace_root,
        tool_allowlist: config.safety.tool_allowlist.clone(),
        tool_denylist: config.safety.tool_denylist.clone(),
        approval_exempt_tools: config.safety.approval_exempt_tools.clone(),
        git_auto_commit: config.tools.git_auto_commit,
        sandbox_enabled: config.tools.sandbox_enabled,
    }
}

/// Build an async embedding function that calls the Ollama `/api/embeddings`
/// endpoint.  Falls back to `None` silently so the system continues to work
/// when Ollama is unavailable.
fn make_ollama_embed_fn(model: &str, base_url: &str) -> EmbedFn {
    let model = model.to_string();
    // Allow the environment variable to override the config value at runtime.
    let base_url = std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| base_url.to_string());
    let url = format!("{}/api/embeddings", base_url.trim_end_matches('/'));

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap_or_default();

    Arc::new(move |text: String| {
        let client = client.clone();
        let url = url.clone();
        let model = model.clone();
        Box::pin(async move {
            let body = serde_json::json!({ "model": model, "prompt": text });
            let resp = client.post(&url).json(&body).send().await.ok()?;
            let json: serde_json::Value = resp.json().await.ok()?;
            let embedding = json["embedding"]
                .as_array()?
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect::<Vec<f32>>();
            if embedding.is_empty() { None } else { Some(embedding) }
        })
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

    // Ensure the workspace directory exists and has a git repo so the agent
    // can use git for file recovery/rollback inside its sandbox.
    std::fs::create_dir_all(&workspace_root).ok();
    if let Err(e) = aigent_exec::git::git_init_if_needed(&workspace_root).await {
        warn!(?e, "failed to auto-init workspace git repo (non-fatal)");
    }
    let agent_data_dir = std::path::Path::new(".aigent").to_path_buf();
    let brave_api_key = {
        let key = config.tools.brave_api_key.trim().to_string();
        if key.is_empty() { None } else { Some(key) }
    };
    let tool_registry = aigent_exec::default_registry(workspace_root, agent_data_dir, brave_api_key, &config);
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
    let proactive_cooldown_minutes = config.memory.proactive_cooldown_minutes as i64;

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
        last_multi_agent_sleep_at,
        last_passive_sleep_at: None,
        proactive_total_sent: 0,
        last_proactive_at: None,
        proactive_handle: None,
    }));

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
                    safe_truncate(&ev.content, 800)
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
                info!("passive distillation starting");
                // Take memory out while briefly locked, then release the lock
                // before the (potentially long) distillation call so incoming
                // connections are never blocked here.
                let mut memory = {
                    let mut s = sleep_state.lock().await;
                    std::mem::take(&mut s.memory)
                };
                match memory.run_sleep_cycle().await {
                    Ok(ref summary) if !summary.promoted_ids.is_empty() => {
                        info!(
                            promoted = summary.promoted_ids.len(),
                            "background passive distillation complete"
                        );
                    }
                    Ok(_) => {
                        info!("passive distillation complete (no promotions)");
                    }
                    Err(ref err) => warn!(?err, "background passive distillation failed"),
                }
                // Lightweight forgetting: prune old low-confidence episodic entries.
                if forget_after_days > 0 {
                    let pruned = memory.run_forgetting_pass(forget_after_days, forget_min_confidence);
                    if pruned > 0 {
                        info!(pruned, forget_after_days, "lightweight forgetting applied during sleep");
                    }
                }
                {
                    let mut s = sleep_state.lock().await;
                    s.memory = memory;
                    s.last_passive_sleep_at = Some(std::time::Instant::now());
                }
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
                if !is_in_window(Utc::now(), sleep_tz, sleep_quiet_start, sleep_quiet_end) {
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
                info!("nightly multi-agent sleep cycle starting");
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
        let proactive_join = tokio::spawn(async move {
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

                // DND window guard — uses the same timezone as the sleep window.
                if is_in_window(Utc::now(), sleep_tz, proactive_dnd_start, proactive_dnd_end) {
                    continue;
                }

                // Cooldown guard — skip if a message was sent too recently.
                if proactive_cooldown_minutes > 0 {
                    let in_cooldown = {
                        let s = proactive_state.lock().await;
                        s.last_proactive_at
                            .map(|t| (Utc::now() - t).num_minutes() < proactive_cooldown_minutes)
                            .unwrap_or(false)
                    };
                    if in_cooldown {
                        continue;
                    }
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
        // Store the abort handle so shutdown can cancel Task C cleanly.
        state.lock().await.proactive_handle = Some(proactive_join.abort_handle());
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
                    if let Err(err) = connection::handle_connection(stream, state, shutdown_tx).await {
                        error!(?err, "daemon connection handler failed");
                    }
                });
            }
        }
    }

    info!("daemon shutting down gracefully");
    {
        // Cancel Task C before the final sleep so it cannot fire mid-shutdown.
        let handle = state.lock().await.proactive_handle.take();
        if let Some(h) = handle {
            h.abort();
            info!("proactive task stopped");
        }
    }
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

