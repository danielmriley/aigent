//! Unified daemon server — orchestrates memory, tools, and the agent runtime.

mod connection;
mod sleep;

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use chrono::{DateTime, Timelike, Utc};
use chrono_tz::Tz;
use tokio::net::UnixListener;
use tokio::sync::{Mutex, broadcast, mpsc, watch};
use tracing::{info, warn};

use aigent_config::AppConfig;
use aigent_exec::{ExecutionPolicy, ToolExecutor};
use aigent_memory::{EmbedFn, MemoryManager};
use aigent_tools::ToolRegistry;

use crate::{AgentRuntime, BackendEvent, ConversationTurn, DaemonStatus};

/// Broadcast channel capacity. Old events are dropped when subscribers lag.
const BROADCAST_CAP: usize = 256;

/// UTF-8-safe truncation — ensures we never slice in the middle of a multi-byte
/// character.
#[allow(dead_code)]
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

/// Returns `true` when `now` (UTC) falls within the `[start_hour, end_hour)`
/// window expressed in the given timezone.  Handles midnight-wrap correctly
/// (e.g. 22:00–06:00 spans midnight and wraps around 0).
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

    let mut memory = MemoryManager::with_event_log(memory_log_path).await?;

    // Wire in the Ollama embedding backend so all new entries are automatically
    // embedded and retrieval uses hybrid lexical+vector scoring.
    {
        let embed_model = config.llm.ollama_model.clone();
        let embed_base_url = config.llm.ollama_base_url.clone();
        memory.set_embed_fn(make_ollama_embed_fn(&embed_model, &embed_base_url));
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

    let agent_data_dir = workspace_root.join(".aigent");
    let brave_api_key = {
        let k = &config.tools.brave_api_key;
        if k.is_empty() { None } else { Some(k.clone()) }
    };
    let tool_registry =
        aigent_exec::default_registry(workspace_root, agent_data_dir, brave_api_key, &config)
            .await;
    let tool_executor = ToolExecutor::new(policy);

    // Extract sleep scheduling config before `config` is moved into the runtime.
    let sleep_quiet_start = config.memory.night_sleep_start_hour as u32;
    let sleep_quiet_end = config.memory.night_sleep_end_hour as u32;
    let sleep_interval_hours: u64 = std::env::var("AIGENT_SLEEP_INTERVAL_HOURS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);

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
    }));

    let listener = UnixListener::bind(&socket_path)?;
    let (shutdown_tx, mut shutdown_rx) = watch::channel(false);
    info!(path = %socket_path.display(), "unified daemon listening");

    // Spawn background tasks.
    sleep::spawn_compaction_task(state.clone(), &shutdown_tx);
    sleep::spawn_passive_distillation(state.clone(), &shutdown_tx, sleep_interval_hours);
    sleep::spawn_nightly_consolidation(
        state.clone(),
        &shutdown_tx,
        sleep_quiet_start,
        sleep_quiet_end,
    );

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
                        tracing::error!(?err, "daemon connection handler failed");
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
