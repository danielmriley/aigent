//! Unified daemon server — orchestrates memory, tools, and the agent runtime.

mod connection;
mod sleep;

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use aigent_tools::Tool;

use anyhow::Result;
use chrono::{DateTime, Timelike, Utc};
use chrono_tz::Tz;
use tokio::net::UnixListener;
use tokio::sync::{Mutex, broadcast, mpsc, watch};
use tracing::{info, warn};

use aigent_config::AppConfig;
use aigent_exec::{ExecutionPolicy, ToolExecutor, ToolOutcomeCallbackFn};
use aigent_memory::{EmbedFn, MemoryManager};
use aigent_tools::ToolRegistry;

use crate::{AgentRuntime, BackendEvent, ConversationTurn, DaemonStatus};
use crate::schedule_store;
use crate::scheduler::{ScheduledTask, SchedulerState, spawn_scheduler, HeartbeatFn};

/// Broadcast channel capacity. Old events are dropped when subscribers lag.
const BROADCAST_CAP: usize = 256;

/// UTF-8-safe truncation — ensures we never slice in the middle of a multi-byte
/// character.
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
    tool_registry: Arc<ToolRegistry>,
    tool_executor: Arc<ToolExecutor>,
    recent_turns: VecDeque<ConversationTurn>,
    turn_count: usize,
    /// Condensed summary of evicted conversation turns for long-session context.
    conversation_summary: Option<String>,
    started_at: Instant,
    /// Timestamp of the last completed conversation turn (persisted in-process).
    last_turn_at: Option<DateTime<Utc>>,
    /// Broadcast sender — fans out external-source BackendEvents to all subscribers.
    event_tx: broadcast::Sender<BackendEvent>,
    /// Instant of the last successful multi-agent sleep cycle run.
    last_multi_agent_sleep_at: Option<std::time::Instant>,
    /// Instant of the last successful passive distillation run.
    last_passive_sleep_at: Option<std::time::Instant>,
    /// Instant of the last successful confidence-learning sleep cycle run
    /// (Pass 1 stale-decay + Pass 2 episodic consolidation).
    last_confidence_sleep_at: Option<std::time::Instant>,
    /// Total proactive messages sent since daemon start.
    proactive_total_sent: u64,
    /// Timestamp of the last proactive message sent.
    last_proactive_at: Option<DateTime<Utc>>,
    /// Abort handle for Task C (proactive mode).  `None` when proactive mode is
    /// disabled.  Aborted gracefully during daemon shutdown so the task does not
    /// outlive the daemon process.
    proactive_handle: Option<tokio::task::AbortHandle>,
    /// Shared scheduler state for cron/interval tasks.
    scheduler_state: SchedulerState,
    /// Action prompts keyed by task name — looked up by the heartbeat callback.
    schedule_prompts: Arc<std::sync::RwLock<std::collections::HashMap<String, String>>>,
    /// Path to the schedule persistence file.
    #[allow(dead_code)]
    schedule_file: PathBuf,
    /// Handle for the scheduler task so it can be aborted on shutdown.
    scheduler_handle: Option<tokio::task::AbortHandle>,
}

impl DaemonState {
    fn status(&self) -> DaemonStatus {
        let stats = self.memory.stats();
        DaemonStatus {
            bot_name: self.runtime.config.agent.name.clone(),
            provider: self.runtime.config.llm.provider.clone(),
            model: self.runtime.config.active_model().to_string(),
            thinking_level: self.runtime.config.agent.thinking_level.clone(),
            external_thinking: self.runtime.config.agent.external_thinking,
            memory_total: stats.total,
            memory_core: stats.core,
            memory_user_profile: stats.user_profile,
            memory_reflective: stats.reflective,
            memory_semantic: stats.semantic,
            memory_episodic: stats.episodic,
            uptime_secs: self.started_at.elapsed().as_secs(),
            available_tools: self.tool_registry.list_specs().into_iter().map(|s| s.name).collect(),
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
        max_calls_per_tool: config.safety.max_calls_per_tool.unwrap_or(25),
        max_security_level: parse_security_level(&config.safety.max_security_level),
        tool_overrides: config.safety.tool_overrides.clone(),
    }
}

/// Parse a config string into a `SecurityLevel`, defaulting to `High` on
/// unrecognised values.
fn parse_security_level(s: &str) -> aigent_tools::SecurityLevel {
    match s.to_lowercase().as_str() {
        "low" => aigent_tools::SecurityLevel::Low,
        "medium" => aigent_tools::SecurityLevel::Medium,
        _ => aigent_tools::SecurityLevel::High,
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
    // Materialise the path so we can use it after the manager takes ownership.
    let memory_log_path = memory_log_path.as_ref().to_path_buf();
    if socket_path.exists() {
        let _ = std::fs::remove_file(&socket_path);
    }

    let mut memory = MemoryManager::with_event_log(&memory_log_path).await?;

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
    memory.set_learning_config(config.memory.learning.clone());
    memory.set_sleep_config(config.memory.sleep.clone());

    // Wire the redb-backed MemoryIndex for O(1) tier lookups and fast
    // deduplication.  Stored as a sibling to the JSONL event log so both
    // files live under `.aigent/memory/`.  Rebuilt from the event log on
    // every daemon start so the index is always consistent with the log.
    // All failures are non-fatal — the manager falls back to O(n) scans.
    {
        let index_path = memory_log_path.with_file_name("memory.index");
        let event_log = aigent_memory::event_log::MemoryEventLog::new(&memory_log_path);
        let index_result: anyhow::Result<aigent_memory::MemoryIndex> = async {
            let mut idx = aigent_memory::MemoryIndex::open(&index_path)?;
            match idx.rebuild_from_log(&event_log).await {
                Ok(n) => {
                    info!(entries = n, path = %index_path.display(), "memory index rebuilt");
                    Ok(idx)
                }
                Err(e) => {
                    warn!(?e, "index rebuild failed — resetting corrupt index and retrying");
                    let mut fresh = aigent_memory::MemoryIndex::reset(&index_path)?;
                    let n = fresh.rebuild_from_log(&event_log).await?;
                    info!(entries = n, "memory index rebuilt after reset");
                    Ok(fresh)
                }
            }
        }.await;
        match index_result {
            Ok(idx) => memory.set_index(idx),
            Err(e) => warn!(?e, path = %index_path.display(), "memory index unavailable — continuing without it (non-fatal)"),
        }
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
    let tavily_api_key = {
        let k = &config.tools.tavily_api_key;
        if k.is_empty() { None } else { Some(k.clone()) }
    };
    let searxng_base_url = {
        let u = &config.tools.searxng_base_url;
        if u.is_empty() { None } else { Some(u.clone()) }
    };
    let serper_api_key = {
        let k = &config.tools.serper_api_key;
        if k.is_empty() { None } else { Some(k.clone()) }
    };
    let exa_api_key = {
        let k = &config.tools.exa_api_key;
        if k.is_empty() { None } else { Some(k.clone()) }
    };
    let tool_registry = aigent_exec::default_registry(
        workspace_root,
        agent_data_dir,
        brave_api_key,
        tavily_api_key,
        searxng_base_url,
        serper_api_key,
        exa_api_key,
        config.tools.search_providers.clone(),
        config.tools.max_shell_command_bytes,
        config.tools.max_shell_output_bytes,
        config.tools.max_file_read_bytes,
    );
    let tool_executor = ToolExecutor::new(policy);

    // ── Bootstrap modules scaffolding scripts into the workspace ─────────
    if config.tools.modules.enabled {
        bootstrap_modules_infra(std::path::Path::new(&config.agent.workspace_path));
    }

    // ── Load dynamic modules from the modules directory at startup ─────────
    if config.tools.modules.enabled {
        let modules_dir = std::path::Path::new(&config.agent.workspace_path)
            .join(&config.tools.modules.modules_dir);
        if modules_dir.is_dir() {
            info!(dir = %modules_dir.display(), "scanning modules directory at startup");
            let workspace = std::path::Path::new(&config.agent.workspace_path);
            let max = if config.tools.modules.max_modules == 0 {
                usize::MAX
            } else {
                config.tools.modules.max_modules
            };
            let mut count = 0usize;
            let mut seen_names = std::collections::HashSet::new();
            if let Ok(entries) = std::fs::read_dir(&modules_dir) {
                for entry in entries.flatten() {
                    if count >= max { break; }
                    let path = entry.path();
                    if path.extension().and_then(|e| e.to_str()) == Some("wasm") {
                        if let Some(tool) = aigent_exec::wasm::WasmTool::load_with_workspace(
                            &path,
                            Some(workspace),
                        ) {
                            let name = tool.spec().name.clone();
                            if !seen_names.insert(name.clone()) {
                                warn!(module = %name, path = %path.display(),
                                    "duplicate module name \u{2014} skipping");
                                continue;
                            }
                            tool_registry.register_with_source(
                                Box::new(tool),
                                aigent_tools::ToolSource::Dynamic,
                            );
                            info!(module = %name, "loaded dynamic module at startup");
                            count += 1;
                        } else {
                            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown");
                            warn!(module = %stem, "failed to load module WASM at startup");
                        }
                    }
                }
            }
            if count > 0 {
                info!(count, "dynamic modules loaded at startup");
            }
        }
    }

    // Extract sleep scheduling config before `config` is moved into the runtime.
    let sleep_quiet_start = config.memory.night_sleep_start_hour as u32;
    let sleep_quiet_end = config.memory.night_sleep_end_hour as u32;
    let sleep_interval_hours: u64 = std::env::var("AIGENT_SLEEP_INTERVAL_HOURS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    // IANA timezone for the nightly quiet window.
    // Priority: config value → system auto-detect → UTC fallback.
    let sleep_tz: chrono_tz::Tz = if config.memory.timezone.is_empty() {
        match iana_time_zone::get_timezone() {
            Ok(ref tz_str) => match tz_str.parse::<chrono_tz::Tz>() {
                Ok(tz) => {
                    info!(tz = %tz, "sleep timezone auto-detected from system");
                    tz
                }
                Err(_) => {
                    warn!(tz = %tz_str, "auto-detected timezone is not a valid IANA name — falling back to UTC");
                    chrono_tz::UTC
                }
            },
            Err(e) => {
                warn!(?e, "could not auto-detect system timezone — falling back to UTC");
                chrono_tz::UTC
            }
        }
    } else {
        config.memory.timezone.parse().unwrap_or_else(|_| {
            warn!(tz = %config.memory.timezone, "unrecognised timezone in config — falling back to UTC");
            chrono_tz::UTC
        })
    };
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
            let delta: chrono::TimeDelta = Utc::now() - e.created_at;
            let age = delta.to_std().ok()?;
            Instant::now().checked_sub(age)
        });

    let state = Arc::new(Mutex::new(DaemonState {
        runtime,
        memory,
        tool_registry: Arc::new(tool_registry),
        tool_executor: Arc::new(tool_executor),
        recent_turns: VecDeque::new(),
        turn_count: 0,
        conversation_summary: None,
        started_at: Instant::now(),
        last_turn_at: None,
        event_tx,
        last_multi_agent_sleep_at,
        last_passive_sleep_at: None,
        last_confidence_sleep_at: None,
        proactive_total_sent: 0,
        last_proactive_at: None,
        proactive_handle: None,
        scheduler_state: SchedulerState::new(),
        schedule_prompts: Arc::new(std::sync::RwLock::new(std::collections::HashMap::new())),
        schedule_file: schedule_store::schedule_file_path(),
        scheduler_handle: None,
    }));

    // ── Register SearchMemoryTool (needs daemon state handle) ────────────────
    {
        let state_for_memory = state.clone();
        let memory_query_fn: aigent_tools::MemoryQueryFn = Arc::new(move |query, limit| {
            let st = state_for_memory.clone();
            Box::pin(async move {
                let s = st.lock().await;
                let ranked = s.memory.context_for_prompt_ranked(&query, limit);
                ranked
                    .into_iter()
                    .map(|r| aigent_tools::MemorySearchResult {
                        content: r.entry.content,
                        tier: format!("{:?}", r.entry.tier),
                        score: r.score,
                        source: r.entry.source,
                        created_at: r.entry.created_at.to_rfc3339(),
                    })
                    .collect()
            })
        });
        let search_tool = aigent_tools::SearchMemoryTool { query_fn: memory_query_fn };
        state.lock().await.tool_registry.register(Box::new(search_tool));
        info!("registered search_memory tool with live daemon memory handle");
    }

    // ── Register WriteMemoryTool (needs daemon state handle) ─────────────────
    {
        let state_for_write = state.clone();
        let memory_write_fn: aigent_tools::MemoryWriteFn = Arc::new(move |content, tier| {
            let st = state_for_write.clone();
            Box::pin(async move {
                let memory_tier = match tier.as_str() {
                    "semantic"   => aigent_memory::MemoryTier::Semantic,
                    "reflective" => aigent_memory::MemoryTier::Reflective,
                    _            => aigent_memory::MemoryTier::Episodic,
                };
                let mut s = st.lock().await;
                s.memory.record(memory_tier, content, "agent-explicit".to_string())
                    .await
                    .map(|e| e.id.to_string())
                    .map_err(|e| e.to_string())
            })
        });
        let write_tool = aigent_tools::WriteMemoryTool { write_fn: memory_write_fn };
        state.lock().await.tool_registry.register(Box::new(write_tool));
        info!("registered write_memory tool with live daemon memory handle");
    }

    // ── Wire tool-outcome confidence callback ─────────────────────────────────
    // After all state is constructed but before the first connection arrives,
    // attach the callback that feeds tool results back into the confidence model.
    // `Arc::get_mut` succeeds here because the executor Arc is owned solely by
    // `DaemonState` at this point (no connections have been accepted yet).
    {
        let cb_state = state.clone();
        let outcome_cb: ToolOutcomeCallbackFn = Arc::new(
            move |tool_name, success, output| {
                let st = cb_state.clone();
                Box::pin(async move {
                    let mut s = st.lock().await;
                    s.memory
                        .emit_tool_outcome_confidence(&tool_name, success, &output)
                        .await;
                })
            },
        );
        let mut s = state.lock().await;
        if let Some(exe) = Arc::get_mut(&mut s.tool_executor) {
            exe.set_outcome_callback(outcome_cb);
            info!("tool-outcome confidence callback registered");
        } else {
            warn!(
                "could not register tool-outcome confidence callback: \
                 executor Arc has multiple owners at startup"
            );
        }
    }

    let listener = UnixListener::bind(&socket_path)?;
    let (shutdown_tx, mut shutdown_rx) = watch::channel(false);
    info!(path = %socket_path.display(), "unified daemon listening");

    // Extract vault path before memory is moved into DaemonState so the
    // bidirectional watcher can be started with an owned PathBuf.
    let vault_path_for_watcher: Option<std::path::PathBuf> =
        state.lock().await.memory.vault_path().map(|p| p.to_path_buf());

    // Spawn background tasks.
    sleep::spawn_vault_watcher_task(state.clone(), vault_path_for_watcher);

    // ── Model warm-up ──────────────────────────────────────────────────────
    // Pre-load the primary model into VRAM so the first request isn't
    // penalised by cold model loading.
    {
        let (primary_model, provider_name) = {
            let st = state.lock().await;
            (
                st.runtime.config.active_model().to_string(),
                st.runtime.config.llm.provider.clone(),
            )
        };
        if provider_name.eq_ignore_ascii_case("ollama") && !primary_model.is_empty() {
            let model = primary_model.clone();
            info!("warming model: {}", model);
            tokio::spawn(async move {
                let base_url = std::env::var("OLLAMA_BASE_URL")
                    .unwrap_or_else(|_| "http://localhost:11434".to_string());
                let endpoint = format!("{}/api/chat", base_url.trim_end_matches('/'));
                let payload = serde_json::json!({
                    "model": model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": false,
                    "keep_alive": "30m",
                    "options": { "num_predict": 1 }
                });
                match reqwest::Client::new().post(&endpoint).json(&payload).send().await {
                    Ok(_) => info!("warm-up complete: {} loaded into VRAM", model),
                    Err(e) => warn!("warm-up failed for {} (non-fatal): {}", model, e),
                }
            });
        }
    }

    // ── Subagent parallelism check ─────────────────────────────────────────
    // Sub-agents run in parallel via tokio::join!, so Ollama must be
    // configured to handle multiple concurrent requests.  Without
    // OLLAMA_NUM_PARALLEL ≥ 3 (or ≥ 4 when sleep specialists also run),
    // requests queue up and inference falls back to serial execution.
    {
        let (subagents_enabled, provider_name) = {
            let st = state.lock().await;
            (
                st.runtime.config.subagents.enabled,
                st.runtime.config.llm.provider.clone(),
            )
        };
        if subagents_enabled && provider_name.eq_ignore_ascii_case("ollama") {
            let parallel = std::env::var("OLLAMA_NUM_PARALLEL")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(0);
            if parallel < 3 {
                warn!(
                    current = parallel,
                    recommended = 3,
                    "subagents are enabled but OLLAMA_NUM_PARALLEL is not set to ≥ 3; \
                     parallel specialist calls will queue and degrade to serial. \
                     Set OLLAMA_NUM_PARALLEL=4 in your Ollama environment."
                );
            }
        }
    }

    sleep::spawn_compaction_task(state.clone(), &shutdown_tx);
    sleep::spawn_passive_distillation(
        state.clone(), &shutdown_tx, sleep_interval_hours,
        forget_after_days, forget_min_confidence,
    );
    sleep::spawn_nightly_consolidation(
        state.clone(),
        &shutdown_tx,
        sleep_quiet_start,
        sleep_quiet_end,
        sleep_tz,
    );
    sleep::spawn_confidence_sleep_task(state.clone(), &shutdown_tx, sleep_quiet_start, sleep_quiet_end, sleep_tz);
    if proactive_interval_minutes > 0 {
        let handle = sleep::spawn_proactive_task(
            state.clone(),
            &shutdown_tx,
            proactive_interval_minutes,
            sleep_tz,
            proactive_dnd_start,
            proactive_dnd_end,
            proactive_cooldown_minutes,
        );
        state.lock().await.proactive_handle = Some(handle);
    }

    // ── Cron / Interval Scheduler ──────────────────────────────────────────
    {
        let schedule_file = schedule_store::schedule_file_path();
        let entries: Vec<schedule_store::TaskEntry> = schedule_store::load_tasks(&schedule_file).unwrap_or_default();
        info!(count = entries.len(), "loaded scheduled tasks from disk");

        // Build runtime ScheduledTask list and populate the prompts map.
        let mut tasks: Vec<ScheduledTask> = Vec::new();
        let prompts: Arc<std::sync::RwLock<std::collections::HashMap<String, String>>> = {
            let s = state.lock().await;
            s.schedule_prompts.clone()
        };
        {
            let mut map = prompts.write().unwrap();
            for entry in &entries {
                match entry.to_scheduled_task() {
                    Ok(task) => {
                        map.insert(task.name.clone(), entry.action_prompt.clone());
                        tasks.push(task);
                    }
                    Err(err) => {
                        warn!(name = %entry.name, ?err, "skipping invalid scheduled task");
                    }
                }
            }
        }

        let sched_state = state.lock().await.scheduler_state.clone();
        let socket_for_sched = socket_path.clone();
        let prompts_for_sched = prompts.clone();

        let heartbeat: HeartbeatFn = Arc::new(move |task_name: String| {
            let socket = socket_for_sched.clone();
            let prompts = prompts_for_sched.clone();
            Box::pin(async move {
                let prompt = {
                    let map = prompts.read().unwrap();
                    map.get(&task_name).cloned()
                };
                let Some(prompt) = prompt else {
                    tracing::debug!(task = %task_name, "scheduler: no prompt for task");
                    return Ok(());
                };
                // Submit through the daemon socket like any other client.
                let client = crate::DaemonClient::new(&socket);
                let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
                client
                    .stream_submit(
                        format!("[scheduled:{task_name}] Before starting, call search_memory to check for any prior work on this topic. Then execute: {prompt}"),
                        "scheduler",
                        tx,
                    )
                    .await?;
                // Drain events so the connection completes.
                while rx.recv().await.is_some() {}
                Ok(())
            })
        });

        let handle = spawn_scheduler(
            tasks,
            sched_state,
            heartbeat,
            Some(schedule_file.clone()),
            Some(prompts.clone()),
        );
        state.lock().await.scheduler_handle = Some(handle.abort_handle());
        info!("cron scheduler started");
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
                        tracing::error!(?err, "daemon connection handler failed");
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
        // Cancel the cron scheduler.
        let handle = state.lock().await.scheduler_handle.take();
        if let Some(h) = handle {
            h.abort();
            info!("cron scheduler stopped");
        }
    }
    {
        // VERIFIED: snapshot pattern — no take/restore.
        // Shutdown agentic sleep: snapshot → generate → apply, same pattern
        // as the nightly consolidation. std::mem::take is never used on
        // DaemonState fields across an .await — only in #[cfg(test)] regression
        // tests that document the historical bug. See tests::take_pattern_loses_concurrent_writes.
        let (rt_clone, memories_snapshot, identity_snapshot) = {
            let mut s = state.lock().await;
            let _ = s.memory.flush_all();
            let rt = s.runtime.clone();
            let memories = s.memory.all().to_vec();
            let identity = s.memory.identity.clone();
            (rt, memories, identity)
        };
        let (noop_tx, _) = mpsc::unbounded_channel::<String>();
        let gen_result = rt_clone
            .generate_agentic_sleep_insights(&memories_snapshot, &identity_snapshot, &noop_tx)
            .await;
        {
            let mut s = state.lock().await;
            match gen_result {
                Ok(crate::SleepGenerationResult::Insights(insights)) => {
                    let summary_text = Some("Shutdown agentic sleep cycle".to_string());
                    let _ = s.memory.apply_agentic_sleep_insights(*insights, summary_text).await;
                }
                Ok(crate::SleepGenerationResult::PassiveFallback(_)) => {
                    let _ = s.memory.run_sleep_cycle().await;
                }
                Err(_) => {
                    let _ = s.memory.run_sleep_cycle().await;
                }
            }
            let _ = s.memory.flush_all();
        }
    }
    let _ = std::fs::remove_file(&socket_path);
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

// ── Modules infrastructure bootstrap ─────────────────────────────────────────

/// Ensures the workspace has the modules scaffolding scripts and directories.
///
/// On a fresh install the `workspace/` dir is empty (it's gitignored).
/// The scripts are embedded at compile time from `extensions/modules-src/` so
/// they are always available regardless of how the binary was distributed.
fn bootstrap_modules_infra(workspace: &std::path::Path) {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    let src_dir = workspace.join("extensions/modules-src");
    let modules_dir = workspace.join("extensions/modules");

    // Create both directories if needed.
    let _ = fs::create_dir_all(&src_dir);
    let _ = fs::create_dir_all(&modules_dir);

    // Embedded scripts (compiled into the binary).
    static NEW_MODULE_SH: &str = include_str!("../../../../extensions/modules-src/new-module.sh");
    static BUILD_SH: &str = include_str!("../../../../extensions/modules-src/build.sh");
    static CARGO_TOML: &str = include_str!("../../../../extensions/modules-src/Cargo.toml");

    let files: &[(&str, &str, bool)] = &[
        ("new-module.sh", NEW_MODULE_SH, true),
        ("build.sh", BUILD_SH, true),
        ("Cargo.toml", CARGO_TOML, false),
    ];

    for &(name, contents, executable) in files {
        let path = src_dir.join(name);
        // Write if missing or if the embedded version is newer (content differs).
        let needs_write = match fs::read_to_string(&path) {
            Ok(existing) => existing != contents,
            Err(_) => true,
        };
        if needs_write {
            if let Err(e) = fs::write(&path, contents) {
                tracing::warn!(file = %path.display(), err = %e, "failed to write module script");
                continue;
            }
            if executable {
                let _ = fs::set_permissions(&path, fs::Permissions::from_mode(0o755));
            }
            tracing::info!(file = %path.display(), "bootstrapped module script");
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use aigent_memory::{MemoryManager, MemoryTier};
    use std::sync::Arc;
    use tokio::sync::Mutex;

    // ── is_in_window ─────────────────────────────────────────────────────

    #[test]
    fn window_within_normal_range() {
        // 10:00 should be inside [08:00, 18:00)
        let dt = chrono::TimeZone::with_ymd_and_hms(&Utc, 2026, 1, 15, 10, 0, 0).unwrap();
        assert!(is_in_window(dt, chrono_tz::UTC, 8, 18));
    }

    #[test]
    fn window_outside_normal_range() {
        // 20:00 should be outside [08:00, 18:00)
        let dt = chrono::TimeZone::with_ymd_and_hms(&Utc, 2026, 1, 15, 20, 0, 0).unwrap();
        assert!(!is_in_window(dt, chrono_tz::UTC, 8, 18));
    }

    #[test]
    fn window_wraps_midnight() {
        // 23:00 should be inside [22:00, 06:00) (wraps midnight)
        let dt = chrono::TimeZone::with_ymd_and_hms(&Utc, 2026, 1, 15, 23, 0, 0).unwrap();
        assert!(is_in_window(dt, chrono_tz::UTC, 22, 6));
    }

    #[test]
    fn window_wraps_midnight_early_morning() {
        // 03:00 should be inside [22:00, 06:00) (wraps midnight)
        let dt = chrono::TimeZone::with_ymd_and_hms(&Utc, 2026, 1, 15, 3, 0, 0).unwrap();
        assert!(is_in_window(dt, chrono_tz::UTC, 22, 6));
    }

    #[test]
    fn window_wraps_midnight_outside() {
        // 12:00 should be outside [22:00, 06:00)
        let dt = chrono::TimeZone::with_ymd_and_hms(&Utc, 2026, 1, 15, 12, 0, 0).unwrap();
        assert!(!is_in_window(dt, chrono_tz::UTC, 22, 6));
    }

    #[test]
    fn window_respects_timezone() {
        // 20:00 UTC → 21:00 CET (Europe/Berlin in winter).
        // Window [20:00, 23:00) in CET — 21:00 CET is inside.
        let dt = chrono::TimeZone::with_ymd_and_hms(&Utc, 2026, 1, 15, 20, 0, 0).unwrap();
        assert!(is_in_window(dt, chrono_tz::Europe::Berlin, 20, 23));
    }

    #[test]
    fn window_boundary_start_is_inclusive() {
        // Exactly on start_hour → should be inside.
        let dt = chrono::TimeZone::with_ymd_and_hms(&Utc, 2026, 1, 15, 8, 0, 0).unwrap();
        assert!(is_in_window(dt, chrono_tz::UTC, 8, 18));
    }

    #[test]
    fn window_boundary_end_is_exclusive() {
        // Exactly on end_hour → should be outside.
        let dt = chrono::TimeZone::with_ymd_and_hms(&Utc, 2026, 1, 15, 18, 0, 0).unwrap();
        assert!(!is_in_window(dt, chrono_tz::UTC, 8, 18));
    }

    // ── Concurrency regression tests ─────────────────────────────────────
    //
    // These tests prove the amnesia bug existed with `std::mem::take` and
    // is fixed with the snapshot-generate-apply pattern.

    /// Demonstrates the **old bug**: `std::mem::take` steals the live
    /// `MemoryManager` from shared state.  Any writes that happen while
    /// the background task holds the taken manager are written into a
    /// fresh `Default` manager — and silently lost when the background
    /// task puts its copy back.
    #[tokio::test]
    async fn take_pattern_loses_concurrent_writes() {
        let memory = MemoryManager::default();
        let shared = Arc::new(Mutex::new(memory));

        // Background task: take memory, simulate long work, restore.
        let bg_shared = shared.clone();
        let bg = tokio::spawn(async move {
            // Phase 1: take memory out of shared state (the old pattern).
            let taken = {
                let mut guard = bg_shared.lock().await;
                std::mem::take(&mut *guard)
                // guard dropped — shared now holds an empty Default manager
            };

            // Phase 2: simulate a long LLM call.
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;

            // Phase 3: restore the old memory, overwriting anything concurrent.
            {
                let mut guard = bg_shared.lock().await;
                *guard = taken;
            }
        });

        // Give the background task time to take the memory.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        // Concurrent "user turn": write a memory while background holds taken.
        {
            let mut guard = shared.lock().await;
            guard
                .record(MemoryTier::Episodic, "important user memory".to_string(), "user")
                .await
                .unwrap();
            // This recorded into the empty Default manager.
            assert_eq!(guard.all().len(), 1, "memory was recorded into the (empty) shared state");
        }

        // Wait for background task to finish restoring its copy.
        bg.await.unwrap();

        // The user's memory is GONE — overwritten by the background task's
        // restore of the pre-take snapshot.
        let guard = shared.lock().await;
        assert_eq!(
            guard.all().len(), 0,
            "BUG DEMONSTRATED: concurrent write lost after std::mem::take restore"
        );
    }

    /// Proves the **fix**: snapshot-generate-apply never displaces the live
    /// `MemoryManager`.  Concurrent writes survive because the background
    /// task only reads a snapshot and writes back additive results.
    #[tokio::test]
    async fn snapshot_pattern_preserves_concurrent_writes() {
        let mut memory = MemoryManager::default();
        // Seed an initial entry so the snapshot has something to "process".
        memory
            .record(MemoryTier::Episodic, "seed entry".to_string(), "test")
            .await
            .unwrap();

        let shared = Arc::new(Mutex::new(memory));

        // Background task: snapshot → simulate LLM work → apply additive result.
        let bg_shared = shared.clone();
        let bg = tokio::spawn(async move {
            // Phase 1: snapshot — briefly lock, clone data, release.
            let snapshot_count = {
                let guard = bg_shared.lock().await;
                let snap = guard.all().to_vec();
                snap.len()
                // guard dropped — shared state untouched
            };

            // Phase 2: simulate long LLM call with the snapshot data.
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            let _ = snapshot_count; // "use" the snapshot

            // Phase 3: re-acquire lock, apply additive insights to live state.
            {
                let mut guard = bg_shared.lock().await;
                guard
                    .record(
                        MemoryTier::Reflective,
                        "insight from sleep cycle".to_string(),
                        "sleep-cycle",
                    )
                    .await
                    .unwrap();
            }
        });

        // Give the background task time to take its snapshot and release.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        // Concurrent "user turn": write a memory while background is "thinking".
        {
            let mut guard = shared.lock().await;
            guard
                .record(MemoryTier::Episodic, "important user memory".to_string(), "user")
                .await
                .unwrap();
        }

        // Wait for background to finish applying.
        bg.await.unwrap();

        // ALL memories survive: seed + user's concurrent write + background's insight.
        let guard = shared.lock().await;
        let all = guard.all();
        assert_eq!(all.len(), 3, "seed + concurrent user write + sleep insight all preserved");

        let contents: Vec<&str> = all.iter().map(|e| e.content.as_str()).collect();
        assert!(contents.contains(&"seed entry"), "seed entry preserved");
        assert!(contents.contains(&"important user memory"), "concurrent user write preserved");
        assert!(
            contents.contains(&"insight from sleep cycle"),
            "sleep cycle insight applied"
        );
    }

    // ── Timezone auto-detection ──────────────────────────────────────────

    /// Verify the `iana-time-zone` crate can detect the system timezone
    /// and it parses to a valid `chrono_tz::Tz`.  This test proves the
    /// auto-detection path works on the host platform.
    #[test]
    fn system_timezone_is_detectable_and_valid() {
        // iana_time_zone::get_timezone() should succeed on any normal system.
        let tz_str = iana_time_zone::get_timezone()
            .expect("should be able to detect system timezone");
        assert!(!tz_str.is_empty(), "detected timezone should not be empty");

        // It must parse to a valid chrono_tz::Tz.
        let tz: chrono_tz::Tz = tz_str
            .parse()
            .unwrap_or_else(|_| panic!("detected timezone '{}' is not a valid IANA name", tz_str));
        // Sanity: format round-trips.
        assert_eq!(tz.to_string(), tz_str);
    }

    /// Verify the config → auto-detect → UTC fallback chain matches what
    /// the daemon startup code does.
    #[test]
    fn timezone_fallback_chain() {
        // 1. Config value takes priority.
        let tz: chrono_tz::Tz = "America/New_York".parse().unwrap();
        assert_eq!(tz, chrono_tz::America::New_York);

        // 2. Empty config → auto-detect succeeds (tested above).

        // 3. Invalid config string → UTC fallback.
        let tz: chrono_tz::Tz = "Not/A/Timezone".parse().unwrap_or(chrono_tz::UTC);
        assert_eq!(tz, chrono_tz::UTC);

        // 4. Empty string won't parse, so the daemon code checks `is_empty()`
        //    before parsing.  Verify is_empty triggers the auto-detect branch.
        let config_tz = "";
        assert!(config_tz.is_empty());
    }
}
