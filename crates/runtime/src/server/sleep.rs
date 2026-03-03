//! Background sleep cycle orchestration.
//!
//! Contains the spawned background tasks for memory compaction, passive
//! distillation, nightly multi-agent consolidation, proactive mode, and the
//! vault watcher.

use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use chrono_tz::Tz;
use tokio::sync::{Mutex, mpsc, watch};
use tracing::{debug, info, warn};

use aigent_memory::{MemoryTier, VaultEditEvent, spawn_vault_watcher as spawn_fs_vault_watcher};

use crate::BackendEvent;

use super::{DaemonState, is_in_window, safe_truncate};

/// Spawn background compaction: remove old episodic entries every 24 hours so
/// the event log stays bounded even when the main sleep cycle doesn't fire.
pub(super) fn spawn_compaction_task(
    state: Arc<Mutex<DaemonState>>,
    shutdown_tx: &watch::Sender<bool>,
) {
    let mut rx = shutdown_tx.subscribe();
    tokio::spawn(async move {
        let interval = Duration::from_secs(24 * 60 * 60);
        loop {
            tokio::select! {
                _ = tokio::time::sleep(interval) => {
                    let mut s = state.lock().await;
                    match s.memory.compact_episodic(7).await {
                        Ok(removed) if removed > 0 => {
                            info!(removed, "background episodic compaction complete");
                        }
                        Ok(_) => {}
                        Err(err) => warn!(?err, "background episodic compaction failed"),
                    }
                }
                changed = rx.changed() => {
                    if changed.is_ok() && *rx.borrow() {
                        break;
                    }
                }
            }
        }
    });
}

/// Task A — Frequent passive distillation (every N hours, no quiet window
/// required).  Runs `memory.run_sleep_cycle()` (heuristic only, no LLM) so it
/// is safe to run at any time.  Skips only if a conversation happened in the
/// last 5 minutes.
///
/// This task is lightweight: it acquires the lock, runs the fast heuristic
/// pass, and releases the lock.  No `std::mem::take` needed.
pub(super) fn spawn_passive_distillation(
    state: Arc<Mutex<DaemonState>>,
    shutdown_tx: &watch::Sender<bool>,
    sleep_interval_hours: u64,
    forget_after_days: u64,
    forget_min_confidence: f32,
) {
    let mut rx = shutdown_tx.subscribe();
    tokio::spawn(async move {
        let passive_interval = Duration::from_secs(sleep_interval_hours * 60 * 60);
        let poll_interval = Duration::from_secs(5 * 60);
        let mut last_passive_at = std::time::Instant::now()
            .checked_sub(passive_interval / 2)
            .unwrap_or_else(std::time::Instant::now);

        loop {
            tokio::select! {
                _ = tokio::time::sleep(poll_interval) => {}
                changed = rx.changed() => {
                    if changed.is_ok() && *rx.borrow() { break; }
                    continue;
                }
            }

            if last_passive_at.elapsed() < passive_interval {
                continue;
            }

            // Skip if conversation is actively ongoing.
            let recently_active = {
                let s = state.lock().await;
                s.last_turn_at
                    .map(|t| (Utc::now() - t).num_minutes() < 5)
                    .unwrap_or(false)
            };
            if recently_active {
                continue;
            }

            last_passive_at = std::time::Instant::now();
            info!("passive heuristic distillation: starting");
            {
                let mut s = state.lock().await;
                match s.memory.run_sleep_cycle().await {
                    Ok(ref summary) if !summary.promoted_ids.is_empty() => {
                        info!(
                            promoted = summary.promoted_ids.len(),
                            "passive heuristic distillation: complete"
                        );
                    }
                    Ok(_) => {
                        info!("passive heuristic distillation: complete (no promotions)");
                    }
                    Err(ref err) => warn!(?err, "passive heuristic distillation: failed"),
                }
                // Lightweight forgetting: prune old low-confidence episodic entries.
                if forget_after_days > 0 {
                    match s.memory.run_forgetting_pass(forget_after_days, forget_min_confidence).await {
                        Ok(pruned) if pruned > 0 => {
                            info!(pruned, forget_after_days, "passive heuristic distillation: forgetting pass applied");
                        }
                        Err(ref err) => warn!(?err, "forgetting pass failed"),
                        _ => {}
                    }
                }
                s.last_passive_sleep_at = Some(std::time::Instant::now());
            }
        }
    });
}

/// Task B — Nightly multi-agent consolidation (once per night, in quiet
/// window).  Runs the multi-agent LLM pipeline which can take several minutes.
///
/// **Concurrency-safe**: the lock is held only to snapshot data and to apply
/// insights afterwards — never across LLM network calls.  Any conversation
/// that occurs while the LLM is thinking operates on the live
/// `MemoryManager` undisturbed, and new memories are preserved.
///
/// Gated by:
///   1. Must be within the quiet window (night_sleep_start_hour..night_sleep_end_hour)
///   2. At least 22 hours since the last multi-agent cycle
///   3. No conversation in the last 15 minutes
pub(super) fn spawn_nightly_consolidation(
    state: Arc<Mutex<DaemonState>>,
    shutdown_tx: &watch::Sender<bool>,
    default_quiet_start: u32,
    default_quiet_end: u32,
    default_tz: Tz,
) {
    let mut rx = shutdown_tx.subscribe();
    tokio::spawn(async move {
        let min_gap = Duration::from_secs(22 * 60 * 60);
        let poll_interval = Duration::from_secs(5 * 60);

        loop {
            tokio::select! {
                _ = tokio::time::sleep(poll_interval) => {}
                changed = rx.changed() => {
                    if changed.is_ok() && *rx.borrow() { break; }
                    continue;
                }
            }

            // ── Dynamic quiet-window resolution ──────────────────────
            // Read timezone and quiet-window hours from UserProfile
            // entries (written during onboarding).  Fall back to the
            // config defaults if the profile hasn't been populated yet.
            let (sleep_tz, sleep_quiet_start, sleep_quiet_end) = {
                let s = state.lock().await;
                let tz = profile_tz(&s.memory, default_tz);
                let (qs, qe) = profile_quiet_window(
                    &s.memory,
                    default_quiet_start,
                    default_quiet_end,
                );
                (tz, qs, qe)
            };

            // Time-of-day guard: only consolidate in the quiet window.
            if !is_in_window(Utc::now(), sleep_tz, sleep_quiet_start, sleep_quiet_end) {
                debug!(
                    tz = %sleep_tz,
                    start = sleep_quiet_start,
                    end = sleep_quiet_end,
                    "nightly consolidation: outside quiet window, skipping"
                );
                continue;
            }

            // Rate-limit: at least 22h since last multi-agent cycle.
            let already_ran = {
                let s = state.lock().await;
                s.last_multi_agent_sleep_at
                    .map(|t| t.elapsed() < min_gap)
                    .unwrap_or(false)
            };
            if already_ran {
                debug!("nightly consolidation: ran recently (<22h), skipping");
                continue;
            }

            // Conversation recency guard.
            let recently_active = {
                let s = state.lock().await;
                s.last_turn_at
                    .map(|t| (Utc::now() - t).num_minutes() < 15)
                    .unwrap_or(false)
            };
            if recently_active {
                debug!("nightly consolidation: conversation active (<15min), skipping");
                continue;
            }

            // All guards passed — run the nightly multi-agent cycle.
            info!("nightly multi-agent LLM consolidation: starting");

            // ── Phase 1: Snapshot — briefly lock to clone read-only data ──
            let (rt_clone, memories_snapshot, identity_snapshot) = {
                let s = state.lock().await;
                let rt = s.runtime.clone();
                let memories = s.memory.all().to_vec();
                let identity = s.memory.identity.clone();
                (rt, memories, identity)
            };
            // Lock is released — a user conversation can proceed normally.

            // ── Phase 2: Generate — long-running LLM calls, no lock held ──
            // Overall timeout so a hung LLM can never block the nightly
            // pipeline indefinitely (individual request timeout is 5 min
            // via the reqwest client; this caps the whole multi-agent run).
            let llm_start = std::time::Instant::now();
            let (noop_tx, _) = mpsc::unbounded_channel::<String>();
            let gen_result = match tokio::time::timeout(
                Duration::from_secs(10 * 60), // 10 min hard cap
                rt_clone.generate_multi_agent_sleep_insights(
                    &memories_snapshot,
                    &identity_snapshot,
                    &noop_tx,
                ),
            )
            .await
            {
                Ok(inner) => inner,
                Err(_elapsed) => {
                    warn!(
                        elapsed_secs = llm_start.elapsed().as_secs(),
                        "nightly consolidation: timed out after 10 minutes"
                    );
                    Err(anyhow::anyhow!("nightly consolidation timed out"))
                }
            };

            // ── Phase 3: Apply — re-acquire lock and merge insights ───────
            {
                let mut s = state.lock().await;
                match gen_result {
                    Ok(crate::SleepGenerationResult::Insights(insights)) => {
                        let summary_text = Some(format!(
                            "Nightly multi-agent consolidation: {} learned, {} follow-ups, {} reflections, {} profile updates",
                            insights.learned_about_user.len(),
                            insights.follow_ups.len(),
                            insights.reflective_thoughts.len(),
                            insights.user_profile_updates.len(),
                        ));
                        match s.memory.apply_agentic_sleep_insights(*insights, summary_text).await {
                            Ok(ref summary) if !summary.distilled.is_empty() => {
                                // Write the sentinel so the 22h rate-limit survives restarts.
                                let _ = s.memory.record(
                                    aigent_memory::MemoryTier::Semantic,
                                    "multi-agent sleep cycle completed",
                                    "sleep:multi-agent-cycle",
                                ).await;
                                s.last_multi_agent_sleep_at = Some(std::time::Instant::now());
                                info!(elapsed_secs = llm_start.elapsed().as_secs(), summary = %summary.distilled, "nightly multi-agent LLM consolidation: complete");
                            }
                            Ok(_) => {
                                let _ = s.memory.record(
                                    aigent_memory::MemoryTier::Semantic,
                                    "multi-agent sleep cycle completed",
                                    "sleep:multi-agent-cycle",
                                ).await;
                                s.last_multi_agent_sleep_at = Some(std::time::Instant::now());
                                info!("nightly multi-agent LLM consolidation: complete (no summary text)");
                            }
                            Err(ref err) => {
                                warn!(?err, "nightly multi-agent LLM consolidation: failed to apply insights");
                            }
                        }
                    }
                    Ok(crate::SleepGenerationResult::PassiveFallback(_)) => {
                        // LLM was unavailable — fall back to passive heuristic.
                        info!("nightly consolidation: LLM unavailable, running passive heuristic distillation as fallback");
                        match s.memory.run_sleep_cycle().await {
                            Ok(ref summary) => {
                                s.last_multi_agent_sleep_at = Some(std::time::Instant::now());
                                info!(
                                    promoted = summary.promoted_ids.len(),
                                    "nightly consolidation: passive fallback complete"
                                );
                            }
                            Err(ref err) => warn!(?err, "nightly consolidation: passive fallback failed"),
                        }
                    }
                    Err(ref err) => {
                        warn!(?err, "nightly multi-agent LLM consolidation: generation failed");
                    }
                }
            }
        }
    });
}

// ── Task C — Proactive mode ──────────────────────────────────────────────────

/// Spawn the proactive background loop that periodically fires
/// `run_proactive_check` and broadcasts a `ProactiveMessage` event when the
/// agent decides to speak unprompted.  Disabled when
/// `proactive_interval_minutes == 0`.
///
/// Returns the [`tokio::task::AbortHandle`] so the caller can cancel Task C
/// during shutdown.
pub(super) fn spawn_proactive_task(
    state: Arc<Mutex<DaemonState>>,
    shutdown_tx: &watch::Sender<bool>,
    proactive_interval_minutes: u64,
    sleep_tz: Tz,
    proactive_dnd_start: u32,
    proactive_dnd_end: u32,
    proactive_cooldown_minutes: i64,
) -> tokio::task::AbortHandle {
    let mut rx = shutdown_tx.subscribe();
    let handle = tokio::spawn(async move {
        let interval = Duration::from_secs(proactive_interval_minutes * 60);
        let poll = Duration::from_secs(60);
        let mut last_check = std::time::Instant::now()
            .checked_sub(interval / 2)
            .unwrap_or_else(std::time::Instant::now);

        loop {
            tokio::select! {
                _ = tokio::time::sleep(poll) => {}
                changed = rx.changed() => {
                    if changed.is_ok() && *rx.borrow() { break; }
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
                    let s = state.lock().await;
                    s.last_proactive_at
                        .map(|t| (Utc::now() - t).num_minutes() < proactive_cooldown_minutes)
                        .unwrap_or(false)
                };
                if in_cooldown {
                    continue;
                }
            }

            last_check = std::time::Instant::now();

            // ── Full AgentLoop proactive tick ────────────────────────────────
            // Instead of a rigid JSON prompt, we run a complete tool loop so
            // the agent can search the web, query memory, run shell commands,
            // etc. while the user is away.  Only the *final* text (if any) is
            // broadcast as a ProactiveMessage.
            let proactive_user_msg = "\
[PROACTIVE WAKE-UP] You have woken up autonomously. Follow the protocol:\n\
\n\
1. ORIENT — Run these two calls first:\n\
   - search_memory(query=\"recent topics open questions user goals\", limit=10)\n\
   - list_cron_jobs()\n\
\n\
2. DECIDE — Based on what you find, pick ONE high-value action:\n\
   Priority order: unresolved user problems > active scheduled research > \n\
   curiosity-driven exploration of user interests.\n\
\n\
3. ACT — Execute your chosen action using as many tool calls as needed. \n\
   Chain web_search -> browse_page -> search_memory as appropriate.\n\
\n\
4. RECORD — Summarize what you learned or accomplished in your final message. \n\
   Use specific keywords so search_memory can find this later. If your \n\
   investigation raised follow-up questions, call create_cron_job to \n\
   schedule the next step.\n\
\n\
5. OUTPUT — If you discovered something the user needs to know, output a \n\
   concise message. If your work was routine maintenance with nothing \n\
   urgent, respond with an empty string to return to sleep silently.";

            // ── Snapshot: clone runtime + memory + tools (lock released fast) ──
            let (rt_clone, mut memory, recent, tool_specs, registry, executor) = {
                let mut s = state.lock().await;
                let rt = s.runtime.clone();
                let recent = s.recent_turns.iter().cloned().collect::<Vec<_>>();
                let mem = std::mem::take(&mut s.memory);
                let specs = s.tool_registry.list_specs();
                let reg = Arc::clone(&s.tool_registry);
                let exe = Arc::clone(&s.tool_executor);
                (rt, mem, recent, specs, reg, exe)
            };
            // Lock released — user turns proceed normally.

            // Build system prompt with full context (identity, beliefs, tools, memories).
            let context = memory.context_for_prompt_ranked_with_embed("proactive background check", 8, None);
            let stats = memory.stats();

            // Pre-compute memory blocks for the prompt builder (pure function).
            let identity_block = memory.cached_identity_block().to_string();
            let beliefs_block = memory.cached_beliefs_block(rt_clone.config.memory.max_beliefs_in_prompt).to_string();
            let user_name = memory.user_name_from_core();
            let relational_block = memory.relational_state_block();

            let prompt_inputs = aigent_prompt::PromptInputs {
                config: &rt_clone.config,
                user_message: proactive_user_msg,
                recent_turns: &recent,
                tool_specs: &[],   // tools go via API, not baked into text
                pending_follow_ups: &[],
                context_items: &context,
                stats,
                identity_block,
                beliefs_block,
                user_name,
                relational_block,
            };
            let system_prompt = aigent_prompt::build_chat_prompt(&prompt_inputs);

            let mut messages = vec![aigent_llm::ChatMessage::system(&system_prompt)];
            for turn in &recent {
                messages.push(aigent_llm::ChatMessage::user(&turn.user));
                messages.push(aigent_llm::ChatMessage::assistant(&turn.assistant));
            }
            messages.push(aigent_llm::ChatMessage::user(proactive_user_msg));

            // Build tool definitions for the API.
            let tools_json = if tool_specs.is_empty() {
                None
            } else {
                Some(aigent_thinker::build_tools_json(&tool_specs))
            };

            let primary = aigent_llm::Provider::from(rt_clone.config.llm.provider.as_str());

            // Create a sink channel — proactive tokens are NOT shown to the user.
            let (sink_tx, mut sink_rx) = tokio::sync::mpsc::channel::<String>(64);
            tokio::spawn(async move { while sink_rx.recv().await.is_some() {} });

            // Run the full tool loop (silent — no event broadcast to TUI).
            let loop_result = aigent_thinker::run_tool_loop(
                &rt_clone.llm,
                primary,
                &rt_clone.config.llm.ollama_model,
                &rt_clone.config.llm.openrouter_model,
                &mut messages,
                tools_json.as_ref(),
                &registry,
                &executor,
                sink_tx,
                None,   // no event_tx → tool events are hidden from the user
            
                std::time::Duration::from_secs(rt_clone.config.agent.step_timeout_seconds),
            ).await;

            // ── Apply phase: re-acquire lock, record results ──
            {
                let mut s = state.lock().await;
                s.memory = memory;
                let _ = s.memory.flush_all();

                match loop_result {
                    Ok(ref result) => {
                        // Record tool executions to procedural memory.
                        for exec in &result.tool_executions {
                            let outcome = format!(
                                "[proactive] Tool '{}' executed. Output (first 400 chars): {}",
                                exec.tool_name,
                                safe_truncate(&exec.output, 400),
                            );
                            let _: Result<_, _> = s.memory.record(
                                MemoryTier::Procedural,
                                outcome,
                                format!("proactive-tool:{}", exec.tool_name),
                            ).await;
                        }

                        let reply = result.content.trim();
                        if !reply.is_empty() {
                            let event = BackendEvent::ProactiveMessage { content: reply.to_string() };
                            let _ = s.event_tx.send(event);
                            let _ = s
                                .memory
                                .record(
                                    MemoryTier::Episodic,
                                    format!("[proactive] {reply}"),
                                    "proactive:agent-loop",
                                )
                                .await;
                            s.proactive_total_sent += 1;
                            s.last_proactive_at = Some(Utc::now());
                            info!(
                                message_len = reply.len(),
                                tools_used = result.tool_executions.len(),
                                "Task C: proactive agent loop completed with message"
                            );
                        } else {
                            debug!(
                                tools_used = result.tool_executions.len(),
                                "Task C: proactive agent loop completed silently"
                            );
                        }
                    }
                    Err(err) => {
                        warn!(?err, "Task C: proactive agent loop failed");
                    }
                }
            }
        }
    });
    handle.abort_handle()
}

// ── Bidirectional vault watcher ──────────────────────────────────────────────

/// Watch the vault summary files for human edits and ingest any changes as
/// high-confidence memories with `source="human-edit"`.
pub(super) fn spawn_vault_watcher_task(
    state: Arc<Mutex<DaemonState>>,
    vault_path: Option<std::path::PathBuf>,
) {
    let Some(vault_path) = vault_path else { return };

    let (vault_edit_tx, mut vault_edit_rx) =
        tokio::sync::mpsc::unbounded_channel::<VaultEditEvent>();
    let _watcher_handle = spawn_fs_vault_watcher(vault_path, vault_edit_tx);

    tokio::spawn(async move {
        while let Some(ev) = vault_edit_rx.recv().await {
            let mut s = state.lock().await;
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


// ── Profile-based quiet window helpers ───────────────────────────────────────

/// Read the user's IANA timezone from `UserProfile` entries, falling back to
/// `default_tz` if nothing is found or the value doesn't parse.
///
/// Looks for profile entries whose source contains `timezone` or whose content
/// starts with `timezone:`.
fn profile_tz(memory: &aigent_memory::MemoryManager, default_tz: Tz) -> Tz {
    for e in memory.all().iter().rev() {
        if e.tier != aigent_memory::MemoryTier::UserProfile {
            continue;
        }
        let low = e.content.to_lowercase();
        let src_low = e.source.to_lowercase();
        if low.starts_with("timezone:") || src_low.contains("timezone") {
            let val = e
                .content
                .split_once(':')
                .map(|x| x.1)
                .unwrap_or(&e.content)
                .trim();
            if let Ok(tz) = val.parse::<Tz>() {
                return tz;
            }
        }
    }
    default_tz
}

/// Read the user's preferred quiet-window hours from `UserProfile` entries.
///
/// Scans for entries whose source or content contains the keys
/// `sleep_start` / `sleep_end` (or `quiet_start` / `quiet_end`).
/// Returns `(start_hour, end_hour)` falling back to config defaults.
fn profile_quiet_window(
    memory: &aigent_memory::MemoryManager,
    default_start: u32,
    default_end: u32,
) -> (u32, u32) {
    let mut start: Option<u32> = None;
    let mut end: Option<u32> = None;

    for e in memory.all().iter().rev() {
        if e.tier != aigent_memory::MemoryTier::UserProfile {
            continue;
        }
        let low = e.content.to_lowercase();
        let src_low = e.source.to_lowercase();

        let is_start = src_low.contains("sleep_start")
            || src_low.contains("quiet_start")
            || low.starts_with("sleep_start_hour:")
            || low.starts_with("quiet_start:");
        let is_end = src_low.contains("sleep_end")
            || src_low.contains("quiet_end")
            || low.starts_with("sleep_end_hour:")
            || low.starts_with("quiet_end:");

        if is_start && start.is_none() {
            if let Some(val) = extract_trailing_u32(&e.content) {
                if val < 24 {
                    start = Some(val);
                }
            }
        }
        if is_end && end.is_none() {
            if let Some(val) = extract_trailing_u32(&e.content) {
                if val < 24 {
                    end = Some(val);
                }
            }
        }
        if start.is_some() && end.is_some() {
            break;
        }
    }

    (start.unwrap_or(default_start), end.unwrap_or(default_end))
}

/// Extract a trailing integer from a "key: value" or "key :: value" string.
fn extract_trailing_u32(s: &str) -> Option<u32> {
    let val_str = s
        .split_once("::")
        .map(|x| x.1)
        .or_else(|| s.split_once(':').map(|x| x.1))?
        .trim();
    val_str.parse().ok()
}
