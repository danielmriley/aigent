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
use tracing::{info, warn};

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
            info!("passive distillation starting");
            // Take memory out while briefly locked, then release the lock
            // before the (potentially long) distillation call so incoming
            // connections are never blocked here.
            let mut memory = {
                let mut s = state.lock().await;
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
                let mut s = state.lock().await;
                s.memory = memory;
                s.last_passive_sleep_at = Some(std::time::Instant::now());
            }
        }
    });
}

/// Task B — Nightly multi-agent consolidation (once per night, in quiet
/// window).  Runs `runtime.run_multi_agent_sleep_cycle()` which calls 4 LLM
/// specialists plus a synthesis agent.  Gated by:
///   1. Must be within the quiet window (night_sleep_start_hour..night_sleep_end_hour)
///   2. At least 22 hours since the last multi-agent cycle
///   3. No conversation in the last 15 minutes
pub(super) fn spawn_nightly_consolidation(
    state: Arc<Mutex<DaemonState>>,
    shutdown_tx: &watch::Sender<bool>,
    sleep_quiet_start: u32,
    sleep_quiet_end: u32,
    sleep_tz: Tz,
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

            // Time-of-day guard: only consolidate in the quiet window.
            if !is_in_window(Utc::now(), sleep_tz, sleep_quiet_start, sleep_quiet_end) {
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
                continue;
            }

            // All guards passed — run the nightly multi-agent cycle.
            info!("nightly multi-agent sleep cycle starting");
            // Clone runtime and take memory, then release lock before the
            // LLM call so incoming connections are never blocked here.
            let (rt_clone, mut memory) = {
                let mut s = state.lock().await;
                let rt = s.runtime.clone();
                let mem = std::mem::take(&mut s.memory);
                (rt, mem)
            };
            let (noop_tx, _) = mpsc::unbounded_channel::<String>();
            let result = rt_clone.run_multi_agent_sleep_cycle(&mut memory, &noop_tx).await;
            {
                let mut s = state.lock().await;
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

            let (rt_clone, mut memory) = {
                let mut s = state.lock().await;
                let rt = s.runtime.clone();
                let mem = std::mem::take(&mut s.memory);
                (rt, mem)
            };

            let outcome = rt_clone.run_proactive_check(&mut memory).await;

            {
                let mut s = state.lock().await;
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
