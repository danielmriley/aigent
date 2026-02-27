//! Background sleep cycle orchestration.
//!
//! Contains the spawned background tasks for memory compaction, passive
//! distillation, and nightly multi-agent consolidation.

use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use tokio::sync::{Mutex, mpsc, watch};
use tracing::{info, warn};

use super::DaemonState;

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
            let mut s = state.lock().await;
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
            s.memory = memory;
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
            let hour = chrono::Timelike::hour(&Utc::now());
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
