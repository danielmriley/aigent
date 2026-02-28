//! Cron-style task scheduler for periodic agent activities.
//!
//! Extends the existing proactive loop with a generic scheduler that can
//! manage multiple named tasks with independent intervals, DND windows,
//! and cooldowns.
//!
//! Built on top of `tokio::time` — no external cron library needed.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

// ── Configuration ──────────────────────────────────────────────────────────────

/// A single scheduled task definition.
#[derive(Debug, Clone)]
pub struct ScheduledTask {
    /// Unique task name (e.g. `"sleep-cycle"`, `"proactive"`, `"embedding-backfill"`).
    pub name: String,
    /// How often to run the task.
    pub interval: Duration,
    /// Minimum time between executions (cooldown).
    pub cooldown: Duration,
    /// Do-not-disturb window: `(start_hour, end_hour)` in 24h format.
    /// When `None`, the task runs at any hour.
    pub dnd_window: Option<(u32, u32)>,
    /// Whether the task is enabled.
    pub enabled: bool,
}

impl ScheduledTask {
    /// Create a new task with the given name and interval.
    pub fn new(name: impl Into<String>, interval: Duration) -> Self {
        Self {
            name: name.into(),
            interval,
            cooldown: Duration::ZERO,
            dnd_window: None,
            enabled: true,
        }
    }

    /// Set the cooldown period.
    pub fn with_cooldown(mut self, cooldown: Duration) -> Self {
        self.cooldown = cooldown;
        self
    }

    /// Set the DND window (24h format).
    pub fn with_dnd(mut self, start_hour: u32, end_hour: u32) -> Self {
        self.dnd_window = Some((start_hour, end_hour));
        self
    }

    /// Enable or disable the task.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

// ── Runtime state ──────────────────────────────────────────────────────────────

/// Per-task execution state.
#[derive(Debug, Clone)]
pub struct TaskState {
    /// When the task last executed successfully.
    pub last_run: Option<DateTime<Utc>>,
    /// Total successful executions.
    pub run_count: u64,
    /// Total failures.
    pub fail_count: u64,
    /// Whether the task is currently running.
    pub running: bool,
}

impl Default for TaskState {
    fn default() -> Self {
        Self {
            last_run: None,
            run_count: 0,
            fail_count: 0,
            running: false,
        }
    }
}

/// Thread-safe scheduler state shared between the scheduler loop and the
/// rest of the application.
#[derive(Clone)]
pub struct SchedulerState {
    inner: Arc<RwLock<HashMap<String, TaskState>>>,
}

impl SchedulerState {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get a snapshot of all task states.
    pub async fn snapshot(&self) -> HashMap<String, TaskState> {
        self.inner.read().await.clone()
    }

    /// Get a single task state.
    pub async fn get(&self, name: &str) -> Option<TaskState> {
        self.inner.read().await.get(name).cloned()
    }

    /// Mark a task as started.
    pub async fn mark_started(&self, name: &str) {
        let mut map = self.inner.write().await;
        let state = map.entry(name.to_string()).or_default();
        state.running = true;
    }

    /// Mark a task as completed successfully.
    pub async fn mark_completed(&self, name: &str) {
        let mut map = self.inner.write().await;
        let state = map.entry(name.to_string()).or_default();
        state.running = false;
        state.last_run = Some(Utc::now());
        state.run_count += 1;
    }

    /// Mark a task as failed.
    pub async fn mark_failed(&self, name: &str) {
        let mut map = self.inner.write().await;
        let state = map.entry(name.to_string()).or_default();
        state.running = false;
        state.fail_count += 1;
    }
}

impl Default for SchedulerState {
    fn default() -> Self {
        Self::new()
    }
}

// ── Scheduler ──────────────────────────────────────────────────────────────────

/// The heartbeat callback type.  The scheduler calls this function for each
/// task that is due.  The function receives the task name and should return
/// Ok(()) on success.
pub type HeartbeatFn = Arc<
    dyn Fn(String) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send>>
        + Send
        + Sync,
>;

/// Spawn a scheduler loop that periodically checks which tasks are due and
/// invokes the heartbeat callback for each one.
///
/// Returns a `JoinHandle` that can be aborted to stop the scheduler.
pub fn spawn_scheduler(
    tasks: Vec<ScheduledTask>,
    state: SchedulerState,
    heartbeat: HeartbeatFn,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        // The tick interval is the GCD of all task intervals, clamped to
        // a minimum of 10 seconds and maximum of 60 seconds.
        let tick = tasks
            .iter()
            .filter(|t| t.enabled)
            .map(|t| t.interval.as_secs().max(10))
            .fold(60u64, gcd);
        let tick = Duration::from_secs(tick.clamp(10, 60));

        info!(
            tick_secs = tick.as_secs(),
            num_tasks = tasks.len(),
            "scheduler started"
        );

        let mut interval = tokio::time::interval(tick);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            interval.tick().await;
            let now = Utc::now();

            for task in &tasks {
                if !task.enabled {
                    continue;
                }

                // DND check.
                if let Some((start, end)) = task.dnd_window {
                    let hour = now.time().hour();
                    if is_in_dnd(hour, start, end) {
                        debug!(task = %task.name, "skipping — DND window");
                        continue;
                    }
                }

                let task_state = state.get(&task.name).await.unwrap_or_default();

                // Already running?
                if task_state.running {
                    debug!(task = %task.name, "skipping — already running");
                    continue;
                }

                // Interval check.
                if let Some(last) = task_state.last_run {
                    let elapsed = (now - last).to_std().unwrap_or(Duration::ZERO);
                    if elapsed < task.interval {
                        continue;
                    }
                    // Cooldown check.
                    if elapsed < task.cooldown {
                        debug!(task = %task.name, "skipping — cooldown");
                        continue;
                    }
                }

                // Fire!
                info!(task = %task.name, "scheduler: firing task");
                state.mark_started(&task.name).await;

                let name = task.name.clone();
                let state_clone = state.clone();
                let heartbeat_clone = heartbeat.clone();

                tokio::spawn(async move {
                    match heartbeat_clone(name.clone()).await {
                        Ok(()) => {
                            state_clone.mark_completed(&name).await;
                            debug!(task = %name, "scheduler: task completed");
                        }
                        Err(e) => {
                            state_clone.mark_failed(&name).await;
                            warn!(task = %name, err = %e, "scheduler: task failed");
                        }
                    }
                });
            }
        }
    })
}

/// Check if the given hour falls within a DND window.
fn is_in_dnd(hour: u32, start: u32, end: u32) -> bool {
    if start <= end {
        hour >= start && hour < end
    } else {
        // Wraps midnight, e.g. 22..06.
        hour >= start || hour < end
    }
}

/// Greatest common divisor (for computing optimal tick interval).
fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

// Bring Timelike into scope for .hour()
use chrono::Timelike;

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dnd_simple_window() {
        assert!(is_in_dnd(23, 22, 6));
        assert!(is_in_dnd(2, 22, 6));
        assert!(!is_in_dnd(12, 22, 6));
    }

    #[test]
    fn dnd_daytime_window() {
        assert!(is_in_dnd(14, 9, 17));
        assert!(!is_in_dnd(8, 9, 17));
        assert!(!is_in_dnd(17, 9, 17));
    }

    #[test]
    fn gcd_values() {
        assert_eq!(gcd(60, 30), 30);
        assert_eq!(gcd(300, 120), 60);
        assert_eq!(gcd(0, 10), 10);
    }

    #[test]
    fn scheduled_task_builder() {
        let task = ScheduledTask::new("test", Duration::from_secs(60))
            .with_cooldown(Duration::from_secs(30))
            .with_dnd(22, 6)
            .with_enabled(true);

        assert_eq!(task.name, "test");
        assert_eq!(task.interval.as_secs(), 60);
        assert_eq!(task.cooldown.as_secs(), 30);
        assert_eq!(task.dnd_window, Some((22, 6)));
        assert!(task.enabled);
    }

    #[tokio::test]
    async fn scheduler_state_lifecycle() {
        let state = SchedulerState::new();

        assert!(state.get("test").await.is_none());

        state.mark_started("test").await;
        let s = state.get("test").await.unwrap();
        assert!(s.running);
        assert_eq!(s.run_count, 0);

        state.mark_completed("test").await;
        let s = state.get("test").await.unwrap();
        assert!(!s.running);
        assert_eq!(s.run_count, 1);
        assert!(s.last_run.is_some());

        state.mark_failed("test").await;
        let s = state.get("test").await.unwrap();
        assert_eq!(s.fail_count, 1);
    }

    #[tokio::test]
    async fn snapshot_returns_all_tasks() {
        let state = SchedulerState::new();
        state.mark_started("a").await;
        state.mark_started("b").await;

        let snap = state.snapshot().await;
        assert_eq!(snap.len(), 2);
        assert!(snap.contains_key("a"));
        assert!(snap.contains_key("b"));
    }
}
