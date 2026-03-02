//! Cron-style task scheduler for periodic agent activities.
//!
//! Extends the existing proactive loop with a generic scheduler that can
//! manage multiple named tasks with independent intervals, DND windows,
//! cooldowns, and **cron expressions**.
//!
//! Supports two scheduling modes:
//! - **Interval**: fixed `Duration` between runs (original mode)
//! - **Cron**: standard 6-field cron expressions via the `cron` crate
//!   (`sec min hour day-of-month month day-of-week`)
//!
//! Built on top of `tokio::time` with the `cron` crate for expression parsing.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use chrono::{DateTime, Utc};
use cron::Schedule as CronSchedule;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

// ── Schedule type ────────────────────────────────────────────────────────────

/// How a task's timing is determined.
#[derive(Debug, Clone)]
pub enum TaskSchedule {
    /// Fixed interval between runs.
    Interval(Duration),
    /// Cron expression (6-field: sec min hour dom month dow).
    ///
    /// Example: `"0 */15 * * * *"` = every 15 minutes.
    Cron(Box<CronSchedule>),
}

impl TaskSchedule {
    /// Parse a cron expression string.
    ///
    /// Accepts standard 6-field cron syntax:
    /// ```text
    /// sec  min  hour  day-of-month  month  day-of-week
    /// 0    */5  *     *             *      *
    /// ```
    pub fn cron(expr: &str) -> Result<Self, cron::error::Error> {
        let schedule: CronSchedule = expr.parse()?;
        Ok(Self::Cron(Box::new(schedule)))
    }

    /// Create a fixed-interval schedule.
    pub fn interval(d: Duration) -> Self {
        Self::Interval(d)
    }

    /// Check if the task is due given the last run time.
    ///
    /// For interval-based tasks, returns true if `now - last_run >= interval`.
    /// For cron-based tasks, returns true if there is a scheduled occurrence
    /// between `last_run` and `now`.
    pub fn is_due(&self, last_run: Option<DateTime<Utc>>, now: DateTime<Utc>) -> bool {
        match self {
            Self::Interval(d) => {
                match last_run {
                    None => true, // Never run before.
                    Some(last) => {
                        let elapsed = (now - last).to_std().unwrap_or(Duration::ZERO);
                        elapsed >= *d
                    }
                }
            }
            Self::Cron(schedule) => {
                let after = last_run.unwrap_or_else(|| now - chrono::Duration::hours(1));
                // Check if there's at least one upcoming occurrence between
                // `after` and `now`.
                schedule
                    .after(&after)
                    .take(1)
                    .any(|next| next <= now)
            }
        }
    }

    /// For interval-based tasks, return the interval in seconds.
    /// For cron-based tasks, estimate the minimum gap.
    pub fn min_interval_secs(&self) -> u64 {
        match self {
            Self::Interval(d) => d.as_secs().max(10),
            Self::Cron(schedule) => {
                // Sample the next two occurrences to estimate the gap.
                let _now = Utc::now();
                let mut upcoming = schedule.upcoming(Utc).take(2);
                if let (Some(a), Some(b)) = (upcoming.next(), upcoming.next()) {
                    let gap = (b - a).num_seconds().unsigned_abs();
                    gap.max(10)
                } else {
                    60 // Fallback.
                }
            }
        }
    }
}

// ── Configuration ────────────────────────────────────────────────────────────

/// A single scheduled task definition.
#[derive(Debug, Clone)]
pub struct ScheduledTask {
    /// Unique task name (e.g. `"sleep-cycle"`, `"proactive"`, `"embedding-backfill"`).
    pub name: String,
    /// How & when to run the task.
    pub schedule: TaskSchedule,
    /// Minimum time between executions (cooldown).
    pub cooldown: Duration,
    /// Do-not-disturb window: `(start_hour, end_hour)` in 24h format.
    /// When `None`, the task runs at any hour.
    pub dnd_window: Option<(u32, u32)>,
    /// Whether the task is enabled.
    pub enabled: bool,
}

impl ScheduledTask {
    /// Create a new task with the given name and fixed interval.
    pub fn new(name: impl Into<String>, interval: Duration) -> Self {
        Self {
            name: name.into(),
            schedule: TaskSchedule::Interval(interval),
            cooldown: Duration::ZERO,
            dnd_window: None,
            enabled: true,
        }
    }

    /// Create a new task from a cron expression.
    ///
    /// ```rust,ignore
    /// ScheduledTask::from_cron("nightly-cleanup", "0 0 3 * * *")?;
    /// ```
    pub fn from_cron(name: impl Into<String>, cron_expr: &str) -> Result<Self, cron::error::Error> {
        Ok(Self {
            name: name.into(),
            schedule: TaskSchedule::cron(cron_expr)?,
            cooldown: Duration::ZERO,
            dnd_window: None,
            enabled: true,
        })
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

    /// Backward-compatible accessor: approximate interval as Duration.
    pub fn interval(&self) -> Duration {
        match &self.schedule {
            TaskSchedule::Interval(d) => *d,
            TaskSchedule::Cron(_) => Duration::from_secs(self.schedule.min_interval_secs()),
        }
    }
}

// ── Runtime state ────────────────────────────────────────────────────────────

/// Per-task execution state.
#[derive(Debug, Clone)]
#[derive(Default)]
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

// ── Scheduler ────────────────────────────────────────────────────────────────

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
/// Supports both fixed-interval and cron-expression tasks.
///
/// Returns a `JoinHandle` that can be aborted to stop the scheduler.
pub fn spawn_scheduler(
    tasks: Vec<ScheduledTask>,
    state: SchedulerState,
    heartbeat: HeartbeatFn,
    schedule_file: Option<PathBuf>,
    schedule_prompts: Option<Arc<std::sync::RwLock<HashMap<String, String>>>>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        // The tick interval is the GCD of all task intervals, clamped to
        // a minimum of 10 seconds and maximum of 60 seconds.
        let tick = tasks
            .iter()
            .filter(|t| t.enabled)
            .map(|t| t.schedule.min_interval_secs())
            .fold(60u64, gcd);
        let tick = Duration::from_secs(tick.clamp(10, 60));

        info!(
            tick_secs = tick.as_secs(),
            num_tasks = tasks.len(),
            "scheduler started"
        );

        let mut interval = tokio::time::interval(tick);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        let mut tasks = tasks;
        let mut last_mtime: Option<SystemTime> = schedule_file
            .as_ref()
            .and_then(|p| std::fs::metadata(p).ok())
            .and_then(|m| m.modified().ok());

        loop {
            interval.tick().await;
            let now = Utc::now();

            // ── Hot-reload: check if schedule file changed ──
            if let Some(ref path) = schedule_file {
                let current_mtime = std::fs::metadata(path)
                    .ok()
                    .and_then(|m| m.modified().ok());
                if current_mtime != last_mtime {
                    last_mtime = current_mtime;
                    match crate::schedule_store::load_tasks(path) {
                        Ok(entries) => {
                            let mut new_tasks = Vec::new();
                            for entry in &entries {
                                match entry.to_scheduled_task() {
                                    Ok(task) => {
                                        new_tasks.push(task);
                                    }
                                    Err(err) => {
                                        warn!(
                                            name = %entry.name,
                                            ?err,
                                            "scheduler: skipping invalid task on reload"
                                        );
                                    }
                                }
                            }
                            // Update the prompts map if available.
                            if let Some(ref prompts) = schedule_prompts {
                                let mut map = prompts.write().unwrap();
                                map.clear();
                                for (task, entry) in new_tasks.iter().zip(entries.iter()) {
                                    map.insert(
                                        task.name.clone(),
                                        entry.action_prompt.clone(),
                                    );
                                }
                            }
                            info!(
                                count = new_tasks.len(),
                                "scheduler: reloaded tasks from disk"
                            );
                            tasks = new_tasks;
                        }
                        Err(err) => {
                            warn!(?err, "scheduler: failed to reload schedule file");
                        }
                    }
                }
            }

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

                // Check if the task is due (works for both interval and cron).
                if !task.schedule.is_due(task_state.last_run, now) {
                    continue;
                }

                // Cooldown check.
                if let Some(last) = task_state.last_run {
                    let elapsed = (now - last).to_std().unwrap_or(Duration::ZERO);
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

// ── Tests ────────────────────────────────────────────────────────────────────

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
        assert_eq!(task.interval().as_secs(), 60);
        assert_eq!(task.cooldown.as_secs(), 30);
        assert_eq!(task.dnd_window, Some((22, 6)));
        assert!(task.enabled);
    }

    #[test]
    fn cron_task_creation() {
        let task = ScheduledTask::from_cron("nightly", "0 0 3 * * *").unwrap();
        assert_eq!(task.name, "nightly");
        assert!(task.enabled);
        assert!(matches!(task.schedule, TaskSchedule::Cron(_)));
    }

    #[test]
    fn cron_invalid_expression() {
        let result = ScheduledTask::from_cron("bad", "not a cron");
        assert!(result.is_err());
    }

    #[test]
    fn interval_is_due() {
        let sched = TaskSchedule::Interval(Duration::from_secs(60));
        let now = Utc::now();

        // Never run before → due.
        assert!(sched.is_due(None, now));

        // Ran 30s ago → not due.
        let recent = now - chrono::Duration::seconds(30);
        assert!(!sched.is_due(Some(recent), now));

        // Ran 90s ago → due.
        let old = now - chrono::Duration::seconds(90);
        assert!(sched.is_due(Some(old), now));
    }

    #[test]
    fn cron_is_due() {
        // Every second ("* * * * * *").
        let sched = TaskSchedule::cron("* * * * * *").unwrap();
        let now = Utc::now();

        // Last ran 2 seconds ago → should be due.
        let last = now - chrono::Duration::seconds(2);
        assert!(sched.is_due(Some(last), now));
    }

    #[test]
    fn cron_min_interval() {
        // Every 5 minutes.
        let sched = TaskSchedule::cron("0 */5 * * * *").unwrap();
        let gap = sched.min_interval_secs();
        // Should be ~300 seconds.
        assert!((290..=310).contains(&gap), "gap was {}", gap);
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
