//! Persistent storage for dynamically created scheduled tasks.
//!
//! Tasks are persisted as a JSON array in `.aigent/schedule.json` so they
//! survive daemon restarts.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::scheduler::{ScheduledTask, TaskSchedule};

/// On-disk representation of a single scheduled task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskEntry {
    /// Unique task name.
    pub name: String,
    /// For interval-based tasks: interval in seconds.
    pub interval_secs: Option<u64>,
    /// For cron-based tasks: the 6-field cron expression.
    pub cron_expr: Option<String>,
    /// The prompt/action the agent should execute when this task fires.
    pub action_prompt: String,
    /// Cooldown in seconds (default 0).
    #[serde(default)]
    pub cooldown_secs: u64,
    /// Optional DND window (start_hour, end_hour).
    #[serde(default)]
    pub dnd_window: Option<(u32, u32)>,
    /// Whether the task is enabled.
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_true() -> bool {
    true
}

impl TaskEntry {
    /// Convert to a runtime `ScheduledTask`.
    pub fn to_scheduled_task(&self) -> Result<ScheduledTask> {
        let schedule = if let Some(ref cron) = self.cron_expr {
            TaskSchedule::cron(cron).context("invalid cron expression")?
        } else if let Some(secs) = self.interval_secs {
            TaskSchedule::interval(Duration::from_secs(secs))
        } else {
            anyhow::bail!("task '{}' has neither cron_expr nor interval_secs", self.name);
        };

        let mut task = ScheduledTask {
            name: self.name.clone(),
            schedule,
            cooldown: Duration::from_secs(self.cooldown_secs),
            dnd_window: self.dnd_window,
            enabled: self.enabled,
        };
        // Attach the action prompt as an extension field via the name convention
        // "name::prompt" isn't great — store it separately in a side-map instead.
        let _ = &mut task; // placeholder — prompt is looked up separately.
        Ok(task)
    }
}

/// Default path: `.aigent/schedule.json` under the current working directory.
pub fn schedule_file_path() -> PathBuf {
    PathBuf::from(".aigent").join("schedule.json")
}

/// Load all task entries from disk.  Returns an empty vec if the file doesn't
/// exist.
pub fn load_tasks(path: &Path) -> Result<Vec<TaskEntry>> {
    if !path.exists() {
        return Ok(vec![]);
    }
    let data = std::fs::read_to_string(path)
        .with_context(|| format!("read schedule file {}", path.display()))?;
    let entries: Vec<TaskEntry> =
        serde_json::from_str(&data).context("parse schedule.json")?;
    Ok(entries)
}

/// Persist the full task list to disk (atomic write).
pub fn save_tasks(path: &Path, entries: &[TaskEntry]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(entries)?;
    // Write to a temp file first, then rename for atomicity.
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json.as_bytes())
        .with_context(|| format!("write schedule temp file {}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .with_context(|| format!("rename schedule file {}", path.display()))?;
    Ok(())
}

/// Add or update a task entry, then persist.
pub fn upsert_task(path: &Path, entry: TaskEntry) -> Result<()> {
    let mut entries = load_tasks(path)?;
    if let Some(existing) = entries.iter_mut().find(|e| e.name == entry.name) {
        *existing = entry;
    } else {
        entries.push(entry);
    }
    save_tasks(path, &entries)
}

/// Remove a task by name, then persist.  Returns true if a task was removed.
pub fn remove_task(path: &Path, name: &str) -> Result<bool> {
    let mut entries = load_tasks(path)?;
    let before = entries.len();
    entries.retain(|e| e.name != name);
    if entries.len() < before {
        save_tasks(path, &entries)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that `TaskEntry` can round-trip a representative JSON fixture
    /// that uses every field.  Keep this in sync with the `schema_parity` test
    /// in `aigent-tools/src/builtins/scheduler.rs` — both must parse the same
    /// JSON without error (schema parity assertion).
    #[test]
    fn schema_parity() {
        let json = r#"{
            "name": "daily-summary",
            "interval_secs": null,
            "cron_expr": "0 0 9 * * *",
            "action_prompt": "Summarise today's todos",
            "cooldown_secs": 300,
            "dnd_window": [22, 7],
            "enabled": true
        }"#;
        let entry: TaskEntry = serde_json::from_str(json).expect("should parse");
        assert_eq!(entry.name, "daily-summary");
        assert_eq!(entry.cron_expr.as_deref(), Some("0 0 9 * * *"));
        assert_eq!(entry.cooldown_secs, 300);
        assert_eq!(entry.dnd_window, Some((22, 7)));
        assert!(entry.enabled);

        // Re-serialise and confirm all expected keys are present.
        let out = serde_json::to_string(&entry).unwrap();
        for key in &["name", "cron_expr", "action_prompt", "cooldown_secs", "dnd_window", "enabled"] {
            assert!(out.contains(key), "missing key {key} in serialised output");
        }
    }

    #[test]
    fn roundtrip_tasks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("schedule.json");

        let entries = vec![
            TaskEntry {
                name: "test-cron".into(),
                interval_secs: None,
                cron_expr: Some("0 */5 * * * *".into()),
                action_prompt: "check the weather".into(),
                cooldown_secs: 60,
                dnd_window: Some((22, 6)),
                enabled: true,
            },
            TaskEntry {
                name: "test-interval".into(),
                interval_secs: Some(300),
                cron_expr: None,
                action_prompt: "review my todos".into(),
                cooldown_secs: 0,
                dnd_window: None,
                enabled: true,
            },
        ];

        save_tasks(&path, &entries).unwrap();
        let loaded = load_tasks(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].name, "test-cron");
        assert_eq!(loaded[1].action_prompt, "review my todos");
    }

    #[test]
    fn upsert_and_remove() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("schedule.json");

        upsert_task(
            &path,
            TaskEntry {
                name: "alpha".into(),
                interval_secs: Some(60),
                cron_expr: None,
                action_prompt: "do alpha".into(),
                cooldown_secs: 0,
                dnd_window: None,
                enabled: true,
            },
        )
        .unwrap();

        upsert_task(
            &path,
            TaskEntry {
                name: "beta".into(),
                interval_secs: None,
                cron_expr: Some("0 0 * * * *".into()),
                action_prompt: "do beta".into(),
                cooldown_secs: 0,
                dnd_window: None,
                enabled: true,
            },
        )
        .unwrap();

        let loaded = load_tasks(&path).unwrap();
        assert_eq!(loaded.len(), 2);

        // Update alpha.
        upsert_task(
            &path,
            TaskEntry {
                name: "alpha".into(),
                interval_secs: Some(120),
                cron_expr: None,
                action_prompt: "do alpha v2".into(),
                cooldown_secs: 10,
                dnd_window: None,
                enabled: true,
            },
        )
        .unwrap();

        let loaded = load_tasks(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].action_prompt, "do alpha v2");

        // Remove beta.
        assert!(remove_task(&path, "beta").unwrap());
        assert!(!remove_task(&path, "beta").unwrap()); // already gone
        assert_eq!(load_tasks(&path).unwrap().len(), 1);
    }

    #[test]
    fn to_scheduled_task_cron() {
        let entry = TaskEntry {
            name: "cron-test".into(),
            interval_secs: None,
            cron_expr: Some("0 */15 * * * *".into()),
            action_prompt: "check things".into(),
            cooldown_secs: 30,
            dnd_window: Some((23, 7)),
            enabled: true,
        };
        let task = entry.to_scheduled_task().unwrap();
        assert_eq!(task.name, "cron-test");
        assert!(matches!(task.schedule, TaskSchedule::Cron(_)));
        assert_eq!(task.cooldown, Duration::from_secs(30));
        assert_eq!(task.dnd_window, Some((23, 7)));
    }

    #[test]
    fn to_scheduled_task_interval() {
        let entry = TaskEntry {
            name: "interval-test".into(),
            interval_secs: Some(600),
            cron_expr: None,
            action_prompt: "do something".into(),
            cooldown_secs: 0,
            dnd_window: None,
            enabled: true,
        };
        let task = entry.to_scheduled_task().unwrap();
        assert!(matches!(task.schedule, TaskSchedule::Interval(d) if d == Duration::from_secs(600)));
    }
}
