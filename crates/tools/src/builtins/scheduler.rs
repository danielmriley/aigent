//! Self-scheduling tool — lets the agent create, remove, and list cron jobs.
//!
//! Operates on `.aigent/schedule.json` as a flat JSON array.  The daemon's
//! scheduler picks up changes on its next tick or after a config reload.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use async_trait::async_trait;

use crate::{Tool, ToolOutput, ToolParam, ToolSpec, ToolMetadata, SecurityLevel, ParamType};

/// Path to the schedule persistence file (same convention as the runtime).
fn schedule_path() -> PathBuf {
    PathBuf::from(".aigent").join("schedule.json")
}

/// On-disk representation of a scheduled task (mirrors `schedule_store::TaskEntry`
/// in the runtime crate but kept standalone to avoid circular dependencies).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct TaskEntry {
    name: String,
    #[serde(default)]
    interval_secs: Option<u64>,
    #[serde(default)]
    cron_expr: Option<String>,
    action_prompt: String,
    #[serde(default)]
    cooldown_secs: u64,
    #[serde(default)]
    dnd_window: Option<(u32, u32)>,
    #[serde(default = "default_true")]
    enabled: bool,
}

fn default_true() -> bool { true }

fn load_entries() -> Result<Vec<TaskEntry>> {
    let path = schedule_path();
    if !path.exists() {
        return Ok(vec![]);
    }
    let data = std::fs::read_to_string(&path)?;
    Ok(serde_json::from_str(&data)?)
}

fn save_entries(entries: &[TaskEntry]) -> Result<()> {
    let path = schedule_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(entries)?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json.as_bytes())?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

// ── create_cron_job ──────────────────────────────────────────────────────────

/// Tool: create or update a cron/interval job.
pub struct CreateCronJobTool;

#[async_trait]
impl Tool for CreateCronJobTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "create_cron_job".to_string(),
            description: "Create or update a recurring scheduled task. \
                          Specify either a cron expression (6-field: sec min hour dom month dow) \
                          or an interval in seconds — not both. \
                          The action_prompt is what the agent will execute when the job fires."
                .to_string(),
            params: vec![
                ToolParam {
                    name: "name".to_string(),
                    description: "Unique task name (e.g. 'weather-check', 'daily-summary')".to_string(),
                    required: true,
                    ..Default::default()
                },
                ToolParam {
                    name: "cron_expr".to_string(),
                    description: "6-field cron expression: sec min hour dom month dow \
                                  (e.g. '0 0 9 * * *' = daily at 09:00, '0 */15 * * * *' = every 15 min)"
                        .to_string(),
                    required: false,
                    ..Default::default()
                },
                ToolParam {
                    name: "interval_secs".to_string(),
                    description: "Fixed interval between runs, in seconds (alternative to cron_expr)"
                        .to_string(),
                    required: false,
                    param_type: ParamType::Integer,
                    ..Default::default()
                },
                ToolParam {
                    name: "action_prompt".to_string(),
                    description: "The prompt/instruction the agent should execute when this task fires"
                        .to_string(),
                    required: true,
                    ..Default::default()
                },
                ToolParam {
                    name: "cooldown_secs".to_string(),
                    description: "Minimum seconds between executions (default: 0)".to_string(),
                    required: false,
                    param_type: ParamType::Integer,
                    default: Some("0".into()),
                    ..Default::default()
                },
                ToolParam {
                    name: "dnd_start_hour".to_string(),
                    description: "Do-not-disturb start hour 0-23 (optional)".to_string(),
                    required: false,
                    param_type: ParamType::Integer,
                    ..Default::default()
                },
                ToolParam {
                    name: "dnd_end_hour".to_string(),
                    description: "Do-not-disturb end hour 0-23 (optional)".to_string(),
                    required: false,
                    param_type: ParamType::Integer,
                    ..Default::default()
                },
            ],
            metadata: ToolMetadata {
                security_level: SecurityLevel::Medium,
                read_only: false,
                group: "scheduler".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let name = args.get("name")
            .ok_or_else(|| anyhow::anyhow!("missing required param: name"))?
            .trim()
            .to_string();
        if name.is_empty() {
            anyhow::bail!("name cannot be empty");
        }

        let cron_expr = args.get("cron_expr").map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
        let interval_secs: Option<u64> = args.get("interval_secs")
            .and_then(|s| s.trim().parse().ok());

        if cron_expr.is_none() && interval_secs.is_none() {
            anyhow::bail!("must specify either cron_expr or interval_secs");
        }
        if cron_expr.is_some() && interval_secs.is_some() {
            anyhow::bail!("specify cron_expr OR interval_secs, not both");
        }

        // Validate cron expression by parsing it.
        if let Some(ref expr) = cron_expr {
            expr.parse::<cron::Schedule>().map_err(|e| {
                anyhow::anyhow!("invalid cron expression '{expr}': {e}")
            })?;
        }

        let action_prompt = args.get("action_prompt")
            .ok_or_else(|| anyhow::anyhow!("missing required param: action_prompt"))?
            .trim()
            .to_string();
        if action_prompt.is_empty() {
            anyhow::bail!("action_prompt cannot be empty");
        }

        let cooldown_secs: u64 = args.get("cooldown_secs")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let dnd_window = match (
            args.get("dnd_start_hour").and_then(|s| s.parse::<u32>().ok()),
            args.get("dnd_end_hour").and_then(|s| s.parse::<u32>().ok()),
        ) {
            (Some(start), Some(end)) => Some((start, end)),
            _ => None,
        };

        let mut entries = load_entries().unwrap_or_default();
        let is_update = entries.iter().any(|e| e.name == name);

        let entry = TaskEntry {
            name: name.clone(),
            interval_secs,
            cron_expr: cron_expr.clone(),
            action_prompt: action_prompt.clone(),
            cooldown_secs,
            dnd_window,
            enabled: true,
        };

        if let Some(existing) = entries.iter_mut().find(|e| e.name == name) {
            *existing = entry;
        } else {
            entries.push(entry);
        }

        save_entries(&entries)?;

        let sched_type = if let Some(ref c) = cron_expr {
            format!("cron '{c}'")
        } else {
            format!("every {}s", interval_secs.unwrap())
        };
        let verb = if is_update { "Updated" } else { "Created" };
        let output = format!(
            "{verb} scheduled task '{name}' ({sched_type})\n\
             Action: {action_prompt}\n\
             The scheduler will pick up this change automatically within the next tick."
        );

        Ok(ToolOutput { output, success: true })
    }
}

// ── remove_cron_job ──────────────────────────────────────────────────────────

/// Tool: remove a scheduled task by name.
pub struct RemoveCronJobTool;

#[async_trait]
impl Tool for RemoveCronJobTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "remove_cron_job".to_string(),
            description: "Remove a scheduled task by name.".to_string(),
            params: vec![
                ToolParam {
                    name: "name".to_string(),
                    description: "Name of the task to remove".to_string(),
                    required: true,
                    ..Default::default()
                },
            ],
            metadata: ToolMetadata {
                security_level: SecurityLevel::Medium,
                read_only: false,
                group: "scheduler".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let name = args.get("name")
            .ok_or_else(|| anyhow::anyhow!("missing required param: name"))?
            .trim();
        if name.is_empty() {
            anyhow::bail!("name cannot be empty");
        }

        let mut entries = load_entries().unwrap_or_default();
        let before = entries.len();
        entries.retain(|e| e.name != name);

        if entries.len() < before {
            save_entries(&entries)?;
            Ok(ToolOutput {
                output: format!("Removed scheduled task '{name}'.\n\
                                 The scheduler will pick up this change automatically within the next tick."),
                success: true,
            })
        } else {
            Ok(ToolOutput {
                output: format!("No scheduled task named '{name}' found."),
                success: false,
            })
        }
    }
}

// ── list_cron_jobs ───────────────────────────────────────────────────────────

/// Tool: list all scheduled tasks.
pub struct ListCronJobsTool;

#[async_trait]
impl Tool for ListCronJobsTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "list_cron_jobs".to_string(),
            description: "List all scheduled tasks with their configuration.".to_string(),
            params: vec![],
            metadata: ToolMetadata {
                security_level: SecurityLevel::Low,
                read_only: true,
                group: "scheduler".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, _args: &HashMap<String, String>) -> Result<ToolOutput> {
        let entries = load_entries().unwrap_or_default();
        if entries.is_empty() {
            return Ok(ToolOutput {
                output: "No scheduled tasks configured.".to_string(),
                success: true,
            });
        }

        let mut lines = Vec::new();
        for e in &entries {
            let schedule = if let Some(ref c) = e.cron_expr {
                format!("cron '{c}'")
            } else if let Some(secs) = e.interval_secs {
                format!("every {secs}s")
            } else {
                "invalid".to_string()
            };
            let status = if e.enabled { "enabled" } else { "disabled" };
            let dnd = e.dnd_window.map_or_else(
                || "none".to_string(),
                |(s, e)| format!("{s}:00-{e}:00"),
            );
            lines.push(format!(
                "• {} [{}] — {}\n  prompt: {}\n  cooldown: {}s, DND: {}",
                e.name, status, schedule, e.action_prompt, e.cooldown_secs, dnd
            ));
        }

        Ok(ToolOutput {
            output: format!("Scheduled tasks ({}):\n{}", entries.len(), lines.join("\n")),
            success: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn list_empty() {
        let tool = ListCronJobsTool;
        let out = tool.run(&HashMap::new()).await.unwrap();
        assert!(out.success);
        assert!(out.output.contains("No scheduled tasks"));
    }

    #[tokio::test]
    async fn create_requires_name() {
        let tool = CreateCronJobTool;
        let mut args = HashMap::new();
        args.insert("cron_expr".into(), "0 0 * * * *".into());
        args.insert("action_prompt".into(), "test".into());
        let out = tool.run(&args).await;
        assert!(out.is_err());
    }

    #[tokio::test]
    async fn create_requires_schedule_type() {
        let tool = CreateCronJobTool;
        let mut args = HashMap::new();
        args.insert("name".into(), "test".into());
        args.insert("action_prompt".into(), "test".into());
        let out = tool.run(&args).await;
        assert!(out.is_err());
    }

    #[tokio::test]
    async fn create_rejects_both_schedule_types() {
        let tool = CreateCronJobTool;
        let mut args = HashMap::new();
        args.insert("name".into(), "test".into());
        args.insert("cron_expr".into(), "0 0 * * * *".into());
        args.insert("interval_secs".into(), "300".into());
        args.insert("action_prompt".into(), "test".into());
        let out = tool.run(&args).await;
        assert!(out.is_err());
    }

    #[tokio::test]
    async fn create_validates_cron() {
        let tool = CreateCronJobTool;
        let mut args = HashMap::new();
        args.insert("name".into(), "bad-cron".into());
        args.insert("cron_expr".into(), "not a cron".into());
        args.insert("action_prompt".into(), "test".into());
        let out = tool.run(&args).await;
        assert!(out.is_err());
    }
}
