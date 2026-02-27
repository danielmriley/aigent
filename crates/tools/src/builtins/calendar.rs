//! Calendar event tool.

use std::collections::HashMap;
use std::io::{Read, Seek, Write};
use std::path::PathBuf;

use anyhow::Result;
use async_trait::async_trait;
use fs2::FileExt;
use serde_json;

use crate::{Tool, ToolSpec, ToolParam, ToolOutput, ToolMetadata, SecurityLevel};

/// Appends an event object to `{data_dir}/calendar.json` (a JSON array).
/// Creates the file if it does not exist.  Uses advisory file locking to
/// prevent data loss from concurrent writes.
pub struct CalendarAddEventTool {
    pub data_dir: PathBuf,
}

#[async_trait]
impl Tool for CalendarAddEventTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "calendar_add_event".to_string(),
            description: "Add an event to the agent's local calendar store.".to_string(),
            params: vec![
                ToolParam {
                    name: "title".to_string(),
                    description: "Event title".to_string(),
                    required: true,
                    ..Default::default()
                },
                ToolParam {
                    name: "date".to_string(),
                    description: "Event date (natural language or ISO-8601)".to_string(),
                    required: true,
                    ..Default::default()
                },
                ToolParam {
                    name: "time".to_string(),
                    description: "Event time (e.g. '14:00' or '2pm')".to_string(),
                    required: false,
                    ..Default::default()
                },
                ToolParam {
                    name: "description".to_string(),
                    description: "Optional description or notes".to_string(),
                    required: false,
                    ..Default::default()
                },
            ],
            metadata: ToolMetadata {
                security_level: SecurityLevel::Low,
                read_only: false,
                group: "calendar".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let title = args
            .get("title")
            .ok_or_else(|| anyhow::anyhow!("missing required param: title"))?;
        let date = args
            .get("date")
            .ok_or_else(|| anyhow::anyhow!("missing required param: date"))?;

        std::fs::create_dir_all(&self.data_dir)?;
        let calendar_path = self.data_dir.join("calendar.json");

        // Open (or create) and lock the file to prevent concurrent corruption.
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&calendar_path)?;
        file.lock_exclusive()?;

        let mut raw = String::new();
        file.read_to_string(&mut raw)?;
        let mut events: Vec<serde_json::Value> =
            if raw.trim().is_empty() { Vec::new() }
            else { serde_json::from_str(&raw).unwrap_or_default() };

        let event = serde_json::json!({
            "title": title,
            "date": date,
            "time": args.get("time").cloned().unwrap_or_default(),
            "description": args.get("description").cloned().unwrap_or_default(),
            "added_at": chrono::Utc::now().to_rfc3339(),
        });
        events.push(event);

        let rendered = serde_json::to_string_pretty(&events)?;
        file.set_len(0)?;
        file.seek(std::io::SeekFrom::Start(0))?;
        file.write_all(rendered.as_bytes())?;
        file.unlock()?;

        Ok(ToolOutput {
            success: true,
            output: format!("event '{}' added for {}", title, date),
        })
    }
}

