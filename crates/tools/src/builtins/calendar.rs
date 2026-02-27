//! Calendar event tool.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use async_trait::async_trait;
use serde_json;

use crate::{Tool, ToolSpec, ToolParam, ToolOutput};

/// Appends an event object to `{data_dir}/calendar.json` (a JSON array).
/// Creates the file if it does not exist.
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
                },
                ToolParam {
                    name: "date".to_string(),
                    description: "Event date (natural language or ISO-8601)".to_string(),
                    required: true,
                },
                ToolParam {
                    name: "time".to_string(),
                    description: "Event time (e.g. '14:00' or '2pm')".to_string(),
                    required: false,
                },
                ToolParam {
                    name: "description".to_string(),
                    description: "Optional description or notes".to_string(),
                    required: false,
                },
            ],
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

        // Load existing events array (or start fresh).
        let mut events: Vec<serde_json::Value> = if calendar_path.exists() {
            let raw = std::fs::read_to_string(&calendar_path)?;
            serde_json::from_str(&raw).unwrap_or_default()
        } else {
            Vec::new()
        };

        let event = serde_json::json!({
            "title": title,
            "date": date,
            "time": args.get("time").cloned().unwrap_or_default(),
            "description": args.get("description").cloned().unwrap_or_default(),
            "added_at": chrono::Utc::now().to_rfc3339(),
        });
        events.push(event);

        let rendered = serde_json::to_string_pretty(&events)?;
        std::fs::write(&calendar_path, rendered)?;

        Ok(ToolOutput {
            success: true,
            output: format!("event '{}' added for {}", title, date),
        })
    }
}

