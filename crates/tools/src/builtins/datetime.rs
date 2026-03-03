//! `get_current_datetime` — returns the current local date/time.

use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;

use crate::{Tool, ToolSpec, ToolOutput, ToolMetadata, SecurityLevel};

/// Zero-cost tool that returns the current local date and time.
pub struct GetCurrentDatetimeTool;

#[async_trait]
impl Tool for GetCurrentDatetimeTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "get_current_datetime".to_string(),
            description: "Return the current local date, time and timezone.".to_string(),
            params: vec![],
            metadata: ToolMetadata {
                security_level: SecurityLevel::Low,
                read_only: true,
                group: "utility".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, _args: &HashMap<String, String>) -> Result<ToolOutput> {
        let now = chrono::Local::now();
        Ok(ToolOutput {
            success: true,
            output: now.format("%Y-%m-%d %H:%M:%S %Z").to_string(),
        })
    }
}
