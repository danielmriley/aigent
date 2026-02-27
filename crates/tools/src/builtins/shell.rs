//! Shell execution tool.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use async_trait::async_trait;

use crate::{Tool, ToolSpec, ToolParam, ToolOutput, ToolMetadata, SecurityLevel};

pub struct RunShellTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for RunShellTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "run_shell".to_string(),
            description: "Execute a shell command within the workspace directory. \
                Each invocation runs in a fresh shell — `cd` does not persist \
                between calls. Chain commands with `&&` if you need them to \
                share working directory or state."
                .to_string(),
            params: vec![
                ToolParam {
                    name: "command".to_string(),
                    description: "Shell command to execute".to_string(),
                    required: true,
                    ..Default::default()
                },
                ToolParam {
                    name: "timeout_secs".to_string(),
                    description: "Max execution time in seconds (default: 30)".to_string(),
                    required: false,
                    ..Default::default()
                },
            ],
            metadata: ToolMetadata {
                security_level: SecurityLevel::High,
                read_only: false,
                group: "shell".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let command = args
            .get("command")
            .ok_or_else(|| anyhow::anyhow!("missing required param: command"))?;
        let timeout_secs: u64 = args
            .get("timeout_secs")
            .and_then(|v| v.parse().ok())
            .unwrap_or(30);

        let output = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            tokio::process::Command::new("sh")
                .arg("-c")
                .arg(command)
                .current_dir(&self.workspace_root)
                .output(),
        )
        .await
        .map_err(|_| anyhow::anyhow!("command timed out after {}s", timeout_secs))??;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = if stderr.is_empty() {
            stdout.to_string()
        } else {
            format!("{stdout}\n[stderr] {stderr}")
        };

        // Truncate output to prevent context explosion
        let max_output = 32768;
        let result = if combined.len() > max_output {
            format!(
                "{}…[truncated at {} bytes]",
                &combined[..max_output],
                max_output
            )
        } else {
            combined
        };

        Ok(ToolOutput {
            success: output.status.success(),
            output: result,
        })
    }
}

