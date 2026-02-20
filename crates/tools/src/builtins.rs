use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Result, bail};
use async_trait::async_trait;

use crate::{Tool, ToolOutput, ToolParam, ToolSpec};

// ── read_file ────────────────────────────────────────────────────────────────

pub struct ReadFileTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for ReadFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read_file".to_string(),
            description: "Read the contents of a file within the workspace.".to_string(),
            params: vec![
                ToolParam {
                    name: "path".to_string(),
                    description: "Relative path from workspace root".to_string(),
                    required: true,
                },
                ToolParam {
                    name: "max_bytes".to_string(),
                    description: "Maximum bytes to read (default: 65536)".to_string(),
                    required: false,
                },
            ],
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let rel_path = args
            .get("path")
            .ok_or_else(|| anyhow::anyhow!("missing required param: path"))?;

        let full = self.workspace_root.join(rel_path);
        let canonical = full
            .canonicalize()
            .map_err(|e| anyhow::anyhow!("cannot resolve path '{}': {}", rel_path, e))?;

        let root_canonical = self.workspace_root.canonicalize()?;
        if !canonical.starts_with(&root_canonical) {
            bail!(
                "path escapes workspace boundary: {}",
                canonical.display()
            );
        }

        let max_bytes: usize = args
            .get("max_bytes")
            .and_then(|v| v.parse().ok())
            .unwrap_or(65536);

        let content = std::fs::read_to_string(&canonical)?;
        let truncated = if content.len() > max_bytes {
            format!("{}…[truncated at {} bytes]", &content[..max_bytes], max_bytes)
        } else {
            content
        };

        Ok(ToolOutput {
            success: true,
            output: truncated,
        })
    }
}

// ── write_file ───────────────────────────────────────────────────────────────

pub struct WriteFileTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for WriteFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "write_file".to_string(),
            description: "Write content to a file within the workspace (creates or overwrites)."
                .to_string(),
            params: vec![
                ToolParam {
                    name: "path".to_string(),
                    description: "Relative path from workspace root".to_string(),
                    required: true,
                },
                ToolParam {
                    name: "content".to_string(),
                    description: "File content to write".to_string(),
                    required: true,
                },
            ],
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let rel_path = args
            .get("path")
            .ok_or_else(|| anyhow::anyhow!("missing required param: path"))?;
        let content = args
            .get("content")
            .ok_or_else(|| anyhow::anyhow!("missing required param: content"))?;

        let full = self.workspace_root.join(rel_path);

        // Prevent escaping workspace even before file exists (can't canonicalize yet)
        let root_canonical = self.workspace_root.canonicalize()?;
        if let Ok(canonical) = full.canonicalize() {
            if !canonical.starts_with(&root_canonical) {
                bail!(
                    "path escapes workspace boundary: {}",
                    canonical.display()
                );
            }
        } else {
            // File doesn't exist yet; check parent
            let parent = full
                .parent()
                .ok_or_else(|| anyhow::anyhow!("invalid path"))?;
            std::fs::create_dir_all(parent)?;
            let parent_canonical = parent.canonicalize()?;
            if !parent_canonical.starts_with(&root_canonical) {
                bail!(
                    "parent escapes workspace boundary: {}",
                    parent_canonical.display()
                );
            }
        }

        std::fs::write(&full, content)?;
        Ok(ToolOutput {
            success: true,
            output: format!("wrote {} bytes to {}", content.len(), rel_path),
        })
    }
}

// ── run_shell ────────────────────────────────────────────────────────────────

pub struct RunShellTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for RunShellTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "run_shell".to_string(),
            description: "Execute a shell command within the workspace directory.".to_string(),
            params: vec![
                ToolParam {
                    name: "command".to_string(),
                    description: "Shell command to execute".to_string(),
                    required: true,
                },
                ToolParam {
                    name: "timeout_secs".to_string(),
                    description: "Max execution time in seconds (default: 30)".to_string(),
                    required: false,
                },
            ],
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
