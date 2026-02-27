//! File system tools: read and write files.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Result, bail};
use async_trait::async_trait;

use crate::{Tool, ToolSpec, ToolParam, ToolOutput};

/// Find the largest byte offset ≤ `max` that falls on a UTF-8 character
/// boundary.  Safe to use as `&s[..truncate_byte_boundary(s, max)]`.
pub(super) fn truncate_byte_boundary(s: &str, max: usize) -> usize {
    if max >= s.len() {
        return s.len();
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    end
}

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
            let end = truncate_byte_boundary(&content, max_bytes);
            format!("{}…[truncated at {} bytes]", &content[..end], max_bytes)
        } else {
            content
        };

        Ok(ToolOutput {
            success: true,
            output: truncated,
        })
    }
}

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

