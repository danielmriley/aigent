//! File system tools: read and write files.

use std::collections::HashMap;
use std::path::{PathBuf, Component};

use anyhow::{Result, bail};
use async_trait::async_trait;

use crate::{Tool, ToolSpec, ToolParam, ToolOutput, ToolMetadata, SecurityLevel};

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

/// Lexically resolve `.` and `..` in a path *without* hitting the filesystem.
///
/// This is essential for write-path validation: `canonicalize()` fails when
/// the file (or its parent directories) don't exist yet, but we still need
/// to verify that the normalized path stays inside the workspace.
fn normalize_path(path: &std::path::Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => { out.pop(); }
            Component::CurDir => {}
            other => out.push(other),
        }
    }
    out
}

/// Verify that `rel_path` (relative to `root`) does not escape the workspace.
///
/// Returns the full normalized path on success.
fn checked_path(root: &std::path::Path, rel_path: &str) -> Result<PathBuf> {
    let full = root.join(rel_path);
    let normalized = normalize_path(&full);
    let root_normalized = normalize_path(root);
    if !normalized.starts_with(&root_normalized) {
        bail!("path escapes workspace boundary: {}", normalized.display());
    }
    Ok(normalized)
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
                    ..Default::default()
                },
                ToolParam {
                    name: "max_bytes".to_string(),
                    description: "Maximum bytes to read (default: 65536)".to_string(),
                    required: false,
                    ..Default::default()
                },
            ],
            metadata: ToolMetadata {
                security_level: SecurityLevel::Low,
                read_only: true,
                group: "filesystem".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let rel_path = args
            .get("path")
            .ok_or_else(|| anyhow::anyhow!("missing required param: path"))?;

        let full = checked_path(&self.workspace_root, rel_path)?;

        let max_bytes: usize = args
            .get("max_bytes")
            .and_then(|v| v.parse().ok())
            .unwrap_or(65536);

        let content = std::fs::read_to_string(&full)?;
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
                    ..Default::default()
                },
                ToolParam {
                    name: "content".to_string(),
                    description: "File content to write".to_string(),
                    required: true,
                    ..Default::default()
                },
            ],
            metadata: ToolMetadata {
                security_level: SecurityLevel::Medium,
                read_only: false,
                group: "filesystem".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let rel_path = args
            .get("path")
            .ok_or_else(|| anyhow::anyhow!("missing required param: path"))?;
        let content = args
            .get("content")
            .ok_or_else(|| anyhow::anyhow!("missing required param: content"))?;

        let full = checked_path(&self.workspace_root, rel_path)?;

        // Ensure parent directories exist.
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::write(&full, content)?;
        Ok(ToolOutput {
            success: true,
            output: format!("wrote {} bytes to {}", content.len(), rel_path),
        })
    }
}

