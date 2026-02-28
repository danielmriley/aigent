//! Native coreutils – pure-Rust replacements for common shell commands.
//!
//! These tools eliminate fragile `sh -c` round-trips for the most frequent
//! filesystem and text processing operations.  Each tool emits compact,
//! LLM-friendly output and supports an optional `--aigent-jsonl` flag for
//! structured machine-readable output.

use std::collections::HashMap;
use std::path::{Path, PathBuf, Component};

use anyhow::{Result, bail};
use async_trait::async_trait;

use crate::{Tool, ToolSpec, ToolParam, ToolOutput, ToolMetadata, SecurityLevel, ParamType};

// ── Shared helpers ───────────────────────────────────────────────────────────

/// Lexically resolve `.` and `..` without touching the filesystem.
fn normalize_path(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::ParentDir => { out.pop(); }
            Component::CurDir    => {}
            other                => out.push(other),
        }
    }
    out
}

/// Validate that `rel_path` stays inside `root`.
fn checked_path(root: &Path, rel_path: &str) -> Result<PathBuf> {
    let full = root.join(rel_path);
    let norm = normalize_path(&full);
    let root_norm = normalize_path(root);
    if !norm.starts_with(&root_norm) {
        bail!("path escapes workspace: {}", norm.display());
    }
    Ok(norm)
}

/// Resolve a path: absolute paths pass through, relative paths are joined
/// to workspace_root.
fn resolve_path(root: &Path, path: &str) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() {
        normalize_path(p)
    } else {
        normalize_path(&root.join(path))
    }
}

/// Format a file size in human-readable units.
fn human_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "K", "M", "G", "T"];
    let mut size = bytes as f64;
    for unit in UNITS {
        if size < 1024.0 || *unit == "T" {
            return if size.fract() < 0.05 {
                format!("{:.0}{}", size, unit)
            } else {
                format!("{:.1}{}", size, unit)
            };
        }
        size /= 1024.0;
    }
    format!("{}B", bytes)
}

/// Format a `SystemTime` as a short date string.
fn short_date(st: std::time::SystemTime) -> String {
    let dt: chrono::DateTime<chrono::Local> = st.into();
    dt.format("%Y-%m-%d %H:%M").to_string()
}

// ── list_dir ─────────────────────────────────────────────────────────────────

pub struct ListDirTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for ListDirTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "list_dir".to_string(),
            description: "List directory contents with optional detail.  \
                Replaces `ls`.  Accepts relative or absolute paths."
                .to_string(),
            params: vec![
                ToolParam::required("path",
                    "Directory path (relative to workspace or absolute)"),
                ToolParam {
                    name: "long".to_string(),
                    description: "Show size, date; like `ls -l` (default: false)".to_string(),
                    required: false,
                    param_type: ParamType::Boolean,
                    default: Some("false".into()),
                    ..Default::default()
                },
                ToolParam {
                    name: "all".to_string(),
                    description: "Include hidden (dot) files (default: false)".to_string(),
                    required: false,
                    param_type: ParamType::Boolean,
                    default: Some("false".into()),
                    ..Default::default()
                },
                ToolParam {
                    name: "json".to_string(),
                    description: "Output as JSON lines (default: false)".to_string(),
                    required: false,
                    param_type: ParamType::Boolean,
                    default: Some("false".into()),
                    ..Default::default()
                },
            ],
            metadata: ToolMetadata {
                security_level: SecurityLevel::Low,
                read_only: true,
                group: "coreutils".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let path = args.get("path").ok_or_else(|| anyhow::anyhow!("missing: path"))?;
        let long = args.get("long").map(|v| v == "true").unwrap_or(false);
        let all  = args.get("all").map(|v| v == "true").unwrap_or(false);
        let json = args.get("json").map(|v| v == "true").unwrap_or(false);

        let dir = resolve_path(&self.workspace_root, path);
        if !dir.is_dir() {
            bail!("not a directory: {}", dir.display());
        }

        let mut entries: Vec<(String, bool, u64, std::time::SystemTime)> = Vec::new();
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();
            if !all && name.starts_with('.') {
                continue;
            }
            let meta = entry.metadata()?;
            let is_dir = meta.is_dir();
            let size = meta.len();
            let modified = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
            entries.push((name, is_dir, size, modified));
        }
        entries.sort_by(|a, b| {
            // Directories first, then alphabetical.
            b.1.cmp(&a.1).then_with(|| a.0.to_lowercase().cmp(&b.0.to_lowercase()))
        });

        if json {
            let lines: Vec<String> = entries.iter().map(|(name, is_dir, size, modified)| {
                serde_json::json!({
                    "name": name,
                    "type": if *is_dir { "dir" } else { "file" },
                    "size": size,
                    "modified": short_date(*modified),
                }).to_string()
            }).collect();
            return Ok(ToolOutput { success: true, output: lines.join("\n") });
        }

        let mut lines: Vec<String> = Vec::new();
        for (name, is_dir, size, modified) in &entries {
            let suffix = if *is_dir { "/" } else { "" };
            if long {
                let kind = if *is_dir { "d" } else { "-" };
                lines.push(format!("{} {:>8}  {}  {}{}", kind, human_size(*size), short_date(*modified), name, suffix));
            } else {
                lines.push(format!("{}{}", name, suffix));
            }
        }

        Ok(ToolOutput {
            success: true,
            output: if lines.is_empty() { "(empty directory)".into() } else { lines.join("\n") },
        })
    }
}

// ── mkdir ────────────────────────────────────────────────────────────────────

pub struct MkdirTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for MkdirTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "mkdir".to_string(),
            description: "Create directories (including parents).".to_string(),
            params: vec![
                ToolParam::required("path", "Directory path to create (relative to workspace)"),
            ],
            metadata: ToolMetadata {
                security_level: SecurityLevel::Medium,
                read_only: false,
                group: "coreutils".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let rel = args.get("path").ok_or_else(|| anyhow::anyhow!("missing: path"))?;
        let full = checked_path(&self.workspace_root, rel)?;
        std::fs::create_dir_all(&full)?;
        Ok(ToolOutput { success: true, output: format!("created {}", rel) })
    }
}

// ── touch ────────────────────────────────────────────────────────────────────

pub struct TouchTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for TouchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "touch".to_string(),
            description: "Create an empty file or update its modification time.".to_string(),
            params: vec![
                ToolParam::required("path", "File path (relative to workspace)"),
            ],
            metadata: ToolMetadata {
                security_level: SecurityLevel::Medium,
                read_only: false,
                group: "coreutils".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let rel = args.get("path").ok_or_else(|| anyhow::anyhow!("missing: path"))?;
        let full = checked_path(&self.workspace_root, rel)?;
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if full.exists() {
            // Update mtime by opening and closing.
            std::fs::OpenOptions::new().append(true).open(&full)?;
            Ok(ToolOutput { success: true, output: format!("touched {}", rel) })
        } else {
            std::fs::File::create(&full)?;
            Ok(ToolOutput { success: true, output: format!("created {}", rel) })
        }
    }
}

// ── rm ───────────────────────────────────────────────────────────────────────

pub struct RmTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for RmTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "rm".to_string(),
            description: "Remove a file or directory (recursively).".to_string(),
            params: vec![
                ToolParam::required("path", "Path to remove (relative to workspace)"),
                ToolParam {
                    name: "recursive".to_string(),
                    description: "Remove directories recursively (default: false)".to_string(),
                    required: false,
                    param_type: ParamType::Boolean,
                    default: Some("false".into()),
                    ..Default::default()
                },
            ],
            metadata: ToolMetadata {
                security_level: SecurityLevel::High,
                read_only: false,
                group: "coreutils".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let rel = args.get("path").ok_or_else(|| anyhow::anyhow!("missing: path"))?;
        let recursive = args.get("recursive").map(|v| v == "true").unwrap_or(false);
        let full = checked_path(&self.workspace_root, rel)?;

        if !full.exists() {
            bail!("path does not exist: {}", rel);
        }

        if full.is_dir() {
            if recursive {
                std::fs::remove_dir_all(&full)?;
                Ok(ToolOutput { success: true, output: format!("removed directory {} (recursive)", rel) })
            } else {
                std::fs::remove_dir(&full)?;
                Ok(ToolOutput { success: true, output: format!("removed empty directory {}", rel) })
            }
        } else {
            std::fs::remove_file(&full)?;
            Ok(ToolOutput { success: true, output: format!("removed {}", rel) })
        }
    }
}

// ── cp ───────────────────────────────────────────────────────────────────────

pub struct CpTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for CpTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "cp".to_string(),
            description: "Copy a file or directory within the workspace.".to_string(),
            params: vec![
                ToolParam::required("src", "Source path (relative to workspace)"),
                ToolParam::required("dst", "Destination path (relative to workspace)"),
                ToolParam {
                    name: "recursive".to_string(),
                    description: "Copy directories recursively (default: false)".to_string(),
                    required: false,
                    param_type: ParamType::Boolean,
                    default: Some("false".into()),
                    ..Default::default()
                },
            ],
            metadata: ToolMetadata {
                security_level: SecurityLevel::Medium,
                read_only: false,
                group: "coreutils".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let src_rel = args.get("src").ok_or_else(|| anyhow::anyhow!("missing: src"))?;
        let dst_rel = args.get("dst").ok_or_else(|| anyhow::anyhow!("missing: dst"))?;
        let recursive = args.get("recursive").map(|v| v == "true").unwrap_or(false);

        let src = checked_path(&self.workspace_root, src_rel)?;
        let dst = checked_path(&self.workspace_root, dst_rel)?;

        if src.is_dir() {
            if !recursive {
                bail!("source is a directory; set recursive=true");
            }
            copy_dir_recursive(&src, &dst)?;
            Ok(ToolOutput { success: true, output: format!("copied {} → {} (recursive)", src_rel, dst_rel) })
        } else {
            if let Some(parent) = dst.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::copy(&src, &dst)?;
            Ok(ToolOutput { success: true, output: format!("copied {} → {}", src_rel, dst_rel) })
        }
    }
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let target = dst.join(entry.file_name());
        if entry.metadata()?.is_dir() {
            copy_dir_recursive(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), &target)?;
        }
    }
    Ok(())
}

// ── mv ───────────────────────────────────────────────────────────────────────

pub struct MvTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for MvTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "mv".to_string(),
            description: "Move or rename a file/directory within the workspace.".to_string(),
            params: vec![
                ToolParam::required("src", "Source path (relative to workspace)"),
                ToolParam::required("dst", "Destination path (relative to workspace)"),
            ],
            metadata: ToolMetadata {
                security_level: SecurityLevel::Medium,
                read_only: false,
                group: "coreutils".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let src_rel = args.get("src").ok_or_else(|| anyhow::anyhow!("missing: src"))?;
        let dst_rel = args.get("dst").ok_or_else(|| anyhow::anyhow!("missing: dst"))?;

        let src = checked_path(&self.workspace_root, src_rel)?;
        let dst = checked_path(&self.workspace_root, dst_rel)?;

        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::rename(&src, &dst)?;
        Ok(ToolOutput { success: true, output: format!("moved {} → {}", src_rel, dst_rel) })
    }
}

// ── find ─────────────────────────────────────────────────────────────────────

pub struct FindTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for FindTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "find".to_string(),
            description: "Find files/directories matching a glob pattern.  \
                Respects .gitignore by default.  Returns paths relative to the \
                search root."
                .to_string(),
            params: vec![
                ToolParam::required("pattern",
                    "Glob pattern to match (e.g. '*.rs', 'src/**/*.ts')"),
                ToolParam::optional("path",
                    "Start directory (default: workspace root)"),
                ToolParam {
                    name: "type".to_string(),
                    description: "Filter: 'f' for files, 'd' for directories, \
                        omit for both"
                        .to_string(),
                    required: false,
                    enum_values: vec!["f".into(), "d".into()],
                    ..Default::default()
                },
                ToolParam {
                    name: "max_results".to_string(),
                    description: "Cap on results returned (default: 200)".to_string(),
                    required: false,
                    param_type: ParamType::Integer,
                    default: Some("200".into()),
                    ..Default::default()
                },
            ],
            metadata: ToolMetadata {
                security_level: SecurityLevel::Low,
                read_only: true,
                group: "coreutils".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let pattern = args.get("pattern").ok_or_else(|| anyhow::anyhow!("missing: pattern"))?;
        let start = args.get("path")
            .map(|p| resolve_path(&self.workspace_root, p))
            .unwrap_or_else(|| self.workspace_root.clone());
        let type_filter = args.get("type").map(|s| s.as_str());
        let max: usize = args.get("max_results")
            .and_then(|v| v.parse().ok())
            .unwrap_or(200);

        let glob = globset::GlobBuilder::new(pattern)
            .literal_separator(false)
            .build()
            .map_err(|e| anyhow::anyhow!("bad glob: {}", e))?
            .compile_matcher();

        let walker = ignore::WalkBuilder::new(&start)
            .hidden(false)
            .git_ignore(true)
            .build();

        let mut results: Vec<String> = Vec::new();
        for entry in walker {
            let entry = match entry {
                Ok(e)  => e,
                Err(_) => continue,
            };
            let path = entry.path();

            // Apply type filter.
            match type_filter {
                Some("f") if !path.is_file() => continue,
                Some("d") if !path.is_dir()  => continue,
                _ => {}
            }

            let rel = path.strip_prefix(&start).unwrap_or(path);
            let rel_str = rel.to_string_lossy();
            if rel_str.is_empty() {
                continue; // skip the root itself
            }

            // Match glob against the relative path OR just the filename.
            let file_name = rel.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            if glob.is_match(&*rel_str) || glob.is_match(&file_name) {
                let suffix = if path.is_dir() { "/" } else { "" };
                results.push(format!("{}{}", rel_str, suffix));
            }

            if results.len() >= max {
                break;
            }
        }

        let output = if results.is_empty() {
            format!("no matches for '{}'", pattern)
        } else {
            let count = results.len();
            let truncated = if count >= max { format!("\n…({} results, capped at {})", count, max) } else { String::new() };
            format!("{}{}", results.join("\n"), truncated)
        };

        Ok(ToolOutput { success: true, output })
    }
}

// ── grep ─────────────────────────────────────────────────────────────────────

pub struct GrepTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for GrepTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "grep".to_string(),
            description: "Search file contents for a pattern (regex or literal).  \
                Respects .gitignore.  Returns matching lines with file:line prefix."
                .to_string(),
            params: vec![
                ToolParam::required("pattern", "Search pattern (regex)"),
                ToolParam::optional("path",
                    "File or directory to search (default: workspace root)"),
                ToolParam {
                    name: "case_insensitive".to_string(),
                    description: "Case-insensitive match (default: true)".to_string(),
                    required: false,
                    param_type: ParamType::Boolean,
                    default: Some("true".into()),
                    ..Default::default()
                },
                ToolParam {
                    name: "max_results".to_string(),
                    description: "Max matching lines (default: 100)".to_string(),
                    required: false,
                    param_type: ParamType::Integer,
                    default: Some("100".into()),
                    ..Default::default()
                },
                ToolParam {
                    name: "context".to_string(),
                    description: "Lines of context around each match (default: 0)".to_string(),
                    required: false,
                    param_type: ParamType::Integer,
                    default: Some("0".into()),
                    ..Default::default()
                },
            ],
            metadata: ToolMetadata {
                security_level: SecurityLevel::Low,
                read_only: true,
                group: "coreutils".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let pattern = args.get("pattern").ok_or_else(|| anyhow::anyhow!("missing: pattern"))?;
        let start = args.get("path")
            .map(|p| resolve_path(&self.workspace_root, p))
            .unwrap_or_else(|| self.workspace_root.clone());
        let case_insensitive = args.get("case_insensitive")
            .map(|v| v != "false")
            .unwrap_or(true);
        let max: usize = args.get("max_results")
            .and_then(|v| v.parse().ok())
            .unwrap_or(100);
        let ctx: usize = args.get("context")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);

        let regex_pat = if case_insensitive {
            format!("(?i){}", pattern)
        } else {
            pattern.to_string()
        };
        let re = regex::Regex::new(&regex_pat)
            .map_err(|e| anyhow::anyhow!("bad regex: {}", e))?;

        let mut results: Vec<String> = Vec::new();

        // Single file mode.
        if start.is_file() {
            grep_file(&start, &start, &re, ctx, max, &mut results)?;
        } else {
            let walker = ignore::WalkBuilder::new(&start)
                .hidden(false)
                .git_ignore(true)
                .build();

            for entry in walker {
                let entry = match entry {
                    Ok(e) => e,
                    Err(_) => continue,
                };
                if !entry.path().is_file() {
                    continue;
                }
                grep_file(entry.path(), &start, &re, ctx, max.saturating_sub(results.len()), &mut results)?;
                if results.len() >= max {
                    break;
                }
            }
        }

        let output = if results.is_empty() {
            format!("no matches for '{}'", pattern)
        } else {
            let count = results.len();
            let truncated = if count >= max {
                format!("\n…({} matches, capped at {})", count, max)
            } else {
                String::new()
            };
            format!("{}{}", results.join("\n"), truncated)
        };

        Ok(ToolOutput { success: true, output })
    }
}

/// Grep a single file, appending matches to `results`.
fn grep_file(
    path: &Path,
    root: &Path,
    re: &regex::Regex,
    ctx: usize,
    remaining: usize,
    results: &mut Vec<String>,
) -> Result<()> {
    // Skip binary files.
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Ok(()),
    };
    let rel = path.strip_prefix(root).unwrap_or(path);
    let lines: Vec<&str> = content.lines().collect();

    for (i, line) in lines.iter().enumerate() {
        if re.is_match(line) {
            // Context before.
            let start = i.saturating_sub(ctx);
            for ci in start..i {
                results.push(format!("{}:{}- {}", rel.display(), ci + 1, lines[ci]));
            }
            results.push(format!("{}:{}  {}", rel.display(), i + 1, line));
            // Context after.
            let end = (i + 1 + ctx).min(lines.len());
            for ci in (i + 1)..end {
                results.push(format!("{}:{}- {}", rel.display(), ci + 1, lines[ci]));
            }
            if results.len() >= remaining {
                break;
            }
        }
    }
    Ok(())
}

// ── head ─────────────────────────────────────────────────────────────────────

pub struct HeadTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for HeadTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "head".to_string(),
            description: "Show the first N lines of a file.".to_string(),
            params: vec![
                ToolParam::required("path", "File path (relative or absolute)"),
                ToolParam {
                    name: "lines".to_string(),
                    description: "Number of lines (default: 20)".to_string(),
                    required: false,
                    param_type: ParamType::Integer,
                    default: Some("20".into()),
                    ..Default::default()
                },
            ],
            metadata: ToolMetadata {
                security_level: SecurityLevel::Low,
                read_only: true,
                group: "coreutils".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let path = args.get("path").ok_or_else(|| anyhow::anyhow!("missing: path"))?;
        let n: usize = args.get("lines").and_then(|v| v.parse().ok()).unwrap_or(20);
        let full = resolve_path(&self.workspace_root, path);
        let content = std::fs::read_to_string(&full)?;
        let result: String = content.lines().take(n).collect::<Vec<_>>().join("\n");
        let total = content.lines().count();
        let suffix = if total > n { format!("\n…[{} of {} lines]", n, total) } else { String::new() };
        Ok(ToolOutput { success: true, output: format!("{}{}", result, suffix) })
    }
}

// ── tail ─────────────────────────────────────────────────────────────────────

pub struct TailTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for TailTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "tail".to_string(),
            description: "Show the last N lines of a file.".to_string(),
            params: vec![
                ToolParam::required("path", "File path (relative or absolute)"),
                ToolParam {
                    name: "lines".to_string(),
                    description: "Number of lines (default: 20)".to_string(),
                    required: false,
                    param_type: ParamType::Integer,
                    default: Some("20".into()),
                    ..Default::default()
                },
            ],
            metadata: ToolMetadata {
                security_level: SecurityLevel::Low,
                read_only: true,
                group: "coreutils".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let path = args.get("path").ok_or_else(|| anyhow::anyhow!("missing: path"))?;
        let n: usize = args.get("lines").and_then(|v| v.parse().ok()).unwrap_or(20);
        let full = resolve_path(&self.workspace_root, path);
        let content = std::fs::read_to_string(&full)?;
        let all_lines: Vec<&str> = content.lines().collect();
        let total = all_lines.len();
        let start = total.saturating_sub(n);
        let result = all_lines[start..].join("\n");
        let suffix = if total > n { format!("\n…[last {} of {} lines]", n, total) } else { String::new() };
        Ok(ToolOutput { success: true, output: format!("{}{}", result, suffix) })
    }
}

// ── wc ───────────────────────────────────────────────────────────────────────

pub struct WcTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for WcTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "wc".to_string(),
            description: "Count lines, words, and bytes in a file.".to_string(),
            params: vec![
                ToolParam::required("path", "File path (relative or absolute)"),
            ],
            metadata: ToolMetadata {
                security_level: SecurityLevel::Low,
                read_only: true,
                group: "coreutils".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let path = args.get("path").ok_or_else(|| anyhow::anyhow!("missing: path"))?;
        let full = resolve_path(&self.workspace_root, path);
        let content = std::fs::read_to_string(&full)?;
        let lines = content.lines().count();
        let words = content.split_whitespace().count();
        let bytes = content.len();
        let chars = content.chars().count();
        Ok(ToolOutput {
            success: true,
            output: format!("{} lines  {} words  {} chars  {} bytes  {}", lines, words, chars, bytes, path),
        })
    }
}

// ── tree ─────────────────────────────────────────────────────────────────────

pub struct TreeTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for TreeTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "tree".to_string(),
            description: "Display directory tree.  Respects .gitignore.".to_string(),
            params: vec![
                ToolParam::optional("path", "Root directory (default: workspace root)"),
                ToolParam {
                    name: "max_depth".to_string(),
                    description: "Maximum depth to descend (default: 4)".to_string(),
                    required: false,
                    param_type: ParamType::Integer,
                    default: Some("4".into()),
                    ..Default::default()
                },
                ToolParam {
                    name: "max_entries".to_string(),
                    description: "Stop after this many entries (default: 300)".to_string(),
                    required: false,
                    param_type: ParamType::Integer,
                    default: Some("300".into()),
                    ..Default::default()
                },
            ],
            metadata: ToolMetadata {
                security_level: SecurityLevel::Low,
                read_only: true,
                group: "coreutils".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let start = args.get("path")
            .map(|p| resolve_path(&self.workspace_root, p))
            .unwrap_or_else(|| self.workspace_root.clone());
        let max_depth: usize = args.get("max_depth")
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        let max_entries: usize = args.get("max_entries")
            .and_then(|v| v.parse().ok())
            .unwrap_or(300);

        let mut lines: Vec<String> = Vec::new();
        let root_name = start.file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| ".".into());
        lines.push(format!("{}/", root_name));

        tree_recurse(&start, "", max_depth, 0, max_entries, &mut lines)?;

        let count = lines.len() - 1; // exclude root line
        let truncated = if count >= max_entries {
            format!("\n…[{} entries shown, capped at {}]", count, max_entries)
        } else {
            format!("\n{} entries", count)
        };
        Ok(ToolOutput { success: true, output: format!("{}{}", lines.join("\n"), truncated) })
    }
}

fn tree_recurse(
    dir: &Path,
    prefix: &str,
    max_depth: usize,
    depth: usize,
    max_entries: usize,
    lines: &mut Vec<String>,
) -> Result<()> {
    if depth >= max_depth || lines.len() >= max_entries {
        return Ok(());
    }

    let mut entries: Vec<(String, bool)> = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        let is_dir = entry.metadata().map(|m| m.is_dir()).unwrap_or(false);
        entries.push((name, is_dir));
    }
    entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.to_lowercase().cmp(&b.0.to_lowercase())));

    let count = entries.len();
    for (i, (name, is_dir)) in entries.into_iter().enumerate() {
        if lines.len() >= max_entries {
            break;
        }
        let is_last = i + 1 == count;
        let connector = if is_last { "└── " } else { "├── " };
        let suffix = if is_dir { "/" } else { "" };
        lines.push(format!("{}{}{}{}", prefix, connector, name, suffix));

        if is_dir {
            let child_prefix = format!("{}{}", prefix, if is_last { "    " } else { "│   " });
            tree_recurse(&dir.join(&name), &child_prefix, max_depth, depth + 1, max_entries, lines)?;
        }
    }
    Ok(())
}

// ── workspace_status ─────────────────────────────────────────────────────────

pub struct WorkspaceStatusTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for WorkspaceStatusTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "workspace_status".to_string(),
            description: "Get a summary of the workspace: git status, branch, \
                recent commits, and disk usage.  A single high-signal snapshot \
                instead of running multiple shell commands."
                .to_string(),
            params: vec![],
            metadata: ToolMetadata {
                security_level: SecurityLevel::Low,
                read_only: true,
                group: "coreutils".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, _args: &HashMap<String, String>) -> Result<ToolOutput> {
        let root = &self.workspace_root;
        let mut sections: Vec<String> = Vec::new();

        // Git branch & status.
        if root.join(".git").exists() {
            if let Ok(out) = tokio::process::Command::new("git")
                .args(["branch", "--show-current"])
                .current_dir(root)
                .output()
                .await
            {
                let branch = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !branch.is_empty() {
                    sections.push(format!("branch: {}", branch));
                }
            }

            if let Ok(out) = tokio::process::Command::new("git")
                .args(["status", "--porcelain=v1"])
                .current_dir(root)
                .output()
                .await
            {
                let status = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if status.is_empty() {
                    sections.push("working tree: clean".into());
                } else {
                    let changed = status.lines().count();
                    sections.push(format!("working tree: {} changed file(s)", changed));
                    // Show first 15 lines.
                    let preview: Vec<&str> = status.lines().take(15).collect();
                    sections.push(preview.join("\n"));
                }
            }

            if let Ok(out) = tokio::process::Command::new("git")
                .args(["log", "--oneline", "-5"])
                .current_dir(root)
                .output()
                .await
            {
                let log = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !log.is_empty() {
                    sections.push(format!("recent commits:\n{}", log));
                }
            }
        } else {
            sections.push("(not a git repository)".into());
        }

        // Disk summary.
        let walker = ignore::WalkBuilder::new(root)
            .hidden(false)
            .git_ignore(true)
            .build();
        let mut file_count: u64 = 0;
        let mut total_bytes: u64 = 0;
        for entry in walker.flatten() {
            if entry.path().is_file() {
                file_count += 1;
                total_bytes += entry.metadata().map(|m| m.len()).unwrap_or(0);
            }
        }
        sections.push(format!("workspace: {} files, {} total", file_count, human_size(total_bytes)));

        Ok(ToolOutput {
            success: true,
            output: sections.join("\n"),
        })
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_size_bytes() {
        assert_eq!(human_size(0), "0B");
        assert_eq!(human_size(512), "512B");
    }

    #[test]
    fn human_size_kilo() {
        assert_eq!(human_size(1024), "1K");
        assert_eq!(human_size(1536), "1.5K");
    }

    #[test]
    fn human_size_mega() {
        assert_eq!(human_size(1_048_576), "1M");
    }

    #[test]
    fn checked_path_ok() {
        let root = PathBuf::from("/workspace");
        assert!(checked_path(&root, "src/main.rs").is_ok());
    }

    #[test]
    fn checked_path_escape() {
        let root = PathBuf::from("/workspace");
        assert!(checked_path(&root, "../../etc/passwd").is_err());
    }

    #[tokio::test]
    async fn list_dir_current() {
        let tmp = std::env::temp_dir().join("aigent_test_listdir");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("sub")).unwrap();
        std::fs::write(tmp.join("file.txt"), "hi").unwrap();

        let tool = ListDirTool { workspace_root: tmp.clone() };
        let mut args = HashMap::new();
        args.insert("path".into(), ".".into());
        let out = tool.run(&args).await.unwrap();
        assert!(out.success);
        assert!(out.output.contains("sub/"));
        assert!(out.output.contains("file.txt"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn mkdir_creates_nested() {
        let tmp = std::env::temp_dir().join("aigent_test_mkdir");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let tool = MkdirTool { workspace_root: tmp.clone() };
        let mut args = HashMap::new();
        args.insert("path".into(), "a/b/c".into());
        let out = tool.run(&args).await.unwrap();
        assert!(out.success);
        assert!(tmp.join("a/b/c").is_dir());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn cp_file() {
        let tmp = std::env::temp_dir().join("aigent_test_cp");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("a.txt"), "hello").unwrap();

        let tool = CpTool { workspace_root: tmp.clone() };
        let mut args = HashMap::new();
        args.insert("src".into(), "a.txt".into());
        args.insert("dst".into(), "b.txt".into());
        let out = tool.run(&args).await.unwrap();
        assert!(out.success);
        assert_eq!(std::fs::read_to_string(tmp.join("b.txt")).unwrap(), "hello");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn rm_file() {
        let tmp = std::env::temp_dir().join("aigent_test_rm");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("x.txt"), "bye").unwrap();

        let tool = RmTool { workspace_root: tmp.clone() };
        let mut args = HashMap::new();
        args.insert("path".into(), "x.txt".into());
        let out = tool.run(&args).await.unwrap();
        assert!(out.success);
        assert!(!tmp.join("x.txt").exists());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn wc_counts() {
        let tmp = std::env::temp_dir().join("aigent_test_wc");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("test.txt"), "hello world\nfoo bar\n").unwrap();

        let tool = WcTool { workspace_root: tmp.clone() };
        let mut args = HashMap::new();
        args.insert("path".into(), "test.txt".into());
        let out = tool.run(&args).await.unwrap();
        assert!(out.success);
        assert!(out.output.contains("2 lines"));
        assert!(out.output.contains("4 words"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn head_limits_lines() {
        let tmp = std::env::temp_dir().join("aigent_test_head");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let content = (1..=50).map(|i| format!("line {}", i)).collect::<Vec<_>>().join("\n");
        std::fs::write(tmp.join("big.txt"), &content).unwrap();

        let tool = HeadTool { workspace_root: tmp.clone() };
        let mut args = HashMap::new();
        args.insert("path".into(), "big.txt".into());
        args.insert("lines".into(), "5".into());
        let out = tool.run(&args).await.unwrap();
        assert!(out.success);
        assert!(out.output.contains("line 1"));
        assert!(out.output.contains("line 5"));
        assert!(!out.output.contains("line 6"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn tail_last_lines() {
        let tmp = std::env::temp_dir().join("aigent_test_tail");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let content = (1..=50).map(|i| format!("line {}", i)).collect::<Vec<_>>().join("\n");
        std::fs::write(tmp.join("big.txt"), &content).unwrap();

        let tool = TailTool { workspace_root: tmp.clone() };
        let mut args = HashMap::new();
        args.insert("path".into(), "big.txt".into());
        args.insert("lines".into(), "3".into());
        let out = tool.run(&args).await.unwrap();
        assert!(out.success);
        assert!(out.output.contains("line 48"));
        assert!(out.output.contains("line 50"));
        assert!(!out.output.contains("line 47"));

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
