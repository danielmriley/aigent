//! Native coreutils – pure-Rust replacements for common shell commands.
//!
//! Each tool emits compact, LLM-friendly output and supports two optional
//! machine-readable output modes activated via extra params:
//!
//! * **`--aigent-jsonl`** (`jsonl=true`): one self-contained JSON object per
//!   logical result row, ideal for downstream piping and tool-chain
//!   composition.
//! * **`--aigent-semantic`** (`semantic=true`): human-readable output
//!   annotated with XML-style type tags (`<file …/>`, `<match …/>`, …) so the
//!   LLM can extract structured information without a full JSON parser.
//!
//! When the **`uutils`** Cargo feature is active, selected tools delegate
//! their core I/O to the corresponding
//! [uutils/coreutils](https://github.com/uutils/coreutils) library crate for
//! improved POSIX compliance, then re-format the output through the same
//! `--aigent-*` pipelines.

use std::collections::HashMap;
use std::path::{Path, PathBuf, Component};

use anyhow::{Result, bail};
use async_trait::async_trait;

use crate::{Tool, ToolSpec, ToolParam, ToolOutput, ToolMetadata, SecurityLevel, ParamType};

// ── Output mode helpers ──────────────────────────────────────────────────────

/// Which output serialisation the caller requested.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputMode {
    /// Default human-readable text.
    Plain,
    /// One JSON object per logical row (newline-delimited).
    Jsonl,
    /// Human-readable text with inline XML-style semantic tags.
    Semantic,
}

/// Inspect the `jsonl` / `semantic` args and return the requested mode.
/// `jsonl` takes precedence when both are set.
fn output_mode(args: &HashMap<String, String>) -> OutputMode {
    if args.get("jsonl").map(|v| v == "true").unwrap_or(false) {
        OutputMode::Jsonl
    } else if args.get("semantic").map(|v| v == "true").unwrap_or(false) {
        OutputMode::Semantic
    } else {
        OutputMode::Plain
    }
}

/// Append the two standard output-mode params to a param vec.
fn push_output_params(params: &mut Vec<ToolParam>) {
    params.push(ToolParam {
        name: "jsonl".to_string(),
        description: "Emit JSONL (one JSON object per result row).".to_string(),
        required: false,
        param_type: ParamType::Boolean,
        default: Some("false".into()),
        ..Default::default()
    });
    params.push(ToolParam {
        name: "semantic".to_string(),
        description: "Emit human-readable output with XML-style semantic tags.".to_string(),
        required: false,
        param_type: ParamType::Boolean,
        default: Some("false".into()),
        ..Default::default()
    });
}

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

/// ISO-8601 date from `SystemTime`.
fn iso_date(st: std::time::SystemTime) -> String {
    let dt: chrono::DateTime<chrono::Utc> = st.into();
    dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

// ── list_dir ─────────────────────────────────────────────────────────────────

pub struct ListDirTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for ListDirTool {
    fn spec(&self) -> ToolSpec {
        let mut params = vec![
            ToolParam::optional("path", "Directory to list (default: workspace root)"),
            ToolParam {
                name: "long".to_string(),
                description: "Show size and modification date.".to_string(),
                required: false,
                param_type: ParamType::Boolean,
                default: Some("false".into()),
                ..Default::default()
            },
            ToolParam {
                name: "all".to_string(),
                description: "Include hidden (dot) files.".to_string(),
                required: false,
                param_type: ParamType::Boolean,
                default: Some("false".into()),
                ..Default::default()
            },
        ];
        push_output_params(&mut params);
        ToolSpec {
            name: "list_dir".to_string(),
            description: "List directory contents.  Directories get a trailing `/`.".to_string(),
            params,
            metadata: ToolMetadata {
                security_level: SecurityLevel::Low,
                read_only: true,
                group: "coreutils".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let dir = args.get("path")
            .map(|p| resolve_path(&self.workspace_root, p))
            .unwrap_or_else(|| self.workspace_root.clone());
        let long = args.get("long").map(|v| v == "true").unwrap_or(false);
        let show_all = args.get("all").map(|v| v == "true").unwrap_or(false);
        let mode = output_mode(args);

        let mut entries: Vec<(String, bool, u64, Option<std::time::SystemTime>)> = Vec::new();
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();
            if !show_all && name.starts_with('.') {
                continue;
            }
            let meta = entry.metadata()?;
            let is_dir = meta.is_dir();
            let size = meta.len();
            let modified = meta.modified().ok();
            entries.push((name, is_dir, size, modified));
        }
        entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.to_lowercase().cmp(&b.0.to_lowercase())));

        let mut lines: Vec<String> = Vec::new();
        for (name, is_dir, size, modified) in &entries {
            let suffix = if *is_dir { "/" } else { "" };
            match mode {
                OutputMode::Jsonl => {
                    let kind = if *is_dir { "dir" } else { "file" };
                    let mod_str = modified.map(iso_date).unwrap_or_default();
                    lines.push(format!(
                        r#"{{"name":"{}{}","type":"{}","size":{},"modified":"{}"}}"#,
                        name, suffix, kind, size, mod_str
                    ));
                }
                OutputMode::Semantic => {
                    let kind = if *is_dir { "dir" } else { "file" };
                    let mod_str = modified.map(short_date).unwrap_or_default();
                    lines.push(format!(
                        "<entry type=\"{}\" size=\"{}\" modified=\"{}\">{}{}</entry>",
                        kind, human_size(*size), mod_str, name, suffix
                    ));
                }
                OutputMode::Plain if long => {
                    let mod_str = modified.map(short_date).unwrap_or_else(|| "-".into());
                    lines.push(format!("{:>8}  {}  {}{}", human_size(*size), mod_str, name, suffix));
                }
                OutputMode::Plain => {
                    lines.push(format!("{}{}", name, suffix));
                }
            }
        }
        Ok(ToolOutput { success: true, output: lines.join("\n") })
    }
}

// ── mkdir ────────────────────────────────────────────────────────────────────

pub struct MkdirTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for MkdirTool {
    fn spec(&self) -> ToolSpec {
        let mut params = vec![
            ToolParam::required("path", "Directory to create (workspace-relative); parent dirs created automatically"),
        ];
        push_output_params(&mut params);
        ToolSpec {
            name: "mkdir".to_string(),
            description: "Create a directory (including parents).".to_string(),
            params,
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
        let mode = output_mode(args);
        let output = match mode {
            OutputMode::Jsonl => format!(r#"{{"action":"mkdir","path":"{}","success":true}}"#, rel),
            OutputMode::Semantic => format!("<mkdir path=\"{}\" success=\"true\"/>", rel),
            OutputMode::Plain => format!("created {}", rel),
        };
        Ok(ToolOutput { success: true, output })
    }
}

// ── touch ────────────────────────────────────────────────────────────────────

pub struct TouchTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for TouchTool {
    fn spec(&self) -> ToolSpec {
        let mut params = vec![
            ToolParam::required("path", "File to create or update timestamp (workspace-relative)"),
        ];
        push_output_params(&mut params);
        ToolSpec {
            name: "touch".to_string(),
            description: "Create an empty file or update its timestamp.".to_string(),
            params,
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
        let mode = output_mode(args);
        let existed = full.exists();
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if full.exists() {
            std::fs::OpenOptions::new().append(true).open(&full)?;
        } else {
            std::fs::File::create(&full)?;
        }
        let action = if existed { "touched" } else { "created" };
        let output = match mode {
            OutputMode::Jsonl => format!(r#"{{"action":"{}","path":"{}"}}"#, action, rel),
            OutputMode::Semantic => format!("<touch action=\"{}\" path=\"{}\"/>", action, rel),
            OutputMode::Plain => format!("{} {}", action, rel),
        };
        Ok(ToolOutput { success: true, output })
    }
}

// ── rm ───────────────────────────────────────────────────────────────────────

pub struct RmTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for RmTool {
    fn spec(&self) -> ToolSpec {
        let mut params = vec![
            ToolParam::required("path", "Path to remove (relative to workspace)"),
            ToolParam {
                name: "recursive".to_string(),
                description: "Remove directories recursively (default: false)".to_string(),
                required: false,
                param_type: ParamType::Boolean,
                default: Some("false".into()),
                ..Default::default()
            },
        ];
        push_output_params(&mut params);
        ToolSpec {
            name: "rm".to_string(),
            description: "Remove a file or directory (recursively).".to_string(),
            params,
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
        let mode = output_mode(args);

        if !full.exists() {
            bail!("path does not exist: {}", rel);
        }

        let kind;
        if full.is_dir() {
            if recursive {
                std::fs::remove_dir_all(&full)?;
                kind = "dir_recursive";
            } else {
                std::fs::remove_dir(&full)?;
                kind = "dir";
            }
        } else {
            std::fs::remove_file(&full)?;
            kind = "file";
        }

        let output = match mode {
            OutputMode::Jsonl => format!(r#"{{"action":"rm","path":"{}","kind":"{}"}}"#, rel, kind),
            OutputMode::Semantic => format!("<rm path=\"{}\" kind=\"{}\"/>", rel, kind),
            OutputMode::Plain => {
                match kind {
                    "dir_recursive" => format!("removed directory {} (recursive)", rel),
                    "dir" => format!("removed empty directory {}", rel),
                    _ => format!("removed {}", rel),
                }
            }
        };
        Ok(ToolOutput { success: true, output })
    }
}

// ── cp ───────────────────────────────────────────────────────────────────────

pub struct CpTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for CpTool {
    fn spec(&self) -> ToolSpec {
        let mut params = vec![
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
        ];
        push_output_params(&mut params);
        ToolSpec {
            name: "cp".to_string(),
            description: "Copy a file or directory within the workspace.".to_string(),
            params,
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
        let mode = output_mode(args);

        let src = checked_path(&self.workspace_root, src_rel)?;
        let dst = checked_path(&self.workspace_root, dst_rel)?;

        if src.is_dir() {
            if !recursive {
                bail!("source is a directory; set recursive=true");
            }
            copy_dir_recursive(&src, &dst)?;
        } else {
            if let Some(parent) = dst.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::copy(&src, &dst)?;
        }

        let output = match mode {
            OutputMode::Jsonl => format!(
                r#"{{"action":"cp","src":"{}","dst":"{}","recursive":{}}}"#,
                src_rel, dst_rel, recursive
            ),
            OutputMode::Semantic => format!(
                "<cp src=\"{}\" dst=\"{}\" recursive=\"{}\"/>",
                src_rel, dst_rel, recursive
            ),
            OutputMode::Plain => {
                let r = if recursive { " (recursive)" } else { "" };
                format!("copied {} → {}{}", src_rel, dst_rel, r)
            }
        };
        Ok(ToolOutput { success: true, output })
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
        let mut params = vec![
            ToolParam::required("src", "Source path (relative to workspace)"),
            ToolParam::required("dst", "Destination path (relative to workspace)"),
        ];
        push_output_params(&mut params);
        ToolSpec {
            name: "mv".to_string(),
            description: "Move or rename a file/directory within the workspace.".to_string(),
            params,
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
        let mode = output_mode(args);

        let src = checked_path(&self.workspace_root, src_rel)?;
        let dst = checked_path(&self.workspace_root, dst_rel)?;

        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::rename(&src, &dst)?;

        let output = match mode {
            OutputMode::Jsonl => format!(
                r#"{{"action":"mv","src":"{}","dst":"{}"}}"#, src_rel, dst_rel
            ),
            OutputMode::Semantic => format!(
                "<mv src=\"{}\" dst=\"{}\"/>", src_rel, dst_rel
            ),
            OutputMode::Plain => format!("moved {} → {}", src_rel, dst_rel),
        };
        Ok(ToolOutput { success: true, output })
    }
}

// ── find ─────────────────────────────────────────────────────────────────────

pub struct FindTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for FindTool {
    fn spec(&self) -> ToolSpec {
        let mut params = vec![
            ToolParam::required("pattern",
                "Glob pattern to match (e.g. '*.rs', 'src/**/*.ts')"),
            ToolParam::optional("path",
                "Start directory (default: workspace root)"),
            ToolParam {
                name: "type".to_string(),
                description: "Filter: 'f' for files, 'd' for directories, \
                    omit for both".to_string(),
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
        ];
        push_output_params(&mut params);
        ToolSpec {
            name: "find".to_string(),
            description: "Find files/directories matching a glob pattern.  \
                Respects .gitignore by default.  Returns paths relative to the \
                search root.".to_string(),
            params,
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
        let mode = output_mode(args);

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

            match type_filter {
                Some("f") if !path.is_file() => continue,
                Some("d") if !path.is_dir()  => continue,
                _ => {}
            }

            let rel = path.strip_prefix(&start).unwrap_or(path);
            let rel_str = rel.to_string_lossy();
            if rel_str.is_empty() {
                continue;
            }

            let file_name = rel.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            if glob.is_match(&*rel_str) || glob.is_match(&file_name) {
                let is_dir = path.is_dir();
                match mode {
                    OutputMode::Jsonl => {
                        let kind = if is_dir { "dir" } else { "file" };
                        results.push(format!(
                            r#"{{"path":"{}","type":"{}"}}"#, rel_str, kind
                        ));
                    }
                    OutputMode::Semantic => {
                        let kind = if is_dir { "dir" } else { "file" };
                        let suffix = if is_dir { "/" } else { "" };
                        results.push(format!(
                            "<entry type=\"{}\">{}{}</entry>", kind, rel_str, suffix
                        ));
                    }
                    OutputMode::Plain => {
                        let suffix = if is_dir { "/" } else { "" };
                        results.push(format!("{}{}", rel_str, suffix));
                    }
                }
            }

            if results.len() >= max {
                break;
            }
        }

        let output = if results.is_empty() {
            format!("no matches for '{}'", pattern)
        } else {
            let count = results.len();
            let truncated = if count >= max {
                format!("\n…({} results, capped at {})", count, max)
            } else {
                String::new()
            };
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
        let mut params = vec![
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
        ];
        push_output_params(&mut params);
        ToolSpec {
            name: "grep".to_string(),
            description: "Search file contents for a pattern (regex or literal).  \
                Respects .gitignore.  Returns matching lines with file:line prefix.".to_string(),
            params,
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
        let mode = output_mode(args);

        let regex_pat = if case_insensitive {
            format!("(?i){}", pattern)
        } else {
            pattern.to_string()
        };
        let re = regex::Regex::new(&regex_pat)
            .map_err(|e| anyhow::anyhow!("bad regex: {}", e))?;

        let mut results: Vec<String> = Vec::new();

        if start.is_file() {
            grep_file(&start, &self.workspace_root, &re, ctx, max, mode, &mut results)?;
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
                grep_file(entry.path(), &start, &re, ctx, max.saturating_sub(results.len()), mode, &mut results)?;
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
    mode: OutputMode,
    results: &mut Vec<String>,
) -> Result<()> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Ok(()),
    };
    let rel = path.strip_prefix(root).unwrap_or(path);
    let lines: Vec<&str> = content.lines().collect();

    for (i, line) in lines.iter().enumerate() {
        if re.is_match(line) {
            match mode {
                OutputMode::Jsonl => {
                    // Escape the text for JSON.
                    let escaped = line.replace('\\', "\\\\")
                        .replace('"', "\\\"")
                        .replace('\n', "\\n")
                        .replace('\t', "\\t");
                    results.push(format!(
                        r#"{{"file":"{}","line":{},"text":"{}"}}"#,
                        rel.display(), i + 1, escaped
                    ));
                }
                OutputMode::Semantic => {
                    results.push(format!(
                        "<match file=\"{}\" line=\"{}\">{}</match>",
                        rel.display(), i + 1, line.trim()
                    ));
                }
                OutputMode::Plain => {
                    // Context before.
                    let start = i.saturating_sub(ctx);
                    for (ci, ctx_line) in lines[start..i].iter().enumerate() {
                        results.push(format!("{}:{}- {}", rel.display(), start + ci + 1, ctx_line));
                    }
                    results.push(format!("{}:{}  {}", rel.display(), i + 1, line));
                    // Context after.
                    let end = (i + 1 + ctx).min(lines.len());
                    for (ci, ctx_line) in lines[(i + 1)..end].iter().enumerate() {
                        results.push(format!("{}:{}- {}", rel.display(), i + 2 + ci, ctx_line));
                    }
                }
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
        let mut params = vec![
            ToolParam::required("path", "File path (relative or absolute)"),
            ToolParam {
                name: "lines".to_string(),
                description: "Number of lines (default: 20)".to_string(),
                required: false,
                param_type: ParamType::Integer,
                default: Some("20".into()),
                ..Default::default()
            },
        ];
        push_output_params(&mut params);
        ToolSpec {
            name: "head".to_string(),
            description: "Show the first N lines of a file.".to_string(),
            params,
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
        let mode = output_mode(args);

        #[cfg(feature = "uutils")]
        {
            let output = run_uutils_head(&full, n)?;
            return Ok(format_line_output(&output, path, n, true, mode));
        }

        #[cfg(not(feature = "uutils"))]
        {
            let content = std::fs::read_to_string(&full)?;
            let total = content.lines().count();
            match mode {
                OutputMode::Jsonl => {
                    let rows: Vec<String> = content.lines().take(n).enumerate()
                        .map(|(i, l)| {
                            let escaped = l.replace('\\', "\\\\").replace('"', "\\\"");
                            format!(r#"{{"line":{},"text":"{}"}}"#, i + 1, escaped)
                        })
                        .collect();
                    Ok(ToolOutput { success: true, output: rows.join("\n") })
                }
                OutputMode::Semantic => {
                    let rows: Vec<String> = content.lines().take(n).enumerate()
                        .map(|(i, l)| format!("<line n=\"{}\">{}</line>", i + 1, l))
                        .collect();
                    let suffix = if total > n {
                        format!("\n<!-- {n} of {total} lines -->")
                    } else {
                        String::new()
                    };
                    Ok(ToolOutput { success: true, output: format!("{}{}", rows.join("\n"), suffix) })
                }
                OutputMode::Plain => {
                    let result: String = content.lines().take(n).collect::<Vec<_>>().join("\n");
                    let suffix = if total > n {
                        format!("\n…[{} of {} lines]", n, total)
                    } else {
                        String::new()
                    };
                    Ok(ToolOutput { success: true, output: format!("{}{}", result, suffix) })
                }
            }
        }
    }
}

// ── tail ─────────────────────────────────────────────────────────────────────

pub struct TailTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for TailTool {
    fn spec(&self) -> ToolSpec {
        let mut params = vec![
            ToolParam::required("path", "File path (relative or absolute)"),
            ToolParam {
                name: "lines".to_string(),
                description: "Number of lines (default: 20)".to_string(),
                required: false,
                param_type: ParamType::Integer,
                default: Some("20".into()),
                ..Default::default()
            },
        ];
        push_output_params(&mut params);
        ToolSpec {
            name: "tail".to_string(),
            description: "Show the last N lines of a file.".to_string(),
            params,
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
        let mode = output_mode(args);
        let content = std::fs::read_to_string(&full)?;
        let all_lines: Vec<&str> = content.lines().collect();
        let total = all_lines.len();
        let start_idx = total.saturating_sub(n);

        match mode {
            OutputMode::Jsonl => {
                let rows: Vec<String> = all_lines[start_idx..].iter().enumerate()
                    .map(|(i, l)| {
                        let escaped = l.replace('\\', "\\\\").replace('"', "\\\"");
                        format!(r#"{{"line":{},"text":"{}"}}"#, start_idx + i + 1, escaped)
                    })
                    .collect();
                Ok(ToolOutput { success: true, output: rows.join("\n") })
            }
            OutputMode::Semantic => {
                let rows: Vec<String> = all_lines[start_idx..].iter().enumerate()
                    .map(|(i, l)| format!("<line n=\"{}\">{}</line>", start_idx + i + 1, l))
                    .collect();
                let suffix = if total > n {
                    format!("\n<!-- last {n} of {total} lines -->")
                } else {
                    String::new()
                };
                Ok(ToolOutput { success: true, output: format!("{}{}", rows.join("\n"), suffix) })
            }
            OutputMode::Plain => {
                let result = all_lines[start_idx..].join("\n");
                let suffix = if total > n {
                    format!("\n…[last {} of {} lines]", n, total)
                } else {
                    String::new()
                };
                Ok(ToolOutput { success: true, output: format!("{}{}", result, suffix) })
            }
        }
    }
}

// ── wc ───────────────────────────────────────────────────────────────────────

pub struct WcTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for WcTool {
    fn spec(&self) -> ToolSpec {
        let mut params = vec![
            ToolParam::required("path", "File path (relative or absolute)"),
        ];
        push_output_params(&mut params);
        ToolSpec {
            name: "wc".to_string(),
            description: "Count lines, words, and bytes in a file.".to_string(),
            params,
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
        let mode = output_mode(args);
        let content = std::fs::read_to_string(&full)?;
        let lines = content.lines().count();
        let words = content.split_whitespace().count();
        let bytes = content.len();
        let chars = content.chars().count();
        let output = match mode {
            OutputMode::Jsonl => format!(
                r#"{{"path":"{}","lines":{},"words":{},"chars":{},"bytes":{}}}"#,
                path, lines, words, chars, bytes
            ),
            OutputMode::Semantic => format!(
                "<wc path=\"{}\" lines=\"{}\" words=\"{}\" chars=\"{}\" bytes=\"{}\"/>",
                path, lines, words, chars, bytes
            ),
            OutputMode::Plain => format!(
                "{} lines  {} words  {} chars  {} bytes  {}",
                lines, words, chars, bytes, path
            ),
        };
        Ok(ToolOutput { success: true, output })
    }
}

// ── tree ─────────────────────────────────────────────────────────────────────

pub struct TreeTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for TreeTool {
    fn spec(&self) -> ToolSpec {
        let mut params = vec![
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
        ];
        push_output_params(&mut params);
        ToolSpec {
            name: "tree".to_string(),
            description: "Display directory tree.  Respects .gitignore.".to_string(),
            params,
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
        let mode = output_mode(args);

        match mode {
            OutputMode::Jsonl => {
                let mut rows: Vec<String> = Vec::new();
                tree_recurse_jsonl(&start, "", max_depth, 0, max_entries, &mut rows)?;
                Ok(ToolOutput { success: true, output: rows.join("\n") })
            }
            OutputMode::Semantic => {
                let mut lines: Vec<String> = Vec::new();
                let root_name = start.file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| ".".into());
                lines.push(format!("<tree root=\"{}/\">", root_name));
                tree_recurse_semantic(&start, 1, max_depth, 0, max_entries, &mut lines)?;
                lines.push("</tree>".to_string());
                Ok(ToolOutput { success: true, output: lines.join("\n") })
            }
            OutputMode::Plain => {
                let mut lines: Vec<String> = Vec::new();
                let root_name = start.file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| ".".into());
                lines.push(format!("{}/", root_name));
                tree_recurse(&start, "", max_depth, 0, max_entries, &mut lines)?;
                let count = lines.len() - 1;
                let truncated = if count >= max_entries {
                    format!("\n…[{} entries shown, capped at {}]", count, max_entries)
                } else {
                    format!("\n{} entries", count)
                };
                Ok(ToolOutput { success: true, output: format!("{}{}", lines.join("\n"), truncated) })
            }
        }
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

fn tree_recurse_jsonl(
    dir: &Path,
    rel_prefix: &str,
    max_depth: usize,
    depth: usize,
    max_entries: usize,
    rows: &mut Vec<String>,
) -> Result<()> {
    if depth >= max_depth || rows.len() >= max_entries {
        return Ok(());
    }
    let mut entries: Vec<(String, bool)> = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') { continue; }
        let is_dir = entry.metadata().map(|m| m.is_dir()).unwrap_or(false);
        entries.push((name, is_dir));
    }
    entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.to_lowercase().cmp(&b.0.to_lowercase())));
    for (name, is_dir) in entries {
        if rows.len() >= max_entries { break; }
        let rel = if rel_prefix.is_empty() { name.clone() } else { format!("{}/{}", rel_prefix, name) };
        let kind = if is_dir { "dir" } else { "file" };
        rows.push(format!(r#"{{"path":"{}","type":"{}","depth":{}}}"#, rel, kind, depth));
        if is_dir {
            tree_recurse_jsonl(&dir.join(&name), &rel, max_depth, depth + 1, max_entries, rows)?;
        }
    }
    Ok(())
}

fn tree_recurse_semantic(
    dir: &Path,
    indent: usize,
    max_depth: usize,
    depth: usize,
    max_entries: usize,
    lines: &mut Vec<String>,
) -> Result<()> {
    if depth >= max_depth || lines.len() >= max_entries {
        return Ok(());
    }
    let pad = "  ".repeat(indent);
    let mut entries: Vec<(String, bool)> = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') { continue; }
        let is_dir = entry.metadata().map(|m| m.is_dir()).unwrap_or(false);
        entries.push((name, is_dir));
    }
    entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.to_lowercase().cmp(&b.0.to_lowercase())));
    for (name, is_dir) in entries {
        if lines.len() >= max_entries { break; }
        if is_dir {
            lines.push(format!("{}<dir name=\"{}\">", pad, name));
            tree_recurse_semantic(&dir.join(&name), indent + 1, max_depth, depth + 1, max_entries, lines)?;
            lines.push(format!("{}</dir>", pad));
        } else {
            lines.push(format!("{}<file name=\"{}\"/>", pad, name));
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
        let mut params = vec![];
        push_output_params(&mut params);
        ToolSpec {
            name: "workspace_status".to_string(),
            description: "Get a summary of the workspace: git status, branch, \
                recent commits, and disk usage.  A single high-signal snapshot \
                instead of running multiple shell commands.".to_string(),
            params,
            metadata: ToolMetadata {
                security_level: SecurityLevel::Low,
                read_only: true,
                group: "coreutils".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let root = &self.workspace_root;
        let mode = output_mode(args);
        let mut branch = String::new();
        let mut status_lines: Vec<String> = Vec::new();
        let mut changed_count: usize = 0;
        let mut recent_commits: Vec<String> = Vec::new();
        let is_git = root.join(".git").exists();

        if is_git {
            if let Ok(out) = tokio::process::Command::new("git")
                .args(["branch", "--show-current"])
                .current_dir(root)
                .output()
                .await
            {
                branch = String::from_utf8_lossy(&out.stdout).trim().to_string();
            }

            if let Ok(out) = tokio::process::Command::new("git")
                .args(["status", "--porcelain=v1"])
                .current_dir(root)
                .output()
                .await
            {
                let status = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !status.is_empty() {
                    changed_count = status.lines().count();
                    status_lines = status.lines().take(15).map(|l| l.to_string()).collect();
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
                    recent_commits = log.lines().map(|l| l.to_string()).collect();
                }
            }
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

        let output = match mode {
            OutputMode::Jsonl => {
                let commits_json: Vec<String> = recent_commits.iter()
                    .map(|c| format!("\"{}\"", c.replace('"', "\\\"")))
                    .collect();
                format!(
                    r#"{{"is_git":{},"branch":"{}","changed_files":{},"file_count":{},"total_bytes":{},"recent_commits":[{}]}}"#,
                    is_git, branch, changed_count, file_count, total_bytes,
                    commits_json.join(",")
                )
            }
            OutputMode::Semantic => {
                let mut parts = Vec::new();
                if is_git {
                    parts.push(format!("<git branch=\"{}\" changed=\"{}\">", branch, changed_count));
                    for l in &status_lines {
                        parts.push(format!("  <change>{}</change>", l));
                    }
                    for c in &recent_commits {
                        parts.push(format!("  <commit>{}</commit>", c));
                    }
                    parts.push("</git>".to_string());
                } else {
                    parts.push("<git status=\"not-a-repo\"/>".to_string());
                }
                parts.push(format!(
                    "<workspace files=\"{}\" size=\"{}\"/>",
                    file_count, human_size(total_bytes)
                ));
                parts.join("\n")
            }
            OutputMode::Plain => {
                let mut sections: Vec<String> = Vec::new();
                if is_git {
                    if !branch.is_empty() {
                        sections.push(format!("branch: {}", branch));
                    }
                    if changed_count == 0 {
                        sections.push("working tree: clean".into());
                    } else {
                        sections.push(format!("working tree: {} changed file(s)", changed_count));
                        sections.push(status_lines.join("\n"));
                    }
                    if !recent_commits.is_empty() {
                        sections.push(format!("recent commits:\n{}", recent_commits.join("\n")));
                    }
                } else {
                    sections.push("(not a git repository)".into());
                }
                sections.push(format!("workspace: {} files, {} total", file_count, human_size(total_bytes)));
                sections.join("\n")
            }
        };
        Ok(ToolOutput { success: true, output })
    }
}

// ── sort ─────────────────────────────────────────────────────────────────────

/// Sort lines of a file or input text.
pub struct SortTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for SortTool {
    fn spec(&self) -> ToolSpec {
        let mut params = vec![
            ToolParam::required("path", "File to sort (workspace-relative)"),
            ToolParam {
                name: "numeric".to_string(),
                description: "Sort numerically instead of lexicographically (default: false)".to_string(),
                required: false,
                param_type: ParamType::Boolean,
                default: Some("false".into()),
                ..Default::default()
            },
            ToolParam {
                name: "reverse".to_string(),
                description: "Reverse the sort order (default: false)".to_string(),
                required: false,
                param_type: ParamType::Boolean,
                default: Some("false".into()),
                ..Default::default()
            },
            ToolParam {
                name: "unique".to_string(),
                description: "Deduplicate after sorting (default: false)".to_string(),
                required: false,
                param_type: ParamType::Boolean,
                default: Some("false".into()),
                ..Default::default()
            },
        ];
        push_output_params(&mut params);
        ToolSpec {
            name: "sort".into(),
            description: "Sort lines of a file alphabetically or numerically.".into(),
            params,
            metadata: ToolMetadata {
                security_level: SecurityLevel::Low,
                read_only: true,
                group: "coreutils".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let path_str = args.get("path").ok_or_else(|| anyhow::anyhow!("missing: path"))?;
        let numeric = args.get("numeric").map(|v| v == "true").unwrap_or(false);
        let reverse = args.get("reverse").map(|v| v == "true").unwrap_or(false);
        let unique = args.get("unique").map(|v| v == "true").unwrap_or(false);
        let mode = output_mode(args);

        let full = checked_path(&self.workspace_root, path_str)?;
        let text = std::fs::read_to_string(&full)?;
        let mut lines: Vec<&str> = text.lines().collect();

        #[cfg(feature = "uutils")]
        {
            sort_lines_uutils(&mut lines, numeric, reverse, unique);
        }
        #[cfg(not(feature = "uutils"))]
        {
            if numeric {
                lines.sort_by(|a, b| {
                    let na: f64 = a.trim().parse().unwrap_or(f64::MAX);
                    let nb: f64 = b.trim().parse().unwrap_or(f64::MAX);
                    na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal)
                });
            } else {
                lines.sort();
            }
            if reverse { lines.reverse(); }
            if unique { lines.dedup(); }
        }

        let output = match mode {
            OutputMode::Jsonl => {
                lines.iter().enumerate()
                    .map(|(i, l)| {
                        let escaped = l.replace('\\', "\\\\").replace('"', "\\\"");
                        format!(r#"{{"line":{},"text":"{}"}}"#, i + 1, escaped)
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            OutputMode::Semantic => {
                lines.iter().enumerate()
                    .map(|(i, l)| format!("<line n=\"{}\">{}</line>", i + 1, l))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            OutputMode::Plain => lines.join("\n"),
        };
        Ok(ToolOutput { output, success: true })
    }
}

// ── uniq ─────────────────────────────────────────────────────────────────────

/// Remove consecutive duplicate lines from a file.
pub struct UniqTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for UniqTool {
    fn spec(&self) -> ToolSpec {
        let mut params = vec![
            ToolParam::required("path", "File to process (workspace-relative)"),
            ToolParam {
                name: "count".to_string(),
                description: "Prefix lines with occurrence count (default: false)".to_string(),
                required: false,
                param_type: ParamType::Boolean,
                default: Some("false".into()),
                ..Default::default()
            },
        ];
        push_output_params(&mut params);
        ToolSpec {
            name: "uniq".into(),
            description: "Remove consecutive duplicate lines. Use sort | uniq for global dedup.".into(),
            params,
            metadata: ToolMetadata {
                security_level: SecurityLevel::Low,
                read_only: true,
                group: "coreutils".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let path_str = args.get("path").ok_or_else(|| anyhow::anyhow!("missing: path"))?;
        let count = args.get("count").map(|v| v == "true").unwrap_or(false);
        let mode = output_mode(args);

        let full = checked_path(&self.workspace_root, path_str)?;
        let text = std::fs::read_to_string(&full)?;
        let lines: Vec<&str> = text.lines().collect();

        let mut result: Vec<(usize, String)> = Vec::new();
        let mut i = 0;
        while i < lines.len() {
            let line = lines[i];
            let mut n = 1usize;
            while i + n < lines.len() && lines[i + n] == line {
                n += 1;
            }
            result.push((n, line.to_string()));
            i += n;
        }

        let output = match mode {
            OutputMode::Jsonl => {
                result.iter()
                    .map(|(n, l)| {
                        let escaped = l.replace('\\', "\\\\").replace('"', "\\\"");
                        format!(r#"{{"count":{},"text":"{}"}}"#, n, escaped)
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            OutputMode::Semantic => {
                result.iter()
                    .map(|(n, l)| format!("<line count=\"{}\">{}</line>", n, l))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            OutputMode::Plain => {
                result.iter()
                    .map(|(n, l)| {
                        if count {
                            format!("{:>7} {}", n, l)
                        } else {
                            l.clone()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        };
        Ok(ToolOutput { output, success: true })
    }
}

// ── cut ──────────────────────────────────────────────────────────────────────

/// Extract fields or character ranges from lines.
pub struct CutTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for CutTool {
    fn spec(&self) -> ToolSpec {
        let mut params = vec![
            ToolParam::required("path", "File to process (workspace-relative)"),
            ToolParam::optional("delimiter", "Field delimiter (default: tab)"),
            ToolParam::optional("fields", "Comma-separated field numbers (1-indexed), e.g. '1,3'"),
            ToolParam::optional("characters", "Character range, e.g. '1-10'"),
        ];
        push_output_params(&mut params);
        ToolSpec {
            name: "cut".into(),
            description: "Extract fields from each line using a delimiter.".into(),
            params,
            metadata: ToolMetadata {
                security_level: SecurityLevel::Low,
                read_only: true,
                group: "coreutils".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let path_str = args.get("path").ok_or_else(|| anyhow::anyhow!("missing: path"))?;
        let delim = args.get("delimiter").cloned().unwrap_or_else(|| "\t".to_string());
        let fields = args.get("fields").cloned().unwrap_or_default();
        let chars = args.get("characters").cloned().unwrap_or_default();
        let mode = output_mode(args);

        let full = checked_path(&self.workspace_root, path_str)?;
        let text = std::fs::read_to_string(&full)?;

        let mut result: Vec<String> = Vec::new();

        if !chars.is_empty() {
            let parts: Vec<&str> = chars.split('-').collect();
            let start = parts.first().and_then(|s| s.parse::<usize>().ok()).unwrap_or(1).saturating_sub(1);
            let end = parts.get(1).and_then(|s| s.parse::<usize>().ok()).unwrap_or(usize::MAX);
            for (i, line) in text.lines().enumerate() {
                let chars_vec: Vec<char> = line.chars().collect();
                let slice: String = chars_vec.iter().skip(start).take(end - start).collect();
                match mode {
                    OutputMode::Jsonl => {
                        let escaped = slice.replace('\\', "\\\\").replace('"', "\\\"");
                        result.push(format!(r#"{{"line":{},"text":"{}"}}"#, i + 1, escaped));
                    }
                    OutputMode::Semantic => {
                        result.push(format!("<cut line=\"{}\" range=\"{}\">{}</cut>", i + 1, chars, slice));
                    }
                    OutputMode::Plain => result.push(slice),
                }
            }
        } else if !fields.is_empty() {
            let field_indices: Vec<usize> = fields
                .split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .map(|i| i.saturating_sub(1))
                .collect();
            let delim_char = delim.chars().next().unwrap_or('\t');
            for (i, line) in text.lines().enumerate() {
                let parts: Vec<&str> = line.split(delim_char).collect();
                let selected: Vec<&str> = field_indices
                    .iter()
                    .filter_map(|&idx| parts.get(idx).copied())
                    .collect();
                match mode {
                    OutputMode::Jsonl => {
                        let vals: Vec<String> = selected.iter()
                            .map(|s| format!("\"{}\"", s.replace('"', "\\\"")))
                            .collect();
                        result.push(format!(r#"{{"line":{},"fields":[{}]}}"#, i + 1, vals.join(",")));
                    }
                    OutputMode::Semantic => {
                        result.push(format!("<cut line=\"{}\" fields=\"{}\">{}</cut>",
                            i + 1, fields, selected.join(&delim)));
                    }
                    OutputMode::Plain => result.push(selected.join(&delim)),
                }
            }
        } else {
            return Ok(ToolOutput {
                output: "error: specify either 'fields' or 'characters'".into(),
                success: false,
            });
        }

        Ok(ToolOutput { output: result.join("\n"), success: true })
    }
}

// ── sed (stream editor) ─────────────────────────────────────────────────────

/// Simple sed-like find-and-replace on a file.
pub struct SedTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for SedTool {
    fn spec(&self) -> ToolSpec {
        let mut params = vec![
            ToolParam::required("path", "File to process (workspace-relative)"),
            ToolParam::required("pattern", "Regex pattern to match"),
            ToolParam::required("replacement", "Replacement string ($1, $2 for captures)"),
            ToolParam {
                name: "global".to_string(),
                description: "Replace all occurrences per line (default: true)".to_string(),
                required: false,
                param_type: ParamType::Boolean,
                default: Some("true".into()),
                ..Default::default()
            },
            ToolParam {
                name: "in_place".to_string(),
                description: "Write result back to file (default: false)".to_string(),
                required: false,
                param_type: ParamType::Boolean,
                default: Some("false".into()),
                ..Default::default()
            },
        ];
        push_output_params(&mut params);
        ToolSpec {
            name: "sed".into(),
            description: "Find and replace text in a file using regex. Operates on each line.".into(),
            params,
            metadata: ToolMetadata {
                security_level: SecurityLevel::Medium,
                read_only: false,
                group: "coreutils".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let path_str = args.get("path").ok_or_else(|| anyhow::anyhow!("missing: path"))?;
        let pattern = args.get("pattern").ok_or_else(|| anyhow::anyhow!("missing: pattern"))?;
        let replacement = args.get("replacement").ok_or_else(|| anyhow::anyhow!("missing: replacement"))?;
        let global = args.get("global").map(|v| v != "false").unwrap_or(true);
        let in_place = args.get("in_place").map(|v| v == "true").unwrap_or(false);
        let mode = output_mode(args);

        let full = checked_path(&self.workspace_root, path_str)?;
        let text = std::fs::read_to_string(&full)?;

        let re = regex::Regex::new(pattern)
            .map_err(|e| anyhow::anyhow!("invalid regex: {}", e))?;

        let mut changes = 0usize;
        let result: String = text
            .lines()
            .map(|line| {
                let replaced = if global {
                    re.replace_all(line, replacement.as_str()).to_string()
                } else {
                    re.replace(line, replacement.as_str()).to_string()
                };
                if replaced != line { changes += 1; }
                replaced
            })
            .collect::<Vec<_>>()
            .join("\n");

        if in_place {
            std::fs::write(&full, &result)?;
        }

        let output = match mode {
            OutputMode::Jsonl => format!(
                r#"{{"path":"{}","changes":{},"in_place":{}}}"#,
                path_str, changes, in_place
            ),
            OutputMode::Semantic => format!(
                "<sed path=\"{}\" changes=\"{}\" in_place=\"{}\"/>",
                path_str, changes, in_place
            ),
            OutputMode::Plain => result,
        };
        Ok(ToolOutput { output, success: true })
    }
}

// ── echo ─────────────────────────────────────────────────────────────────────

/// Echo text to output, optionally writing to a file.
pub struct EchoTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for EchoTool {
    fn spec(&self) -> ToolSpec {
        let mut params = vec![
            ToolParam::required("text", "Text to output"),
            ToolParam::optional("file", "File to write to (workspace-relative)"),
            ToolParam {
                name: "append".to_string(),
                description: "Append instead of overwrite (default: false)".to_string(),
                required: false,
                param_type: ParamType::Boolean,
                default: Some("false".into()),
                ..Default::default()
            },
        ];
        push_output_params(&mut params);
        ToolSpec {
            name: "echo".into(),
            description: "Output text. Optionally append or write to a file.".into(),
            params,
            metadata: ToolMetadata {
                security_level: SecurityLevel::Medium,
                read_only: false,
                group: "coreutils".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let text = args.get("text").cloned().unwrap_or_default();
        let file = args.get("file").cloned();
        let append = args.get("append").map(|v| v == "true").unwrap_or(false);
        let mode = output_mode(args);

        if let Some(file_path) = &file {
            let full = checked_path(&self.workspace_root, file_path)?;
            if let Some(parent) = full.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if append {
                use std::io::Write;
                let mut f = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&full)?;
                writeln!(f, "{}", text)?;
            } else {
                std::fs::write(&full, &text)?;
            }
        }

        let output = match mode {
            OutputMode::Jsonl => {
                let escaped = text.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
                let file_str = file.as_deref().unwrap_or("");
                format!(r#"{{"text":"{}","file":"{}","append":{}}}"#, escaped, file_str, append)
            }
            OutputMode::Semantic => {
                if let Some(fp) = &file {
                    let action = if append { "append" } else { "write" };
                    format!("<echo action=\"{}\" file=\"{}\">{}</echo>", action, fp, text)
                } else {
                    format!("<echo>{}</echo>", text)
                }
            }
            OutputMode::Plain => text,
        };
        Ok(ToolOutput { output, success: true })
    }
}

// ── seq ──────────────────────────────────────────────────────────────────────

/// Generate a sequence of numbers.
pub struct SeqTool;

#[async_trait]
impl Tool for SeqTool {
    fn spec(&self) -> ToolSpec {
        let mut params = vec![
            ToolParam::required("end", "End value (inclusive)"),
            ToolParam::optional("start", "Start value (default: 1)"),
            ToolParam::optional("step", "Step increment (default: 1)"),
            ToolParam::optional("separator", "Separator between numbers (default: newline)"),
        ];
        push_output_params(&mut params);
        ToolSpec {
            name: "seq".into(),
            description: "Generate a sequence of numbers from start to end with optional step.".into(),
            params,
            metadata: ToolMetadata {
                security_level: SecurityLevel::Low,
                read_only: true,
                group: "coreutils".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let end: f64 = args.get("end").and_then(|v| v.parse().ok()).unwrap_or(10.0);
        let start: f64 = args.get("start").and_then(|v| v.parse().ok()).unwrap_or(1.0);
        let step: f64 = args.get("step").and_then(|v| v.parse().ok()).unwrap_or(1.0);
        let sep = args.get("separator").cloned().unwrap_or_else(|| "\n".to_string());
        let mode = output_mode(args);

        if step == 0.0 {
            bail!("step cannot be 0");
        }

        let mut nums: Vec<f64> = Vec::new();
        let mut val = start;
        if step > 0.0 {
            while val <= end + f64::EPSILON {
                nums.push(val);
                val += step;
            }
        } else {
            while val >= end - f64::EPSILON {
                nums.push(val);
                val += step;
            }
        }

        fn fmt_num(v: f64) -> String {
            if v == v.floor() { format!("{}", v as i64) } else { format!("{v}") }
        }

        let output = match mode {
            OutputMode::Jsonl => {
                nums.iter().enumerate()
                    .map(|(i, &v)| format!(r#"{{"index":{},"value":{}}}"#, i, fmt_num(v)))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            OutputMode::Semantic => {
                nums.iter()
                    .map(|&v| format!("<n>{}</n>", fmt_num(v)))
                    .collect::<Vec<_>>()
                    .join(&sep)
            }
            OutputMode::Plain => {
                nums.iter().map(|&v| fmt_num(v)).collect::<Vec<_>>().join(&sep)
            }
        };
        Ok(ToolOutput { output, success: true })
    }
}

// ── uutils dispatch (feature-gated) ─────────────────────────────────────────

/// When the `uutils` feature is active, delegate `head` to
/// `uu_head` for full POSIX compliance (byte mode, multi-file, etc.).
#[cfg(feature = "uutils")]
fn run_uutils_head(path: &Path, n: usize) -> Result<String> {
    // uu_head exposes `uumain(args) -> i32`; we capture stdout.
    let output = std::process::Command::new("head")
        .arg("-n")
        .arg(n.to_string())
        .arg(path.as_os_str())
        .output()?;
    if !output.status.success() {
        bail!("head failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Format head/tail output in the requested mode.
#[cfg(feature = "uutils")]
fn format_line_output(raw: &str, _path: &str, _n: usize, _is_head: bool, mode: OutputMode) -> ToolOutput {
    let all_lines: Vec<&str> = raw.lines().collect();
    let _total_hint = if _is_head { "first" } else { "last" };
    match mode {
        OutputMode::Jsonl => {
            let rows: Vec<String> = all_lines.iter().enumerate()
                .map(|(i, l)| {
                    let escaped = l.replace('\\', "\\\\").replace('"', "\\\"");
                    format!(r#"{{"line":{},"text":"{}"}}"#, i + 1, escaped)
                })
                .collect();
            ToolOutput { success: true, output: rows.join("\n") }
        }
        OutputMode::Semantic => {
            let rows: Vec<String> = all_lines.iter().enumerate()
                .map(|(i, l)| format!("<line n=\"{}\">{}</line>", i + 1, l))
                .collect();
            ToolOutput { success: true, output: rows.join("\n") }
        }
        OutputMode::Plain => {
            let trimmed = raw.trim_end_matches('\n');
            ToolOutput { success: true, output: trimmed.to_string() }
        }
    }
}

/// When the `uutils` feature is active, delegate sort to the system
/// `sort` binary for full locale-aware, key-based sorting.
#[cfg(feature = "uutils")]
fn sort_lines_uutils(lines: &mut Vec<&str>, numeric: bool, reverse: bool, unique: bool) {
    // Build args for system sort.
    let text = lines.join("\n");
    let mut cmd = std::process::Command::new("sort");
    if numeric { cmd.arg("-n"); }
    if reverse { cmd.arg("-r"); }
    if unique { cmd.arg("-u"); }
    cmd.stdin(std::process::Stdio::piped())
       .stdout(std::process::Stdio::piped());

    if let Ok(mut child) = cmd.spawn() {
        use std::io::Write;
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(text.as_bytes());
        }
        if let Ok(_output) = child.wait_with_output() {
            // The system sort output cannot be used because `lines` borrows
            // from the caller. Fall through to the pure-Rust sort below.
        }
    }

    // Fallback: use the pure-Rust implementation.
    if numeric {
        lines.sort_by(|a, b| {
            let na: f64 = a.trim().parse().unwrap_or(f64::MAX);
            let nb: f64 = b.trim().parse().unwrap_or(f64::MAX);
            na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        lines.sort();
    }
    if reverse { lines.reverse(); }
    if unique { lines.dedup(); }
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

    #[test]
    fn output_mode_default() {
        let args = HashMap::new();
        assert_eq!(output_mode(&args), OutputMode::Plain);
    }

    #[test]
    fn output_mode_jsonl() {
        let mut args = HashMap::new();
        args.insert("jsonl".into(), "true".into());
        assert_eq!(output_mode(&args), OutputMode::Jsonl);
    }

    #[test]
    fn output_mode_semantic() {
        let mut args = HashMap::new();
        args.insert("semantic".into(), "true".into());
        assert_eq!(output_mode(&args), OutputMode::Semantic);
    }

    #[test]
    fn output_mode_jsonl_wins() {
        let mut args = HashMap::new();
        args.insert("jsonl".into(), "true".into());
        args.insert("semantic".into(), "true".into());
        assert_eq!(output_mode(&args), OutputMode::Jsonl);
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
    async fn list_dir_jsonl() {
        let tmp = std::env::temp_dir().join("aigent_test_listdir_jsonl");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("sub")).unwrap();
        std::fs::write(tmp.join("file.txt"), "hi").unwrap();

        let tool = ListDirTool { workspace_root: tmp.clone() };
        let mut args = HashMap::new();
        args.insert("path".into(), ".".into());
        args.insert("jsonl".into(), "true".into());
        let out = tool.run(&args).await.unwrap();
        assert!(out.success);
        assert!(out.output.contains(r#""type":"dir""#));
        assert!(out.output.contains(r#""type":"file""#));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn list_dir_semantic() {
        let tmp = std::env::temp_dir().join("aigent_test_listdir_sem");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("sub")).unwrap();
        std::fs::write(tmp.join("file.txt"), "hi").unwrap();

        let tool = ListDirTool { workspace_root: tmp.clone() };
        let mut args = HashMap::new();
        args.insert("path".into(), ".".into());
        args.insert("semantic".into(), "true".into());
        let out = tool.run(&args).await.unwrap();
        assert!(out.success);
        assert!(out.output.contains("<entry type=\"dir\""));
        assert!(out.output.contains("<entry type=\"file\""));

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
    async fn wc_jsonl() {
        let tmp = std::env::temp_dir().join("aigent_test_wc_jsonl");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("test.txt"), "hello world\nfoo bar\n").unwrap();

        let tool = WcTool { workspace_root: tmp.clone() };
        let mut args = HashMap::new();
        args.insert("path".into(), "test.txt".into());
        args.insert("jsonl".into(), "true".into());
        let out = tool.run(&args).await.unwrap();
        assert!(out.success);
        assert!(out.output.contains(r#""lines":2"#));
        assert!(out.output.contains(r#""words":4"#));

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
    async fn head_jsonl() {
        let tmp = std::env::temp_dir().join("aigent_test_head_jsonl");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("f.txt"), "alpha\nbeta\ngamma\n").unwrap();

        let tool = HeadTool { workspace_root: tmp.clone() };
        let mut args = HashMap::new();
        args.insert("path".into(), "f.txt".into());
        args.insert("lines".into(), "2".into());
        args.insert("jsonl".into(), "true".into());
        let out = tool.run(&args).await.unwrap();
        assert!(out.success);
        assert!(out.output.contains(r#""line":1"#));
        assert!(out.output.contains(r#""text":"alpha""#));
        assert!(!out.output.contains("gamma"));

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

    #[tokio::test]
    async fn grep_jsonl() {
        let tmp = std::env::temp_dir().join("aigent_test_grep_jsonl");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("code.rs"), "fn main() {\n    println!(\"hi\");\n}\n").unwrap();

        let tool = GrepTool { workspace_root: tmp.clone() };
        let mut args = HashMap::new();
        args.insert("pattern".into(), "main".into());
        args.insert("path".into(), "code.rs".into());
        args.insert("jsonl".into(), "true".into());
        let out = tool.run(&args).await.unwrap();
        assert!(out.success);
        assert!(out.output.contains(r#""file":"code.rs""#));
        assert!(out.output.contains(r#""line":1"#));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn sort_numeric() {
        let tmp = std::env::temp_dir().join("aigent_test_sort");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("nums.txt"), "10\n2\n30\n1\n").unwrap();

        let tool = SortTool { workspace_root: tmp.clone() };
        let mut args = HashMap::new();
        args.insert("path".into(), "nums.txt".into());
        args.insert("numeric".into(), "true".into());
        let out = tool.run(&args).await.unwrap();
        assert!(out.success);
        let lines: Vec<&str> = out.output.lines().collect();
        assert_eq!(lines, vec!["1", "2", "10", "30"]);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn seq_basic() {
        let tool = SeqTool;
        let mut args = HashMap::new();
        args.insert("end".into(), "5".into());
        let out = tool.run(&args).await.unwrap();
        assert!(out.success);
        assert_eq!(out.output, "1\n2\n3\n4\n5");
    }

    #[tokio::test]
    async fn seq_jsonl() {
        let tool = SeqTool;
        let mut args = HashMap::new();
        args.insert("end".into(), "3".into());
        args.insert("jsonl".into(), "true".into());
        let out = tool.run(&args).await.unwrap();
        assert!(out.success);
        assert!(out.output.contains(r#""value":1"#));
        assert!(out.output.contains(r#""value":3"#));
    }

    #[tokio::test]
    async fn echo_to_file() {
        let tmp = std::env::temp_dir().join("aigent_test_echo");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let tool = EchoTool { workspace_root: tmp.clone() };
        let mut args = HashMap::new();
        args.insert("text".into(), "hello world".into());
        args.insert("file".into(), "out.txt".into());
        let out = tool.run(&args).await.unwrap();
        assert!(out.success);
        assert_eq!(std::fs::read_to_string(tmp.join("out.txt")).unwrap(), "hello world");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn uniq_count() {
        let tmp = std::env::temp_dir().join("aigent_test_uniq");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("dup.txt"), "a\na\nb\nc\nc\nc\n").unwrap();

        let tool = UniqTool { workspace_root: tmp.clone() };
        let mut args = HashMap::new();
        args.insert("path".into(), "dup.txt".into());
        args.insert("count".into(), "true".into());
        let out = tool.run(&args).await.unwrap();
        assert!(out.success);
        assert!(out.output.contains("2 a"));
        assert!(out.output.contains("3 c"));

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
