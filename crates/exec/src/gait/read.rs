//! Read-only git operations.

use std::fmt::Write as _;
use std::path::Path;

use anyhow::{Result, bail};

use super::open_repo;
use super::types::{GitOperation, GaitPolicy, assert_inside_trusted};

pub(super) fn do_status(repo_path: &Path) -> Result<String> {
    let repo = open_repo(repo_path)?;
    let statuses = repo.statuses(Some(
        git2::StatusOptions::new()
            .include_untracked(true)
            .recurse_untracked_dirs(true),
    ))?;

    if statuses.is_empty() {
        return Ok("Working tree clean — nothing to commit.".to_string());
    }

    let mut out = String::from("Git status:\n");
    for entry in statuses.iter() {
        let path = entry.path().unwrap_or("?");
        let st = entry.status();
        let label = if st.contains(git2::Status::WT_NEW) {
            "untracked"
        } else if st.contains(git2::Status::WT_MODIFIED) | st.contains(git2::Status::INDEX_MODIFIED)
        {
            "modified"
        } else if st.contains(git2::Status::WT_DELETED) | st.contains(git2::Status::INDEX_DELETED)
        {
            "deleted"
        } else if st.contains(git2::Status::INDEX_NEW) {
            "staged (new)"
        } else if st.contains(git2::Status::WT_RENAMED) | st.contains(git2::Status::INDEX_RENAMED)
        {
            "renamed"
        } else {
            "changed"
        };
        let _ = writeln!(out, "  {label}: {path}");
    }
    Ok(out)
}

pub(super) fn do_log(repo_path: &Path, op: &GitOperation) -> Result<String> {
    let repo = open_repo(repo_path)?;
    let mut revwalk = repo.revwalk()?;
    revwalk.push_head()?;
    revwalk.set_sorting(git2::Sort::TIME)?;

    let max_entries = 15;
    let mut out = String::from("Recent commits:\n");
    let mut count = 0;

    for oid in revwalk {
        if count >= max_entries {
            let _ = writeln!(out, "  … (showing first {max_entries})");
            break;
        }
        let oid = oid?;
        let commit = repo.find_commit(oid)?;
        let short = &oid.to_string()[..7];
        let summary = commit.summary().unwrap_or("(no message)");
        let time = commit.time();
        let ts = chrono::DateTime::from_timestamp(time.seconds(), 0)
            .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
            .unwrap_or_else(|| "?".to_string());
        let author = commit.author();
        let name = author.name().unwrap_or("?");
        let _ = writeln!(out, "  {short} {ts} ({name}) {summary}");
        count += 1;
    }

    if count == 0 {
        return Ok("No commits found.".to_string());
    }
    let _ = op.branch; // reserved for future branch-specific log
    Ok(out)
}

pub(super) fn do_diff(repo_path: &Path, _op: &GitOperation) -> Result<String> {
    let repo = open_repo(repo_path)?;
    let diff = repo.diff_index_to_workdir(None, None)?;
    let stats = diff.stats()?;

    if stats.files_changed() == 0 {
        return Ok("No differences found.".to_string());
    }

    let mut out = format!(
        "Diff: {} file(s) changed, {} insertion(s), {} deletion(s)\n",
        stats.files_changed(),
        stats.insertions(),
        stats.deletions(),
    );

    // Collect patch text (capped at 8 KiB to keep LLM context manageable).
    let mut patch_buf = String::new();
    diff.print(git2::DiffFormat::Patch, |_delta, _hunk, line| {
        if patch_buf.len() < 8192 {
            let origin = line.origin();
            if origin == '+' || origin == '-' || origin == ' ' {
                patch_buf.push(origin);
            }
            if let Ok(content) = std::str::from_utf8(line.content()) {
                patch_buf.push_str(content);
            }
        }
        true
    })?;

    if patch_buf.len() >= 8192 {
        patch_buf.push_str("\n…[patch truncated at 8 KiB]");
    }
    out.push_str(&patch_buf);
    Ok(out)
}

pub(super) fn do_show(repo_path: &Path, op: &GitOperation) -> Result<String> {
    let repo = open_repo(repo_path)?;
    let rev = op.branch.as_deref().unwrap_or("HEAD");
    let obj = repo.revparse_single(rev)?;
    let commit = obj.peel_to_commit()?;

    let author = commit.author();
    let time = commit.time();
    let ts = chrono::DateTime::from_timestamp(time.seconds(), 0)
        .map(|dt| dt.format("%Y-%m-%d %H:%M UTC").to_string())
        .unwrap_or_else(|| "?".to_string());

    let mut out = format!(
        "commit {}\nAuthor: {} <{}>\nDate:   {}\n\n    {}\n",
        commit.id(),
        author.name().unwrap_or("?"),
        author.email().unwrap_or("?"),
        ts,
        commit.message().unwrap_or("(no message)"),
    );

    // Show diff stats for this commit.
    if let Ok(parent) = commit.parent(0) {
        let a = parent.tree()?;
        let b = commit.tree()?;
        let diff = repo.diff_tree_to_tree(Some(&a), Some(&b), None)?;
        let stats = diff.stats()?;
        let _ = writeln!(
            out,
            " {} file(s) changed, {} insertion(s), {} deletion(s)",
            stats.files_changed(),
            stats.insertions(),
            stats.deletions(),
        );
    }
    Ok(out)
}

pub(super) async fn do_ls_remote(op: &GitOperation, policy: &GaitPolicy) -> Result<String> {
    let url = op.repo.trim();
    // ls-remote is read-only but hits the network.  For local paths, check
    // access according to policy.
    if !url.starts_with("https://") && !url.starts_with("http://") && !url.starts_with("git@") {
        let path = tokio::fs::canonicalize(url).await?;
        if !policy.allow_system_read {
            assert_inside_trusted(&path, policy)?;
        }
    }

    let output = tokio::process::Command::new("git")
        .args(["ls-remote", "--heads", "--tags", url])
        .output()
        .await?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(format!("Refs for '{url}':\n{}", stdout.trim()))
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("git ls-remote failed: {}", stderr.trim());
    }
}

pub(super) async fn do_fetch(repo_path: &Path, _op: &GitOperation) -> Result<String> {
    let output = tokio::process::Command::new("git")
        .args(["fetch", "--all", "--prune"])
        .current_dir(repo_path)
        .output()
        .await?;

    if output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Ok(format!(
            "Fetched all remotes.\n{}",
            if stderr.is_empty() {
                "(no new data)"
            } else {
                stderr.trim()
            }
        ))
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("git fetch failed: {}", stderr.trim());
    }
}

pub(super) async fn do_blame(repo_path: &Path, op: &GitOperation) -> Result<String> {
    let file = op
        .paths
        .as_ref()
        .and_then(|p| p.first())
        .ok_or_else(|| anyhow::anyhow!("blame requires a file path in 'paths'"))?;

    let output = tokio::process::Command::new("git")
        .args(["blame", "--line-porcelain", file])
        .current_dir(repo_path)
        .output()
        .await?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        // Compact the porcelain output for the LLM.
        let compact = compact_blame(&stdout);
        Ok(compact)
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("git blame failed: {}", stderr.trim());
    }
}

/// Compact git blame porcelain output into a readable summary.
pub(super) fn compact_blame(raw: &str) -> String {
    let mut out = String::from("Blame:\n");
    let mut lineno = 1u32;
    let mut current_author = String::new();
    for line in raw.lines() {
        if let Some(author) = line.strip_prefix("author ") {
            current_author = author.to_string();
        }
        if let Some(code) = line.strip_prefix('\t') {
            let _ = writeln!(out, "  {lineno:>4} ({current_author}) {code}");
            lineno += 1;
        }
        // Cap output at 4 KiB.
        if out.len() > 4096 {
            out.push_str("\n…[blame truncated]");
            break;
        }
    }
    out
}

