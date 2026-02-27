//! Write / mutating git operations.

use std::path::{Path, PathBuf};
use std::fmt::Write as _;

use anyhow::{Result, bail};
use tracing::info;

use super::open_repo;
use super::types::{GitOperation, GaitPolicy, assert_inside_trusted};

pub(super) fn do_commit_sync(repo_path: &Path, op: &GitOperation) -> Result<String> {
    let msg = op
        .message
        .as_deref()
        .unwrap_or("Aigent: gait auto-commit");

    let repo = open_repo(repo_path)?;

    // Stage all changes (mirrors `git add -A`).  If `paths` is provided,
    // stage only those paths.
    let mut index = repo.index()?;
    if let Some(ref paths) = op.paths {
        for p in paths {
            index.add_path(Path::new(p))?;
        }
    } else {
        index.add_all(["*"].iter(), git2::IndexAddOption::DEFAULT, None)?;
    }
    index.write()?;
    let tree_oid = index.write_tree()?;
    let tree = repo.find_tree(tree_oid)?;

    let sig = repo.signature().unwrap_or_else(|_| {
        git2::Signature::now("Aigent", "aigent@localhost").expect("valid signature")
    });

    let parent = repo
        .head()
        .ok()
        .and_then(|h| h.peel_to_commit().ok());
    let parents: Vec<&git2::Commit<'_>> = parent.iter().collect();

    let oid = repo.commit(Some("HEAD"), &sig, &sig, msg, &tree, &parents)?;
    let short = &oid.to_string()[..7];

    info!(sha = %short, path = %repo_path.display(), "gait: committed");
    Ok(format!("Committed {short}: {msg}"))
}

pub(super) fn do_checkout(repo_path: &Path, op: &GitOperation) -> Result<String> {
    let branch_name = op
        .branch
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("checkout requires a branch name"))?;

    let repo = open_repo(repo_path)?;

    // Try as local branch first, then as remote tracking branch.
    let refname = format!("refs/heads/{branch_name}");
    let obj = repo
        .revparse_single(&refname)
        .or_else(|_| repo.revparse_single(&format!("refs/remotes/origin/{branch_name}")))?;

    repo.checkout_tree(&obj, Some(git2::build::CheckoutBuilder::new().force()))?;
    repo.set_head(&refname)
        .or_else(|_| repo.set_head_detached(obj.id()))?;

    Ok(format!("Checked out '{branch_name}'."))
}

pub(super) fn do_branch(repo_path: &Path, op: &GitOperation) -> Result<String> {
    let repo = open_repo(repo_path)?;

    if let Some(ref name) = op.branch {
        // Create new branch at HEAD.
        let head = repo.head()?.peel_to_commit()?;
        let force = op.force.unwrap_or(false);
        repo.branch(name, &head, force)?;
        Ok(format!("Created branch '{name}'."))
    } else {
        // List branches.
        let branches = repo.branches(Some(git2::BranchType::Local))?;
        let mut out = String::from("Branches:\n");
        for b in branches {
            let (branch, _) = b?;
            let name = branch.name()?.unwrap_or("?");
            let marker = if branch.is_head() { " *" } else { "" };
            let _ = writeln!(out, "  {name}{marker}");
        }
        Ok(out)
    }
}

pub(super) fn do_reset(repo_path: &Path, op: &GitOperation) -> Result<String> {
    let repo = open_repo(repo_path)?;
    let target = op.branch.as_deref().unwrap_or("HEAD~1");
    let obj = repo.revparse_single(target)?;
    let commit = obj
        .peel_to_commit()
        .map_err(|e| anyhow::anyhow!("cannot resolve '{}' to a commit: {}", target, e))?;

    let reset_type = if op.force.unwrap_or(false) {
        git2::ResetType::Hard
    } else {
        git2::ResetType::Mixed
    };

    repo.reset(commit.as_object(), reset_type, None)?;

    let type_label = if op.force.unwrap_or(false) {
        "hard"
    } else {
        "mixed"
    };
    Ok(format!(
        "Reset ({type_label}) to '{target}' ({})",
        &commit.id().to_string()[..7]
    ))
}

pub(super) async fn do_clone(op: &GitOperation, policy: &GaitPolicy) -> Result<String> {
    let url = op.repo.trim();
    let target_dir = op
        .target_dir
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!(
            "clone requires 'target_dir' — specifies where to put the cloned repo. \
             Must be inside trusted_write_paths."
        ))?;

    // Resolve target_dir: if relative, resolve against workspace root.
    let target = if Path::new(target_dir).is_absolute() {
        PathBuf::from(target_dir)
    } else {
        policy.workspace_root.join(target_dir)
    };

    // SECURITY: trust check BEFORE creating any directories.
    // We canonicalize the existing portion of the path (its nearest existing
    // ancestor) to perform a reliable prefix check.
    let canonical_target = {
        // Walk up to find the nearest existing ancestor, canonicalize it,
        // then re-append the remaining components.
        let mut existing = target.clone();
        let mut suffix_parts: Vec<std::ffi::OsString> = Vec::new();
        while !existing.exists() {
            if let Some(name) = existing.file_name() {
                suffix_parts.push(name.to_os_string());
            } else {
                break;
            }
            existing = match existing.parent() {
                Some(p) => p.to_path_buf(),
                None => break,
            };
        }
        let canonical_base = tokio::fs::canonicalize(&existing).await
            .map_err(|e| anyhow::anyhow!("cannot resolve clone target base '{}': {}", existing.display(), e))?;
        let mut result = canonical_base;
        for part in suffix_parts.into_iter().rev() {
            result.push(part);
        }
        result
    };

    assert_inside_trusted(&canonical_target, policy)?;

    // When allow_system_read is false, also verify local clone sources.
    if !url.starts_with("https://") && !url.starts_with("http://") && !url.starts_with("git@")
        && !policy.allow_system_read
    {
        let source = tokio::fs::canonicalize(url).await
            .map_err(|e| anyhow::anyhow!("cannot resolve clone source '{}': {}", url, e))?;
        assert_inside_trusted(&source, policy)?;
    }

    // Now that the trust check passed, create parent directories.
    if let Some(parent) = canonical_target.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    info!(url = %url, target = %canonical_target.display(), "gait: cloning");

    // Use git CLI for clone — libgit2 clone with TLS/SSH auth is complex
    // and the CLI handles credential helpers and SSH agent natively.
    let output = tokio::process::Command::new("git")
        .args(["clone", "--progress", url])
        .arg(&canonical_target)
        .output()
        .await?;

    if output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Ok(format!(
            "Cloned '{}' → '{}'\n{}",
            url,
            canonical_target.display(),
            stderr.trim()
        ))
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("git clone failed: {}", stderr.trim());
    }
}

pub(super) async fn do_pull(repo_path: &Path, op: &GitOperation) -> Result<String> {
    let mut args = vec!["pull"];
    if let Some(ref branch) = op.branch {
        args.push("origin");
        args.push(branch);
    }

    let output = tokio::process::Command::new("git")
        .args(&args)
        .current_dir(repo_path)
        .output()
        .await?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        Ok(format!("{}{}", stdout.trim(), if stderr.is_empty() { String::new() } else { format!("\n{}", stderr.trim()) }))
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("git pull failed: {}", stderr.trim());
    }
}

pub(super) async fn do_push(repo_path: &Path, op: &GitOperation) -> Result<String> {
    let mut args = vec!["push"];
    if op.force.unwrap_or(false) {
        args.push("--force-with-lease");
    }
    if let Some(ref branch) = op.branch {
        args.push("origin");
        args.push(branch);
    }

    let output = tokio::process::Command::new("git")
        .args(&args)
        .current_dir(repo_path)
        .output()
        .await?;

    if output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Ok(format!("Pushed successfully.\n{}", stderr.trim()))
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("git push failed: {}", stderr.trim());
    }
}

pub(super) fn do_stash(repo_path: &Path, op: &GitOperation) -> Result<String> {
    let mut repo = open_repo(repo_path)?;
    let sig = repo.signature().unwrap_or_else(|_| {
        git2::Signature::now("Aigent", "aigent@localhost").expect("valid signature")
    });

    let msg = op.message.as_deref().unwrap_or("gait stash");
    let _oid = repo.stash_save(&sig, msg, None)?;
    Ok(format!("Stashed working directory: '{msg}'"))
}

pub(super) fn do_tag(repo_path: &Path, op: &GitOperation) -> Result<String> {
    let repo = open_repo(repo_path)?;

    if let Some(ref tag_name) = op.branch {
        // Create a lightweight tag at HEAD.
        let head = repo.head()?.peel(git2::ObjectType::Commit)?;
        let force = op.force.unwrap_or(false);
        repo.tag_lightweight(tag_name, &head, force)?;
        Ok(format!("Created tag '{tag_name}'."))
    } else {
        // List tags.
        let tags = repo.tag_names(None)?;
        let mut out = String::from("Tags:\n");
        for i in 0..tags.len() {
            if let Some(name) = tags.get(i) {
                let _ = writeln!(out, "  {name}");
            }
        }
        if tags.is_empty() {
            out.push_str("  (none)");
        }
        Ok(out)
    }
}

