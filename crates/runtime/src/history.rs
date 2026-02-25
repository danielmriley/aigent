//! TUI chat history persistence — daily JSONL files in `~/.aigent/history/`.
//!
//! This module is **completely separate** from the 6-tier memory system.  It is
//! a thin append-only transcript so the TUI can restore the last N turns when
//! the user opens a new session.

use anyhow::{Context, Result};
use chrono::{Local, Utc};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
};

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// A single turn persisted to disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnRecord {
    /// `"user"` or `"assistant"`.
    pub role: String,
    pub content: String,
    pub timestamp: chrono::DateTime<Utc>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Path helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Returns `.aigent/history/` relative to the current working directory.
/// This matches the convention used everywhere else in the project.
pub fn history_dir() -> PathBuf {
    PathBuf::from(".aigent").join("history")
}

/// Returns `~/.aigent/history/YYYY-MM-DD.jsonl` for today's date.
pub fn history_file_path() -> PathBuf {
    let today = Local::now().format("%Y-%m-%d").to_string();
    history_dir().join(format!("{today}.jsonl"))
}

// ──────────────────────────────────────────────────────────────────────────────
// Core operations
// ──────────────────────────────────────────────────────────────────────────────

/// Atomically append one turn to today's history file.
/// Creates the file (and parent directories) if they don't exist.
pub fn append_turn(role: impl Into<String>, content: impl Into<String>) -> Result<()> {
    let path = history_file_path();
    fs::create_dir_all(path.parent().unwrap())?;

    let record = TurnRecord {
        role: role.into(),
        content: content.into(),
        timestamp: Utc::now(),
    };
    let line = serde_json::to_string(&record).context("serialize TurnRecord")?;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("open history file {}", path.display()))?;
    writeln!(file, "{line}")?;
    Ok(())
}

/// Load up to `max_turns` most-recent turns from today's history file.
/// If the file doesn't exist, returns an empty `Vec`.
pub fn load_recent(max_turns: usize) -> Result<Vec<TurnRecord>> {
    let path = history_file_path();
    if !path.exists() {
        return Ok(vec![]);
    }
    let file = fs::File::open(&path)
        .with_context(|| format!("open history file {}", path.display()))?;
    let reader = BufReader::new(file);

    let mut records: Vec<TurnRecord> = reader
        .lines()
        .filter_map(|line| {
            let line = line.ok()?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                return None;
            }
            serde_json::from_str(trimmed).ok()
        })
        .collect();

    // Keep only the last `max_turns` entries.
    if records.len() > max_turns {
        let skip = records.len() - max_turns;
        records.drain(..skip);
    }
    Ok(records)
}

/// Delete today's history file.
pub fn clear_history() -> Result<()> {
    let path = history_file_path();
    if path.exists() {
        fs::remove_file(&path)
            .with_context(|| format!("remove history file {}", path.display()))?;
    }
    Ok(())
}

/// Copy today's history file to `dest`.
pub fn export_history(dest: &Path) -> Result<()> {
    let path = history_file_path();
    if !path.exists() {
        anyhow::bail!("no history for today ({} does not exist)", path.display());
    }
    fs::copy(&path, dest)
        .with_context(|| format!("copy {} -> {}", path.display(), dest.display()))?;
    Ok(())
}
