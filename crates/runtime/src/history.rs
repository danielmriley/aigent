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

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turn_record_serde_roundtrip() {
        let record = TurnRecord {
            role: "user".to_string(),
            content: "Hello there".to_string(),
            timestamp: Utc::now(),
        };
        let json = serde_json::to_string(&record).unwrap();
        let back: TurnRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back.role, "user");
        assert_eq!(back.content, "Hello there");
        assert_eq!(back.timestamp, record.timestamp);
    }

    #[test]
    fn turn_record_all_fields_present_in_json() {
        let record = TurnRecord {
            role: "assistant".to_string(),
            content: "Hi!".to_string(),
            timestamp: Utc::now(),
        };
        let json = serde_json::to_string(&record).unwrap();
        assert!(json.contains("\"role\":\"assistant\""));
        assert!(json.contains("\"content\":\"Hi!\""));
        assert!(json.contains("\"timestamp\""));
    }

    #[test]
    fn history_dir_is_relative() {
        let dir = history_dir();
        assert!(!dir.is_absolute());
        assert!(dir.to_string_lossy().contains("history"));
    }

    #[test]
    fn history_file_path_has_date_format() {
        let path = history_file_path();
        let filename = path.file_name().unwrap().to_string_lossy();
        // Should match YYYY-MM-DD.jsonl pattern
        assert!(filename.ends_with(".jsonl"));
        assert!(filename.len() == "YYYY-MM-DD.jsonl".len());
    }

    /// Test the JSONL read logic by writing records directly to a temp file and
    /// then reading them back with the same deserializer `load_recent` uses.
    #[test]
    fn jsonl_write_and_read_roundtrip() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.jsonl");

        // Write two records as JSONL (the same format append_turn uses)
        let records = vec![
            TurnRecord {
                role: "user".into(),
                content: "hello".into(),
                timestamp: Utc::now(),
            },
            TurnRecord {
                role: "assistant".into(),
                content: "hi there".into(),
                timestamp: Utc::now(),
            },
            TurnRecord {
                role: "user".into(),
                content: "how are you?".into(),
                timestamp: Utc::now(),
            },
        ];
        {
            let mut file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .unwrap();
            for r in &records {
                let line = serde_json::to_string(r).unwrap();
                writeln!(file, "{line}").unwrap();
            }
        }

        // Read back using the same BufReader logic from load_recent
        let file = fs::File::open(&path).unwrap();
        let reader = std::io::BufReader::new(file);
        let loaded: Vec<TurnRecord> = reader
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

        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded[0].role, "user");
        assert_eq!(loaded[0].content, "hello");
        assert_eq!(loaded[1].role, "assistant");
        assert_eq!(loaded[1].content, "hi there");
        assert_eq!(loaded[2].content, "how are you?");
    }

    #[test]
    fn jsonl_read_with_limit() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("limited.jsonl");

        {
            let mut file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .unwrap();
            for i in 0..5 {
                let r = TurnRecord {
                    role: "user".into(),
                    content: format!("msg {i}"),
                    timestamp: Utc::now(),
                };
                writeln!(file, "{}", serde_json::to_string(&r).unwrap()).unwrap();
            }
        }

        // Read all then keep last N (same logic as load_recent)
        let file = fs::File::open(&path).unwrap();
        let reader = std::io::BufReader::new(file);
        let mut records: Vec<TurnRecord> = reader
            .lines()
            .filter_map(|l| serde_json::from_str(l.ok()?.trim()).ok())
            .collect();

        let max_turns = 2;
        if records.len() > max_turns {
            let skip = records.len() - max_turns;
            records.drain(..skip);
        }

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].content, "msg 3");
        assert_eq!(records[1].content, "msg 4");
    }

    #[test]
    fn empty_jsonl_file_parses_to_empty_vec() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("empty.jsonl");
        fs::write(&path, "").unwrap();

        let file = fs::File::open(&path).unwrap();
        let reader = std::io::BufReader::new(file);
        let records: Vec<TurnRecord> = reader
            .lines()
            .filter_map(|l| serde_json::from_str(l.ok()?.trim()).ok())
            .collect();
        assert!(records.is_empty());
    }

    #[test]
    fn jsonl_skips_blank_and_malformed_lines() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("messy.jsonl");

        let valid = TurnRecord {
            role: "user".into(),
            content: "valid".into(),
            timestamp: Utc::now(),
        };
        let mut content = String::new();
        content.push_str(&serde_json::to_string(&valid).unwrap());
        content.push('\n');
        content.push_str("not json at all\n");
        content.push('\n');
        content.push_str(&serde_json::to_string(&valid).unwrap());
        content.push('\n');
        fs::write(&path, content).unwrap();

        let file = fs::File::open(&path).unwrap();
        let reader = std::io::BufReader::new(file);
        let records: Vec<TurnRecord> = reader
            .lines()
            .filter_map(|l| {
                let l = l.ok()?;
                let t = l.trim();
                if t.is_empty() { return None; }
                serde_json::from_str(t).ok()
            })
            .collect();
        assert_eq!(records.len(), 2);
    }
}
