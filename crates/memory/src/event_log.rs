use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

use crate::schema::MemoryEntry;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecordEvent {
    pub event_id: Uuid,
    pub occurred_at: DateTime<Utc>,
    pub entry: MemoryEntry,
}

#[derive(Debug, Clone)]
pub struct MemoryEventLog {
    path: PathBuf,
}

impl MemoryEventLog {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub async fn append(&self, event: &MemoryRecordEvent) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .await?;
        let line = serde_json::to_string(event)?;
        file.write_all(line.as_bytes()).await?;
        file.write_all(b"\n").await?;
        // Flush userspace buffers and fsync to disk so the entry survives a
        // process crash or power loss immediately after append.
        file.flush().await?;
        file.sync_all().await?;
        Ok(())
    }

    /// Atomically replace the event log with a new set of events.
    ///
    /// Crash-safety guarantee: the new content is written to a `.tmp` sibling
    /// file, `fsync`'d, then renamed over the original.  A crash at any point
    /// before the rename leaves the original file untouched.  A crash after
    /// the rename leaves a consistent new file.  The `.tmp` file is cleaned up
    /// on any error path.
    pub async fn overwrite(&self, events: &[MemoryRecordEvent]) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Derive `.tmp` path by appending `.tmp` to the full filename.
        let tmp_path = {
            let filename = self
                .path
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_else(|| "events.jsonl".to_string());
            self.path.with_file_name(format!("{filename}.tmp"))
        };

        // Write to the temp file first.
        let write_result: Result<()> = async {
            let mut file = tokio::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&tmp_path)
                .await?;
            for event in events {
                let line = serde_json::to_string(event)?;
                file.write_all(line.as_bytes()).await?;
                file.write_all(b"\n").await?;
            }
            // Flush userspace buffers and sync to disk before rename.
            file.flush().await?;
            file.sync_all().await?;
            Ok(())
        }
        .await;

        if let Err(err) = write_result {
            let _ = tokio::fs::remove_file(&tmp_path).await;
            return Err(err);
        }

        // Atomic rename: if this returns Ok the new file is fully consistent.
        if let Err(err) = tokio::fs::rename(&tmp_path, &self.path).await {
            let _ = tokio::fs::remove_file(&tmp_path).await;
            return Err(err.into());
        }

        Ok(())
    }

    /// Copy the live event log to a `.bak` sibling file.
    ///
    /// Called at the start of each sleep cycle so a consistent snapshot is
    /// available even if the cycle writes new entries or the process crashes
    /// mid-cycle.  If the source file does not yet exist the call is a no-op.
    pub fn backup(&self) -> Result<()> {
        if !self.path.exists() {
            return Ok(());
        }

        let bak_path = {
            let filename = self
                .path
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_else(|| "events.jsonl".to_string());
            self.path.with_file_name(format!("{filename}.bak"))
        };

        fs::copy(&self.path, &bak_path)?;
        Ok(())
    }

    pub fn load(&self) -> Result<Vec<MemoryRecordEvent>> {
        use std::fs::OpenOptions;
        if !self.path.exists() {
            return Ok(Vec::new());
        }

        let file = OpenOptions::new().read(true).open(&self.path)?;
        let reader = BufReader::new(file);
        let mut events = Vec::new();
        let mut corrupt_count = 0usize;

        for (line_idx, line_result) in reader.lines().enumerate() {
            let line = line_result?;
            if line.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<MemoryRecordEvent>(&line) {
                Ok(event) => events.push(event),
                Err(err) => {
                    corrupt_count += 1;
                    tracing::warn!(
                        line = line_idx + 1,
                        error = %err,
                        path = %self.path.display(),
                        "corrupt JSONL record — skipping line (original preserved in .corrupt file)"
                    );
                    // Append the bad line to a sidecar file for forensics.
                    let corrupt_path = self.path.with_extension("jsonl.corrupt");
                    let mut bad = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&corrupt_path)
                        .unwrap_or_else(|_| {
                            // If we can't open the sidecar, just continue.
                            // The warning above is the only signal the user gets.
                            std::fs::File::open("/dev/null").expect("/dev/null always exists")
                        });
                    use std::io::Write as _;
                    let _ = writeln!(bad, "{line}");
                }
            }
        }

        if corrupt_count > 0 {
            tracing::warn!(
                corrupt_lines = corrupt_count,
                path = %self.path.display(),
                "event log loaded with skipped corrupt lines — inspect .corrupt sidecar"
            );
        }

        Ok(events)
    }
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use uuid::Uuid;

    use super::{MemoryEventLog, MemoryRecordEvent};
    use crate::schema::{MemoryEntry, MemoryTier};

    fn make_event(content: &str) -> MemoryRecordEvent {
        MemoryRecordEvent {
            event_id: Uuid::new_v4(),
            occurred_at: Utc::now(),
            entry: MemoryEntry {
                id: Uuid::new_v4(),
                tier: MemoryTier::Episodic,
                content: content.to_string(),
                source: "test".to_string(),
                confidence: 0.8,
                valence: 0.0,
                created_at: Utc::now(),
                provenance_hash: "test-hash".to_string(),
                tags: vec!["tag1".to_string()],
                embedding: None,
            },
        }
    }

    fn temp_path() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("aigent-elog-test-{}.jsonl", Uuid::new_v4()))
    }

    #[tokio::test]
    async fn append_and_load_round_trip() {
        let path = temp_path();
        let log = MemoryEventLog::new(&path);
        let event = make_event("hello");
        log.append(&event).await.unwrap();
        let events = log.load().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].entry.content, "hello");
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn append_multiple_events() {
        let path = temp_path();
        let log = MemoryEventLog::new(&path);
        for i in 0..5 {
            log.append(&make_event(&format!("event-{i}"))).await.unwrap();
        }
        let events = log.load().unwrap();
        assert_eq!(events.len(), 5);
        assert_eq!(events[0].entry.content, "event-0");
        assert_eq!(events[4].entry.content, "event-4");
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn overwrite_replaces_all_events() {
        let path = temp_path();
        let log = MemoryEventLog::new(&path);
        log.append(&make_event("old")).await.unwrap();
        log.append(&make_event("also old")).await.unwrap();
        assert_eq!(log.load().unwrap().len(), 2);
        let new_events = vec![make_event("new")];
        log.overwrite(&new_events).await.unwrap();
        let loaded = log.load().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].entry.content, "new");
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn overwrite_to_empty() {
        let path = temp_path();
        let log = MemoryEventLog::new(&path);
        log.append(&make_event("a")).await.unwrap();
        log.overwrite(&[]).await.unwrap();
        let loaded = log.load().unwrap();
        assert!(loaded.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_nonexistent_returns_empty() {
        let path = temp_path();
        let log = MemoryEventLog::new(&path);
        let events = log.load().unwrap();
        assert!(events.is_empty());
    }

    #[tokio::test]
    async fn load_skips_corrupt_lines() {
        let path = temp_path();
        // Write a valid event followed by a corrupt line.
        let log = MemoryEventLog::new(&path);
        log.append(&make_event("valid")).await.unwrap();
        // Append garbage directly.
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .map(|mut f| {
                use std::io::Write;
                writeln!(f, "{{invalid json garbage}}").unwrap();
            })
            .unwrap();
        log.append(&make_event("also valid")).await.unwrap();
        let events = log.load().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].entry.content, "valid");
        assert_eq!(events[1].entry.content, "also valid");
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn backup_creates_bak_file() {
        let path = temp_path();
        let log = MemoryEventLog::new(&path);
        log.append(&make_event("backup me")).await.unwrap();
        log.backup().unwrap();
        let bak_path = path.with_file_name(
            format!("{}.bak", path.file_name().unwrap().to_string_lossy()),
        );
        assert!(bak_path.exists(), "backup file should exist");
        // Backup should be loadable too.
        let bak_log = MemoryEventLog::new(&bak_path);
        let bak_events = bak_log.load().unwrap();
        assert_eq!(bak_events.len(), 1);
        assert_eq!(bak_events[0].entry.content, "backup me");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&bak_path);
    }

    #[test]
    fn backup_nonexistent_file_is_no_op() {
        let path = temp_path();
        let log = MemoryEventLog::new(&path);
        // Should not error when the file doesn't exist.
        log.backup().unwrap();
    }

    #[tokio::test]
    async fn tags_survive_round_trip() {
        let path = temp_path();
        let log = MemoryEventLog::new(&path);
        let mut event = make_event("tagged");
        event.entry.tags = vec!["belief".to_string(), "important".to_string()];
        log.append(&event).await.unwrap();
        let loaded = log.load().unwrap();
        assert_eq!(loaded[0].entry.tags, vec!["belief", "important"]);
        let _ = std::fs::remove_file(&path);
    }
}