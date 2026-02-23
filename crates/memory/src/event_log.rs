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

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            events.push(serde_json::from_str::<MemoryRecordEvent>(&line)?);
        }

        Ok(events)
    }
}
