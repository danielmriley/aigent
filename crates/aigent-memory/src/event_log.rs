use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
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

    pub fn append(&self, event: &MemoryRecordEvent) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        let line = serde_json::to_string(event)?;
        writeln!(file, "{line}")?;
        Ok(())
    }

    pub fn overwrite(&self, events: &[MemoryRecordEvent]) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.path)?;

        for event in events {
            let line = serde_json::to_string(event)?;
            writeln!(file, "{line}")?;
        }

        Ok(())
    }

    pub fn load(&self) -> Result<Vec<MemoryRecordEvent>> {
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
