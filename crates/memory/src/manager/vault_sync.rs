//! Obsidian vault export and YAML KV syncing for [`MemoryManager`].

use std::path::{Path, PathBuf};
use anyhow::Result;
use tracing::debug;

use crate::vault::{VaultExportSummary, export_obsidian_vault, sync_kv_summaries};

use super::MemoryManager;

impl MemoryManager {
    pub fn export_vault(&self, path: impl AsRef<Path>) -> Result<VaultExportSummary> {
        export_obsidian_vault(self.all(), path)
    }

    pub(super) fn sync_vault_projection(&self) -> Result<()> {
        if let Some(path) = &self.vault_path {
            // Full Obsidian note/index rebuild (incremental: only sub-dirs rebuilt).
            export_obsidian_vault(self.all(), path)?;
            // Write / update the three YAML KV summary files + MEMORY.md.
            // Uses SHA-256 checksums — unchanged files are not touched.
            let written = sync_kv_summaries(self.all(), path, self.kv_tier_limit)?;
            if written > 0 {
                debug!(files_written = written, path = %path.display(), "vault KV summaries updated");
            }
        }
        Ok(())
    }

}

pub(super) fn derive_default_vault_path(event_log_path: &Path) -> Option<PathBuf> {
    let file_name = event_log_path.file_name()?.to_string_lossy();
    if file_name != "events.jsonl" {
        return None;
    }

    let memory_dir = event_log_path.parent()?;
    if memory_dir.file_name()?.to_string_lossy() != "memory" {
        return None;
    }

    let root = memory_dir.parent()?;
    Some(root.join("vault"))
}
