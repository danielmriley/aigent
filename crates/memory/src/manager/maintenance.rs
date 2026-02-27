//! Forgetting passes, deduplication, and cleanup for [`MemoryManager`].

use std::collections::HashSet;
use anyhow::Result;
use chrono::{Duration, Utc};
use tracing::info;
use uuid::Uuid;

use crate::schema::MemoryTier;

use super::MemoryManager;

impl MemoryManager {
    pub async fn wipe_all(&mut self) -> Result<usize> {
        let removed = self.store.len();
        self.store.clear();

        if let Some(event_log) = &self.event_log {
            event_log.overwrite(&[]).await?;
        }

        self.sync_vault_projection()?;

        Ok(removed)
    }

    pub async fn wipe_tiers(&mut self, tiers: &[MemoryTier]) -> Result<usize> {
        if tiers.is_empty() {
            return Ok(0);
        }

        let removed = self.store.retain(|entry| !tiers.contains(&entry.tier));

        if let Some(event_log) = &self.event_log {
            let events = event_log.load()?;
            let kept = events
                .into_iter()
                .filter(|event| !tiers.contains(&event.entry.tier))
                .collect::<Vec<_>>();
            event_log.overwrite(&kept).await?;
        }

        self.sync_vault_projection()?;

        Ok(removed)
    }

    // ── Lightweight forgetting ─────────────────────────────────────────────

    /// Remove Episodic entries that are older than `forget_after_days` days
    /// **and** have confidence below `min_confidence`.
    ///
    /// Call this after [`run_sleep_cycle`] when
    /// `config.memory.forget_episodic_after_days > 0`.
    ///
    /// Returns the number of entries removed.
    pub fn run_forgetting_pass(&mut self, forget_after_days: u64, min_confidence: f32) -> usize {
        if forget_after_days == 0 {
            return 0;
        }
        let cutoff = Utc::now() - Duration::days(forget_after_days as i64);
        let before = self.store.len();
        self.store.retain(|e| {
            // Keep if wrong tier, too recent, or confidence is high enough.
            e.tier != MemoryTier::Episodic
                || e.created_at > cutoff
                || e.confidence >= min_confidence
        });
        let removed = before.saturating_sub(self.store.len());
        if removed > 0 {
            info!(
                removed,
                forget_after_days,
                min_confidence,
                "lightweight forgetting: pruned stale episodic entries"
            );
        }
        removed
    }

    // ── Content deduplication ──────────────────────────────────────────────

    /// Remove content-duplicate entries from the in-memory store **and** the
    /// persistent event log.
    ///
    /// For every group of entries that share the same `(tier, normalised_content)`,
    /// the **newest** entry (by `created_at`) is kept and all older copies are
    /// purged.  Returns the number of entries removed.
    ///
    /// This is called automatically at the start of every sleep cycle and can
    /// also be triggered manually via `/dedup` or the daemon API.
    pub async fn deduplicate_by_content(&mut self) -> Result<usize> {
        let dupe_ids = self.store.find_content_duplicates();
        if dupe_ids.is_empty() {
            return Ok(0);
        }

        let id_set: HashSet<Uuid> = dupe_ids.iter().copied().collect();
        let removed = self.store.retain(|e| !id_set.contains(&e.id));

        // Purge corresponding event-log entries so duplicates don't reappear
        // on the next daemon restart.
        if let Some(event_log) = &self.event_log {
            let events = event_log.load()?;
            let kept = events
                .into_iter()
                .filter(|ev| !id_set.contains(&ev.entry.id))
                .collect::<Vec<_>>();
            event_log.overwrite(&kept).await?;
        }

        if removed > 0 {
            info!(removed, "content deduplication: purged duplicate entries");
        }
        Ok(removed)
    }

    // ── Compaction ─────────────────────────────────────────────────────────

    /// Remove Episodic entries older than `max_age_days` days from both the
    /// in-memory store and the persistent event log.
    ///
    /// Returns the number of entries removed.
    pub async fn compact_episodic(&mut self, max_age_days: i64) -> Result<usize> {
        let cutoff = Utc::now() - chrono::Duration::days(max_age_days);
        let to_remove: Vec<Uuid> = self
            .entries_by_tier(MemoryTier::Episodic)
            .iter()
            .filter(|e| e.created_at < cutoff)
            .map(|e| e.id)
            .collect();
        if to_remove.is_empty() {
            return Ok(0);
        }
        let id_set: HashSet<Uuid> = to_remove.iter().copied().collect();
        self.store.retain(|e| !id_set.contains(&e.id));
        if let Some(event_log) = &self.event_log {
            let kept = event_log
                .load()?
                .into_iter()
                .filter(|ev| !id_set.contains(&ev.entry.id))
                .collect::<Vec<_>>();
            event_log.overwrite(&kept).await?;
        }
        Ok(to_remove.len())
    }

}
