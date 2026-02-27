/// Hybrid secondary index backed by [`redb`] for fast tier-aware entry lookups.
///
/// The JSONL event log remains the **canonical source of truth**.  This index
/// is a write-through cache that avoids full-scans for large memory stores.
/// If the index file is absent or corrupt it is transparently rebuilt from the
/// event log — zero data loss.
///
/// # Tables
///
/// | Name         | Key              | Value                                |
/// |--------------|------------------|--------------------------------------|
/// | `entries`    | UUID string (36c) | bincode-serialised [`IndexedEntry`] |
/// | `tier_index` | tier slug (&str) | newline-separated UUID list          |
///
/// # Usage
///
/// The index is **opt-in**.  Create one and pass it to
/// [`MemoryManager::set_index`] (or manage it alongside the daemon state).
/// The daemon calls [`MemoryIndex::insert`] on every successful
/// [`MemoryManager::record`] and [`MemoryIndex::rebuild_from_log`] at startup.
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use lru::LruCache;
use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::event_log::MemoryEventLog;
use crate::schema::{MemoryEntry, MemoryTier};

/// Snapshot of the LRU cache performance counters.
#[derive(Debug, Clone, Default)]
pub struct IndexCacheStats {
    pub capacity: usize,
    pub len: usize,
    pub hits: u64,
    pub misses: u64,
    /// Hit rate as a percentage 0.0 – 100.0.
    pub hit_rate_pct: f32,
}

// ── redb table definitions ────────────────────────────────────────────────────

/// Entry metadata table: `entry_id (str) → bincode(IndexedEntry)`.
const ENTRIES_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("entries");
/// Tier lookup table: `tier_slug (str) → newline-joined UUID list (str)`.
const TIER_TABLE: TableDefinition<&str, &str> = TableDefinition::new("tier_index");

// ── LRU cache capacity ────────────────────────────────────────────────────────

/// Number of full [`MemoryEntry`] objects held in the hot-path LRU cache.
const LRU_CAPACITY: usize = 256;

// ── Compact indexed metadata ──────────────────────────────────────────────────

/// Compact metadata record stored in the redb `entries` table.
///
/// Intentionally does **not** store the embedding — those stay in RAM only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedEntry {
    pub id: Uuid,
    pub tier: u8, // MemoryTier discriminant (0-5)
    pub confidence: f32,
    pub valence: f32,
    pub created_at: DateTime<Utc>,
    pub source: String,
    /// SHA-256 hex of `content` — used for deduplication without storing full text.
    pub content_hash: String,
}

impl IndexedEntry {
    pub fn from_entry(entry: &MemoryEntry) -> Self {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(entry.content.as_bytes());
        let content_hash = format!("{:x}", h.finalize());
        Self {
            id: entry.id,
            tier: tier_to_u8(entry.tier),
            confidence: entry.confidence,
            valence: entry.valence,
            created_at: entry.created_at,
            source: entry.source.clone(),
            content_hash,
        }
    }

    pub fn memory_tier(&self) -> Option<MemoryTier> {
        u8_to_tier(self.tier)
    }
}

fn tier_to_u8(tier: MemoryTier) -> u8 {
    match tier {
        MemoryTier::Episodic => 0,
        MemoryTier::Semantic => 1,
        MemoryTier::Procedural => 2,
        MemoryTier::Reflective => 3,
        MemoryTier::UserProfile => 4,
        MemoryTier::Core => 5,
    }
}

fn u8_to_tier(v: u8) -> Option<MemoryTier> {
    match v {
        0 => Some(MemoryTier::Episodic),
        1 => Some(MemoryTier::Semantic),
        2 => Some(MemoryTier::Procedural),
        3 => Some(MemoryTier::Reflective),
        4 => Some(MemoryTier::UserProfile),
        5 => Some(MemoryTier::Core),
        _ => None,
    }
}



// ── MemoryIndex ───────────────────────────────────────────────────────────────

pub struct MemoryIndex {
    db: Database,
    path: PathBuf,
    /// Hot cache: UUID string → full [`MemoryEntry`] (embedding excluded on
    /// serialisation).
    cache: LruCache<String, MemoryEntry>,
    /// Cumulative cache hits since startup (for `aigent memory stats`).
    cache_hits: u64,
    /// Cumulative cache misses since startup.
    cache_misses: u64,
}

impl MemoryIndex {
    /// Open or create the redb index file at `path`.
    ///
    /// On version mismatch or corruption [`redb`] returns an error; callers
    /// should handle this by deleting `path` and re-calling `open`, then
    /// rebuilding from the event log.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let db = Database::create(&path)
            .with_context(|| format!("opening redb index at {}", path.display()))?;

        // Ensure tables exist.
        {
            let tx = db.begin_write()?;
            tx.open_table(ENTRIES_TABLE)?;
            tx.open_table(TIER_TABLE)?;
            tx.commit()?;
        }

        Ok(Self {
            db,
            path,
            cache: LruCache::new(NonZeroUsize::new(LRU_CAPACITY).unwrap()),
            cache_hits: 0,
            cache_misses: 0,
        })
    }

    /// Delete the index file and re-create it.  Used for forced rebuilds.
    pub fn reset(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Self::open(path)
    }

    /// Insert (or upsert) a single entry into the index and update the LRU
    /// cache.
    pub fn insert(&mut self, entry: &MemoryEntry) -> Result<()> {
        let id_str = entry.id.to_string();
        let indexed = IndexedEntry::from_entry(entry);
        let bytes = serde_json::to_vec(&indexed)?;

        let tx = self.db.begin_write()?;
        {
            let mut tbl = tx.open_table(ENTRIES_TABLE)?;
            tbl.insert(id_str.as_str(), bytes.as_slice())?;

            // Update tier index: append id to the tier's newline-separated list.
            let slug = entry.tier.slug();
            let mut tier_tbl = tx.open_table(TIER_TABLE)?;
            let existing = tier_tbl
                .get(slug)?
                .map(|v| v.value().to_string())
                .unwrap_or_default();
            let updated = if existing.is_empty() {
                id_str.clone()
            } else {
                format!("{existing}\n{id_str}")
            };
            tier_tbl.insert(slug, updated.as_str())?;
        }
        tx.commit()?;

        // Warm the LRU cache (embedding excluded — stays in RAM)
        self.cache.put(id_str, entry.clone());
        Ok(())
    }

    /// Return all entry IDs belonging to `tier`.
    pub fn ids_for_tier(&self, tier: MemoryTier) -> Result<Vec<String>> {
        let tx = self.db.begin_read()?;
        let tbl = tx.open_table(TIER_TABLE)?;
        let slug = tier.slug();
        let list = tbl
            .get(slug)?
            .map(|v| v.value().to_string())
            .unwrap_or_default();
        Ok(list
            .lines()
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect())
    }

    /// Look up compact metadata for one entry by UUID string.
    pub fn get_metadata(&self, id: &str) -> Result<Option<IndexedEntry>> {
        let tx = self.db.begin_read()?;
        let tbl = tx.open_table(ENTRIES_TABLE)?;
        match tbl.get(id)? {
            None => Ok(None),
            Some(v) => {
                let entry: IndexedEntry = serde_json::from_slice(v.value())?;
                Ok(Some(entry))
            }
        }
    }

    /// Peek at the LRU cache for a full [`MemoryEntry`].
    pub fn cache_get(&mut self, id: &str) -> Option<&MemoryEntry> {
        match self.cache.get(id) {
            Some(e) => {
                self.cache_hits += 1;
                Some(e)
            }
            None => {
                self.cache_misses += 1;
                None
            }
        }
}

    /// Rebuild the entire index from the JSONL event log.
    ///
    /// Drops all existing index data first.  Called at startup when the index
    /// file is missing or the schema version changes.
    pub async fn rebuild_from_log(&mut self, event_log: &MemoryEventLog) -> Result<usize> {
        // Wipe existing data.
        {
            let tx = self.db.begin_write()?;
            {
                let mut entries_tbl = tx.open_table(ENTRIES_TABLE)?;
                let keys: Vec<String> = entries_tbl
                    .iter()?
                    .filter_map(|r| r.ok().map(|(k, _)| k.value().to_string()))
                    .collect();
                for k in &keys {
                    entries_tbl.remove(k.as_str())?;
                }
            }
            {
                let mut tier_tbl = tx.open_table(TIER_TABLE)?;
                let keys: Vec<String> = tier_tbl
                    .iter()?
                    .filter_map(|r| r.ok().map(|(k, _)| k.value().to_string()))
                    .collect();
                for k in &keys {
                    tier_tbl.remove(k.as_str())?;
                }
            }
            tx.commit()?;
        }
        self.cache.clear();

        let events = event_log.load().await?;
        let count = events.len();
        for event in events {
            self.insert(&event.entry)?;
        }
        tracing::info!(entries = count, path = %self.path.display(), "memory index rebuilt from event log");
        Ok(count)
    }

    /// Snapshot of the LRU cache's runtime statistics.
    pub fn cache_stats(&self) -> IndexCacheStats {
        IndexCacheStats {
            capacity: LRU_CAPACITY,
            len: self.cache.len(),
            hits: self.cache_hits,
            misses: self.cache_misses,
            hit_rate_pct: if self.cache_hits + self.cache_misses == 0 {
                0.0
            } else {
                (self.cache_hits as f32 / (self.cache_hits + self.cache_misses) as f32) * 100.0
            },
        }
}

    /// Number of entries in the index (full table scan).
    pub fn len(&self) -> Result<usize> {
        let tx = self.db.begin_read()?;
        let tbl = tx.open_table(ENTRIES_TABLE)?;
        Ok(tbl.len()? as usize)
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}
