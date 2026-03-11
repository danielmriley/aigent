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
use tracing::warn;
use uuid::Uuid;

use crate::event_log::MemoryEventLog;
use crate::schema::{BeliefKind, EdgeKind, MemoryEntry, MemoryTier};

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

/// Entry metadata table: `entry_id (str) → json(IndexedEntry)`.
const ENTRIES_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("entries");
/// Tier lookup table: `tier_slug (str) → newline-joined UUID list (str)`.
const TIER_TABLE: TableDefinition<&str, &str> = TableDefinition::new("tier_index");

// ── Phase-1 belief-graph tables ───────────────────────────────────────────────

/// Node registry: `entry_id (str) → json(NodeRegistryEntry)`.
///
/// Every belief UUID (Active or historical) has exactly one record here.
/// Forwarding pointers live here too; resolving a consolidated UUID chains
/// through registry lookups until an Active record is found.
const NODE_REGISTRY_TABLE: TableDefinition<&str, &[u8]> =
    TableDefinition::new("node_registry");

/// Forward adjacency: `"{source_uuid}:{edge_kind_slug}" (str) → newline-joined target UUIDs (str)`.
const FWD_ADJ_TABLE: TableDefinition<&str, &str> = TableDefinition::new("fwd_adj");

/// Reverse adjacency: `"{target_uuid}:{edge_kind_slug}" (str) → newline-joined source UUIDs (str)`.
///
/// Used for orphan-prevention checks (O(1) lookup) and dampened confidence
/// propagation without a full graph scan.
const REV_ADJ_TABLE: TableDefinition<&str, &str> = TableDefinition::new("rev_adj");

/// Confidence checkpoints: `entry_id (str) → json(ConfidenceCheckpoint)`.
///
/// Periodic snapshots of the computed confidence so startup only needs to
/// replay events *after* the checkpoint rather than the entire log.
const CONF_CHECKPOINTS_TABLE: TableDefinition<&str, &[u8]> =
    TableDefinition::new("conf_checkpoints");

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

// ── Node lifecycle ────────────────────────────────────────────────────────────

/// Lifecycle state of a belief node.  Transitions are strictly one-way.
///
/// ```text
/// Active → Consolidated (absorbed by a higher-tier belief)
/// Active → Decayed      (confidence reached 0.0)
/// Active → Archived     (aged past hot window, no edges, low signal)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum NodeState {
    /// Present in the working store and redb index; eligible for retrieval
    /// and confidence updates.  All newly written beliefs start here.
    #[default]
    Active,
    /// Absorbed by a higher-tier belief during sleep consolidation.
    /// A forwarding pointer in [`NodeRegistryEntry::forwarding_to`] redirects
    /// future lookups to the canonical replacement.
    Consolidated,
    /// Confidence reached 0.0.  Removed from working store and redb index.
    /// Graph edges marked inactive in the reverse adjacency table.
    Decayed,
    /// Old episodic belief aged past the archival window without consolidating
    /// or decaying.  Kept in redb as a non-queryable historical record.
    Archived,
}

/// Compact per-node record in the redb node registry.
///
/// This is not a duplication of `IndexedEntry` — it tracks the *lifecycle and
/// graph position* of the belief, whereas `IndexedEntry` tracks retrieval
/// metadata (tier, confidence snapshot, source, content hash).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeRegistryEntry {
    pub id: Uuid,
    pub belief_kind: BeliefKind,
    pub state: NodeState,
    /// If `state == Consolidated`, points to the canonical replacement UUID.
    /// Follow the chain until an `Active` node is found.  Cycles are impossible
    /// because consolidation always produces a higher-tier belief.
    pub forwarding_to: Option<Uuid>,
    /// The `confidence` field of the original `MemoryEntry` at write time.
    /// All `ConfidenceUpdateEvent` deltas are added on top of this anchor.
    pub initial_confidence: f32,
    /// `MemoryTier` discriminant (0-5) at the time of last state transition.
    pub tier_at_last_state: u8,
    pub created_at: DateTime<Utc>,
    /// Last time this node was accessed by retrieval or received a confidence
    /// update.  Used by Pass 1 (stale decay) to decide which beliefs are stale.
    pub last_accessed_at: DateTime<Utc>,
}

/// Periodic confidence snapshot stored in the `conf_checkpoints` table.
///
/// On startup the index loads the most recent checkpoint per UUID, then
/// replays only the `ConfidenceUpdateEvent` records that occurred after
/// `after_event_seq`.  This keeps startup O(recent events) rather than O(log).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceCheckpoint {
    pub id: Uuid,
    /// Computed confidence at the time of this snapshot.
    pub confidence: f32,
    /// Monotonic event log sequence number after which this snapshot was taken.
    /// Events at indices > this value must be replayed to bring confidence
    /// up to date.
    pub after_event_seq: u64,
    pub created_at: DateTime<Utc>,
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
        let db = match Database::create(&path) {
            Ok(db) => db,
            Err(err) => {
                // The index file may be corrupt or from an incompatible redb
                // version.  Delete it and re-create a fresh empty index; it
                // will be repopulated on the next write.  The event log is the
                // source of truth, so no data is lost.
                warn!(
                    ?err,
                    path = %path.display(),
                    "redb index failed to open — deleting and recreating (will \
                     repopulate from event log on next write)"
                );
                if path.exists() {
                    std::fs::remove_file(&path)
                        .with_context(|| format!("removing corrupt redb index at {}", path.display()))?;
                }
                Database::create(&path)
                    .with_context(|| format!("recreating redb index at {}", path.display()))?
            }
        };

        // Ensure all tables exist (idempotent — redb creates them if absent).
        {
            let tx = db.begin_write()?;
            tx.open_table(ENTRIES_TABLE)?;
            tx.open_table(TIER_TABLE)?;
            tx.open_table(NODE_REGISTRY_TABLE)?;
            tx.open_table(FWD_ADJ_TABLE)?;
            tx.open_table(REV_ADJ_TABLE)?;
            tx.open_table(CONF_CHECKPOINTS_TABLE)?;
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
    /// cache.  Deduplicates by ID: if the entry already exists in a tier
    /// list, the old reference is removed before inserting the new one.
    pub fn insert(&mut self, entry: &MemoryEntry) -> Result<()> {
        let id_str = entry.id.to_string();
        let indexed = IndexedEntry::from_entry(entry);
        let bytes = serde_json::to_vec(&indexed)?;

        let tx = self.db.begin_write()?;
        {
            let mut tbl = tx.open_table(ENTRIES_TABLE)?;
            tbl.insert(id_str.as_str(), bytes.as_slice())?;

            // Update tier index: append id to the tier's newline-separated list,
            // but first strip any existing occurrence to avoid duplicates.
            let slug = entry.tier.slug();
            let mut tier_tbl = tx.open_table(TIER_TABLE)?;
            let existing = tier_tbl
                .get(slug)?
                .map(|v| v.value().to_string())
                .unwrap_or_default();
            let deduped: Vec<&str> = existing
                .lines()
                .filter(|line| !line.is_empty() && *line != id_str)
                .collect();
            let mut updated = deduped.join("\n");
            if updated.is_empty() {
                updated = id_str.clone();
            } else {
                updated = format!("{updated}\n{id_str}");
            }
            tier_tbl.insert(slug, updated.as_str())?;
        }
        tx.commit()?;

        // Warm the LRU cache (embedding excluded — stays in RAM)
        self.cache.put(id_str, entry.clone());
        Ok(())
    }

    /// Remove a single entry by UUID from the entries table, all tier lists,
    /// and the LRU cache.
    pub fn remove(&mut self, id: &Uuid) -> Result<()> {
        let id_str = id.to_string();
        let tx = self.db.begin_write()?;
        {
            let mut tbl = tx.open_table(ENTRIES_TABLE)?;
            tbl.remove(id_str.as_str())?;

            // Remove from all tier lists (we don't know which tier it belonged
            // to after deletion, so scan all slugs).
            let mut tier_tbl = tx.open_table(TIER_TABLE)?;
            let slugs: Vec<String> = tier_tbl
                .iter()?
                .filter_map(|r| r.ok().map(|(k, _)| k.value().to_string()))
                .collect();
            for slug in &slugs {
                let list_opt = tier_tbl
                    .get(slug.as_str())?
                    .map(|g| g.value().to_string());
                if let Some(list) = list_opt {
                    let filtered: Vec<&str> = list
                        .lines()
                        .filter(|line| !line.is_empty() && *line != id_str)
                        .collect();
                    tier_tbl.insert(slug.as_str(), filtered.join("\n").as_str())?;
                }
            }
        }
        tx.commit()?;
        self.cache.pop(&id_str);
        Ok(())
    }

    /// Remove all entries from the index, clearing both tables and the cache.
    pub fn clear(&mut self) -> Result<()> {
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
        self.cache.clear();
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

    // ── Node registry ─────────────────────────────────────────────────────────

    /// Insert or replace a [`NodeRegistryEntry`] for the given UUID.
    pub fn upsert_node_registry(&mut self, entry: &NodeRegistryEntry) -> Result<()> {
        let id_str = entry.id.to_string();
        let bytes = serde_json::to_vec(entry)?;
        let tx = self.db.begin_write()?;
        {
            let mut tbl = tx.open_table(NODE_REGISTRY_TABLE)?;
            tbl.insert(id_str.as_str(), bytes.as_slice())?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Look up the [`NodeRegistryEntry`] for a UUID.  O(log n) redb read.
    pub fn get_node_registry(&self, id: &Uuid) -> Result<Option<NodeRegistryEntry>> {
        let id_str = id.to_string();
        let tx = self.db.begin_read()?;
        let tbl = tx.open_table(NODE_REGISTRY_TABLE)?;
        match tbl.get(id_str.as_str())? {
            None => Ok(None),
            Some(v) => Ok(Some(serde_json::from_slice(v.value())?)),
        }
    }

    /// Transition a node to a new state.  Writes a forwarding pointer when
    /// `new_state == Consolidated`.  Idempotent if called with the same state.
    pub fn set_node_state(
        &mut self,
        id: &Uuid,
        new_state: NodeState,
        forwarding_to: Option<Uuid>,
    ) -> Result<()> {
        let id_str = id.to_string();
        // Read first, then write — avoids holding an immutable borrow while
        // taking a mutable one on the same redb table handle.
        let updated_bytes: Option<Vec<u8>> = {
            let rtx = self.db.begin_read()?;
            let rtbl = rtx.open_table(NODE_REGISTRY_TABLE)?;
            match rtbl.get(id_str.as_str())? {
                None => None,
                Some(v) => {
                    let mut entry: NodeRegistryEntry = serde_json::from_slice(v.value())?;
                    entry.state = new_state;
                    entry.forwarding_to = forwarding_to;
                    Some(serde_json::to_vec(&entry)?)
                }
            }
        };
        if let Some(bytes) = updated_bytes {
            let tx = self.db.begin_write()?;
            {
                let mut tbl = tx.open_table(NODE_REGISTRY_TABLE)?;
                tbl.insert(id_str.as_str(), bytes.as_slice())?;
            }
            tx.commit()?;
        }
        Ok(())
    }

    /// Resolve a UUID through any chain of forwarding pointers to the current
    /// canonical Active node.  Returns `id` unchanged if already Active or if
    /// no registry entry exists.  Bounded by the number of memory tiers (≤6).
    pub fn resolve_forwarding(&self, id: &Uuid) -> Result<Uuid> {
        let mut current = *id;
        // Max chain depth = number of tiers (6); iterate with a safety cap.
        for _ in 0..8 {
            match self.get_node_registry(&current)? {
                Some(entry) if entry.state == NodeState::Consolidated => {
                    match entry.forwarding_to {
                        Some(next) => current = next,
                        None => break, // Consolidated but no pointer — return as-is.
                    }
                }
                _ => break,
            }
        }
        Ok(current)
    }

    // ── Belief graph adjacency ─────────────────────────────────────────────────

    /// Add a directed edge of `edge_kind` from `source_id` to `target_id`,
    /// updating both the forward and reverse adjacency tables.  Idempotent.
    pub fn add_edge(
        &mut self,
        source_id: &Uuid,
        edge_kind: EdgeKind,
        target_id: &Uuid,
    ) -> Result<()> {
        let fwd_key = format!("{}:{}", source_id, edge_kind.slug());
        let rev_key = format!("{}:{}", target_id, edge_kind.slug());
        let src_str = source_id.to_string();
        let tgt_str = target_id.to_string();

        let tx = self.db.begin_write()?;
        {
            let mut fwd = tx.open_table(FWD_ADJ_TABLE)?;
            let existing = fwd.get(fwd_key.as_str())?.map(|v| v.value().to_string()).unwrap_or_default();
            if !existing.lines().any(|l| l == tgt_str) {
                let updated = if existing.is_empty() { tgt_str.clone() } else { format!("{existing}\n{tgt_str}") };
                fwd.insert(fwd_key.as_str(), updated.as_str())?;
            }

            let mut rev = tx.open_table(REV_ADJ_TABLE)?;
            let existing = rev.get(rev_key.as_str())?.map(|v| v.value().to_string()).unwrap_or_default();
            if !existing.lines().any(|l| l == src_str) {
                let updated = if existing.is_empty() { src_str.clone() } else { format!("{existing}\n{src_str}") };
                rev.insert(rev_key.as_str(), updated.as_str())?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Return all target UUIDs reachable from `source_id` via `edge_kind`.
    pub fn forward_edges(&self, source_id: &Uuid, edge_kind: EdgeKind) -> Result<Vec<Uuid>> {
        let key = format!("{}:{}", source_id, edge_kind.slug());
        let tx = self.db.begin_read()?;
        let tbl = tx.open_table(FWD_ADJ_TABLE)?;
        let list = tbl.get(key.as_str())?.map(|v| v.value().to_string()).unwrap_or_default();
        Ok(list.lines().filter(|s| !s.is_empty()).filter_map(|s| s.parse().ok()).collect())
    }

    /// Return all source UUIDs that point *to* `target_id` via `edge_kind`.
    /// Used by orphan-prevention and dampened confidence propagation.
    pub fn reverse_edges(&self, target_id: &Uuid, edge_kind: EdgeKind) -> Result<Vec<Uuid>> {
        let key = format!("{}:{}", target_id, edge_kind.slug());
        let tx = self.db.begin_read()?;
        let tbl = tx.open_table(REV_ADJ_TABLE)?;
        let list = tbl.get(key.as_str())?.map(|v| v.value().to_string()).unwrap_or_default();
        Ok(list.lines().filter(|s| !s.is_empty()).filter_map(|s| s.parse().ok()).collect())
    }

    // ── Confidence checkpoints ─────────────────────────────────────────────────

    /// Write a confidence checkpoint for a UUID after processing events up to
    /// `after_event_seq`.  Overwrites any previous checkpoint for this UUID.
    pub fn write_confidence_checkpoint(
        &mut self,
        id: &Uuid,
        confidence: f32,
        after_event_seq: u64,
    ) -> Result<()> {
        let id_str = id.to_string();
        let checkpoint = ConfidenceCheckpoint {
            id: *id,
            confidence,
            after_event_seq,
            created_at: chrono::Utc::now(),
        };
        let bytes = serde_json::to_vec(&checkpoint)?;
        let tx = self.db.begin_write()?;
        {
            let mut tbl = tx.open_table(CONF_CHECKPOINTS_TABLE)?;
            tbl.insert(id_str.as_str(), bytes.as_slice())?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Retrieve the most recent confidence checkpoint for a UUID, if any.
    pub fn get_confidence_checkpoint(&self, id: &Uuid) -> Result<Option<ConfidenceCheckpoint>> {
        let id_str = id.to_string();
        let tx = self.db.begin_read()?;
        let tbl = tx.open_table(CONF_CHECKPOINTS_TABLE)?;
        match tbl.get(id_str.as_str())? {
            None => Ok(None),
            Some(v) => Ok(Some(serde_json::from_slice(v.value())?)),
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
        let now = chrono::Utc::now();
        for event in events {
            self.insert(&event.entry)?;
            // Seed the node registry for all existing entries as Active.
            // Entries written before Phase 1 will default to BeliefKind::Empirical.
            let reg = NodeRegistryEntry {
                id: event.entry.id,
                belief_kind: event.entry.belief_kind,
                state: NodeState::Active,
                forwarding_to: None,
                initial_confidence: event.entry.confidence,
                tier_at_last_state: tier_to_u8(event.entry.tier),
                created_at: event.entry.created_at,
                last_accessed_at: now,
            };
            self.upsert_node_registry(&reg)?;
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

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use uuid::Uuid;

    use super::*;
    use crate::schema::{BeliefKind, EdgeKind, MemoryTier};

    fn temp_index_path() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("aigent-index-test-{}.redb", Uuid::new_v4()))
    }

    fn make_node(id: Uuid) -> NodeRegistryEntry {
        NodeRegistryEntry {
            id,
            belief_kind: BeliefKind::Empirical,
            state: NodeState::Active,
            forwarding_to: None,
            initial_confidence: 0.5,
            tier_at_last_state: 0, // Episodic
            created_at: Utc::now(),
            last_accessed_at: Utc::now(),
        }
    }

    // ── NodeState transitions stored and retrieved ────────────────────────────

    #[test]
    fn node_registry_insert_and_retrieve() {
        let path = temp_index_path();
        let mut idx = MemoryIndex::open(&path).unwrap();
        let id = Uuid::new_v4();
        let node = make_node(id);
        idx.upsert_node_registry(&node).unwrap();
        let retrieved = idx.get_node_registry(&id).unwrap().expect("node must exist");
        assert_eq!(retrieved.id, id);
        assert_eq!(retrieved.state, NodeState::Active);
        assert_eq!(retrieved.belief_kind, BeliefKind::Empirical);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn node_state_transition_active_to_consolidated() {
        let path = temp_index_path();
        let mut idx = MemoryIndex::open(&path).unwrap();
        let id = Uuid::new_v4();
        let canonical = Uuid::new_v4();
        idx.upsert_node_registry(&make_node(id)).unwrap();
        idx.set_node_state(&id, NodeState::Consolidated, Some(canonical)).unwrap();
        let node = idx.get_node_registry(&id).unwrap().unwrap();
        assert_eq!(node.state, NodeState::Consolidated);
        assert_eq!(node.forwarding_to, Some(canonical));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn node_state_transition_active_to_decayed() {
        let path = temp_index_path();
        let mut idx = MemoryIndex::open(&path).unwrap();
        let id = Uuid::new_v4();
        idx.upsert_node_registry(&make_node(id)).unwrap();
        idx.set_node_state(&id, NodeState::Decayed, None).unwrap();
        let node = idx.get_node_registry(&id).unwrap().unwrap();
        assert_eq!(node.state, NodeState::Decayed);
        assert_eq!(node.forwarding_to, None);
        let _ = std::fs::remove_file(&path);
    }

    // ── Forwarding pointer resolution ─────────────────────────────────────────

    #[test]
    fn resolve_forwarding_follows_chain() {
        // A (Consolidated → B) → B (Consolidated → C) → C (Active)
        let path = temp_index_path();
        let mut idx = MemoryIndex::open(&path).unwrap();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();
        idx.upsert_node_registry(&make_node(a)).unwrap();
        idx.upsert_node_registry(&make_node(b)).unwrap();
        idx.upsert_node_registry(&make_node(c)).unwrap();
        idx.set_node_state(&a, NodeState::Consolidated, Some(b)).unwrap();
        idx.set_node_state(&b, NodeState::Consolidated, Some(c)).unwrap();
        // c remains Active
        let resolved = idx.resolve_forwarding(&a).unwrap();
        assert_eq!(resolved, c, "resolve_forwarding must follow the chain to the Active canonical");
        // Resolving b directly should also reach c.
        assert_eq!(idx.resolve_forwarding(&b).unwrap(), c);
        // Resolving c directly should return c unchanged.
        assert_eq!(idx.resolve_forwarding(&c).unwrap(), c);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn resolve_forwarding_unknown_uuid_returns_itself() {
        let path = temp_index_path();
        let idx = MemoryIndex::open(&path).unwrap();
        let unknown = Uuid::new_v4();
        // No registry entry — should return the UUID unchanged.
        assert_eq!(idx.resolve_forwarding(&unknown).unwrap(), unknown);
        let _ = std::fs::remove_file(&path);
    }

    // ── Adjacency table insert and retrieve ───────────────────────────────────

    #[test]
    fn add_edge_and_retrieve_forward_and_reverse() {
        let path = temp_index_path();
        let mut idx = MemoryIndex::open(&path).unwrap();
        let source = Uuid::new_v4();
        let target = Uuid::new_v4();
        idx.add_edge(&source, EdgeKind::Supports, &target).unwrap();
        let fwd = idx.forward_edges(&source, EdgeKind::Supports).unwrap();
        assert_eq!(fwd, vec![target], "forward edge must be retrievable");
        let rev = idx.reverse_edges(&target, EdgeKind::Supports).unwrap();
        assert_eq!(rev, vec![source], "reverse edge must be retrievable");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn add_edge_is_idempotent() {
        let path = temp_index_path();
        let mut idx = MemoryIndex::open(&path).unwrap();
        let source = Uuid::new_v4();
        let target = Uuid::new_v4();
        idx.add_edge(&source, EdgeKind::Contradicts, &target).unwrap();
        idx.add_edge(&source, EdgeKind::Contradicts, &target).unwrap();
        let fwd = idx.forward_edges(&source, EdgeKind::Contradicts).unwrap();
        assert_eq!(fwd.len(), 1, "duplicate edge insert must not create duplicate entries");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn edges_are_per_edge_kind() {
        let path = temp_index_path();
        let mut idx = MemoryIndex::open(&path).unwrap();
        let source = Uuid::new_v4();
        let target1 = Uuid::new_v4();
        let target2 = Uuid::new_v4();
        idx.add_edge(&source, EdgeKind::Supports, &target1).unwrap();
        idx.add_edge(&source, EdgeKind::Contradicts, &target2).unwrap();
        assert_eq!(idx.forward_edges(&source, EdgeKind::Supports).unwrap(), vec![target1]);
        assert_eq!(idx.forward_edges(&source, EdgeKind::Contradicts).unwrap(), vec![target2]);
        assert!(idx.forward_edges(&source, EdgeKind::DerivedFrom).unwrap().is_empty());
        let _ = std::fs::remove_file(&path);
    }

    // ── Confidence checkpoints ────────────────────────────────────────────────

    #[test]
    fn confidence_checkpoint_write_and_read() {
        let path = temp_index_path();
        let mut idx = MemoryIndex::open(&path).unwrap();
        let id = Uuid::new_v4();
        idx.write_confidence_checkpoint(&id, 0.73, 42).unwrap();
        let cp = idx.get_confidence_checkpoint(&id).unwrap().expect("checkpoint must exist");
        assert_eq!(cp.id, id);
        assert!((cp.confidence - 0.73).abs() < f32::EPSILON);
        assert_eq!(cp.after_event_seq, 42);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn confidence_checkpoint_overwrite_replaces_previous() {
        let path = temp_index_path();
        let mut idx = MemoryIndex::open(&path).unwrap();
        let id = Uuid::new_v4();
        idx.write_confidence_checkpoint(&id, 0.50, 10).unwrap();
        idx.write_confidence_checkpoint(&id, 0.65, 20).unwrap();
        let cp = idx.get_confidence_checkpoint(&id).unwrap().unwrap();
        assert!((cp.confidence - 0.65).abs() < f32::EPSILON, "checkpoint must hold latest value");
        assert_eq!(cp.after_event_seq, 20);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn get_confidence_checkpoint_returns_none_for_unknown() {
        let path = temp_index_path();
        let idx = MemoryIndex::open(&path).unwrap();
        let result = idx.get_confidence_checkpoint(&Uuid::new_v4()).unwrap();
        assert!(result.is_none());
        let _ = std::fs::remove_file(&path);
    }

    // ── All six tables are created on fresh open ──────────────────────────────

    #[test]
    fn all_tables_created_on_fresh_database() {
        // A freshly opened index must have all six tables usable.
        // We verify this by performing a read on each new table to check they exist.
        let path = temp_index_path();
        let idx = MemoryIndex::open(&path).unwrap();
        // Node registry: empty lookup should return None, not an error.
        assert!(idx.get_node_registry(&Uuid::new_v4()).unwrap().is_none());
        // Adjacency: empty edge lookups should return empty vecs, not errors.
        let src = Uuid::new_v4();
        assert!(idx.forward_edges(&src, EdgeKind::Supports).unwrap().is_empty());
        assert!(idx.reverse_edges(&src, EdgeKind::Supports).unwrap().is_empty());
        // Confidence checkpoints: unknown UUID should return None.
        assert!(idx.get_confidence_checkpoint(&Uuid::new_v4()).unwrap().is_none());
        let _ = std::fs::remove_file(&path);
    }

    // ── NodeRegistryEntry belief_kind preserved ───────────────────────────────

    #[test]
    fn node_registry_preserves_belief_kind() {
        let path = temp_index_path();
        let mut idx = MemoryIndex::open(&path).unwrap();
        let id = Uuid::new_v4();
        let mut node = make_node(id);
        node.belief_kind = BeliefKind::Opinion;
        idx.upsert_node_registry(&node).unwrap();
        let retrieved = idx.get_node_registry(&id).unwrap().unwrap();
        assert_eq!(retrieved.belief_kind, BeliefKind::Opinion);
        let _ = std::fs::remove_file(&path);
    }
}
