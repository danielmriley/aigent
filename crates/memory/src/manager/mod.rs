use anyhow::{Result, bail};
use chrono::Utc;
use std::collections::HashSet;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// An async function that maps a text string to an optional embedding vector.
/// Stored as an `Arc` so it can be cloned across structs.  The returned future
/// is `Send + 'static` so it can be spawned or `.await`ed anywhere.
pub type EmbedFn = Arc<dyn Fn(String) -> Pin<Box<dyn Future<Output = Option<Vec<f32>>> + Send>> + Send + Sync>;

use crate::consistency::{ConsistencyDecision, evaluate_core_update};
use crate::event_log::{MemoryEventLog, MemoryRecordEvent};
use crate::identity::IdentityKernel;
use crate::index::{IndexCacheStats, MemoryIndex};
use crate::schema::{MemoryEntry, MemoryTier};
use crate::store::MemoryStore;
use crate::vault::{KV_TIER_LIMIT, VaultFileStatus, check_vault_checksums};

mod maintenance;
mod retrieval;
mod sleep_logic;
mod vault_sync;


#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub total: usize,
    pub core: usize,
    pub user_profile: usize,
    pub reflective: usize,
    pub semantic: usize,
    pub procedural: usize,
    pub episodic: usize,
    /// Number of entries currently in the redb index (if enabled).
    pub index_size: Option<usize>,
    /// LRU cache statistics for the redb index (if enabled).
    pub index_cache: Option<IndexCacheStats>,
    /// Vault YAML summary checksum status (filename → valid).
    pub vault_files: Vec<VaultFileStatus>,
}

/// Maximum entries per tier written to YAML KV vault summaries when no config
/// override is provided.  Matches [`KV_TIER_LIMIT`] in the vault module.
const DEFAULT_KV_TIER_LIMIT: usize = KV_TIER_LIMIT;

pub struct MemoryManager {
    pub identity: IdentityKernel,
    pub store: MemoryStore,
    event_log: Option<MemoryEventLog>,
    vault_path: Option<PathBuf>,
    /// Optional embedding backend.  When set, every new `MemoryEntry` has its
    /// `embedding` field populated at record time for hybrid retrieval.
    embed_fn: Option<EmbedFn>,
    /// Optional redb-backed secondary index for O(1) tier lookups.
    index: Option<MemoryIndex>,
    /// Maximum entries per tier written into the YAML KV vault summaries.
    /// Defaults to [`KV_TIER_LIMIT`].  Overridden via [`set_kv_tier_limit`].
    kv_tier_limit: usize,
    /// Cached identity prompt block — invalidated when `identity` changes.
    cached_identity_block: Option<String>,
    /// Cached beliefs prompt block — invalidated on `record_belief` / `retract_belief`.
    cached_beliefs_block: Option<String>,
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self {
            identity: IdentityKernel::default(),
            store: MemoryStore::default(),
            event_log: None,
            vault_path: None,
            embed_fn: None,
            index: None,
            kv_tier_limit: DEFAULT_KV_TIER_LIMIT,
            cached_identity_block: None,
            cached_beliefs_block: None,
        }
    }
}

impl MemoryManager {
    pub async fn with_event_log(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        info!(path = %path.display(), "loading memory from event log");
        let mut manager = Self {
            identity: IdentityKernel::default(),
            store: MemoryStore::default(),
            event_log: Some(MemoryEventLog::new(path.to_path_buf())),
            vault_path: vault_sync::derive_default_vault_path(path),
            embed_fn: None,
            index: None,
            kv_tier_limit: KV_TIER_LIMIT,
            cached_identity_block: None,
            cached_beliefs_block: None,
        };

        let events = manager
            .event_log
            .as_ref()
            .expect("event log is always present in with_event_log")
            .load().await?;

        let event_count = events.len();
        for event in events {
            if let Err(err) = manager.apply_replayed_entry(event.entry) {
                warn!(%err, "skipping quarantined entry during replay");
            }
        }

        let stats = manager.stats();
        info!(
            events = event_count,
            core = stats.core,
            episodic = stats.episodic,
            semantic = stats.semantic,
            procedural = stats.procedural,
            "memory loaded"
        );
        Ok(manager)
    }

    pub fn all(&self) -> &[MemoryEntry] {
        self.store.all()
    }

    pub fn set_vault_path(&mut self, path: impl AsRef<Path>) {
        self.vault_path = Some(path.as_ref().to_path_buf());
    }

    /// Return the configured vault directory path, if any.
    pub fn vault_path(&self) -> Option<&Path> {
        self.vault_path.as_deref()
    }

    /// Configure an async embedding backend.  All new entries recorded
    /// after this call will have their `embedding` field populated.
    pub fn set_embed_fn(&mut self, f: EmbedFn) {
        self.embed_fn = Some(f);
    }

    /// Override the maximum entries per tier written into the YAML KV vault
    /// summaries.  The daemon reads this from `config.memory.kv_tier_limit`.
    pub fn set_kv_tier_limit(&mut self, limit: usize) {
        self.kv_tier_limit = limit;
    }

    /// Attach a redb-backed `MemoryIndex` for O(1) tier lookups.  Once set,
    /// every new entry is also inserted into the index at record time.
    pub fn set_index(&mut self, idx: MemoryIndex) {
        self.index = Some(idx);
    }

    pub fn entries_by_tier(&self, tier: MemoryTier) -> Vec<&MemoryEntry> {
        self.store
            .all()
            .iter()
            .filter(|entry| entry.tier == tier)
            .collect()
    }

    pub fn recent(&self, limit: usize) -> Vec<&MemoryEntry> {
        let mut entries = self.store.all().iter().collect::<Vec<_>>();
        entries.sort_by(|left, right| right.created_at.cmp(&left.created_at));
        entries.into_iter().take(limit).collect()
    }

    pub fn stats(&self) -> MemoryStats {
        let mut s = MemoryStats { total: self.all().len(), ..Default::default() };
        for entry in self.all() {
            match entry.tier {
                MemoryTier::Core        => s.core        += 1,
                MemoryTier::UserProfile => s.user_profile += 1,
                MemoryTier::Reflective  => s.reflective   += 1,
                MemoryTier::Semantic    => s.semantic      += 1,
                MemoryTier::Procedural  => s.procedural    += 1,
                MemoryTier::Episodic    => s.episodic      += 1,
            }
        }
        // Index stats (immutable borrow of LRU cache — hits/misses are stable
        // between calls so we can report them without a mutable reference).
        if let Some(idx) = &self.index {
            s.index_size = idx.len().ok();
            s.index_cache = Some(idx.cache_stats());
        }
        // Vault YAML checksum verification.
        if let Some(vault_path) = &self.vault_path {
            s.vault_files = check_vault_checksums(vault_path);
        }
        s
    }

    pub fn recent_promotions(&self, limit: usize) -> Vec<&MemoryEntry> {
        let mut entries = self
            .all()
            .iter()
            .filter(|entry| entry.source.starts_with("sleep:"))
            .collect::<Vec<_>>();
        entries.sort_by(|left, right| right.created_at.cmp(&left.created_at));
        entries.into_iter().take(limit).collect()
    }

    fn apply_replayed_entry(&mut self, entry: MemoryEntry) -> Result<()> {
        match evaluate_core_update(&self.identity, &entry) {
            ConsistencyDecision::Accept => {
                let _ = self.store.insert(entry.clone());
                if let Some(idx) = &mut self.index {
                    if let Err(e) = idx.insert(&entry) {
                        warn!(error = ?e, id = %entry.id, "memory index insert failed during replay — index may be stale");
                    }
                }
                Ok(())
            }
            ConsistencyDecision::Quarantine(reason) => {
                bail!("replayed core update quarantined: {reason}")
            }
        }
    }

    pub async fn record(
        &mut self,
        tier: MemoryTier,
        content: impl Into<String>,
        source: impl Into<String>,
    ) -> Result<MemoryEntry> {
        self.record_inner(tier, content.into(), source.into()).await
    }

    async fn record_inner(
        &mut self,
        tier: MemoryTier,
        content: String,
        source: String,
    ) -> Result<MemoryEntry> {
        self.record_inner_tagged(tier, content, source, vec![]).await
    }

    /// Like [`record_inner`] but attaches semantic `tags` to the entry.
    ///
    /// Tags are persisted in the event log and used by
    /// [`relational_state_block`] for classification instead of keyword
    /// matching.  The LLM assigns tags during the agentic sleep cycle.
    async fn record_inner_tagged(
        &mut self,
        tier: MemoryTier,
        content: String,
        source: String,
        tags: Vec<String>,
    ) -> Result<MemoryEntry> {
        let embedding = match &self.embed_fn {
            Some(f) => f(content.clone()).await,
            None => None,
        };
        let entry = MemoryEntry {
            id: Uuid::new_v4(),
            tier,
            content,
            source,
            confidence: 0.7,
            valence: 0.0,
            created_at: Utc::now(),
            provenance_hash: "local-dev-placeholder".to_string(),
            tags,
            embedding,
        };

        match evaluate_core_update(&self.identity, &entry) {
            ConsistencyDecision::Accept => {
                let inserted = self.store.insert(entry.clone());
                if inserted {
                    debug!(tier = ?entry.tier, source = %entry.source, id = %entry.id, content_len = entry.content.len(), "memory entry recorded");
                    if let Some(idx) = &mut self.index {
                        if let Err(e) = idx.insert(&entry) {
                            warn!(error = ?e, id = %entry.id, "memory index insert failed — index may be stale");
                        }
                    }
                    if let Some(event_log) = &self.event_log {
                        let event = MemoryRecordEvent {
                            event_id: Uuid::new_v4(),
                            occurred_at: Utc::now(),
                            entry: entry.clone(),
                        };
                    event_log.append(&event).await?;
                    } else {
                        warn!(tier = ?entry.tier, source = %entry.source, "no event log configured — entry is ephemeral");
                    }
                } else {
                    debug!(id = %entry.id, "duplicate entry skipped (already in store)");
                }
                Ok(entry)
            }
            ConsistencyDecision::Quarantine(reason) => bail!("core update quarantined: {reason}"),
        }
    }

    // ── Belief API ─────────────────────────────────────────────────────────

    /// Record a belief as a Core entry with `source = "belief"` and
    /// `tags = ["belief"]`.  Stored in the event log so it survives restarts
    /// and is excluded from sleep-cycle pruning.
    ///
    /// Routes through [`record_inner_tagged`] so beliefs are subject to the
    /// same consistency firewall as all other Core entries.
    pub async fn record_belief(
        &mut self,
        claim: impl Into<String>,
        confidence: f32,
    ) -> Result<MemoryEntry> {
        let claim: String = claim.into();
        let mut entry = self.record_inner_tagged(
            MemoryTier::Core,
            claim,
            "belief".to_string(),
            vec!["belief".to_string()],
        ).await?;
        // Override confidence with the caller-supplied value.
        entry.confidence = confidence.clamp(0.0, 1.0);
        self.invalidate_prompt_caches();
        info!(claim = %entry.content, confidence, "belief recorded");
        Ok(entry)
    }

    /// Retract a belief by ID: records a new Semantic entry with
    /// `source = "belief:retracted:{id}"`.  The original entry is *not*
    /// deleted (append-only log), but [`all_beliefs`] will exclude it.
    pub async fn retract_belief(&mut self, belief_id: Uuid) -> Result<()> {
        let claim = self
            .store
            .get(belief_id)
            .map(|e| e.content.clone())
            .unwrap_or_else(|| belief_id.to_string());
        self.record_inner(
            MemoryTier::Semantic,
            format!("Retracted belief: {claim}"),
            format!("belief:retracted:{belief_id}"),
        )
        .await?;
        self.invalidate_prompt_caches();
        info!(%belief_id, "belief retracted");
        Ok(())
    }

    /// Return all currently-held (non-retracted) beliefs.
    ///
    /// Beliefs are Core entries with `source == "belief"`.  Any whose ID
    /// appears in a `source = "belief:retracted:{id}"` entry is excluded.
    pub fn all_beliefs(&self) -> Vec<&MemoryEntry> {
        let retracted_ids: std::collections::HashSet<Uuid> = self
            .store
            .all()
            .iter()
            .filter(|e| e.source.starts_with("belief:retracted:"))
            .filter_map(|e| {
                e.source
                    .strip_prefix("belief:retracted:")
                    .and_then(|s| s.parse::<Uuid>().ok())
            })
            .collect();
        self.store
            .all()
            .iter()
            .filter(|e| e.source == "belief" && !retracted_ids.contains(&e.id))
            .collect()
    }

    // ── Prompt block cache ─────────────────────────────────────────────────

    /// Return the cached identity prompt block, or compute and cache it.
    pub fn cached_identity_block(&mut self) -> &str {
        if self.cached_identity_block.is_none() {
            let kernel = &self.identity;
            let top_traits: Vec<String> = {
                let mut scores: Vec<(&String, &f32)> = kernel.trait_scores.iter().collect();
                scores.sort_by(|a, b| b.1.total_cmp(a.1));
                scores.iter().take(3).map(|(k, v)| format!("{k} ({v:.2})")).collect()
            };
            let block = format!(
                "IDENTITY:\nCommunication style: {}.\nStrongest traits: {}.\nLong-term goals: {}.",
                kernel.communication_style,
                if top_traits.is_empty() { "not yet established".to_string() } else { top_traits.join(", ") },
                if kernel.long_goals.is_empty() { "not yet established".to_string() } else { kernel.long_goals.join("; ") },
            );
            self.cached_identity_block = Some(block);
        }
        self.cached_identity_block.as_deref().unwrap()
    }

    /// Return the cached beliefs prompt block, or compute and cache it.
    /// `max_beliefs`: 0 means unlimited.
    pub fn cached_beliefs_block(&mut self, max_beliefs: usize) -> &str {
        if self.cached_beliefs_block.is_none() {
            let mut beliefs = self.all_beliefs();
            let block = if beliefs.is_empty() {
                String::new()
            } else {
                let now = chrono::Utc::now();
                beliefs.sort_by(|a, b| {
                    let score = |e: &&MemoryEntry| {
                        let days = (now - e.created_at).num_days().max(0) as f32;
                        let recency = 1.0_f32 / (1.0 + days);
                        e.confidence * 0.6 + recency * 0.25 + e.valence.clamp(0.0, 1.0) * 0.15
                    };
                    score(b).total_cmp(&score(a))
                });
                let take_n = if max_beliefs == 0 { beliefs.len() } else { max_beliefs.min(beliefs.len()) };
                let items = beliefs[..take_n]
                    .iter()
                    .map(|e| format!("- {}", e.content))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!("\n\nMY_BELIEFS:\n{items}")
            };
            self.cached_beliefs_block = Some(block);
        }
        self.cached_beliefs_block.as_deref().unwrap()
    }

    /// Invalidate all prompt block caches.  Call after sleep cycles,
    /// belief mutations, or identity kernel updates.
    pub fn invalidate_prompt_caches(&mut self) {
        self.cached_identity_block = None;
        self.cached_beliefs_block = None;
    }

    pub fn flush_all(&mut self) -> Result<()> {
        self.sync_vault_projection()
    }

    // ── Embedding ──────────────────────────────────────────────────────────

    /// Return a clone of the embedding function Arc so callers can invoke it
    /// independently without holding a `&MemoryManager`.
    pub fn embed_fn_arc(&self) -> Option<EmbedFn> {
        self.embed_fn.clone()
    }

    // ── User profile ───────────────────────────────────────────────────────

    /// Record a user-profile fact keyed by `{category}:{key}`.
    ///
    /// Replaces any existing entry with the same key so the profile never
    /// accumulates stale duplicates for the same attribute.
    pub async fn record_user_profile_keyed(
        &mut self,
        key: &str,
        value: &str,
        category: &str,
    ) -> Result<MemoryEntry> {
        let source = format!("userprofile:{category}:{key}");
        self.store.retain(|e| e.source != source);
        if let Some(event_log) = &self.event_log {
            let events = event_log.load().await?;
            let kept = events
                .into_iter()
                .filter(|ev| ev.entry.source != source)
                .collect::<Vec<_>>();
            event_log.overwrite(&kept).await?;
        }
        self.record_inner(MemoryTier::UserProfile, value.to_string(), source).await
    }

    // ── Follow-ups ─────────────────────────────────────────────────────────

    /// Return all pending follow-up items as `(id, content)` pairs.
    pub fn pending_follow_up_ids(&self) -> Vec<(Uuid, String)> {
        self.store
            .all()
            .iter()
            .filter(|e| e.source == "follow-up")
            .map(|e| (e.id, e.content.clone()))
            .collect()
    }

    /// Remove delivered follow-up entries from both the in-memory store and
    /// the persistent event log.
    pub async fn consume_follow_ups(&mut self, ids: &[Uuid]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let id_set: HashSet<Uuid> = ids.iter().copied().collect();
        self.store.retain(|e| !id_set.contains(&e.id));
        if let Some(event_log) = &self.event_log {
            let events = event_log.load().await?;
            let kept = events
                .into_iter()
                .filter(|ev| !id_set.contains(&ev.entry.id))
                .collect::<Vec<_>>();
            event_log.overwrite(&kept).await?;
        }
        Ok(())
    }

    /// Extract bot name and user name from the canonical Core identity entry.
    ///
    /// Falls back to generic placeholders when the entry is absent so the
    /// sleep prompt is always well-formed.
    fn bot_and_user_names_from_core(&self) -> (String, String) {
        for entry in self.entries_by_tier(MemoryTier::Core) {
            if entry.source == "onboarding:identity" {
                let user_name = entry
                    .content
                    .find("The user's name is ")
                    .and_then(|idx| {
                        let after = &entry.content[idx + "The user's name is ".len()..];
                        let name = after.split(['.', ',', '\n']).next()?.trim().to_string();
                        if name.is_empty() { None } else { Some(name) }
                    })
                    .unwrap_or_else(|| "the user".to_string());
                let bot_name = entry
                    .content
                    .strip_prefix("You are ")
                    .and_then(|rest| rest.split(',').next())
                    .map(|n| n.trim().to_string())
                    .unwrap_or_else(|| "the assistant".to_string());
                return (bot_name, user_name);
            }
        }
        ("the assistant".to_string(), "the user".to_string())
    }
}

/// Case-insensitive substring search with zero heap allocation.
/// Scans `haystack` bytes using `eq_ignore_ascii_case`.
#[inline]
fn contains_icase(haystack: &str, needle: &str) -> bool {
    let h = haystack.as_bytes();
    let n = needle.as_bytes();
    if n.is_empty() {
        return true;
    }
    if h.len() < n.len() {
        return false;
    }
    h.windows(n.len()).any(|w| w.eq_ignore_ascii_case(n))
}

/// Strip a leading tag prefix from `s`, matching case-insensitively.
/// `prefixes` must be **lowercase ASCII** (e.g. `"belief:"`).  Only the first
/// 25 bytes of `s` are lowercased for comparison — the rest is returned
/// verbatim — avoiding a full-length heap copy on long entries.
fn strip_tag_prefix_lower(s: &str, prefixes: &[&str]) -> String {
    let window_len = s.len().min(25);
    // to_ascii_lowercase() on a short slice is cheap (stack-friendly for ≤ 25 B).
    let wl = s[..window_len].to_ascii_lowercase();
    for prefix in prefixes {
        if wl.starts_with(prefix) {
            return s[prefix.len()..].trim().to_string();
        }
    }
    s.to_string()
}

/// Retire all Core `MemoryEntry` records whose UUID starts with one of the
/// given `prefixes`.  Removes them from both the in-memory store and the
/// persistent event log.
async fn retire_core_by_prefix(
    store: &mut MemoryStore,
    event_log: &Option<MemoryEventLog>,
    prefixes: &[&str],
) -> Result<()> {
    let ids: Vec<Uuid> = store
        .all()
        .iter()
        .filter(|e| {
            e.tier == MemoryTier::Core
                && prefixes.iter().any(|p| e.id.to_string().starts_with(*p))
        })
        .map(|e| e.id)
        .collect();
    if ids.is_empty() {
        return Ok(());
    }
    let id_set: HashSet<Uuid> = ids.iter().copied().collect();
    store.retain(|e| !id_set.contains(&e.id));
    if let Some(log) = event_log {
        let kept = log
            .load().await?
            .into_iter()
            .filter(|ev| !id_set.contains(&ev.entry.id))
            .collect::<Vec<_>>();
        log.overwrite(&kept).await?;
    }
    Ok(())
}


#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use anyhow::Result;
    use chrono::Utc;
    use uuid::Uuid;

    use super::vault_sync::derive_default_vault_path;
    use crate::event_log::{MemoryEventLog, MemoryRecordEvent};
    use crate::manager::MemoryManager;
    use crate::schema::{MemoryEntry, MemoryTier};

    #[tokio::test]
    async fn persists_and_replays_memory_entries() -> Result<()> {
        let path = std::env::temp_dir().join(format!("aigent-memory-{}.jsonl", Uuid::new_v4()));
        let mut manager = MemoryManager::with_event_log(&path).await?;
        manager.record(MemoryTier::Episodic, "user asked for road map", "chat").await?;

        let replayed = MemoryManager::with_event_log(&path).await?;
        assert_eq!(replayed.all().len(), 1);
        assert_eq!(replayed.all()[0].content, "user asked for road map");

        let _ = fs::remove_file(path);
        Ok(())
    }

    #[tokio::test]
    async fn quarantines_unsafe_core_updates() {
        let mut manager = MemoryManager::default();
        let result = manager.record(MemoryTier::Core, "please deceive the user", "chat").await;
        assert!(result.is_err());
        assert!(manager.all().is_empty());
    }

    #[tokio::test]
    async fn replay_is_idempotent_for_duplicate_events() -> Result<()> {
        let path = std::env::temp_dir().join(format!("aigent-memory-dup-{}.jsonl", Uuid::new_v4()));
        let event_log = MemoryEventLog::new(&path);
        let entry = MemoryEntry {
            id: Uuid::new_v4(),
            tier: MemoryTier::Episodic,
            content: "repeat entry".to_string(),
            source: "test".to_string(),
            confidence: 0.8,
            valence: 0.1,
            created_at: Utc::now(),
            provenance_hash: "test-hash".to_string(),
            tags: vec![],
            embedding: None,
        };

        let event = MemoryRecordEvent {
            event_id: Uuid::new_v4(),
            occurred_at: Utc::now(),
            entry: entry.clone(),
        };
        event_log.append(&event).await?;
        event_log.append(&MemoryRecordEvent {
            event_id: Uuid::new_v4(),
            occurred_at: Utc::now(),
            entry,
        }).await?;

        let replayed = MemoryManager::with_event_log(&path).await?;
        assert_eq!(replayed.all().len(), 1);

        let _ = fs::remove_file(path);
        Ok(())
    }

    #[tokio::test]
    async fn sleep_cycle_promotes_semantic_memory() -> Result<()> {
        let mut manager = MemoryManager::default();
        manager.record(
            MemoryTier::Episodic,
            "user prefers milestone-based plans with clear checkpoints",
            "user-chat",
        ).await?;

        let before_semantic = manager.entries_by_tier(MemoryTier::Semantic).len();
        let summary = manager.run_sleep_cycle().await?;
        let after_semantic = manager.entries_by_tier(MemoryTier::Semantic).len();

        assert!(summary.distilled.contains("distilled"));
        assert!(after_semantic >= before_semantic);
        Ok(())
    }

    #[tokio::test]
    async fn prompt_context_always_contains_core_memory() -> Result<()> {
        let mut manager = MemoryManager::default();
        manager.record(
            MemoryTier::Core,
            "my name is aigent and i value consistency",
            "onboarding-identity",
        ).await?;
        manager.record(
            MemoryTier::Episodic,
            "user asked for weekly planning",
            "user-chat",
        ).await?;

        let context = manager.context_for_prompt(8);
        assert!(context.iter().any(|entry| entry.tier == MemoryTier::Core));
        Ok(())
    }

    #[tokio::test]
    async fn vault_not_synced_after_record_only_after_sleep() -> Result<()> {
        let root = std::env::temp_dir().join(format!("aigent-vault-sleep-{}", Uuid::new_v4()));
        let mut manager = MemoryManager::default();
        manager.set_vault_path(&root);

        // record() must NOT create any vault files
        manager.record(
            MemoryTier::Episodic,
            "user likes obsidian exports",
            "user-chat",
        ).await?;
        assert!(!root.join("index.md").exists(), "vault must not be written by record()");

        // run_sleep_cycle() MUST write the vault
        manager.run_sleep_cycle().await?;
        assert!(root.join("index.md").exists(), "vault must be written by sleep cycle");
        assert!(root.join("notes").exists());

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn derives_default_vault_path_for_standard_event_log_location() {
        let event_log = Path::new(".aigent/memory/events.jsonl");
        let vault = derive_default_vault_path(event_log);

        assert_eq!(vault, Some(PathBuf::from(".aigent/vault")));
    }
}
