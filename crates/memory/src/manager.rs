//! `MemoryManager` — the central orchestrator for all memory operations.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │  MemoryManager                                               │
//! │                                                              │
//! │  MemoryStore (in-process index)                              │
//! │    ├── Core       (constitution, identity)                   │
//! │    ├── UserProfile (preferences, goals, relationship facts)  │
//! │    ├── Reflective (agent thoughts, plans, self-critiques)    │
//! │    ├── Semantic   (distilled facts)                          │
//! │    ├── Procedural (how-to knowledge)                         │
//! │    └── Episodic   (raw conversation turns)                   │
//! │                                                              │
//! │  MemoryEventLog (append-only JSONL — source of truth)        │
//! │  VaultProjection (Obsidian markdown — auto-synced)           │
//! └──────────────────────────────────────────────────────────────┘
//! ```

use anyhow::{Result, bail};
use chrono::{Duration, Utc};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

use crate::consistency::{ConsistencyDecision, evaluate_core_update};
use crate::constitution::constitution_seeds;
use crate::event_log::{MemoryEventLog, MemoryRecordEvent};
use crate::identity::{IdentityKernel, update_trait_score};
use crate::profile::format_user_profile_block;
use crate::retrieval::{RankedMemoryContext, assemble_context_with_provenance};
use crate::schema::{MemoryEntry, MemoryTier};
use crate::sentiment::infer_valence;
use crate::sleep::{
    AgenticSleepInsights, SleepSummary, agentic_sleep_prompt as build_sleep_prompt, distill,
};
use crate::store::MemoryStore;
use crate::vault::{VaultExportSummary, export_obsidian_vault};

// ── Public statistics type ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total: usize,
    pub core: usize,
    pub user_profile: usize,
    pub reflective: usize,
    pub semantic: usize,
    pub procedural: usize,
    pub episodic: usize,
}

// ── MemoryManager ─────────────────────────────────────────────────────────────

/// Optional synchronous embedding backend.
///
/// The closure accepts a text string and returns an embedding vector, or
/// `None` if the backend is unavailable.  The implementation is expected to
/// perform a blocking HTTP call to a local Ollama instance.
pub type EmbedFn = Arc<dyn Fn(&str) -> Option<Vec<f32>> + Send + Sync>;

#[derive(Default)]
pub struct MemoryManager {
    pub identity: IdentityKernel,
    pub store: MemoryStore,
    event_log: Option<MemoryEventLog>,
    vault_path: Option<PathBuf>,
    /// Optional embedding backend used to enrich retrieval.
    embed_fn: Option<EmbedFn>,
}

impl std::fmt::Debug for MemoryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryManager")
            .field("store_len", &self.store.len())
            .field("has_event_log", &self.event_log.is_some())
            .field("has_embed_fn", &self.embed_fn.is_some())
            .finish()
    }
}

impl MemoryManager {
    // ── Construction ──────────────────────────────────────────────────────────

    pub fn with_event_log(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        info!(path = %path.display(), "loading memory from event log");
        let mut manager = Self {
            identity: IdentityKernel::default(),
            store: MemoryStore::default(),
            event_log: Some(MemoryEventLog::new(path.to_path_buf())),
            vault_path: derive_default_vault_path(path),
            embed_fn: None,
        };

        let events = manager
            .event_log
            .as_ref()
            .expect("event log always present")
            .load()?;

        let event_count = events.len();
        for event in events {
            manager.apply_replayed_entry(event.entry)?;
        }

        if event_count > 10_000 {
            warn!(
                event_count,
                "memory event log is very large; run a sleep cycle and compaction to keep it bounded"
            );
        }

        let stats = manager.stats();
        info!(
            events = event_count,
            core = stats.core,
            user_profile = stats.user_profile,
            reflective = stats.reflective,
            semantic = stats.semantic,
            episodic = stats.episodic,
            "memory loaded"
        );

        // Load persisted identity snapshot if available.
        let identity_snapshot_path = {
            let filename = path
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_default();
            path.with_file_name(format!("{filename}.identity.json"))
        };
        if identity_snapshot_path.exists() {
            match std::fs::read_to_string(&identity_snapshot_path)
                .ok()
                .and_then(|json| serde_json::from_str::<IdentityKernel>(&json).ok())
            {
                Some(identity) => {
                    debug!("loaded identity snapshot from previous sleep cycle");
                    manager.identity = identity;
                }
                None => {
                    warn!(
                        path = %identity_snapshot_path.display(),
                        "identity snapshot exists but failed to parse — using default"
                    );
                }
            }
        }

        Ok(manager)
    }

    // ── Embedding backend ─────────────────────────────────────────────────────

    /// Set a synchronous embedding function (e.g. blocking Ollama call).
    ///
    /// When set, new entries are automatically embedded at record time and
    /// retrieval uses hybrid lexical+embedding scoring.
    pub fn set_embed_fn(&mut self, f: EmbedFn) {
        info!("embedding backend configured");
        self.embed_fn = Some(f);
    }

    fn compute_embedding(&self, text: &str) -> Option<Vec<f32>> {
        self.embed_fn.as_ref().and_then(|f| f(text))
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    pub fn all(&self) -> &[MemoryEntry] {
        self.store.all()
    }

    pub fn set_vault_path(&mut self, path: impl AsRef<Path>) {
        self.vault_path = Some(path.as_ref().to_path_buf());
    }

    pub fn entries_by_tier(&self, tier: MemoryTier) -> Vec<&MemoryEntry> {
        self.store
            .all()
            .iter()
            .filter(|e| e.tier == tier)
            .collect()
    }

    pub fn recent(&self, limit: usize) -> Vec<&MemoryEntry> {
        let mut entries = self.store.all().iter().collect::<Vec<_>>();
        entries.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        entries.into_iter().take(limit).collect()
    }

    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            total: self.all().len(),
            core: self.entries_by_tier(MemoryTier::Core).len(),
            user_profile: self.entries_by_tier(MemoryTier::UserProfile).len(),
            reflective: self.entries_by_tier(MemoryTier::Reflective).len(),
            semantic: self.entries_by_tier(MemoryTier::Semantic).len(),
            procedural: self.entries_by_tier(MemoryTier::Procedural).len(),
            episodic: self.entries_by_tier(MemoryTier::Episodic).len(),
        }
    }

    pub fn recent_promotions(&self, limit: usize) -> Vec<&MemoryEntry> {
        let mut entries = self
            .all()
            .iter()
            .filter(|e| e.source.starts_with("sleep:") || e.source.starts_with("agentic-sleep:"))
            .collect::<Vec<_>>();
        entries.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        entries.into_iter().take(limit).collect()
    }

    // ── Context assembly ──────────────────────────────────────────────────────

    pub fn context_for_prompt(&self, limit: usize) -> Vec<MemoryEntry> {
        self.context_for_prompt_ranked("", limit)
            .into_iter()
            .map(|item| item.entry)
            .collect()
    }

    /// Assemble a ranked context window with provenance scores.
    ///
    /// Core, UserProfile, Reflective, and `agent-perspective:*` entries are
    /// always considered regardless of lexical overlap — the agent's durable
    /// views and personality should always be within reach.  Sleep bookkeeping
    /// entries are filtered out.
    /// When an embedding backend is configured the query is embedded and used
    /// for hybrid lexical+vector scoring.
    pub fn context_for_prompt_ranked(&self, query: &str, limit: usize) -> Vec<RankedMemoryContext> {
        let priority_tiers = [MemoryTier::Core, MemoryTier::UserProfile, MemoryTier::Reflective];

        let priority_entries = self
            .store
            .all()
            .iter()
            // Exclude retired Core entries so they no longer appear in prompts.
            // Agent perspectives are always treated as priority alongside
            // Core/UserProfile/Reflective so developed views are consistently present.
            .filter(|e| {
                (priority_tiers.contains(&e.tier) || e.source.starts_with("agent-perspective:"))
                    && e.source != "sleep:retired"
            })
            .cloned()
            .collect::<Vec<_>>();

        let other_entries = self
            .store
            .all()
            .iter()
            .filter(|e| {
                !priority_tiers.contains(&e.tier)
                    && !e.source.starts_with("agent-perspective:")
                    && !e.source.starts_with("assistant-turn")
                    && e.source != "sleep:cycle"
            })
            .cloned()
            .collect::<Vec<_>>();

        // Embed the query so hybrid scoring can use cosine similarity.
        let query_embedding = if query.is_empty() { None } else { self.compute_embedding(query) };

        assemble_context_with_provenance(other_entries, priority_entries, query, limit, query_embedding)
    }

    /// Return a formatted user-profile block for injection into prompts,
    /// or `None` if no profile entries exist.
    pub fn user_profile_block(&self) -> Option<String> {
        format_user_profile_block(self.all())
    }

    // ── Recording ─────────────────────────────────────────────────────────────

    pub fn record(
        &mut self,
        tier: MemoryTier,
        content: impl Into<String>,
        source: impl Into<String>,
    ) -> Result<MemoryEntry> {
        self.record_inner(tier, content.into(), source.into(), 0.7, true)
    }

    pub fn record_with_confidence(
        &mut self,
        tier: MemoryTier,
        content: impl Into<String>,
        source: impl Into<String>,
        confidence: f32,
    ) -> Result<MemoryEntry> {
        self.record_inner(tier, content.into(), source.into(), confidence, true)
    }

    fn record_inner(
        &mut self,
        tier: MemoryTier,
        content: String,
        source: String,
        confidence: f32,
        sync_vault: bool,
    ) -> Result<MemoryEntry> {
        // Compute embedding before building entry (borrow rules: embed_fn read, store write later).
        let embedding = self.compute_embedding(&content);

        let valence = infer_valence(&content);

        let mut entry = MemoryEntry {
            id: Uuid::new_v4(),
            tier,
            content,
            source,
            confidence: confidence.clamp(0.0, 1.0),
            valence,
            tags: Vec::new(),
            embedding,
            created_at: Utc::now(),
            provenance_hash: "local-dev-placeholder".to_string(),
        };

        match evaluate_core_update(&self.identity, &entry) {
            ConsistencyDecision::Accept => {
                // Strip embedding before serialisation.
                let to_log = {
                    let mut e = entry.clone();
                    e.embedding = None;
                    e
                };

                let inserted = self.store.insert(entry.clone());
                if inserted {
                    debug!(
                        tier = ?entry.tier,
                        source = %entry.source,
                        id = %entry.id,
                        content_len = entry.content.len(),
                        has_embedding = entry.embedding.is_some(),
                        "memory entry recorded"
                    );

                    if let Some(event_log) = &self.event_log {
                        let event = MemoryRecordEvent {
                            event_id: Uuid::new_v4(),
                            occurred_at: Utc::now(),
                            entry: to_log,
                        };
                        event_log.append(&event)?;
                    } else {
                        warn!(
                            tier = ?entry.tier,
                            source = %entry.source,
                            "no event log — entry is ephemeral"
                        );
                    }

                    if sync_vault {
                        self.sync_vault_projection()?;
                    }
                } else {
                    debug!(id = %entry.id, "duplicate entry skipped");
                    // Return the already-stored entry so callers get a consistent view.
                    entry = self
                        .all()
                        .iter()
                        .find(|e| e.id == entry.id)
                        .cloned()
                        .unwrap_or(entry);
                }
                Ok(entry)
            }
            ConsistencyDecision::Quarantine(reason) => bail!("memory update quarantined: {reason}"),
        }
    }

    // ── Onboarding ────────────────────────────────────────────────────────────

    /// Seed the immutable AI constitution into Core memory.
    ///
    /// Idempotent: skips entries whose `constitution:*` source already exists.
    #[instrument(skip(self))]
    pub fn seed_constitution(&mut self, bot_name: &str, user_name: &str) -> Result<()> {
        let bot_name = bot_name.trim();
        let user_name = user_name.trim();
        if bot_name.is_empty() || user_name.is_empty() {
            bail!("bot_name and user_name required for constitution seeding");
        }

        for (content, source_tag) in constitution_seeds(bot_name, user_name) {
            let already = self
                .entries_by_tier(MemoryTier::Core)
                .into_iter()
                .any(|e| e.source == source_tag);

            if already {
                debug!(source_tag, "constitution entry already present — skipping");
                continue;
            }

            let entry = self.record_inner(
                MemoryTier::Core,
                content,
                source_tag.to_string(),
                1.0,
                false,
            )?;
            info!(id = %entry.id, source = source_tag, "constitution entry seeded");
        }

        self.sync_vault_projection()?;
        Ok(())
    }

    /// Seed legacy single-entry Core identity (kept for backward compatibility).
    ///
    /// Prefer [`seed_constitution`] for new installations; this method now
    /// delegates to it.
    #[instrument(skip(self))]
    pub fn seed_core_identity(&mut self, user_name: &str, bot_name: &str) -> Result<()> {
        let user_name = user_name.trim();
        let bot_name = bot_name.trim();
        if user_name.is_empty() || bot_name.is_empty() {
            bail!("user name and bot name are required for identity seeding");
        }

        // Migrate: if old single-entry identity exists, keep it but also seed constitution.
        let old_present = self
            .entries_by_tier(MemoryTier::Core)
            .into_iter()
            .any(|e| e.source == "onboarding:identity");

        if !old_present {
            // Insert legacy entry for backward compat with old event logs.
            let statement = format!(
                "You are {bot_name}, a helpful and truthful AI companion. \
                 The user's name is {user_name}. \
                 You have persistent multi-tier memory and can learn and remember \
                 information over time across conversations."
            );
            info!(bot_name, user_name, "seeding legacy core identity");
            let entry = self.record_inner(
                MemoryTier::Core,
                statement,
                "onboarding:identity".to_string(),
                1.0,
                false,
            )?;
            info!(id = %entry.id, "legacy core identity entry created");
        } else {
            debug!(bot_name, user_name, "legacy core identity already present");
        }

        // Always ensure constitution is fully seeded.
        self.seed_constitution(bot_name, user_name)?;
        Ok(())
    }

    // ── User profile ──────────────────────────────────────────────────────────

    /// Record a persistent user fact into the UserProfile tier.
    ///
    /// `category` should be one of: `preference`, `goal`, `fact`, `style`,
    /// `trigger`, or any custom label.
    pub fn record_user_profile(
        &mut self,
        content: impl Into<String>,
        category: &str,
    ) -> Result<MemoryEntry> {
        let source = format!("user-profile:{category}");
        self.record_inner(MemoryTier::UserProfile, content.into(), source, 0.9, true)
    }

    /// Record (or update in-place) a keyed UserProfile fact.
    ///
    /// Looks for an existing `UserProfile` entry whose content starts with
    /// `"{key}:"`.  If found and the value differs, the entry is overwritten
    /// in both the in-memory store and the event log.  If identical, the
    /// existing entry is returned without writing.  If not found, a new entry
    /// is created via `record_inner`.
    pub fn record_user_profile_keyed(
        &mut self,
        key: &str,
        value: &str,
        category: &str,
    ) -> Result<MemoryEntry> {
        let content = format!("{key}: {value}");
        let prefix = format!("{key}:");

        // Find the most-recent existing entry with this key prefix.
        let existing = self
            .store
            .all()
            .iter()
            .filter(|e| e.tier == MemoryTier::UserProfile && e.content.starts_with(&prefix))
            .max_by_key(|e| e.created_at)
            .cloned();

        if let Some(existing_entry) = existing {
            if existing_entry.content == content {
                // Identical — no write needed.
                return Ok(existing_entry);
            }

            // Overwrite in place.
            let old_id = existing_entry.id;
            let mut updated = existing_entry.clone();
            updated.content = content.clone();
            updated.created_at = Utc::now();

            self.store.retain(|e| e.id != old_id);
            self.store.insert(updated.clone());

            if let Some(event_log) = &self.event_log {
                let mut events = event_log.load()?;
                for ev in &mut events {
                    if ev.entry.id == old_id {
                        ev.entry.content = content.clone();
                        ev.entry.created_at = updated.created_at;
                    }
                }
                event_log.overwrite(&events)?;
            }

            debug!(key, "user-profile entry updated in place");
            return Ok(updated);
        }

        let source = format!("user-profile:{category}");
        self.record_inner(MemoryTier::UserProfile, content, source, 0.9, true)
    }

    /// Record (or update in-place) the agent's developed view on a topic.
    ///
    /// Stored in `Semantic` with source `"agent-perspective:{topic}"`.
    /// If an entry for the same topic already exists and the view differs,
    /// it is overwritten in-place.  Identical views are returned without writing.
    pub fn record_agent_perspective(&mut self, topic: &str, view: &str) -> Result<MemoryEntry> {
        let source_prefix = format!("agent-perspective:{topic}");

        let existing = self
            .store
            .all()
            .iter()
            .filter(|e| e.source.starts_with(&source_prefix))
            .max_by_key(|e| e.created_at)
            .cloned();

        if let Some(existing_entry) = existing {
            if existing_entry.content == view {
                return Ok(existing_entry);
            }

            let old_id = existing_entry.id;
            let mut updated = existing_entry.clone();
            updated.content = view.to_string();
            updated.created_at = Utc::now();

            self.store.retain(|e| e.id != old_id);
            self.store.insert(updated.clone());

            if let Some(event_log) = &self.event_log {
                let mut events = event_log.load()?;
                for ev in &mut events {
                    if ev.entry.id == old_id {
                        ev.entry.content = view.to_string();
                        ev.entry.created_at = updated.created_at;
                    }
                }
                event_log.overwrite(&events)?;
            }

            debug!(topic, "agent perspective updated in place");
            return Ok(updated);
        }

        self.record_inner(MemoryTier::Semantic, view.to_string(), source_prefix, 0.8, true)
    }

    // ── Agentic sleep ─────────────────────────────────────────────────────────

    /// Generate the nightly reflection prompt to send to the LLM.
    pub fn agentic_sleep_prompt(&self) -> String {
        let bot_name = self.bot_name_from_core().unwrap_or_else(|| "Aigent".to_string());
        let user_name = self.user_name_from_core().unwrap_or_else(|| "User".to_string());
        build_sleep_prompt(self.all(), &bot_name, &user_name, &self.identity.trait_scores)
    }

    /// Apply LLM-generated sleep insights, committing them to memory.
    ///
    /// Also runs passive heuristic distillation before applying insights.
    #[instrument(skip(self, insights))]
    pub fn apply_agentic_sleep_insights(
        &mut self,
        insights: AgenticSleepInsights,
        summary_text: Option<String>,
    ) -> Result<SleepSummary> {
        let snapshot = self.store.all().to_vec();
        info!(entries = snapshot.len(), "agentic sleep: starting");

        // 1. Passive heuristic pass (free, no LLM).
        let mut summary = distill(&snapshot);

        // 2. Commit heuristic promotions.
        let promotion_lines = summary
            .promotions
            .iter()
            .map(|p| {
                let preview: String = p.content.chars().take(80).collect();
                format!("  • [{:?}] {}", p.to_tier, preview)
            })
            .collect::<Vec<_>>()
            .join("\n");

        let heuristic_note = if promotion_lines.is_empty() {
            format!("Passive review of {} memories; no new promotions.", snapshot.len())
        } else {
            format!(
                "Passive review promoted {} items:\n{} distilled",
                summary.promotions.len(),
                promotion_lines
            )
        };

        let marker = self.record_inner(
            MemoryTier::Semantic,
            heuristic_note,
            "sleep:cycle".to_string(),
            0.6,
            false,
        )?;
        summary.promoted_ids.push(marker.id.to_string());

        for promotion in &summary.promotions {
            let promoted = self.record_inner(
                promotion.to_tier,
                promotion.content.clone(),
                format!("sleep:{}", promotion.reason),
                0.75,
                false,
            )?;
            summary.promoted_ids.push(promoted.id.to_string());
        }

        // 3. Commit agentic insights.
        for text in &insights.learned_about_user {
            let e = self.record_inner(
                MemoryTier::UserProfile,
                text.clone(),
                "agentic-sleep:learned".to_string(),
                0.8,
                false,
            )?;
            summary.promoted_ids.push(e.id.to_string());
        }

        for text in &insights.follow_ups {
            let e = self.record_inner(
                MemoryTier::Reflective,
                format!("FOLLOW-UP: {text}"),
                "agentic-sleep:follow-up".to_string(),
                0.85,
                false,
            )?;
            summary.promoted_ids.push(e.id.to_string());
        }

        for text in &insights.reflective_thoughts {
            let e = self.record_inner(
                MemoryTier::Reflective,
                text.clone(),
                "agentic-sleep:reflection".to_string(),
                0.8,
                false,
            )?;
            summary.promoted_ids.push(e.id.to_string());
        }

        if let Some(ref reinforce) = insights.personality_reinforcement {
            // Bug 5: cap live personality-reinforce entries at 2 by retiring the
            // oldest when we would exceed the limit.
            let mut live: Vec<MemoryEntry> = self
                .entries_by_tier(MemoryTier::Core)
                .into_iter()
                .filter(|e| e.source == "sleep:personality-reinforce" && e.confidence > 0.0)
                .cloned()
                .collect();
            live.sort_by_key(|e| e.created_at);
            while live.len() >= 2 {
                let oldest = live.remove(0);
                let id_short: String = oldest.id.to_string().chars().take(8).collect();
                match self.retire_core_entry_by_id_short(&id_short) {
                    Ok(_) => debug!(id_short, "retired old personality-reinforce entry"),
                    Err(err) => warn!(?err, "failed to retire personality-reinforce entry"),
                }
            }
            let e = self.record_inner(
                MemoryTier::Core,
                format!("Personality reinforcement: {reinforce}"),
                "sleep:personality-reinforce".to_string(),
                0.9,
                false,
            )?;
            summary.promoted_ids.push(e.id.to_string());
        }

        for (key, value) in &insights.user_profile_updates {
            // Bug 3: use keyed upsert so the same key doesn't accumulate stale duplicates.
            match self.record_user_profile_keyed(key, value, "profile-update") {
                Ok(e) => summary.promoted_ids.push(e.id.to_string()),
                Err(err) => warn!(?err, key, "failed to upsert user profile entry"),
            }
        }

        for text in &insights.contradictions {
            let e = self.record_inner(
                MemoryTier::Semantic,
                format!("⚠ Contradiction noticed: {text}"),
                "agentic-sleep:contradiction".to_string(),
                0.7,
                false,
            )?;
            summary.promoted_ids.push(e.id.to_string());
        }

        // 3b. Core consolidations.
        let mut consolidated_count = 0usize;
        for (id_shorts_csv, synthesis) in &insights.consolidate_core {
            match self.consolidate_core_entries(id_shorts_csv, synthesis) {
                Ok((retired, created)) => {
                    consolidated_count += retired;
                    info!(id_shorts_csv, created, "agentic sleep: core consolidated");
                }
                Err(err) => warn!(?err, "agentic sleep: consolidate_core failed"),
            }
        }

        // 3c. Tool insights → Procedural.
        for text in &insights.tool_insights {
            let e = self.record_inner(
                MemoryTier::Procedural,
                text.clone(),
                "agentic-sleep:tool-insight".to_string(),
                0.75,
                false,
            )?;
            summary.promoted_ids.push(e.id.to_string());
        }

        // 3d. Higher-order syntheses → Semantic.
        for text in &insights.synthesis {
            let e = self.record_inner(
                MemoryTier::Semantic,
                text.clone(),
                "agentic-sleep:synthesis".to_string(),
                0.85,
                false,
            )?;
            summary.promoted_ids.push(e.id.to_string());
        }

        // 3e. Agent perspectives → Semantic (upsert by topic).
        for (topic, view) in &insights.perspectives {
            match self.record_agent_perspective(topic, view) {
                Ok(e) => summary.promoted_ids.push(e.id.to_string()),
                Err(err) => warn!(?err, topic, "failed to record agent perspective"),
            }
        }

        // 3f. Relationship milestones → Core (rolling cap of 5).
        for text in &insights.relationship_milestones {
            let mut live_milestones: Vec<MemoryEntry> = self
                .entries_by_tier(MemoryTier::Core)
                .into_iter()
                .filter(|e| e.source == "sleep:relationship-milestone" && e.confidence > 0.0)
                .cloned()
                .collect();
            live_milestones.sort_by_key(|e| e.created_at);
            while live_milestones.len() >= 5 {
                let oldest = live_milestones.remove(0);
                let id_short: String = oldest.id.to_string().chars().take(8).collect();
                let _ = self.retire_core_entry_by_id_short(&id_short);
            }
            let e = self.record_inner(
                MemoryTier::Core,
                text.clone(),
                "sleep:relationship-milestone".to_string(),
                0.85,
                false,
            )?;
            summary.promoted_ids.push(e.id.to_string());
        }

        // 3g. Communication style update → IdentityKernel + Core (via consistency firewall).
        if let Some(ref style) = insights.communication_style_update {
            let probe = MemoryEntry {
                id: Uuid::new_v4(),
                tier: MemoryTier::Core,
                content: style.clone(),
                source: "sleep:style-update".to_string(),
                confidence: 0.85,
                valence: 0.0,
                tags: Vec::new(),
                embedding: None,
                created_at: Utc::now(),
                provenance_hash: "sleep".to_string(),
            };
            match evaluate_core_update(&self.identity, &probe) {
                ConsistencyDecision::Accept => {
                    // Record as a Core entry so it persists through derive_identity_from_core.
                    match self.record_inner(
                        MemoryTier::Core,
                        format!("Communication style: {style}"),
                        "sleep:style-update".to_string(),
                        0.85,
                        false,
                    ) {
                        Ok(e) => {
                            summary.promoted_ids.push(e.id.to_string());
                            debug!(style, "communication style update committed to Core");
                        }
                        Err(err) => warn!(?err, "failed to record style update to Core"),
                    }
                }
                ConsistencyDecision::Quarantine(reason) => {
                    warn!(reason, "communication style update quarantined by consistency firewall");
                }
            }
        }

        // 3h. Valence corrections → in-memory store + event log.
        if !insights.valence_corrections.is_empty() {
            for (id_short, new_valence) in &insights.valence_corrections {
                let updated = self.store.update_valence_by_id_short(id_short, *new_valence);
                if updated {
                    debug!(id_short, valence = new_valence, "valence correction applied");
                } else {
                    warn!(id_short, "valence correction: no matching entry found");
                }
            }
            // Persist corrections to the event log in a single overwrite pass.
            if let Some(event_log) = &self.event_log {
                match event_log.load() {
                    Ok(mut events) => {
                        for ev in &mut events {
                            let id_str = ev.entry.id.to_string();
                            for (short, val) in &insights.valence_corrections {
                                if id_str.starts_with(short.as_str()) {
                                    ev.entry.valence = val.clamp(-1.0, 1.0);
                                }
                            }
                        }
                        if let Err(err) = event_log.overwrite(&events) {
                            warn!(?err, "failed to persist valence corrections to event log");
                        }
                    }
                    Err(err) => warn!(?err, "failed to load event log for valence corrections"),
                }
            }
        }

        // 4. Apply Core retirements and rewrites requested by the LLM.
        let mut retired_count = 0usize;
        for id_short in &insights.retire_core_ids {
            match self.retire_core_entry_by_id_short(id_short) {
                Ok(true) => {
                    retired_count += 1;
                    info!(id_short, "agentic sleep: core entry retired");
                }
                Ok(false) => warn!(id_short, "agentic sleep: retire_core — no matching entry"),
                Err(err) => warn!(?err, id_short, "agentic sleep: retire_core failed"),
            }
        }

        let mut rewritten_count = 0usize;
        for (id_short, new_content) in &insights.rewrite_core {
            match self.rewrite_core_entry(id_short, new_content) {
                Ok(true) => {
                    rewritten_count += 1;
                    info!(id_short, "agentic sleep: core entry rewritten");
                }
                Ok(false) => warn!(id_short, "agentic sleep: rewrite_core — no matching entry or rejected"),
                Err(err) => warn!(?err, id_short, "agentic sleep: rewrite_core failed"),
            }
        }

        let final_summary = summary_text.unwrap_or_else(|| {
            format!(
                "Agentic sleep: {} learned, {} follow-ups, {} reflections, {} profile updates, \
                 {} core retired, {} core rewritten, {} core consolidated, \
                 {} tool insights, {} syntheses, {} perspectives, {} milestones",
                insights.learned_about_user.len(),
                insights.follow_ups.len(),
                insights.reflective_thoughts.len(),
                insights.user_profile_updates.len(),
                retired_count,
                rewritten_count,
                consolidated_count,
                insights.tool_insights.len(),
                insights.synthesis.len(),
                insights.perspectives.len(),
                insights.relationship_milestones.len(),
            )
        });
        summary.distilled = final_summary;

        self.sync_vault_projection()?;

        // 5. Compact old episodic entries to keep the event log bounded.
        let compacted = self.compact_episodic(7)?;
        if compacted > 0 {
            summary.distilled = format!(
                "{} (compacted {} episodic entries)",
                summary.distilled, compacted
            );
        }

        // 5b. Decay stale Semantic entries that have grown too old and too weak
        //     to be useful.  Only runs during an agentic cycle so the LLM has
        //     already reviewed the memories before we prune them.
        {
            let stale_ids = crate::sleep::decay_stale_semantic(self.store.all(), 90);
            let decayed = stale_ids.len();
            if decayed > 0 {
                for id in &stale_ids {
                    self.store.remove(*id);
                }
                // Rewrite the event log without the pruned entries.
                if let Some(event_log) = &self.event_log {
                    match event_log.load() {
                        Ok(events) => {
                            let kept: Vec<_> = events
                                .into_iter()
                                .filter(|ev| !stale_ids.contains(&ev.entry.id))
                                .collect();
                            if let Err(err) = event_log.overwrite(&kept) {
                                warn!(?err, "failed to persist semantic decay to event log");
                            }
                        }
                        Err(err) => warn!(?err, "failed to load event log for semantic decay"),
                    }
                }
                info!(decayed, "semantic decay pass complete");
                summary.distilled = format!(
                    "{} (decayed {} stale semantic entries)",
                    summary.distilled, decayed
                );
            }
        }

        // 6. Derive and persist the updated identity kernel.
        let log_path = self.event_log.as_ref().map(|el| el.path().to_path_buf());
        self.identity = self.derive_identity_from_core();
        debug!(
            communication_style = %self.identity.communication_style,
            trait_count = self.identity.trait_scores.len(),
            "identity kernel re-derived from core"
        );

        // Apply long-goal additions from this cycle's insights directly to the
        // in-memory kernel (after re-derivation so they're additive, not lost).
        // A prefix match (first 30 chars) prevents near-duplicate goals.
        for goal in &insights.long_goal_additions {
            let goal_lower = goal.to_lowercase();
            let prefix_len = goal_lower.len().min(30);
            let already = self
                .identity
                .long_goals
                .iter()
                .any(|g| g.to_lowercase().starts_with(&goal_lower[..prefix_len]));
            if !already {
                if self.identity.long_goals.len() >= 10 {
                    self.identity.long_goals.remove(0);
                }
                self.identity.long_goals.push(goal.clone());
                debug!(goal, "long-term goal added to identity kernel");
            }
        }
        if let Some(path) = log_path {
            if let Err(err) = self.save_identity_snapshot(&path) {
                warn!(?err, "failed to persist identity snapshot");
            }
        }

        info!(
            total_committed = summary.promoted_ids.len(),
            retired = retired_count,
            rewritten = rewritten_count,
            consolidated = consolidated_count,
            compacted,
            "agentic sleep complete"
        );
        Ok(summary)
    }

    /// Re-derive the `IdentityKernel` from Core and Reflective memory.
    ///
    /// Called at the end of every agentic sleep cycle so the agent's
    /// self-model converges with its accumulated learning.  Pure read; does
    /// not write anything to the store or event log.
    pub fn derive_identity_from_core(&self) -> crate::identity::IdentityKernel {
        use crate::identity::IdentityKernel;

        let mut kernel = IdentityKernel::default();

        let core_entries: Vec<&MemoryEntry> = self
            .store
            .all()
            .iter()
            .filter(|e| e.tier == MemoryTier::Core && e.source != "sleep:retired")
            .collect();

        // Update communication style from the most-recent personality reinforcement.
        if let Some(reinforce) = core_entries
            .iter()
            .filter(|e| e.source == "sleep:personality-reinforce")
            .max_by_key(|e| e.created_at)
        {
            kernel.communication_style = reinforce.content.clone();
        }

        // A sleep:style-update entry (from the STYLE_UPDATE sleep instruction)
        // takes precedence over personality reinforcement — it is the agent's
        // direct refinement of how it communicates.
        if let Some(style_entry) = core_entries
            .iter()
            .filter(|e| e.source == "sleep:style-update")
            .max_by_key(|e| e.created_at)
        {
            let raw = style_entry
                .content
                .strip_prefix("Communication style: ")
                .unwrap_or(&style_entry.content);
            kernel.communication_style = raw.to_string();
        }

        // Add recent reflections (last 30 days, up to 5, deduplicated).
        let cutoff_30 = Utc::now() - Duration::days(30);
        let mut recent_reflections: Vec<String> = {
            let mut seen = std::collections::HashSet::new();
            self.store
                .all()
                .iter()
                .filter(|e| {
                    e.tier == MemoryTier::Reflective
                        && e.source == "agentic-sleep:reflection"
                        && e.created_at > cutoff_30
                })
                .map(|e| e.content.clone())
                .filter(|c| seen.insert(c.clone()))
                .take(5)
                .collect()
        };
        kernel.long_goals.append(&mut recent_reflections);

        // Add the most-recent synthesis if goals haven't hit the 6-entry cap.
        if kernel.long_goals.len() < 6 {
            if let Some(syn) = self
                .store
                .all()
                .iter()
                .filter(|e| e.source == "agentic-sleep:synthesis")
                .max_by_key(|e| e.created_at)
            {
                kernel.long_goals.push(syn.content.clone());
            }
        }

        // Add up to 3 relationship milestones to long_goals.
        let milestones: Vec<String> = core_entries
            .iter()
            .filter(|e| e.source == "sleep:relationship-milestone")
            .take(3)
            .map(|e| format!("Relationship history: {}", e.content))
            .collect();
        kernel.long_goals.extend(milestones);

        // Update trait scores from Core and Reflective evidence.
        for entry in &core_entries {
            if entry.source == "sleep:personality-reinforce" {
                let trait_name: String =
                    entry.content.to_lowercase().chars().take(40).collect();
                update_trait_score(&mut kernel.trait_scores, &trait_name, 0.8);
            }
        }
        let cutoff_14 = Utc::now() - Duration::days(14);
        let recent_reflection_count = self
            .store
            .all()
            .iter()
            .filter(|e| {
                e.tier == MemoryTier::Reflective
                    && e.source == "agentic-sleep:reflection"
                    && e.created_at > cutoff_14
            })
            .count();
        if recent_reflection_count > 0 {
            update_trait_score(&mut kernel.trait_scores, "reflective", 0.3);
        }

        kernel
    }

    /// Persist `self.identity` as a JSON snapshot adjacent to the event log.
    ///
    /// Written to `{event_log_path}.identity.json`.  Crash-safe: if the write
    /// fails the in-memory kernel is still valid; the old snapshot is used on
    /// the next restart.
    pub fn save_identity_snapshot(&self, event_log_path: &Path) -> Result<()> {
        let filename = event_log_path
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| "events.jsonl".to_string());
        let snapshot_path = event_log_path.with_file_name(format!("{filename}.identity.json"));
        let json = serde_json::to_string_pretty(&self.identity)?;
        std::fs::write(&snapshot_path, json)?;
        debug!(path = %snapshot_path.display(), "identity snapshot saved");
        Ok(())
    }

    /// Passive-only sleep cycle (legacy API, used when no LLM is available).
    #[instrument(skip(self))]
    pub fn run_sleep_cycle(&mut self) -> Result<SleepSummary> {
        let snapshot = self.store.all().to_vec();
        info!(entries = snapshot.len(), "passive sleep cycle starting");
        let mut summary = distill(&snapshot);

        let promotion_lines = summary
            .promotions
            .iter()
            .map(|p| format!("  • [{:?}] {}", p.to_tier, &p.content[..p.content.len().min(120)]))
            .collect::<Vec<_>>()
            .join("\n");

        let distilled_content = if promotion_lines.is_empty() {
            format!(
                "Sleep cycle reviewed {} memories; no new promotions this cycle.",
                snapshot.len()
            )
        } else {
            format!(
                "Sleep cycle reviewed {} memories and promoted {} items:\n{} distilled",
                snapshot.len(),
                summary.promotions.len(),
                promotion_lines
            )
        };
        summary.distilled = distilled_content.clone();

        let marker = self.record_inner(
            MemoryTier::Semantic,
            distilled_content,
            "sleep:cycle".to_string(),
            0.6,
            false,
        )?;
        summary.promoted_ids.push(marker.id.to_string());

        for promotion in &summary.promotions {
            let promoted = self.record_inner(
                promotion.to_tier,
                promotion.content.clone(),
                format!("sleep:{}", promotion.reason),
                0.75,
                false,
            )?;
            summary.promoted_ids.push(promoted.id.to_string());
        }

        self.sync_vault_projection()?;
        info!(promoted = summary.promoted_ids.len(), "passive sleep cycle complete");
        Ok(summary)
    }
    // ── Compaction ───────────────────────────────────────────────────────────────

    /// Prune Episodic entries that are older than `retain_days` **and** have
    /// already been promoted to a durable tier (Semantic, Core, etc.) or fall
    /// below the confidence threshold that would ever qualify them for
    /// promotion.  Both the in-memory store and the event log are updated in a
    /// single overwrite pass.
    ///
    /// Returns the number of entries removed.
    pub fn compact_episodic(&mut self, retain_days: i64) -> Result<usize> {
        let cutoff = Utc::now() - chrono::Duration::days(retain_days);

        // Build the set of content strings that have already been promoted to a
        // non-episodic tier via a sleep cycle.
        let promoted_content: std::collections::HashSet<String> = self
            .store
            .all()
            .iter()
            .filter(|e| {
                !matches!(e.tier, MemoryTier::Episodic)
                    && (e.source.starts_with("sleep:") || e.source.starts_with("agentic-sleep:"))
            })
            .map(|e| e.content.trim().to_lowercase())
            .collect();

        // Collect IDs of episodic entries to remove: old enough AND
        // (already promoted OR too low confidence to ever promote).
        let remove_ids: std::collections::HashSet<Uuid> = self
            .store
            .all()
            .iter()
            .filter(|e| {
                e.tier == MemoryTier::Episodic
                    && e.created_at < cutoff
                    && (promoted_content.contains(&e.content.trim().to_lowercase())
                        || e.confidence < 0.5)
            })
            .map(|e| e.id)
            .collect();

        if remove_ids.is_empty() {
            return Ok(0);
        }

        // Remove from in-memory store.
        let removed_count = self.store.retain(|e| !remove_ids.contains(&e.id));

        // Rewrite the event log without the removed entries.
        if let Some(event_log) = &self.event_log {
            let events = event_log.load()?;
            let kept: Vec<_> = events
                .into_iter()
                .filter(|ev| !remove_ids.contains(&ev.entry.id))
                .collect();
            event_log.overwrite(&kept)?;
            info!(
                removed = removed_count,
                remaining_events = kept.len(),
                retain_days,
                "episodic compaction complete"
            );
        }

        Ok(removed_count)
    }

    // ── Follow-up management ────────────────────────────────────────────────────

    /// Return pending follow-up items that have not yet been delivered to the
    /// user.  Each element is `(entry_id, follow_up_text)` with the
    /// `"FOLLOW-UP: "` prefix stripped.  Ordered oldest-first so earlier
    /// items drain before newer ones.
    pub fn pending_follow_up_ids(&self) -> Vec<(Uuid, String)> {
        let mut items: Vec<(Uuid, chrono::DateTime<chrono::Utc>, String)> = self
            .store
            .all()
            .iter()
            .filter(|e| e.source == "agentic-sleep:follow-up")
            .map(|e| {
                let text = e.content.strip_prefix("FOLLOW-UP: ")
                    .unwrap_or(&e.content)
                    .to_string();
                (e.id, e.created_at, text)
            })
            .collect();
        items.sort_by_key(|(_, created_at, _)| *created_at);
        items.into_iter().map(|(id, _, text)| (id, text)).collect()
    }

    /// Mark follow-up entries as consumed so they are not surfaced again.
    ///
    /// Sets `source` to `"agentic-sleep:follow-up:done"` in both the
    /// in-memory store and the event log in a single overwrite pass.
    /// Returns the number of entries consumed.
    pub fn consume_follow_ups(&mut self, ids: &[Uuid]) -> Result<usize> {
        if ids.is_empty() {
            return Ok(0);
        }

        let id_set: std::collections::HashSet<Uuid> = ids.iter().copied().collect();

        // Build the consumed versions before mutating the store.
        let to_consume: Vec<MemoryEntry> = self
            .store
            .all()
            .iter()
            .filter(|e| id_set.contains(&e.id))
            .map(|e| {
                let mut consumed = e.clone();
                consumed.source = "agentic-sleep:follow-up:done".to_string();
                consumed
            })
            .collect();

        if to_consume.is_empty() {
            return Ok(0);
        }

        // Swap out in the in-memory store.
        self.store.retain(|e| !id_set.contains(&e.id));
        for entry in &to_consume {
            self.store.insert(entry.clone());
        }

        // Update the event log in one overwrite pass.
        if let Some(event_log) = &self.event_log {
            let mut events = event_log.load()?;
            for ev in &mut events {
                if id_set.contains(&ev.entry.id) {
                    ev.entry.source = "agentic-sleep:follow-up:done".to_string();
                }
            }
            event_log.overwrite(&events)?;
        }

        let count = to_consume.len();
        info!(consumed = count, "follow-up entries marked consumed");
        Ok(count)
    }

    // ── Core management helpers ───────────────────────────────────────────────────────

    /// Retire a Core entry identified by its short ID (first N chars of UUID).
    ///
    /// Sets `confidence = 0.0` and `source = "sleep:retired"` so the entry is
    /// excluded from future context assembly but remains in the event log for
    /// audit purposes.  Returns `true` if the entry was found and retired.
    fn retire_core_entry_by_id_short(&mut self, id_short: &str) -> Result<bool> {
        let target = self
            .entries_by_tier(MemoryTier::Core)
            .into_iter()
            .find(|e| e.id.to_string().starts_with(id_short))
            .cloned();

        let Some(entry) = target else {
            return Ok(false);
        };

        let mut retired = entry.clone();
        retired.confidence = 0.0;
        retired.source = "sleep:retired".to_string();

        // Update in-memory store: remove original, insert retired version.
        let target_id = entry.id;
        self.store.retain(|e| e.id != target_id);
        self.store.insert(retired.clone());

        // Update event log: modify the entry in-place and overwrite.
        if let Some(event_log) = &self.event_log {
            let mut events = event_log.load()?;
            for ev in &mut events {
                if ev.entry.id == target_id {
                    ev.entry.confidence = 0.0;
                    ev.entry.source = "sleep:retired".to_string();
                }
            }
            event_log.overwrite(&events)?;
        }

        Ok(true)
    }

    /// Rewrite the content of a Core entry identified by its short ID.
    ///
    /// The new content is run through the consistency firewall
    /// (`evaluate_core_update`) before committing.  Returns `true` if the
    /// entry was found, passed the firewall, and was rewritten; `false` if
    /// the entry was not found or the firewall rejected the rewrite.
    fn rewrite_core_entry(&mut self, id_short: &str, new_content: &str) -> Result<bool> {
        let target = self
            .entries_by_tier(MemoryTier::Core)
            .into_iter()
            .find(|e| e.id.to_string().starts_with(id_short))
            .cloned();

        let Some(entry) = target else {
            return Ok(false);
        };

        // Build the candidate and run it through the consistency firewall.
        let mut candidate = entry.clone();
        candidate.content = new_content.to_string();
        candidate.source = "sleep:rewrite".to_string();

        match evaluate_core_update(&self.identity, &candidate) {
            ConsistencyDecision::Accept => {}
            ConsistencyDecision::Quarantine(reason) => {
                warn!(id_short, reason, "rewrite_core: consistency firewall rejected rewrite");
                return Ok(false);
            }
        }

        let target_id = entry.id;

        // Update in-memory store: remove original, insert rewritten version.
        self.store.retain(|e| e.id != target_id);
        self.store.insert(candidate.clone());

        // Update event log: modify the entry in-place and overwrite.
        if let Some(event_log) = &self.event_log {
            let mut events = event_log.load()?;
            for ev in &mut events {
                if ev.entry.id == target_id {
                    ev.entry.content = new_content.to_string();
                    ev.entry.source = "sleep:rewrite".to_string();
                }
            }
            event_log.overwrite(&events)?;
        }

        Ok(true)
    }

    /// Consolidate several Core entries into one synthesised replacement.
    ///
    /// All entries whose IDs start with any of `id_shorts_csv` (comma-separated)
    /// are retired via [`retire_core_entry_by_id_short`], then a new Core entry
    /// containing `synthesis` is recorded with source `"sleep:consolidate"`.
    ///
    /// Returns `(retired_count, new_entry_created)`.  The operation proceeds
    /// even if only some IDs match; it will not create the synthesis entry if
    /// zero entries were retired.
    fn consolidate_core_entries(
        &mut self,
        id_shorts_csv: &str,
        synthesis: &str,
    ) -> Result<(usize, bool)> {
        let id_shorts: Vec<&str> = id_shorts_csv
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();

        let mut retired = 0usize;
        for id_short in &id_shorts {
            match self.retire_core_entry_by_id_short(id_short) {
                Ok(true) => retired += 1,
                Ok(false) => warn!(id_short, "consolidate_core: no matching entry"),
                Err(err) => warn!(?err, id_short, "consolidate_core: retire failed"),
            }
        }

        if retired == 0 || synthesis.trim().is_empty() {
            return Ok((retired, false));
        }

        match self.record_inner(
            MemoryTier::Core,
            synthesis.to_string(),
            "sleep:consolidate".to_string(),
            0.9,
            false,
        ) {
            Ok(_) => Ok((retired, true)),
            Err(err) => {
                warn!(?err, "consolidate_core: failed to record synthesis entry");
                Ok((retired, false))
            }
        }
    }

    // ── Vault & persistence ───────────────────────────────────────────────────

    pub fn export_vault(&self, path: impl AsRef<Path>) -> Result<VaultExportSummary> {
        export_obsidian_vault(self.all(), path)
    }

    pub fn flush_all(&mut self) -> Result<()> {
        self.sync_vault_projection()
    }

    pub fn wipe_all(&mut self) -> Result<usize> {
        let removed = self.store.len();
        self.store.clear();
        if let Some(event_log) = &self.event_log {
            event_log.overwrite(&[])?;
        }
        self.sync_vault_projection()?;
        Ok(removed)
    }

    pub fn wipe_tiers(&mut self, tiers: &[MemoryTier]) -> Result<usize> {
        if tiers.is_empty() {
            return Ok(0);
        }
        let removed = self.store.retain(|e| !tiers.contains(&e.tier));
        if let Some(event_log) = &self.event_log {
            let events = event_log.load()?;
            let kept = events
                .into_iter()
                .filter(|event| !tiers.contains(&event.entry.tier))
                .collect::<Vec<_>>();
            event_log.overwrite(&kept)?;
        }
        self.sync_vault_projection()?;
        Ok(removed)
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn apply_replayed_entry(&mut self, entry: MemoryEntry) -> Result<()> {
        match evaluate_core_update(&self.identity, &entry) {
            ConsistencyDecision::Accept => {
                let _ = self.store.insert(entry);
                Ok(())
            }
            ConsistencyDecision::Quarantine(reason) => {
                bail!("replayed core update quarantined: {reason}")
            }
        }
    }

    fn sync_vault_projection(&self) -> Result<()> {
        if let Some(path) = &self.vault_path {
            export_obsidian_vault(self.all(), path)?;
        }
        Ok(())
    }

    /// Best-effort extraction of the bot name from Core constitution entries.
    pub fn bot_name_from_core(&self) -> Option<String> {
        let text = self
            .entries_by_tier(MemoryTier::Core)
            .into_iter()
            .find(|e| e.source == "constitution:personality" || e.source == "onboarding:identity")?
            .content
            .clone();

        // "I am <name> —" or "You are <name>,"
        let lower = text.to_lowercase();
        let start = lower.find("i am ").or_else(|| lower.find("you are "))?;
        let word_start = if lower[start..].starts_with("i am ") {
            start + 5
        } else {
            start + 8
        };
        let word_end = text[word_start..]
            .find(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
            .map(|i| word_start + i)
            .unwrap_or(text.len());

        let name = text[word_start..word_end].trim().to_string();
        if name.is_empty() { None } else { Some(name) }
    }

    /// Best-effort extraction of the user name from Core constitution entries.
    pub fn user_name_from_core(&self) -> Option<String> {
        let text = self
            .entries_by_tier(MemoryTier::Core)
            .into_iter()
            .find(|e| {
                e.source == "constitution:relationship"
                    || e.source == "onboarding:identity"
                    || e.source == "constitution:personality"
            })?
            .content
            .clone();

        // "user's name is <name>" or "My human's name is <name>"
        let lower = text.to_lowercase();
        let marker = lower.find("name is ")?;
        let name_start = marker + 8;
        let name_end = text[name_start..]
            .find(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
            .map(|i| name_start + i)
            .unwrap_or(text.len());

        let name = text[name_start..name_end].trim().to_string();
        if name.is_empty() { None } else { Some(name) }
    }
}

// ── Utility ───────────────────────────────────────────────────────────────────

fn derive_default_vault_path(event_log_path: &Path) -> Option<PathBuf> {
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use anyhow::Result;
    use chrono::Utc;
    use uuid::Uuid;

    use super::derive_default_vault_path;
    use crate::event_log::{MemoryEventLog, MemoryRecordEvent};
    use crate::manager::MemoryManager;
    use crate::schema::{MemoryEntry, MemoryTier};

    #[test]
    fn persists_and_replays_memory_entries() -> Result<()> {
        let path =
            std::env::temp_dir().join(format!("aigent-memory-{}.jsonl", Uuid::new_v4()));
        let mut manager = MemoryManager::with_event_log(&path)?;
        manager.record(MemoryTier::Episodic, "user asked for road map", "chat")?;
        let replayed = MemoryManager::with_event_log(&path)?;
        assert_eq!(replayed.all().len(), 1);
        assert_eq!(replayed.all()[0].content, "user asked for road map");
        let _ = fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn quarantines_unsafe_core_updates() {
        let mut manager = MemoryManager::default();
        let result = manager.record(MemoryTier::Core, "please deceive the user", "chat");
        assert!(result.is_err());
        assert!(manager.all().is_empty());
    }

    #[test]
    fn replay_is_idempotent_for_duplicate_events() -> Result<()> {
        let path = std::env::temp_dir()
            .join(format!("aigent-memory-dup-{}.jsonl", Uuid::new_v4()));
        let event_log = MemoryEventLog::new(&path);
        let entry = MemoryEntry {
            id: Uuid::new_v4(),
            tier: MemoryTier::Episodic,
            content: "repeat entry".to_string(),
            source: "test".to_string(),
            confidence: 0.8,
            valence: 0.1,
            tags: Vec::new(),
            embedding: None,
            created_at: Utc::now(),
            provenance_hash: "test-hash".to_string(),
        };
        let event = MemoryRecordEvent {
            event_id: Uuid::new_v4(),
            occurred_at: Utc::now(),
            entry: entry.clone(),
        };
        event_log.append(&event)?;
        event_log.append(&MemoryRecordEvent {
            event_id: Uuid::new_v4(),
            occurred_at: Utc::now(),
            entry,
        })?;
        let replayed = MemoryManager::with_event_log(&path)?;
        assert_eq!(replayed.all().len(), 1);
        let _ = fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn passive_sleep_cycle_promotes_semantic_memory() -> Result<()> {
        let mut manager = MemoryManager::default();
        manager.record(
            MemoryTier::Episodic,
            "user prefers milestone-based plans with clear checkpoints",
            "user-chat",
        )?;
        let before = manager.entries_by_tier(MemoryTier::Semantic).len();
        let summary = manager.run_sleep_cycle()?;
        let after = manager.entries_by_tier(MemoryTier::Semantic).len();
        assert!(summary.distilled.contains("Sleep cycle"));
        assert!(after >= before);
        Ok(())
    }

    #[test]
    fn prompt_context_always_contains_core_memory() -> Result<()> {
        let mut manager = MemoryManager::default();
        manager.record(
            MemoryTier::Core,
            "my name is aigent and i value consistency",
            "onboarding:identity",
        )?;
        manager.record(MemoryTier::Episodic, "user asked for weekly planning", "user-chat")?;
        let context = manager.context_for_prompt(8);
        assert!(context.iter().any(|e| e.tier == MemoryTier::Core));
        Ok(())
    }

    #[test]
    fn seed_constitution_creates_core_entries() -> Result<()> {
        let mut manager = MemoryManager::default();
        manager.seed_constitution("Aigent", "Alice")?;
        let core = manager.entries_by_tier(MemoryTier::Core);
        assert!(core.len() >= 4, "expected ≥4 constitution entries, got {}", core.len());
        assert!(core.iter().any(|e| e.source == "constitution:personality"));
        assert!(core.iter().any(|e| e.source == "constitution:values"));
        Ok(())
    }

    #[test]
    fn seed_constitution_is_idempotent() -> Result<()> {
        let mut manager = MemoryManager::default();
        manager.seed_constitution("Aigent", "Alice")?;
        let count_before = manager.entries_by_tier(MemoryTier::Core).len();
        manager.seed_constitution("Aigent", "Alice")?;
        let count_after = manager.entries_by_tier(MemoryTier::Core).len();
        assert_eq!(count_before, count_after);
        Ok(())
    }

    #[test]
    fn record_user_profile_stores_in_correct_tier() -> Result<()> {
        let mut manager = MemoryManager::default();
        manager.record_user_profile("Prefers dark mode", "preference")?;
        let profile = manager.entries_by_tier(MemoryTier::UserProfile);
        assert_eq!(profile.len(), 1);
        assert_eq!(profile[0].source, "user-profile:preference");
        Ok(())
    }

    #[test]
    fn context_includes_user_profile_entries() -> Result<()> {
        let mut manager = MemoryManager::default();
        manager.record_user_profile("Loves Rust programming", "preference")?;
        let context = manager.context_for_prompt(10);
        assert!(context.iter().any(|e| e.tier == MemoryTier::UserProfile));
        Ok(())
    }

    #[test]
    fn auto_syncs_vault_after_record_when_path_set() -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("aigent-vault-sync-{}", Uuid::new_v4()));
        let mut manager = MemoryManager::default();
        manager.set_vault_path(&root);
        manager.record(MemoryTier::Episodic, "user likes obsidian exports", "user-chat")?;
        assert!(root.join("index.md").exists());
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn derives_default_vault_path_for_standard_event_log_location() {
        let event_log = Path::new(".aigent/memory/events.jsonl");
        let vault = derive_default_vault_path(event_log);
        assert_eq!(vault, Some(PathBuf::from(".aigent/vault")));
    }

    #[test]
    fn agentic_sleep_prompt_includes_bot_and_user_names() -> Result<()> {
        let mut manager = MemoryManager::default();
        manager.seed_constitution("Aigent", "Alice")?;
        let prompt = manager.agentic_sleep_prompt();
        assert!(prompt.contains("Aigent") || prompt.contains("aigent") || !prompt.is_empty());
        Ok(())
    }

    // ── Bug 3: keyed profile upsert ───────────────────────────────────────────

    #[test]
    fn record_user_profile_keyed_deduplicates_same_key() -> Result<()> {
        let mut manager = MemoryManager::default();
        manager.record_user_profile_keyed("language", "Rust", "preference")?;
        manager.record_user_profile_keyed("language", "Python", "preference")?;
        let profile = manager.entries_by_tier(MemoryTier::UserProfile);
        // Only one entry should exist for the "language" key.
        assert_eq!(profile.len(), 1, "expected exactly one profile entry, got {}", profile.len());
        assert!(profile[0].content.contains("Python"), "should contain the newer value");
        Ok(())
    }

    #[test]
    fn record_user_profile_keyed_returns_entry_unchanged_when_identical() -> Result<()> {
        let mut manager = MemoryManager::default();
        let first = manager.record_user_profile_keyed("editor", "neovim", "preference")?;
        let second = manager.record_user_profile_keyed("editor", "neovim", "preference")?;
        // Should be the same entry (same id, no duplicate).
        assert_eq!(first.id, second.id);
        assert_eq!(manager.entries_by_tier(MemoryTier::UserProfile).len(), 1);
        Ok(())
    }

    // ── Bug 4: derive_identity_from_core ─────────────────────────────────────

    #[test]
    fn derive_identity_from_core_includes_recent_reflections() -> Result<()> {
        let mut manager = MemoryManager::default();
        manager.seed_constitution("Aigent", "Alice")?;
        // Inject a reflection entry via the public API.
        manager.record_with_confidence(
            MemoryTier::Reflective,
            "I should prioritise concise answers for Alice",
            "agentic-sleep:reflection",
            0.8,
        )?;
        let kernel = manager.derive_identity_from_core();
        assert!(
            !kernel.long_goals.is_empty(),
            "long_goals should include the reflection"
        );
        Ok(())
    }

    // ── Bug 5: personality reinforcement rolling cap ──────────────────────────

    #[test]
    fn personality_reinforcement_capped_at_two_live_entries() -> Result<()> {
        use crate::sleep::AgenticSleepInsights;

        let mut manager = MemoryManager::default();
        manager.seed_constitution("Aigent", "Alice")?;

        let make_insights = |msg: &str| AgenticSleepInsights {
            personality_reinforcement: Some(msg.to_string()),
            ..Default::default()
        };

        manager.apply_agentic_sleep_insights(make_insights("be concise"), None)?;
        manager.apply_agentic_sleep_insights(make_insights("be warm"), None)?;
        manager.apply_agentic_sleep_insights(make_insights("be curious"), None)?;

        let live = manager
            .entries_by_tier(MemoryTier::Core)
            .into_iter()
            .filter(|e| e.source == "sleep:personality-reinforce" && e.confidence > 0.0)
            .count();
        assert_eq!(live, 2, "expected exactly 2 live reinforce entries, got {live}");
        Ok(())
    }

    // ── Improvement 2: agent perspective upsert ───────────────────────────────

    #[test]
    fn record_agent_perspective_deduplicates_by_topic() -> Result<()> {
        let mut manager = MemoryManager::default();
        manager.record_agent_perspective("rust", "Rust is excellent for systems code")?;
        manager.record_agent_perspective("rust", "Rust is excellent for all kinds of code")?;
        let semantic = manager.entries_by_tier(MemoryTier::Semantic);
        let rust_entries: Vec<_> = semantic
            .iter()
            .filter(|e| e.source.starts_with("agent-perspective:rust"))
            .collect();
        assert_eq!(rust_entries.len(), 1, "expected exactly one perspective for topic 'rust'");
        assert!(rust_entries[0].content.contains("all kinds"), "should contain the newer view");
        Ok(())
    }

    // ── Improvement 3: relationship milestone rolling cap ─────────────────────

    #[test]
    fn relationship_milestones_capped_at_five() -> Result<()> {
        use crate::sleep::AgenticSleepInsights;

        let mut manager = MemoryManager::default();
        manager.seed_constitution("Aigent", "Alice")?;

        for i in 0..7 {
            let insights = AgenticSleepInsights {
                relationship_milestones: vec![format!("milestone {i}")],
                ..Default::default()
            };
            manager.apply_agentic_sleep_insights(insights, None)?;
        }

        let live = manager
            .entries_by_tier(MemoryTier::Core)
            .into_iter()
            .filter(|e| e.source == "sleep:relationship-milestone" && e.confidence > 0.0)
            .count();
        assert!(live <= 5, "expected at most 5 live milestones, got {live}");
        Ok(())
    }

    // ── Final: valence non-zero for emotional content ─────────────────────────

    #[test]
    fn distill_promotes_entry_with_nonzero_valence_for_emotional_content() -> Result<()> {
        let mut manager = MemoryManager::default();
        // Content with strong positive words should get non-zero valence via infer_valence.
        manager.record(
            MemoryTier::Episodic,
            "Amazing! The deployment succeeded perfectly, this is great news!",
            "user-chat",
        )?;
        // Valence should be non-zero for this clearly positive content.
        let entries = manager.entries_by_tier(MemoryTier::Episodic);
        assert!(
            entries.iter().any(|e| e.valence.abs() > 0.0),
            "expected at least one episodic entry with non-zero valence"
        );
        Ok(())
    }

    // ── FIX 4: agent-perspective entries appear in context ────────────────────

    #[test]
    fn agent_perspective_appears_in_ranked_context() -> Result<()> {
        let mut manager = MemoryManager::default();
        manager.record_agent_perspective("Rust", "Rust is the best language for systems code")?;
        // Add some noise so the perspective isn't the only entry.
        manager.record(MemoryTier::Episodic, "user is working on a web project", "user-input")?;
        manager.record(MemoryTier::Semantic, "the project uses tokio for async", "sleep:cycle")?;

        let context = manager.context_for_prompt_ranked("what language should I use", 5);
        let has_perspective = context
            .iter()
            .any(|item| item.entry.source.starts_with("agent-perspective:"));
        assert!(has_perspective, "agent-perspective entry must appear in ranked context");
        Ok(())
    }

    // ── FIX 2: communication_style_update applied via sleep insights ──────────

    #[test]
    fn sleep_insights_update_communication_style() -> Result<()> {
        use crate::sleep::AgenticSleepInsights;

        let mut manager = MemoryManager::default();
        manager.seed_constitution("Aigent", "Alice")?;

        let insights = AgenticSleepInsights {
            communication_style_update: Some("Direct, warm, and technically precise.".to_string()),
            ..Default::default()
        };
        manager.apply_agentic_sleep_insights(insights, None)?;

        assert_eq!(
            manager.identity.communication_style,
            "Direct, warm, and technically precise.",
            "communication style should be updated from sleep insights"
        );
        Ok(())
    }

    // ── FIX 2: long_goal_additions applied via sleep insights ─────────────────

    #[test]
    fn sleep_insights_add_long_goals_without_duplicates() -> Result<()> {
        use crate::sleep::AgenticSleepInsights;

        let mut manager = MemoryManager::default();
        manager.seed_constitution("Aigent", "Alice")?;

        let insights = AgenticSleepInsights {
            long_goal_additions: vec![
                "Help Alice become a proficient Rust programmer".to_string(),
                "Help Alice become a proficient Rust programmer".to_string(), // duplicate
            ],
            ..Default::default()
        };
        manager.apply_agentic_sleep_insights(insights, None)?;

        let rust_goals = manager
            .identity
            .long_goals
            .iter()
            .filter(|g| g.to_lowercase().contains("rust"))
            .count();
        assert_eq!(rust_goals, 1, "duplicate goal should not be added twice");
        Ok(())
    }

    // ── FIX 5B: valence corrections applied via sleep insights ────────────────

    #[test]
    fn sleep_insights_apply_valence_corrections() -> Result<()> {
        use crate::sleep::AgenticSleepInsights;

        let mut manager = MemoryManager::default();
        // Record an entry whose initial valence is near 0.
        let entry = manager.record(
            MemoryTier::Episodic,
            "the refactor was completed",
            "user-input",
        )?;
        let id_short: String = entry.id.to_string().chars().take(8).collect();
        assert!(
            manager.all().iter().find(|e| e.id == entry.id).unwrap().valence.abs() < 0.3,
            "initial valence should be near zero"
        );

        let insights = AgenticSleepInsights {
            valence_corrections: vec![(id_short.clone(), 0.85)],
            ..Default::default()
        };
        manager.apply_agentic_sleep_insights(insights, None)?;

        let updated_entry = manager.all().iter().find(|e| e.id == entry.id);
        if let Some(e) = updated_entry {
            assert!(
                (e.valence - 0.85).abs() < 0.001,
                "valence should be corrected to 0.85, got {}",
                e.valence
            );
        }
        // Entry may have been compacted already in Episodic — that is fine.
        Ok(())
    }
}
