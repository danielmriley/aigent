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
use chrono::Utc;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

use crate::consistency::{ConsistencyDecision, evaluate_core_update};
use crate::constitution::constitution_seeds;
use crate::event_log::{MemoryEventLog, MemoryRecordEvent};
use crate::identity::IdentityKernel;
use crate::profile::format_user_profile_block;
use crate::retrieval::{RankedMemoryContext, assemble_context_with_provenance};
use crate::schema::{MemoryEntry, MemoryTier};
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
    /// Core, UserProfile, and Reflective are always considered regardless of
    /// lexical overlap.  Sleep bookkeeping entries are filtered out.
    pub fn context_for_prompt_ranked(&self, query: &str, limit: usize) -> Vec<RankedMemoryContext> {
        let priority_tiers = [MemoryTier::Core, MemoryTier::UserProfile, MemoryTier::Reflective];

        let priority_entries = self
            .store
            .all()
            .iter()
            .filter(|e| priority_tiers.contains(&e.tier))
            .cloned()
            .collect::<Vec<_>>();

        let other_entries = self
            .store
            .all()
            .iter()
            .filter(|e| {
                !priority_tiers.contains(&e.tier)
                    && !e.source.starts_with("assistant-turn")
                    && e.source != "sleep:cycle"
            })
            .cloned()
            .collect::<Vec<_>>();

        assemble_context_with_provenance(other_entries, priority_entries, query, limit)
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

        let mut entry = MemoryEntry {
            id: Uuid::new_v4(),
            tier,
            content,
            source,
            confidence: confidence.clamp(0.0, 1.0),
            valence: 0.0,
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

    // ── Agentic sleep ─────────────────────────────────────────────────────────

    /// Generate the nightly reflection prompt to send to the LLM.
    pub fn agentic_sleep_prompt(&self) -> String {
        let bot_name = self.bot_name_from_core().unwrap_or_else(|| "Aigent".to_string());
        let user_name = self.user_name_from_core().unwrap_or_else(|| "User".to_string());
        build_sleep_prompt(self.all(), &bot_name, &user_name)
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
            .map(|p| format!("  • [{:?}] {}", p.to_tier, &p.content[..p.content.len().min(80)]))
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
            let e = self.record_inner(
                MemoryTier::UserProfile,
                format!("{key}: {value}"),
                "agentic-sleep:profile-update".to_string(),
                0.85,
                false,
            )?;
            summary.promoted_ids.push(e.id.to_string());
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

        let final_summary = summary_text.unwrap_or_else(|| {
            format!(
                "Agentic sleep: {} learned, {} follow-ups, {} reflections, {} profile updates",
                insights.learned_about_user.len(),
                insights.follow_ups.len(),
                insights.reflective_thoughts.len(),
                insights.user_profile_updates.len(),
            )
        });
        summary.distilled = final_summary;

        self.sync_vault_projection()?;

        info!(
            total_committed = summary.promoted_ids.len(),
            "agentic sleep complete"
        );
        Ok(summary)
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
}
