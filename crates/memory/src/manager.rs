use anyhow::{Result, bail};
use chrono::Utc;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

/// A synchronous function that maps a text string to an optional embedding
/// vector.  Stored as an `Arc` so it can be cloned across structs.
pub type EmbedFn = Arc<dyn Fn(&str) -> Option<Vec<f32>> + Send + Sync>;

use crate::consistency::{ConsistencyDecision, evaluate_core_update};
use crate::event_log::{MemoryEventLog, MemoryRecordEvent};
use crate::identity::IdentityKernel;
use crate::retrieval::{RankedMemoryContext, assemble_context_with_provenance};
use crate::schema::{MemoryEntry, MemoryTier};
use crate::sleep::{AgenticSleepInsights, SleepSummary, distill};
use crate::store::MemoryStore;
use crate::vault::{VaultExportSummary, export_obsidian_vault};

#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub total: usize,
    pub core: usize,
    pub user_profile: usize,
    pub reflective: usize,
    pub semantic: usize,
    pub procedural: usize,
    pub episodic: usize,
}

#[derive(Default)]
pub struct MemoryManager {
    pub identity: IdentityKernel,
    pub store: MemoryStore,
    event_log: Option<MemoryEventLog>,
    vault_path: Option<PathBuf>,
    /// Optional embedding backend.  When set, every new `MemoryEntry` has its
    /// `embedding` field populated at record time for hybrid retrieval.
    embed_fn: Option<EmbedFn>,
}

impl MemoryManager {
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
            .expect("event log is always present in with_event_log")
            .load()?;

        let event_count = events.len();
        for event in events {
            manager.apply_replayed_entry(event.entry)?;
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

    /// Configure a synchronous embedding backend.  All new entries recorded
    /// after this call will have their `embedding` field populated.
    pub fn set_embed_fn(&mut self, f: EmbedFn) {
        self.embed_fn = Some(f);
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

    pub fn context_for_prompt(&self, limit: usize) -> Vec<MemoryEntry> {
        self.context_for_prompt_ranked("", limit)
            .into_iter()
            .map(|item| item.entry)
            .collect()
    }

    pub fn context_for_prompt_ranked(&self, query: &str, limit: usize) -> Vec<RankedMemoryContext> {
        let core_entries = self
            .entries_by_tier(MemoryTier::Core)
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        let non_core = self
            .store
            .all()
            .iter()
            .filter(|entry| {
                entry.tier != MemoryTier::Core
                    // Skip assistant-turn metadata (old format)
                    && !entry.source.starts_with("assistant-turn")
                    // Skip sleep cycle bookkeeping entries — their content
                    // contains structured text ("promoted 79 items: • [Core]…")
                    // that the LLM misreads as factual memory counts.
                    && entry.source != "sleep:cycle"
            })
            .cloned()
            .collect::<Vec<_>>();

        assemble_context_with_provenance(non_core, core_entries, query, limit, None)
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

    pub fn export_vault(&self, path: impl AsRef<Path>) -> Result<VaultExportSummary> {
        export_obsidian_vault(self.all(), path)
    }

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
        let embedding = self.embed_fn.as_ref().and_then(|f| f(&content));
        let entry = MemoryEntry {
            id: Uuid::new_v4(),
            tier,
            content,
            source,
            confidence: 0.7,
            valence: 0.0,
            created_at: Utc::now(),
            provenance_hash: "local-dev-placeholder".to_string(),
            tags: vec![],
            embedding,
        };

        match evaluate_core_update(&self.identity, &entry) {
            ConsistencyDecision::Accept => {
                let inserted = self.store.insert(entry.clone());
                if inserted {
                    debug!(tier = ?entry.tier, source = %entry.source, id = %entry.id, content_len = entry.content.len(), "memory entry recorded");
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

    #[instrument(skip(self))]
    pub async fn run_sleep_cycle(&mut self) -> Result<SleepSummary> {
        let snapshot = self.store.all().to_vec();
        info!(entries = snapshot.len(), "sleep cycle starting");
        let mut summary = distill(&snapshot);

        // Build a human-readable summary of what was promoted.
        let promotion_lines = summary
            .promotions
            .iter()
            .map(|p| format!("  • [{:?}] {}", p.to_tier, &p.content[..p.content.len().min(120)]))
            .collect::<Vec<_>>()
            .join("\n");
        let distilled_content = if promotion_lines.is_empty() {
            format!("Sleep cycle reviewed {} memories; no new promotions this cycle. distilled", snapshot.len())
        } else {
            format!(
                "Sleep cycle reviewed {} memories and promoted {} items:\n{} distilled",
                snapshot.len(),
                summary.promotions.len(),
                promotion_lines
            )
        };
        summary.distilled = distilled_content.clone();

        // Back up events.jsonl before writing sleep-cycle entries so there is
        // always a clean pre-sleep snapshot on disk.
        if let Some(event_log) = &self.event_log {
            event_log.backup()?;
        }

        let marker = self.record_inner(
            MemoryTier::Semantic,
            distilled_content,
            "sleep:cycle".to_string(),
        ).await?;
        summary.promoted_ids.push(marker.id.to_string());

        for promotion in &summary.promotions {
            let promoted = self.record_inner(
                promotion.to_tier,
                promotion.content.clone(),
                format!("sleep:{}", promotion.reason),
            ).await?;
            summary.promoted_ids.push(promoted.id.to_string());
        }

        self.sync_vault_projection()?;
        info!(promoted = summary.promoted_ids.len(), "sleep cycle complete");
        Ok(summary)
    }

    pub fn flush_all(&mut self) -> Result<()> {
        self.sync_vault_projection()
    }

    // ── Embedding ──────────────────────────────────────────────────────────

    /// Return a clone of the embedding function Arc so async callers can move
    /// it into `tokio::task::spawn_blocking` without holding a `&MemoryManager`.
    pub fn embed_fn_arc(&self) -> Option<EmbedFn> {
        self.embed_fn.clone()
    }

    // ── Ranked context with pre-computed embedding ─────────────────────────

    /// Like [`context_for_prompt_ranked`] but accepts a pre-computed query
    /// embedding so the embedding HTTP call can happen off the async thread.
    pub fn context_for_prompt_ranked_with_embed(
        &self,
        query: &str,
        limit: usize,
        query_embedding: Option<Vec<f32>>,
    ) -> Vec<RankedMemoryContext> {
        let core_entries = self
            .entries_by_tier(MemoryTier::Core)
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        let non_core = self
            .store
            .all()
            .iter()
            .filter(|entry| {
                entry.tier != MemoryTier::Core
                    && !entry.source.starts_with("assistant-turn")
                    && entry.source != "sleep:cycle"
            })
            .cloned()
            .collect::<Vec<_>>();
        assemble_context_with_provenance(non_core, core_entries, query, limit, query_embedding)
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
            let events = event_log.load()?;
            let kept = events
                .into_iter()
                .filter(|ev| ev.entry.source != source)
                .collect::<Vec<_>>();
            event_log.overwrite(&kept).await?;
        }
        self.record_inner(MemoryTier::UserProfile, value.to_string(), source).await
    }

    /// Build a high-density relational matrix block for prompt injection.
    ///
    /// Merges `UserProfile` and `Reflective` entries into three compressed
    /// buckets:
    ///   `[USER: …]`        — facts and preferences about the user
    ///   `[MY_BELIEFS: …]`  — agent's own opinions / worldview stances
    ///   `[OUR_DYNAMIC: …]` — relationship tone, shared history, inside jokes
    ///
    /// Returns `None` when all buckets are empty.
    pub fn relational_state_block(&self) -> Option<String> {
        let mut user_facts: Vec<String> = Vec::new();
        let mut agent_beliefs: Vec<String> = Vec::new();
        let mut relationship_dynamics: Vec<String> = Vec::new();

        // Single pass — no intermediate Vec allocations from entries_by_tier.
        for entry in self.store.all().iter()
            .filter(|e| e.tier == MemoryTier::UserProfile || e.tier == MemoryTier::Reflective)
        {
            // Source-based routing is zero-allocation (&str comparisons).
            let src = entry.source.as_str();
            let is_belief_src  = src.contains("critic")  || src.contains("belief");
            let is_dynamic_src = src.contains("psychologist") || src == "sleep:relationship";

            if is_belief_src {
                agent_beliefs.push(strip_tag_prefix_lower(
                    &entry.content,
                    &["belief:", "my_belief:", "opinion:"],
                ));
            } else if is_dynamic_src {
                relationship_dynamics.push(strip_tag_prefix_lower(
                    &entry.content,
                    &["dynamic:", "our_dynamic:", "relationship:"],
                ));
            } else {
                // Content-based fallback — lowercase only for ambiguous entries.
                let c = entry.content.to_ascii_lowercase();
                if c.contains("belief") || c.contains("opinion") || c.contains("think")
                    || c.contains("feel about") || c.contains("my_belief")
                {
                    agent_beliefs.push(strip_tag_prefix_lower(
                        &entry.content,
                        &["belief:", "my_belief:", "opinion:"],
                    ));
                } else if c.contains("dynamic") || c.contains("relationship")
                    || c.contains("joke") || c.contains("rapport")
                    || c.contains("our_dynamic") || c.contains("shared")
                {
                    relationship_dynamics.push(strip_tag_prefix_lower(
                        &entry.content,
                        &["dynamic:", "our_dynamic:", "relationship:"],
                    ));
                } else {
                    user_facts.push(entry.content.clone());
                }
            }
        }

        if user_facts.is_empty() && agent_beliefs.is_empty() && relationship_dynamics.is_empty() {
            return None;
        }

        let fmt = |label: &str, items: &[String]| -> String {
            if items.is_empty() {
                return String::new();
            }
            format!("[{}: {}]", label, items.join("; "))
        };

        let parts: Vec<String> = [
            fmt("USER",        &user_facts),
            fmt("MY_BELIEFS",  &agent_beliefs),
            fmt("OUR_DYNAMIC", &relationship_dynamics),
        ]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect();

        Some(parts.join("\n"))
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
            let events = event_log.load()?;
            let kept = events
                .into_iter()
                .filter(|ev| !id_set.contains(&ev.entry.id))
                .collect::<Vec<_>>();
            event_log.overwrite(&kept).await?;
        }
        Ok(())
    }

    // ── Identity helpers ───────────────────────────────────────────────────

    /// Extract the user's first name from the canonical Core identity entry.
    pub fn user_name_from_core(&self) -> Option<String> {
        for entry in self.entries_by_tier(MemoryTier::Core) {
            if let Some(idx) = entry.content.find("The user's name is ") {
                let after = &entry.content[idx + "The user's name is ".len()..];
                let name = after.split(['.', ',', '\n']).next()?.trim().to_string();
                if !name.is_empty() {
                    return Some(name);
                }
            }
        }
        None
    }

    // ── Agentic sleep ──────────────────────────────────────────────────────

    /// Build the LLM reflection prompt that drives the agentic sleep cycle.
    ///
    /// Derives bot and user names from the canonical Core identity entry so
    /// the prompt is always consistent with what was seeded at onboarding.
    pub fn agentic_sleep_prompt(&self) -> String {
        let (bot_name, user_name) = self.bot_and_user_names_from_core();
        crate::sleep::agentic_sleep_prompt(
            self.all(),
            &bot_name,
            &user_name,
            &self.identity.trait_scores,
        )
    }

    /// Apply structured `AgenticSleepInsights` produced by the LLM to memory,
    /// then run the standard heuristic distillation on top.
    ///
    /// Pipeline:
    ///   1. Persist each insight at the appropriate tier (no vault sync per record).
    ///   2. Apply Core rewrites / consolidations / retirements.
    ///   3. Run `run_sleep_cycle()` for heuristic distillation + vault sync.
    ///   4. Return a `SleepSummary` that covers everything.
    pub async fn apply_agentic_sleep_insights(
        &mut self,
        insights: AgenticSleepInsights,
        summary_text: Option<String>,
    ) -> Result<SleepSummary> {
        let mut extra_ids: Vec<String> = Vec::new();

        // learned_about_user → UserProfile
        for fact in &insights.learned_about_user {
            let e = self.record_inner(
                MemoryTier::UserProfile,
                fact.clone(),
                "sleep:learned-about-user".to_string(),
            ).await?;
            extra_ids.push(e.id.to_string());
        }

        // follow_ups → Reflective (source = "follow-up" so pending_follow_up_ids can find them)
        for item in &insights.follow_ups {
            let e = self.record_inner(
                MemoryTier::Reflective,
                item.clone(),
                "follow-up".to_string(),
            ).await?;
            extra_ids.push(e.id.to_string());
        }

        // reflective_thoughts → Reflective
        for thought in &insights.reflective_thoughts {
            let e = self.record_inner(
                MemoryTier::Reflective,
                thought.clone(),
                "sleep:reflection".to_string(),
            ).await?;
            extra_ids.push(e.id.to_string());
        }

        // relationship_milestones → Reflective (source = "sleep:relationship" routes to
        // OUR_DYNAMIC bucket in relational_state_block via source-based detection)
        for milestone in &insights.relationship_milestones {
            let e = self.record_inner(
                MemoryTier::Reflective,
                milestone.clone(),
                "sleep:relationship".to_string(),
            ).await?;
            extra_ids.push(e.id.to_string());
        }

        // keyed user-profile updates (replace by key)
        for (key, value) in &insights.user_profile_updates {
            if let Err(err) = self.record_user_profile_keyed(key, value, "sleep").await {
                warn!(?err, key, "apply_agentic_sleep_insights: user-profile update failed");
            }
        }

        // Core retirements
        for id_prefix in &insights.retire_core_ids {
            retire_core_by_prefix(
                &mut self.store,
                &self.event_log,
                &[id_prefix.as_str()],
            ).await?;
        }

        // Core rewrites: retire old entry + record rewritten version
        for (id_prefix, new_content) in &insights.rewrite_core {
            retire_core_by_prefix(
                &mut self.store,
                &self.event_log,
                &[id_prefix.as_str()],
            ).await?;
            let e = self.record_inner(
                MemoryTier::Core,
                new_content.clone(),
                "sleep:core-rewrite".to_string(),
            ).await?;
            extra_ids.push(e.id.to_string());
        }

        // Core consolidations: retire N originals + record synthesis
        for (ids_csv, synthesis) in &insights.consolidate_core {
            let prefixes: Vec<&str> = ids_csv.split(',').map(str::trim).collect();
            retire_core_by_prefix(
                &mut self.store,
                &self.event_log,
                &prefixes,
            ).await?;
            let e = self.record_inner(
                MemoryTier::Core,
                synthesis.clone(),
                "sleep:core-consolidation".to_string(),
            ).await?;
            extra_ids.push(e.id.to_string());
        }

        // Standard heuristic distillation + vault sync (includes backup).
        let mut summary = self.run_sleep_cycle().await?;
        summary.promoted_ids.extend(extra_ids);

        if let Some(text) = summary_text {
            summary.distilled = text;
        }

        Ok(summary)
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

    #[instrument(skip(self))]
    pub async fn seed_core_identity(&mut self, user_name: &str, bot_name: &str) -> Result<()> {
        let user_name = user_name.trim();
        let bot_name = bot_name.trim();
        if user_name.is_empty() || bot_name.is_empty() {
            bail!("user name and bot name are required for identity seeding");
        }

        let statement = format!(
            "You are {bot_name}, a helpful and truthful AI companion. \
             The user's name is {user_name}. \
             You have persistent multi-tier memory (Core, Semantic, Episodic, Procedural) \
             and can learn and remember information over time across conversations."
        );

        let already_present = self
            .entries_by_tier(MemoryTier::Core)
            .into_iter()
            .any(|entry| entry.source == "onboarding:identity");

        if already_present {
            debug!(bot_name, user_name, "core identity already seeded — skipping");
        } else {
            info!(bot_name, user_name, "seeding core identity");
            let entry = self.record(MemoryTier::Core, statement, "onboarding:identity").await?;
            info!(id = %entry.id, "core identity entry created");
        }

        Ok(())
    }

    fn sync_vault_projection(&self) -> Result<()> {
        if let Some(path) = &self.vault_path {
            export_obsidian_vault(self.all(), path)?;
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

/// Strip a recognised tag prefix (e.g. `"BELIEF:"`, `"MY_BELIEF:"`) from the
/// front of `s` (case-insensitive), trimming surrounding whitespace.
/// If none of the `prefixes` match, `s` is returned unchanged.
/// Strip a leading tag prefix from `s`, matching case-insensitively.
/// `prefixes` must be **lowercase ASCII** (e.g. `"belief:"`).  Only the first
/// 25 bytes of `s` are lowercased for comparison — the rest of the string is
/// returned verbatim — avoiding a full-length heap copy on long entries.
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
            .load()?
            .into_iter()
            .filter(|ev| !id_set.contains(&ev.entry.id))
            .collect::<Vec<_>>();
        log.overwrite(&kept).await?;
    }
    Ok(())
}

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

    #[tokio::test]
    async fn persists_and_replays_memory_entries() -> Result<()> {
        let path = std::env::temp_dir().join(format!("aigent-memory-{}.jsonl", Uuid::new_v4()));
        let mut manager = MemoryManager::with_event_log(&path)?;
        manager.record(MemoryTier::Episodic, "user asked for road map", "chat").await?;

        let replayed = MemoryManager::with_event_log(&path)?;
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

        let replayed = MemoryManager::with_event_log(&path)?;
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
        ).await?;;

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
