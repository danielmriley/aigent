//! Sleep prompt generation, consolidation triggers, and agentic sleep
//! execution for [`MemoryManager`].

use std::collections::HashSet;

use anyhow::{Result, bail};
use chrono::{DateTime, Utc};
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

use crate::event_log::MemoryLogEvent;
use crate::index::NodeState;
use crate::schema::{
    BeliefConsolidatedEvent, BeliefKind, BeliefRelationshipEvent, ConfidenceReason,
    ConfidenceSource, EdgeKind, MemoryTier, SourceKind,
};
use crate::sleep::{AgenticSleepInsights, SleepSummary, distill};

use super::{ConsolidationFn, MemoryManager, retire_core_by_prefix};

// ── Phase 4 clustering helpers ────────────────────────────────────────────────

/// Snapshot of an Episodic entry assembled during the Pass 2 pre-scan.
///
/// Clones only the fields needed for clustering so the `MemoryStore` borrow
/// is released before any `&mut self` methods are called on the manager.
struct EpisodicCandidate {
    id: Uuid,
    belief_kind: BeliefKind,
    content: String,
    /// Raw source string — used to classify clusters as "repetitive"
    /// when all sources share the same prefix (e.g. `"assistant-reply"`).
    source: String,
    embedding: Option<Vec<f32>>,
    tokens: HashSet<String>,
    tags: Vec<String>,
    /// Initial anchor confidence from the `MemoryEntry` — used as the
    /// canonical confidence for the synthesised entry before any post-write
    /// confidence signals are applied.
    initial_conf: f32,
}

/// Cosine similarity between two equal-length `f32` vectors.
/// Returns 0.0 for zero vectors or mismatched lengths.
fn cosine_sim_vecs(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let ma = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mb = b.iter().map(|y| y * y).sum::<f32>().sqrt();
    if ma == 0.0 || mb == 0.0 {
        return 0.0;
    }
    (dot / (ma * mb)).clamp(0.0, 1.0)
}

/// Jaccard similarity between two token sets.
/// Returns 0.0 when the union is empty.
fn jaccard_tokens(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    let inter = a.intersection(b).count();
    let union_size = a.len() + b.len().saturating_sub(inter);
    if union_size == 0 {
        return 0.0;
    }
    inter as f32 / union_size as f32
}

/// Similarity score between two candidates.
///
/// Prefers embedding cosine similarity when **both** have embeddings;
/// falls back to token Jaccard otherwise.
fn cluster_similarity(a: &EpisodicCandidate, b: &EpisodicCandidate) -> f32 {
    if let (Some(ea), Some(eb)) = (&a.embedding, &b.embedding) {
        return cosine_sim_vecs(ea, eb);
    }
    jaccard_tokens(&a.tokens, &b.tokens)
}

/// Returns `true` when a cluster is "repetitive": all entries share the same
/// source prefix AND every all-pairs token Jaccard score is ≥ 0.70.
fn is_cluster_repetitive(cluster: &[&EpisodicCandidate]) -> bool {
    if cluster.len() < 2 {
        return false;
    }
    let prefix0 = cluster[0].source.split(':').next().unwrap_or("");
    if !cluster
        .iter()
        .all(|e| e.source.split(':').next().unwrap_or("") == prefix0)
    {
        return false;
    }
    for i in 0..cluster.len() {
        for j in (i + 1)..cluster.len() {
            if jaccard_tokens(&cluster[i].tokens, &cluster[j].tokens) < 0.70 {
                return false;
            }
        }
    }
    true
}

/// Dominant `BeliefKind` for a cluster: `Procedural` when more than half of
/// the entries are procedural; `Empirical` otherwise.
fn dominant_belief_kind(cluster: &[&EpisodicCandidate]) -> BeliefKind {
    let proc_count = cluster
        .iter()
        .filter(|e| e.belief_kind == BeliefKind::Procedural)
        .count();
    if proc_count > cluster.len() / 2 {
        BeliefKind::Procedural
    } else {
        BeliefKind::Empirical
    }
}

/// Build the LLM consolidation prompt for a non-repetitive cluster.
///
/// Truncates each observation to 300 characters so the prompt stays
/// within a reasonable token budget.
fn build_consolidation_prompt(cluster: &[&EpisodicCandidate]) -> String {
    let observations = cluster
        .iter()
        .enumerate()
        .map(|(i, e)| format!("  {}. {}", i + 1, &e.content[..e.content.len().min(300)]))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "You are a memory consolidation specialist.\n\
         The following episodic observations were recorded recently:\n\
         {observations}\n\n\
         Write ONE concise sentence capturing the key factual insight from these \
         observations, suitable for long-term semantic memory storage. \
         Return ONLY the distilled belief sentence."
    )
}

impl MemoryManager {
    #[instrument(skip(self))]
    pub async fn run_sleep_cycle(&mut self) -> Result<SleepSummary> {
        // ── Pre-sleep deduplication ────────────────────────────────────
        // Remove content-identical entries before distillation so the
        // repetition counter in `distill()` works on a clean slate and
        // the LLM doesn't see bloated memory counts.
        let deduped = self.deduplicate_by_content().await?;
        if deduped > 0 {
            info!(deduped, "pre-sleep deduplication removed entries");
        }

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
            event_log.backup().await?;
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
        self.invalidate_prompt_caches();
        info!(promoted = summary.promoted_ids.len(), "sleep cycle complete");
        Ok(summary)
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
    ///   1. Persist each insight at the appropriate tier with semantic tags.
    ///   2. Apply Core rewrites / consolidations / retirements.
    ///   3. Apply LLM-driven promotions (Phase 2).
    ///   4. Apply free-form memories (Phase 4).
    ///   5. Run `run_sleep_cycle()` for heuristic distillation + vault sync.
    ///   6. Return a `SleepSummary` that covers everything.
    pub async fn apply_agentic_sleep_insights(
        &mut self,
        insights: AgenticSleepInsights,
        summary_text: Option<String>,
    ) -> Result<SleepSummary> {
        let mut extra_ids: Vec<String> = Vec::new();

        // learned_about_user → UserProfile (tagged)
        for fact in &insights.learned_about_user {
            let e = self.record_inner_tagged(
                MemoryTier::UserProfile,
                fact.clone(),
                "sleep:learned-about-user".to_string(),
                vec!["user_fact".to_string()],
                None,
                BeliefKind::default(),
            ).await?;
            extra_ids.push(e.id.to_string());
        }

        // follow_ups → Reflective (tagged)
        for item in &insights.follow_ups {
            let e = self.record_inner_tagged(
                MemoryTier::Reflective,
                item.clone(),
                "follow-up".to_string(),
                vec!["follow_up".to_string()],
                None,
                BeliefKind::default(),
            ).await?;
            extra_ids.push(e.id.to_string());
        }

        // reflective_thoughts → Reflective (tagged)
        for thought in &insights.reflective_thoughts {
            let e = self.record_inner_tagged(
                MemoryTier::Reflective,
                thought.clone(),
                "sleep:reflection".to_string(),
                vec!["reflection".to_string()],
                None,
                BeliefKind::SelfModel,
            ).await?;
            extra_ids.push(e.id.to_string());
        }

        // relationship_milestones → Reflective (tagged with relationship + dynamic)
        for milestone in &insights.relationship_milestones {
            let e = self.record_inner_tagged(
                MemoryTier::Reflective,
                milestone.clone(),
                "sleep:relationship".to_string(),
                vec!["relationship".to_string(), "dynamic".to_string()],
                None,
                BeliefKind::default(),
            ).await?;
            extra_ids.push(e.id.to_string());
        }

        // perspectives → Semantic (tagged with agent_belief + perspective)
        for (topic, view) in &insights.perspectives {
            let content = format!("{topic}: {view}");
            let e = self.record_inner_tagged(
                MemoryTier::Semantic,
                content,
                "sleep:perspective".to_string(),
                vec!["agent_belief".to_string(), "perspective".to_string()],
                None,
                BeliefKind::Opinion,
            ).await?;
            extra_ids.push(e.id.to_string());
        }

        // contradictions → Semantic (tagged)
        for contradiction in &insights.contradictions {
            let e = self.record_inner_tagged(
                MemoryTier::Semantic,
                contradiction.clone(),
                "sleep:contradiction".to_string(),
                vec!["contradiction".to_string()],
                None,
                BeliefKind::default(),
            ).await?;
            extra_ids.push(e.id.to_string());
        }

        // tool_insights → Procedural (tagged)
        for insight_text in &insights.tool_insights {
            let e = self.record_inner_tagged(
                MemoryTier::Procedural,
                insight_text.clone(),
                "sleep:tool-insight".to_string(),
                vec!["tool_pattern".to_string()],
                None,
                BeliefKind::Procedural,
            ).await?;
            extra_ids.push(e.id.to_string());
        }

        // synthesis → Semantic (tagged)
        for synth in &insights.synthesis {
            let e = self.record_inner_tagged(
                MemoryTier::Semantic,
                synth.clone(),
                "sleep:synthesis".to_string(),
                vec!["synthesis".to_string()],
                None,
                BeliefKind::default(),
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
                &mut self.index,
                &[id_prefix.as_str()],
            ).await?;
        }

        // Non-Core memory retirements (Episodic/Semantic/Reflective/Procedural).
        // Delete outright from store + event log so entry counts stay bounded.
        if !insights.retire_memory_ids.is_empty() {
            let retire_ids: Vec<uuid::Uuid> = self
                .store
                .all()
                .iter()
                .filter(|e| {
                    e.tier != MemoryTier::Core
                        && e.tier != MemoryTier::UserProfile
                        && insights
                            .retire_memory_ids
                            .iter()
                            .any(|prefix| e.id.to_string().starts_with(prefix.as_str()))
                })
                .map(|e| e.id)
                .collect();
            if !retire_ids.is_empty() {
                let id_set: std::collections::HashSet<uuid::Uuid> =
                    retire_ids.iter().copied().collect();
                // Persist to disk first.
                if let Some(log) = &self.event_log {
                    let kept = log
                        .load()
                        .await?
                        .into_iter()
                        .filter(|ev| !id_set.contains(&ev.entry.id))
                        .collect::<Vec<_>>();
                    log.overwrite(&kept).await?;
                }
                let removed = self.store.retain(|e| !id_set.contains(&e.id));
                self.index_remove_ids(&id_set);
                info!(
                    requested = insights.retire_memory_ids.len(),
                    removed,
                    "non-Core memory retirement applied"
                );
            }
        }

        // Non-Core memory retractions (soft tombstone — zeroes in-memory
        // confidence so retrieval ignores the entry, appends a tombstone
        // event so the retraction survives a restart).  Use this path for
        // entries that are FACTUALLY WRONG rather than just redundant.
        if !insights.retract_memory_ids.is_empty() {
            for id_prefix in &insights.retract_memory_ids {
                let target = self
                    .store
                    .all()
                    .iter()
                    .find(|e| {
                        !matches!(e.tier, MemoryTier::Core | MemoryTier::UserProfile)
                            && e.id.to_string().starts_with(id_prefix.as_str())
                    })
                    .map(|e| (e.id, e.content.chars().take(80).collect::<String>()));

                if let Some((target_id, preview)) = target {
                    match self
                        .retract_memory(target_id, "sleep:contradicts-observed-behavior")
                        .await
                    {
                        Ok(()) => {
                            info!(%target_id, %preview, "sleep retraction applied");
                        }
                        Err(ref e) => {
                            warn!(%target_id, err = %e, "retract_memory failed — skipping");
                        }
                    }
                } else {
                    warn!(
                        id_prefix,
                        "RETRACT: no matching non-Core entry found — may have been retired already"
                    );
                }
            }
        }


        for (id_prefix, new_content) in &insights.rewrite_core {
            retire_core_by_prefix(
                &mut self.store,
                &self.event_log,
                &mut self.index,
                &[id_prefix.as_str()],
            ).await?;
            let e = self.record_inner_tagged(
                MemoryTier::Core,
                new_content.clone(),
                "sleep:core-rewrite".to_string(),
                vec!["core_rewrite".to_string()],
                None,
                BeliefKind::default(),
            ).await?;
            extra_ids.push(e.id.to_string());
        }

        // Core consolidations: retire N originals + record synthesis
        for (ids_csv, synthesis) in &insights.consolidate_core {
            let prefixes: Vec<&str> = ids_csv.split(',').map(str::trim).collect();
            retire_core_by_prefix(
                &mut self.store,
                &self.event_log,
                &mut self.index,
                &prefixes,
            ).await?;
            let e = self.record_inner_tagged(
                MemoryTier::Core,
                synthesis.clone(),
                "sleep:core-consolidation".to_string(),
                vec!["core_consolidation".to_string()],
                None,
                BeliefKind::default(),
            ).await?;
            extra_ids.push(e.id.to_string());
        }

        // ── Phase 2: LLM-driven promotions ─────────────────────────────────
        //
        // The LLM reviews existing entries and decides which deserve
        // promotion to a higher tier, replacing the heuristic scoring.
        for (id_prefix, tier_label) in &insights.llm_promotions {
            let Some(target_tier) = MemoryTier::from_label(tier_label) else {
                warn!(tier_label, id_prefix, "LLM promotion: unknown tier label — skipping");
                continue;
            };
            // Find the entry by id_short prefix.
            let source_entry = self.store.all().iter().find(|e| {
                e.id.to_string().starts_with(id_prefix.as_str())
            }).cloned();
            let Some(entry) = source_entry else {
                warn!(id_prefix, "LLM promotion: no entry matches id_short — skipping");
                continue;
            };
            // Skip if already at the target tier (or higher).
            if entry.tier == target_tier {
                debug!(id_prefix, "LLM promotion: entry already at target tier — skipping");
                continue;
            }
            // Record the promoted copy at the new tier with provenance.
            let e = self.record_inner_tagged(
                target_tier,
                entry.content.clone(),
                format!("sleep:llm-promoted-from-{:?}", entry.tier),
                {
                    let mut tags = entry.tags.clone();
                    tags.push("llm_promoted".to_string());
                    tags.dedup();
                    tags
                },
                None,
                entry.belief_kind,
            ).await?;
            info!(
                from_tier = ?entry.tier,
                to_tier = ?target_tier,
                id = %e.id,
                "LLM-driven promotion applied"
            );
            extra_ids.push(e.id.to_string());
        }

        // ── Phase 4: Free-form memory proposals ────────────────────────────
        //
        // The LLM can create any memory it wants, specifying tier, content,
        // and tags.  This gives maximal agency over memory formation.
        for (tier_label, content, tags_csv) in &insights.free_memories {
            let Some(tier) = MemoryTier::from_label(tier_label) else {
                warn!(tier_label, "free memory: unknown tier label — skipping");
                continue;
            };
            let tags: Vec<String> = tags_csv
                .split(',')
                .map(|t| t.trim().to_lowercase())
                .filter(|t| !t.is_empty())
                .collect();
            let e = self.record_inner_tagged(
                tier,
                content.clone(),
                "sleep:free-memory".to_string(),
                tags,
                None,
                BeliefKind::default(),
            ).await?;
            info!(
                tier = ?tier,
                id = %e.id,
                tag_count = e.tags.len(),
                "free-form memory recorded"
            );
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
            .any(|entry| matches!(entry.source_kind(), SourceKind::OnboardingIdentity));

        if already_present {
            debug!(bot_name, user_name, "core identity already seeded — skipping");
        } else {
            info!(bot_name, user_name, "seeding core identity");
            let entry = self.record(MemoryTier::Core, statement, "onboarding:identity").await?;
            info!(id = %entry.id, "core identity entry created");
        }

        Ok(())
    }

    // ── Phase 4 — Sleep Pass 1: stale-decay ──────────────────────────────────

    /// Decay the confidence of entries that have not been accessed in the last
    /// 24 hours and immediately transition zero-confidence entries to
    /// `NodeState::Decayed` (removing them from the working store).
    ///
    /// **Decay rates by BeliefKind** (applied as a negative delta per run):
    /// - `Empirical`  → −0.03
    /// - `Procedural` → −0.01  (procedural knowledge is sticky)
    /// - `SelfModel`  → −0.02
    /// - `Opinion`    → −0.005 (opinions fade slowly)
    ///
    /// Returns the number of entries that were fully decayed (transitioned to
    /// `NodeState::Decayed`).
    ///
    /// Complexity: O(n) to snapshot the store (unavoidable — every entry must
    /// be checked against the registry), then O(log n) per entry for the redb
    /// registry read and confidence signal write.
    pub async fn run_sleep_pass_1_decay(&mut self) -> Result<usize> {
        let stale_cutoff: DateTime<Utc> = Utc::now() - chrono::Duration::hours(24);

        // Snapshot (id, belief_kind, created_at) to release the &self.store borrow
        // before calling &mut self methods (record_confidence_signal, decay_node).
        let snapshot: Vec<(Uuid, BeliefKind, DateTime<Utc>)> = self
            .store
            .all()
            .iter()
            .map(|e| (e.id, e.belief_kind, e.created_at))
            .collect();

        // Identify stale entries: created before the cutoff AND last accessed
        // before the cutoff (read from the redb node registry; fall back to
        // created_at when no registry entry is present).
        let stale: Vec<(Uuid, BeliefKind)> = snapshot
            .iter()
            .filter(|(id, _, created_at)| {
                if *created_at >= stale_cutoff {
                    return false; // freshly created — not stale yet
                }
                let last_accessed = self
                    .index
                    .as_ref()
                    .and_then(|idx| idx.get_node_registry(id).ok().flatten())
                    .map(|reg| reg.last_accessed_at)
                    .unwrap_or(*created_at);
                last_accessed < stale_cutoff
            })
            .map(|(id, kind, _)| (*id, *kind))
            .collect();

        let mut decayed_count = 0usize;

        for (id, belief_kind) in stale {
            // Decay rate is parameterised by semantic category.
            let delta: f32 = match belief_kind {
                BeliefKind::Empirical  => -0.03,
                BeliefKind::Procedural => -0.01,
                BeliefKind::SelfModel  => -0.02,
                BeliefKind::Opinion    => -0.005,
            };

            if let Err(e) = self
                .record_confidence_signal(
                    id,
                    delta,
                    ConfidenceReason::StaleDecay,
                    ConfidenceSource::SleepPipeline { pass: 1 },
                )
                .await
            {
                warn!(error = ?e, %id, "Pass 1: record_confidence_signal failed");
                continue;
            }

            // Once confidence reaches zero the entry is no longer useful;
            // transition it to NodeState::Decayed and remove it from the store.
            if self.current_confidence(id) <= 0.0 {
                match self.decay_node(id) {
                    Ok(()) => {
                        decayed_count += 1;
                        debug!(%id, "Pass 1: entry decayed to zero confidence and removed");
                    }
                    Err(e) => warn!(error = ?e, %id, "Pass 1: decay_node failed (non-fatal)"),
                }
            }
        }

        if decayed_count > 0 {
            info!(decayed_count, "sleep pass 1 (stale decay) complete");
        }
        Ok(decayed_count)
    }

    // ── Phase 4 — Sleep Pass 2: episodic consolidation ────────────────────────

    /// Consolidate similar Episodic entries into a single Semantic or
    /// Procedural canonical belief.
    ///
    /// Pipeline:
    /// 1. Collect all Episodic entries older than 24 hours ("cold" window).
    /// 2. Greedy O(n²) clustering: entries are joined into a cluster when their
    ///    similarity score exceeds the relevant threshold
    ///    (embedding cosine ≥ 0.80 **or** Jaccard token overlap ≥ 0.50).
    /// 3. Single-entry clusters are isolation cases — skipped entirely.
    /// 4. Clusters where any source entry has active dependents are deferred
    ///    until those dependents decay or are consolidated first.
    /// 5. A canonical entry is written:
    ///    - *Repetitive* cluster (same source prefix + Jaccard ≥ 0.70): use
    ///      the highest-confidence member's content (heuristic dedup).
    ///    - *Non-repetitive*: call `consolidation_fn` for an LLM-distilled
    ///      sentence; fall back to highest-confidence member on failure.
    /// 6. Source entries are transitioned to `NodeState::Consolidated` with a
    ///    forwarding pointer to the canonical id, removed from the working
    ///    store, and graph edges are written to redb + the event log.
    ///
    /// Returns the number of clusters that were successfully consolidated.
    ///
    /// Complexity: O(n²) clustering pass where n = Episodic entry count.
    /// Acceptable because Pass 2 only runs during nightly sleep; n is expected
    /// to be O(hundreds) of entries at most.
    pub async fn run_sleep_pass_2_consolidation(
        &mut self,
        consolidation_fn: Option<&ConsolidationFn>,
    ) -> Result<usize> {
        // Only operate on entries that have left the "hot" 24-hour window.
        let hot_cutoff: DateTime<Utc> = Utc::now() - chrono::Duration::hours(24);

        // Step 1: snapshot Episodic candidates — releases the &self.store borrow
        // so we can call &mut self methods later without borrow-checker conflicts.
        let candidates: Vec<EpisodicCandidate> = self
            .store
            .all()
            .iter()
            .filter(|e| e.tier == MemoryTier::Episodic && e.created_at < hot_cutoff)
            .map(|e| EpisodicCandidate {
                id: e.id,
                belief_kind: e.belief_kind,
                content: e.content.clone(),
                source: e.source.clone(),
                embedding: e.embedding.clone(),
                tokens: e.tokens.clone(),
                tags: e.tags.clone(),
                initial_conf: e.confidence,
            })
            .collect();

        if candidates.is_empty() {
            return Ok(0);
        }

        // Step 2: greedy O(n²) single-linkage clustering.
        // Use embedding cosine when BOTH entries have embeddings; Jaccard otherwise.
        const EMBED_THRESHOLD: f32 = 0.80;
        const JACCARD_THRESHOLD: f32 = 0.50;

        let n = candidates.len();
        let mut assigned = vec![false; n];
        let mut clusters: Vec<Vec<usize>> = Vec::new();

        for i in 0..n {
            if assigned[i] {
                continue;
            }
            assigned[i] = true;
            let mut cluster = vec![i];
            for j in (i + 1)..n {
                if assigned[j] {
                    continue;
                }
                let sim = cluster_similarity(&candidates[i], &candidates[j]);
                let threshold =
                    if candidates[i].embedding.is_some() && candidates[j].embedding.is_some() {
                        EMBED_THRESHOLD
                    } else {
                        JACCARD_THRESHOLD
                    };
                if sim >= threshold {
                    assigned[j] = true;
                    cluster.push(j);
                }
            }
            clusters.push(cluster);
        }

        let mut consolidated_count = 0usize;

        for cluster_idxs in clusters {
            // Isolation case: a single-entry cluster cannot be consolidated —
            // there is nothing to merge it with.
            if cluster_idxs.len() < 2 {
                continue;
            }

            let cluster: Vec<&EpisodicCandidate> =
                cluster_idxs.iter().map(|&i| &candidates[i]).collect();

            // Orphan guard: defer the whole cluster if any source entry still
            // has active dependents (a downstream belief actively references it).
            let mut defer = false;
            for entry in &cluster {
                match self.has_active_dependents(entry.id) {
                    Ok(true) => {
                        defer = true;
                        break;
                    }
                    Ok(false) => {}
                    Err(e) => {
                        warn!(
                            error = ?e, id = %entry.id,
                            "Pass 2: has_active_dependents lookup failed"
                        );
                    }
                }
            }
            if defer {
                debug!(
                    cluster_size = cluster.len(),
                    "Pass 2: cluster deferred (active dependents)"
                );
                continue;
            }

            // Determine target tier and kind for the consolidated entry.
            let avg_conf: f32 =
                cluster.iter().map(|e| e.initial_conf).sum::<f32>() / cluster.len() as f32;
            let target_kind = dominant_belief_kind(&cluster);
            let target_tier = if target_kind == BeliefKind::Procedural {
                MemoryTier::Procedural
            } else {
                MemoryTier::Semantic
            };
            // Union of all source tags (deduplicated).
            let all_tags: Vec<String> = cluster
                .iter()
                .flat_map(|e| e.tags.iter().cloned())
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();

            // Generate the consolidated content.
            let consolidated_content = if is_cluster_repetitive(&cluster) {
                // Repetitive cluster: use the highest-confidence entry's content.
                cluster
                    .iter()
                    .max_by(|a, b| {
                        a.initial_conf
                            .partial_cmp(&b.initial_conf)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|e| e.content.clone())
                    .unwrap_or_default()
            } else if let Some(fn_ref) = consolidation_fn {
                let prompt = build_consolidation_prompt(&cluster);
                match fn_ref(prompt).await {
                    Ok(distilled) if !distilled.trim().is_empty() => {
                        distilled.trim().to_string()
                    }
                    Ok(_) => {
                        // LLM returned empty — fall back to heuristic.
                        cluster
                            .iter()
                            .max_by(|a, b| {
                                a.initial_conf
                                    .partial_cmp(&b.initial_conf)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|e| e.content.clone())
                            .unwrap_or_default()
                    }
                    Err(e) => {
                        warn!(
                            error = ?e,
                            "Pass 2: LLM consolidation failed — using heuristic fallback"
                        );
                        cluster
                            .iter()
                            .max_by(|a, b| {
                                a.initial_conf
                                    .partial_cmp(&b.initial_conf)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|e| e.content.clone())
                            .unwrap_or_default()
                    }
                }
            } else {
                // No LLM configured — heuristic fallback for non-repetitive clusters.
                cluster
                    .iter()
                    .max_by(|a, b| {
                        a.initial_conf
                            .partial_cmp(&b.initial_conf)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|e| e.content.clone())
                    .unwrap_or_default()
            };

            if consolidated_content.is_empty() {
                continue;
            }

            // Write the canonical consolidated entry.
            // Bounds avg_conf to [0.15, 0.85] so the new entry never inherits
            // the extremes (0.0 would be immediately pruned; 1.0 is reserved
            // for manually verified Core facts).
            let canonical_entry = self
                .record_inner_tagged(
                    target_tier,
                    consolidated_content,
                    "sleep:consolidation:pass2".to_string(),
                    all_tags,
                    Some(avg_conf.clamp(0.15, 0.85)),
                    target_kind,
                )
                .await?;
            let canonical_id = canonical_entry.id;
            let source_ids: Vec<Uuid> = cluster.iter().map(|e| e.id).collect();

            // Append BeliefConsolidatedEvent to the event log.
            if let Some(log) = &self.event_log {
                let _ = log
                    .append_log_event(&MemoryLogEvent::BeliefConsolidated(
                        BeliefConsolidatedEvent {
                            event_id: Uuid::new_v4(),
                            occurred_at: Utc::now(),
                            source_ids: source_ids.clone(),
                            canonical_id,
                            sleep_pass: 2,
                        },
                    ))
                    .await;
            }

            // Append BeliefRelationshipEvent (DerivedFrom) for each source
            // and register the directed edge in redb.
            for &src_id in &source_ids {
                if let Some(log) = &self.event_log {
                    let _ = log
                        .append_log_event(&MemoryLogEvent::BeliefRelationship(
                            BeliefRelationshipEvent {
                                event_id: Uuid::new_v4(),
                                occurred_at: Utc::now(),
                                source_id: canonical_id,
                                target_id: src_id,
                                edge_kind: EdgeKind::DerivedFrom,
                                relationship_confidence: avg_conf,
                            },
                        ))
                        .await;
                }
                if let Some(idx) = &mut self.index {
                    if let Err(e) = idx.add_edge(&canonical_id, EdgeKind::DerivedFrom, &src_id) {
                        warn!(
                            error = ?e, %canonical_id, %src_id,
                            "Pass 2: add_edge(DerivedFrom) failed"
                        );
                    }
                }
            }

            // Transition source entries to NodeState::Consolidated with a
            // forwarding pointer to the canonical, then remove them from the
            // working store and confidence override cache.
            for &src_id in &source_ids {
                if let Some(idx) = &mut self.index {
                    if let Err(e) =
                        idx.set_node_state(&src_id, NodeState::Consolidated, Some(canonical_id))
                    {
                        warn!(error = ?e, %src_id, "Pass 2: set_node_state(Consolidated) failed");
                    }
                    if let Err(e) = idx.remove(&src_id) {
                        warn!(error = ?e, %src_id, "Pass 2: index.remove for consolidated source failed");
                    }
                }
                self.confidence_overrides.remove(&src_id);
            }

            // Remove source entries from the working store in a single pass.
            let src_set: HashSet<Uuid> = source_ids.iter().copied().collect();
            self.store.retain(|e| !src_set.contains(&e.id));

            consolidated_count += 1;
            info!(
                canonical = %canonical_id,
                sources = source_ids.len(),
                tier = ?target_tier,
                "Pass 2: cluster consolidated"
            );
        }

        if consolidated_count > 0 {
            info!(consolidated_count, "sleep pass 2 (consolidation) complete");
        }
        Ok(consolidated_count)
    }

}
