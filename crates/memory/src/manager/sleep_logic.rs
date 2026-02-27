//! Sleep prompt generation, consolidation triggers, and agentic sleep
//! execution for [`MemoryManager`].

use anyhow::{Result, bail};
use tracing::{debug, info, instrument, warn};

use crate::schema::MemoryTier;
use crate::sleep::{AgenticSleepInsights, SleepSummary, distill};

use super::{MemoryManager, retire_core_by_prefix};

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
            let e = self.record_inner_tagged(
                MemoryTier::Core,
                new_content.clone(),
                "sleep:core-rewrite".to_string(),
                vec!["core_rewrite".to_string()],
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
            let e = self.record_inner_tagged(
                MemoryTier::Core,
                synthesis.clone(),
                "sleep:core-consolidation".to_string(),
                vec!["core_consolidation".to_string()],
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

}
