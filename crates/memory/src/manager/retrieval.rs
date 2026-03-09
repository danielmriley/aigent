//! Context fetching and ranking logic for [`MemoryManager`].

use chrono::Utc;
use uuid::Uuid;

use crate::retrieval::{RankedMemoryContext, assemble_context_with_provenance};
use crate::schema::{MemoryEntry, MemoryTier};
use crate::vault::read_kv_for_injection;

use super::{MemoryManager, contains_icase, strip_tag_prefix_lower};

impl MemoryManager {
    pub fn context_for_prompt(&self, limit: usize) -> Vec<MemoryEntry> {
        self.context_for_prompt_ranked("", limit)
            .into_iter()
            .map(|item| item.entry)
            .collect()
    }

    pub fn context_for_prompt_ranked(&self, query: &str, limit: usize) -> Vec<RankedMemoryContext> {
        self.context_for_prompt_ranked_with_embed(query, limit, None)
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
        // Single O(n) pass building both buckets simultaneously, halving
        // the memory working set compared to two independent filter passes.
        let mut core_entries: Vec<&MemoryEntry> = Vec::new();
        let mut non_core: Vec<&MemoryEntry> = Vec::new();
        for entry in self.store.all() {
            if entry.tier == MemoryTier::Core {
                core_entries.push(entry);
            } else if !entry.source.starts_with("assistant-turn")
                && entry.source != "sleep:cycle"
            {
                non_core.push(entry);
            }
        }
        // Pass a live-confidence resolver so entries ranked here reflect dynamic
        // confidence deltas from `record_confidence_signal`, not just the initial
        // anchor stored at record time.  O(1) per entry via the in-memory cache.
        let cf: &dyn Fn(Uuid) -> f32 = &|id: Uuid| self.current_confidence(id);
        let mut ranked = assemble_context_with_provenance(&non_core, &core_entries, query, limit, query_embedding, Some(cf));
        self.prepend_kv_identity_block(&mut ranked);
        ranked
    }

    /// Prepend a pinned KV identity block as score=2.0 entry when the vault
    /// summaries are available.  This guarantees the agent always knows who it
    /// is even when retrieval ranking would otherwise miss Core entries.
    fn prepend_kv_identity_block(&self, ranked: &mut Vec<RankedMemoryContext>) {
        let Some(vault_path) = &self.vault_path else { return };
        let Some(kv_block) = read_kv_for_injection(vault_path) else { return };
        ranked.insert(
            0,
            RankedMemoryContext {
                entry: MemoryEntry {
                    id: Uuid::nil(),
                    tier: MemoryTier::Core,
                    content: kv_block,
                    source: "kv_summary:auto_injected".to_string(),
                    confidence: 1.0,
                    valence: 0.0,
                    created_at: Utc::now(),
                    provenance_hash: "kv_summary".to_string(),
                    belief_kind: Default::default(),
                    tags: vec!["identity".to_string(), "auto_injected".to_string()],
                    embedding: None,
                    // Pinned synthetic entry — not scored lexically, skip tokenisation.
                    tokens: Default::default(),
                },
                score: 2.0,
                rationale: "pinned: KV identity summary auto-injected from vault".to_string(),
                live_confidence: 1.0,
            },
        );
    }

    /// Build a high-density relational matrix block for prompt injection.
    ///
    /// Merges `UserProfile` and `Reflective` entries into three compressed
    /// buckets:
    ///   `[USER: …]`        — facts and preferences about the user
    ///   `[MY_BELIEFS: …]`  — agent's own opinions / worldview stances
    ///   `[OUR_DYNAMIC: …]` — relationship tone, shared history, inside jokes
    ///
    /// **Routing priority** (highest → lowest):
    ///   1. **Tag-based** — entries with semantic tags assigned by the LLM
    ///      during the agentic sleep cycle are routed by tag name.
    ///   2. **Source-based** — entries from known sources (critic, belief,
    ///      psychologist, sleep:relationship) are routed by source string.
    ///   3. **Content-based fallback** — legacy entries without tags are
    ///      scanned for keywords (preserved for backward compatibility).
    ///
    /// `max_per_bucket` caps each bucket independently.  When `0`, all items
    /// are retained (not recommended for production — can produce 300 K+ char
    /// prompts).  Items are deduplicated by content and sorted most-recent-first
    /// before the cap is applied so fresh facts survive pruning.
    ///
    /// Returns `None` when all buckets are empty.
    pub fn relational_state_block(&self, max_per_bucket: usize) -> Option<String> {
        use std::collections::HashSet;

        // Collect (content, created_at) tuples so we can dedup + sort later.
        let mut user_facts: Vec<(String, chrono::DateTime<chrono::Utc>)> = Vec::new();
        let mut agent_beliefs: Vec<(String, chrono::DateTime<chrono::Utc>)> = Vec::new();
        let mut relationship_dynamics: Vec<(String, chrono::DateTime<chrono::Utc>)> = Vec::new();

        // Single pass — no intermediate Vec allocations from entries_by_tier.
        for entry in self.store.all().iter()
            .filter(|e| e.tier == MemoryTier::UserProfile || e.tier == MemoryTier::Reflective)
        {
            // ── Priority 1: Tag-based routing (LLM-assigned) ───────────
            let has_belief_tag = entry.tags.iter().any(|t| {
                t == "agent_belief" || t == "perspective" || t == "opinion"
            });
            let has_dynamic_tag = entry.tags.iter().any(|t| {
                t == "relationship" || t == "dynamic"
            });
            let has_user_fact_tag = entry.tags.iter().any(|t| {
                t == "user_fact" || t == "preference"
            });

            if has_belief_tag {
                agent_beliefs.push((strip_tag_prefix_lower(
                    &entry.content,
                    &["belief:", "my_belief:", "opinion:"],
                ), entry.created_at));
                continue;
            }
            if has_dynamic_tag {
                relationship_dynamics.push((strip_tag_prefix_lower(
                    &entry.content,
                    &["dynamic:", "our_dynamic:", "relationship:"],
                ), entry.created_at));
                continue;
            }
            if has_user_fact_tag {
                user_facts.push((entry.content.clone(), entry.created_at));
                continue;
            }

            // ── Priority 2: Source-based routing (zero-allocation) ─────
            let src = entry.source.as_str();
            let is_belief_src  = src.contains("critic")  || src.contains("belief");
            let is_dynamic_src = src.contains("psychologist") || src == "sleep:relationship";

            if is_belief_src {
                agent_beliefs.push((strip_tag_prefix_lower(
                    &entry.content,
                    &["belief:", "my_belief:", "opinion:"],
                ), entry.created_at));
            } else if is_dynamic_src {
                relationship_dynamics.push((strip_tag_prefix_lower(
                    &entry.content,
                    &["dynamic:", "our_dynamic:", "relationship:"],
                ), entry.created_at));
            } else {
                // ── Priority 3: Content keyword fallback (legacy) ──────
                // "think" and "shared" are intentionally excluded: both are
                // too broad and produce false positives on ordinary
                // user-fact entries.
                if contains_icase(&entry.content, "belief")
                    || contains_icase(&entry.content, "opinion")
                    || contains_icase(&entry.content, "feel about")
                    || contains_icase(&entry.content, "my_belief")
                {
                    agent_beliefs.push((strip_tag_prefix_lower(
                        &entry.content,
                        &["belief:", "my_belief:", "opinion:"],
                    ), entry.created_at));
                } else if contains_icase(&entry.content, "dynamic")
                    || contains_icase(&entry.content, "relationship")
                    || contains_icase(&entry.content, "joke")
                    || contains_icase(&entry.content, "rapport")
                    || contains_icase(&entry.content, "our_dynamic")
                {
                    relationship_dynamics.push((strip_tag_prefix_lower(
                        &entry.content,
                        &["dynamic:", "our_dynamic:", "relationship:"],
                    ), entry.created_at));
                } else {
                    user_facts.push((entry.content.clone(), entry.created_at));
                }
            }
        }

        if user_facts.is_empty() && agent_beliefs.is_empty() && relationship_dynamics.is_empty() {
            return None;
        }

        // ── Dedup + sort + cap each bucket ─────────────────────────────
        let dedup_sort_cap = |bucket: &mut Vec<(String, chrono::DateTime<chrono::Utc>)>| -> Vec<String> {
            // Dedup by content: keep the most recent occurrence of each
            // unique string.  Store u64 hashes instead of full String clones
            // to avoid O(n·L) allocations across all bucket entries.
            use std::hash::{Hash, Hasher};
            let mut seen: HashSet<u64> = HashSet::new();
            bucket.retain(|(content, _)| {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                content.hash(&mut h);
                seen.insert(h.finish())
            });
            // Most recent first.
            bucket.sort_by(|a, b| b.1.cmp(&a.1));
            let take_n = if max_per_bucket == 0 { bucket.len() } else { max_per_bucket.min(bucket.len()) };
            bucket[..take_n].iter().map(|(c, _)| c.clone()).collect()
        };

        let user_facts_final = dedup_sort_cap(&mut user_facts);
        let beliefs_final = dedup_sort_cap(&mut agent_beliefs);
        let dynamics_final = dedup_sort_cap(&mut relationship_dynamics);

        let fmt = |label: &str, items: &[String]| -> String {
            if items.is_empty() {
                return String::new();
            }
            format!("[{}: {}]", label, items.join("; "))
        };

        let parts: Vec<String> = [
            fmt("USER",        &user_facts_final),
            fmt("MY_BELIEFS",  &beliefs_final),
            fmt("OUR_DYNAMIC", &dynamics_final),
        ]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect();

        Some(parts.join("\n"))
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

}
