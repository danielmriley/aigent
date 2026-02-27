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
        let mut ranked = assemble_context_with_provenance(&non_core, &core_entries, query, limit, query_embedding);
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
                    tags: vec!["identity".to_string(), "auto_injected".to_string()],
                    embedding: None,
                },
                score: 2.0,
                rationale: "pinned: KV identity summary auto-injected from vault".to_string(),
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
    /// Returns `None` when all buckets are empty.
    pub fn relational_state_block(&self) -> Option<String> {
        let mut user_facts: Vec<String> = Vec::new();
        let mut agent_beliefs: Vec<String> = Vec::new();
        let mut relationship_dynamics: Vec<String> = Vec::new();

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
                agent_beliefs.push(strip_tag_prefix_lower(
                    &entry.content,
                    &["belief:", "my_belief:", "opinion:"],
                ));
                continue;
            }
            if has_dynamic_tag {
                relationship_dynamics.push(strip_tag_prefix_lower(
                    &entry.content,
                    &["dynamic:", "our_dynamic:", "relationship:"],
                ));
                continue;
            }
            if has_user_fact_tag {
                user_facts.push(entry.content.clone());
                continue;
            }

            // ── Priority 2: Source-based routing (zero-allocation) ─────
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
                // ── Priority 3: Content keyword fallback (legacy) ──────
                // "think" and "shared" are intentionally excluded: both are
                // too broad and produce false positives on ordinary
                // user-fact entries.
                if contains_icase(&entry.content, "belief")
                    || contains_icase(&entry.content, "opinion")
                    || contains_icase(&entry.content, "feel about")
                    || contains_icase(&entry.content, "my_belief")
                {
                    agent_beliefs.push(strip_tag_prefix_lower(
                        &entry.content,
                        &["belief:", "my_belief:", "opinion:"],
                    ));
                } else if contains_icase(&entry.content, "dynamic")
                    || contains_icase(&entry.content, "relationship")
                    || contains_icase(&entry.content, "joke")
                    || contains_icase(&entry.content, "rapport")
                    || contains_icase(&entry.content, "our_dynamic")
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
