use std::collections::{BTreeSet, HashSet};

use chrono::Utc;

use crate::schema::{MemoryEntry, MemoryTier};

#[derive(Debug, Clone)]
pub struct RankedMemoryContext {
    pub entry: MemoryEntry,
    pub score: f32,
    pub rationale: String,
}

pub fn assemble_context_with_provenance(
    matches: Vec<MemoryEntry>,
    core_entries: Vec<MemoryEntry>,
    query: &str,
    limit: usize,
) -> Vec<RankedMemoryContext> {
    let mut combined = core_entries;
    combined.extend(matches);

    let mut seen_ids = HashSet::new();
    combined.retain(|entry| seen_ids.insert(entry.id));

    let query_terms = tokenize(query);
    let now = Utc::now();

    let mut ranked = combined
        .into_iter()
        .map(|entry| {
            let tier_score = tier_priority(entry.tier);
            let recency_score = recency_score(now, entry.created_at);
            let relevance_score = relevance_score(&entry.content, &query_terms);
            let confidence_score = entry.confidence.clamp(0.0, 1.0);

            let score = (tier_score * 0.4)
                + (recency_score * 0.25)
                + (relevance_score * 0.25)
                + (confidence_score * 0.1);

            let rationale = format!(
                "tier={tier_score:.2}; recency={recency_score:.2}; relevance={relevance_score:.2}; confidence={confidence_score:.2}"
            );

            RankedMemoryContext {
                entry,
                score,
                rationale,
            }
        })
        .collect::<Vec<_>>();

    ranked.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| right.entry.created_at.cmp(&left.entry.created_at))
    });

    ranked.into_iter().take(limit).collect()
}

pub fn assemble_context(
    matches: Vec<MemoryEntry>,
    core_entries: Vec<MemoryEntry>,
) -> Vec<MemoryEntry> {
    assemble_context_with_provenance(matches, core_entries, "", 12)
        .into_iter()
        .map(|item| item.entry)
        .collect()
}

fn tier_priority(tier: MemoryTier) -> f32 {
    match tier {
        MemoryTier::Core => 1.0,
        MemoryTier::Semantic => 0.75,
        MemoryTier::Procedural => 0.6,
        MemoryTier::Episodic => 0.5,
    }
}

fn recency_score(now: chrono::DateTime<Utc>, created_at: chrono::DateTime<Utc>) -> f32 {
    let age_seconds = (now - created_at).num_seconds().max(0) as f32;
    let age_hours = age_seconds / 3600.0;
    1.0 / (1.0 + (age_hours / 24.0))
}

fn relevance_score(content: &str, query_terms: &BTreeSet<String>) -> f32 {
    if query_terms.is_empty() {
        return 0.0;
    }

    let content_terms = tokenize(content);
    let overlap = query_terms.intersection(&content_terms).count() as f32;
    overlap / query_terms.len() as f32
}

fn tokenize(text: &str) -> BTreeSet<String> {
    text.split(|ch: char| !ch.is_alphanumeric())
        .filter(|token| token.len() >= 3)
        .map(|token| token.to_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use chrono::{Duration, Utc};
    use uuid::Uuid;

    use super::assemble_context_with_provenance;
    use crate::schema::{MemoryEntry, MemoryTier};

    fn sample_entry(tier: MemoryTier, content: &str, age_hours: i64) -> MemoryEntry {
        MemoryEntry {
            id: Uuid::new_v4(),
            tier,
            content: content.to_string(),
            source: "test".to_string(),
            confidence: 0.8,
            valence: 0.0,
            created_at: Utc::now() - Duration::hours(age_hours),
            provenance_hash: "hash".to_string(),
        }
    }

    #[test]
    fn core_memory_is_ranked_first_when_other_factors_are_similar() -> Result<()> {
        let core = sample_entry(MemoryTier::Core, "my name is aigent", 2);
        let semantic = sample_entry(MemoryTier::Semantic, "my name is aigent", 2);

        let ranked = assemble_context_with_provenance(
            vec![semantic],
            vec![core.clone()],
            "what is your name",
            2,
        );

        assert_eq!(ranked.first().map(|item| item.entry.id), Some(core.id));
        Ok(())
    }

    #[test]
    fn query_overlap_increases_rank_for_relevant_memories() -> Result<()> {
        let unrelated = sample_entry(MemoryTier::Semantic, "i enjoy mountain hiking", 1);
        let relevant = sample_entry(
            MemoryTier::Semantic,
            "user prefers milestone-based plans",
            6,
        );

        let ranked = assemble_context_with_provenance(
            vec![unrelated, relevant.clone()],
            vec![],
            "create milestone project plan",
            2,
        );

        assert_eq!(ranked.first().map(|item| item.entry.id), Some(relevant.id));
        Ok(())
    }
}
