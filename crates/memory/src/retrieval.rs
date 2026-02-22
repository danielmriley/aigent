/// Memory retrieval and context assembly.
///
/// Scoring model (weights sum to 1.0):
/// ```text
/// score = tier(0.35) + recency(0.20) + lexical(0.25) + embedding(0.15) + confidence(0.05)
/// ```
/// When no embedding backend is available the embedding weight is redistributed
/// to lexical and recency.
use std::collections::{BTreeSet, HashSet};

use chrono::Utc;
use tracing::trace;

use crate::schema::{MemoryEntry, MemoryTier};

/// A ranked memory item with its computed score and human-readable rationale.
#[derive(Debug, Clone)]
pub struct RankedMemoryContext {
    pub entry: MemoryEntry,
    pub score: f32,
    pub rationale: String,
}

/// Build a ranked, deduplicated context window for LLM prompt injection.
///
/// * `matches`        – candidate non-Core entries (Episodic, Semantic, Procedural,
///   Reflective, UserProfile)
/// * `core_entries`   – Core entries (always injected; scored for ordering)
/// * `query`          – the user's current message (used for lexical relevance)
/// * `limit`          – maximum number of items returned
/// * `query_embedding`– optional pre-computed embedding of `query`; when
///   `Some`, hybrid lexical+vector scoring is used.
///
/// Core and UserProfile entries are given strong tier boosts so they tend to
/// appear first even with modest lexical overlap.
pub fn assemble_context_with_provenance(
    matches: Vec<MemoryEntry>,
    core_entries: Vec<MemoryEntry>,
    query: &str,
    limit: usize,
    query_embedding: Option<Vec<f32>>,
) -> Vec<RankedMemoryContext> {
    let mut combined = core_entries;
    combined.extend(matches);

    // Deduplicate by ID.
    let mut seen_ids = HashSet::new();
    combined.retain(|entry| seen_ids.insert(entry.id));

    let query_terms = tokenize(query);
    let query_emb_slice: &[f32] = query_embedding.as_deref().unwrap_or(&[]);
    let now = Utc::now();

    let mut ranked: Vec<RankedMemoryContext> = combined
        .into_iter()
        .map(|entry| {
            let embedding_sim = cosine_similarity_if_available(&entry, query_emb_slice);
            score_entry(&entry, &query_terms, now, embedding_sim)
        })
        .collect();

    ranked.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| right.entry.created_at.cmp(&left.entry.created_at))
    });

    ranked.into_iter().take(limit).collect()
}

/// Score a single entry given the current query terms and optional embedding similarity.
pub fn score_entry(
    entry: &MemoryEntry,
    query_terms: &BTreeSet<String>,
    now: chrono::DateTime<Utc>,
    embedding_cos_sim: Option<f32>,
) -> RankedMemoryContext {
    let tier_score = tier_priority(entry.tier);
    let recency = recency_score(now, entry.created_at);
    let lexical = lexical_relevance_score(&entry.content, query_terms);
    let confidence = entry.confidence.clamp(0.0, 1.0);

    let score = if let Some(emb) = embedding_cos_sim {
        // Hybrid: tier + recency + lexical + embedding + confidence
        (tier_score * 0.35) + (recency * 0.20) + (lexical * 0.25) + (emb * 0.15) + (confidence * 0.05)
    } else {
        // Lexical-only fallback: redistribute embedding weight to lexical/recency
        (tier_score * 0.35) + (recency * 0.25) + (lexical * 0.35) + (confidence * 0.05)
    };

    let rationale = match embedding_cos_sim {
        Some(emb) => format!(
            "tier={tier_score:.2}; recency={recency:.2}; lexical={lexical:.2}; emb={emb:.2}; conf={confidence:.2}"
        ),
        None => format!(
            "tier={tier_score:.2}; recency={recency:.2}; lexical={lexical:.2}; conf={confidence:.2}"
        ),
    };

    trace!(
        id = %entry.id,
        tier = ?entry.tier,
        score,
        %rationale,
        "scored memory entry"
    );

    RankedMemoryContext {
        entry: entry.clone(),
        score,
        rationale,
    }
}

/// Legacy helper: assemble without provenance metadata.
pub fn assemble_context(
    matches: Vec<MemoryEntry>,
    core_entries: Vec<MemoryEntry>,
) -> Vec<MemoryEntry> {
    assemble_context_with_provenance(matches, core_entries, "", 12, None)
        .into_iter()
        .map(|item| item.entry)
        .collect()
}

// ── Tier priority ─────────────────────────────────────────────────────────────

fn tier_priority(tier: MemoryTier) -> f32 {
    match tier {
        MemoryTier::Core => 1.00,
        MemoryTier::UserProfile => 0.90,
        MemoryTier::Reflective => 0.75,
        MemoryTier::Semantic => 0.65,
        MemoryTier::Procedural => 0.55,
        MemoryTier::Episodic => 0.40,
    }
}

// ── Recency ───────────────────────────────────────────────────────────────────

fn recency_score(now: chrono::DateTime<Utc>, created_at: chrono::DateTime<Utc>) -> f32 {
    let age_secs = (now - created_at).num_seconds().max(0) as f32;
    let age_hours = age_secs / 3600.0;
    // Half-life ~48 h — very recent memories score ≈1.0, week-old ≈0.35
    1.0 / (1.0 + (age_hours / 48.0))
}

// ── Lexical relevance ─────────────────────────────────────────────────────────

fn lexical_relevance_score(content: &str, query_terms: &BTreeSet<String>) -> f32 {
    if query_terms.is_empty() {
        return 0.0;
    }
    let content_terms = tokenize(content);
    let overlap = query_terms.intersection(&content_terms).count() as f32;
    overlap / query_terms.len() as f32
}

/// Common English stop words excluded from the lexical term set.
/// Filtering these prevents high-frequency words from creating false-positive
/// relevance scores that dilute genuine semantic matches.
const STOP_WORDS: &[&str] = &[
    "the", "and", "for", "was", "has", "are", "not", "this", "that",
    "with", "from", "have", "you", "can", "its", "will", "but", "they",
    "all", "been", "also", "into", "more", "than", "when", "who", "what",
    "how", "out", "our", "new", "now",
];

pub(crate) fn tokenize(text: &str) -> BTreeSet<String> {
    text.split(|ch: char| !ch.is_alphanumeric())
        .filter(|t| t.len() >= 3)
        .map(|t| t.to_lowercase())
        .filter(|t| !STOP_WORDS.contains(&t.as_str()))
        .collect()
}

// ── Embedding cosine similarity ───────────────────────────────────────────────

/// Returns cosine similarity between the entry's stored embedding and a fresh
/// query embedding, or `None` if either is absent.
///
/// `query_vec` is empty in the common case when no embedding backend is configured.
pub fn cosine_similarity_if_available(entry: &MemoryEntry, query_vec: &[f32]) -> Option<f32> {
    if query_vec.is_empty() {
        return None;
    }
    let entry_vec = entry.embedding.as_deref()?;
    Some(cosine_similarity(entry_vec, query_vec))
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    (dot / (mag_a * mag_b)).clamp(0.0, 1.0)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

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
            tags: Vec::new(),
            embedding: None,
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
            None,
        );

        assert_eq!(ranked.first().map(|item| item.entry.id), Some(core.id));
        Ok(())
    }

    #[test]
    fn user_profile_outranks_episodic_without_query_overlap() -> Result<()> {
        let profile = sample_entry(
            MemoryTier::UserProfile,
            "user prefers dark mode and concise answers",
            24,
        );
        let episodic = sample_entry(MemoryTier::Episodic, "some other fact", 1);

        let ranked =
            assemble_context_with_provenance(vec![episodic, profile.clone()], vec![], "", 2, None);

        assert_eq!(ranked.first().map(|item| item.entry.id), Some(profile.id));
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
            None,
        );

        assert_eq!(ranked.first().map(|item| item.entry.id), Some(relevant.id));
        Ok(())
    }

    #[test]
    fn reflective_tier_ranks_above_episodic() -> Result<()> {
        let reflective = sample_entry(
            MemoryTier::Reflective,
            "I should follow up on project X next time",
            2,
        );
        let episodic = sample_entry(MemoryTier::Episodic, "user mentioned project X briefly", 1);

        let ranked =
            assemble_context_with_provenance(vec![reflective.clone(), episodic], vec![], "", 2, None);

        assert_eq!(ranked.first().map(|item| item.entry.id), Some(reflective.id));
        Ok(())
    }

    #[test]
    fn embedding_similarity_lifts_matching_entry_above_unrelated() -> Result<()> {
        // Two Semantic entries with the same tier/recency; the one with a
        // similar embedding to the query should rank first.
        let mut close = sample_entry(MemoryTier::Semantic, "rust async performance tips", 1);
        let mut far = sample_entry(MemoryTier::Semantic, "buying groceries at the market", 1);

        // Query vector: [1, 0, 0]
        // Close entry embedding: [0.9, 0.1, 0]  (high cosine similarity)
        // Far entry embedding:   [0, 0, 1]       (orthogonal = 0 similarity)
        close.embedding = Some(vec![0.9_f32, 0.1, 0.0]);
        far.embedding = Some(vec![0.0_f32, 0.0, 1.0]);
        let query_embedding = Some(vec![1.0_f32, 0.0, 0.0]);

        let ranked = assemble_context_with_provenance(
            vec![close.clone(), far],
            vec![],
            "rust async",
            2,
            query_embedding,
        );

        assert_eq!(
            ranked.first().map(|item| item.entry.id),
            Some(close.id),
            "expected embedding-similar entry to rank first"
        );
        Ok(())
    }

    #[test]
    fn stop_words_are_excluded_from_tokenize() {
        let terms = super::tokenize("the project was a success");
        assert!(!terms.contains("the"), "'the' should be filtered as stop word");
        assert!(!terms.contains("was"), "'was' should be filtered as stop word");
        assert!(terms.contains("project"), "'project' should be present");
        assert!(terms.contains("success"), "'success' should be present");
    }
}
