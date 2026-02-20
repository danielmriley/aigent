use std::collections::HashMap;

use crate::schema::{MemoryEntry, MemoryTier};
use crate::scorer::{PromotionSignals, is_core_eligible};

#[derive(Debug, Clone)]
pub struct SleepPromotion {
    pub source_id: String,
    pub to_tier: MemoryTier,
    pub reason: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct SleepSummary {
    pub distilled: String,
    pub promoted_ids: Vec<String>,
    pub promotions: Vec<SleepPromotion>,
}

pub fn distill(entries: &[MemoryEntry]) -> SleepSummary {
    let mut normalized_count: HashMap<String, usize> = HashMap::new();
    for entry in entries {
        let key = entry.content.trim().to_lowercase();
        if key.is_empty() {
            continue;
        }

        *normalized_count.entry(key).or_default() += 1;
    }

    let mut promotions = Vec::new();
    for entry in entries {
        if entry.tier == MemoryTier::Core {
            continue;
        }

        let repeats = normalized_count
            .get(&entry.content.trim().to_lowercase())
            .copied()
            .unwrap_or(1);
        let repetition_score = (repeats as f32 / 3.0).clamp(0.0, 1.0);
        let signals = PromotionSignals {
            repetition_score,
            emotional_salience: entry.valence.abs().clamp(0.0, 1.0),
            user_confirmed_importance: if entry.source.contains("user") {
                0.8
            } else {
                0.4
            },
            task_utility: if entry.content.len() > 32 { 0.8 } else { 0.5 },
        };

        let promoted_tier = if is_core_eligible(entry, signals) {
            MemoryTier::Core
        } else if entry.confidence >= 0.7 {
            MemoryTier::Semantic
        } else {
            continue;
        };

        promotions.push(SleepPromotion {
            source_id: entry.id.to_string(),
            to_tier: promoted_tier,
            reason: format!(
                "sleep-distilled repetition={repeats} confidence={:.2}",
                entry.confidence
            ),
            content: entry.content.clone(),
        });
    }

    let promoted_ids = promotions
        .iter()
        .map(|promotion| promotion.source_id.clone())
        .collect();
    let distilled = format!(
        "distilled {} memories, proposed {} promotions",
        entries.len(),
        promotions.len()
    );

    SleepSummary {
        distilled,
        promoted_ids,
        promotions,
    }
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use uuid::Uuid;

    use crate::schema::{MemoryEntry, MemoryTier};
    use crate::sleep::distill;

    #[test]
    fn distill_proposes_promotions_for_high_confidence_entries() {
        let entries = vec![MemoryEntry {
            id: Uuid::new_v4(),
            tier: MemoryTier::Episodic,
            content: "user prefers milestone check-ins for progress".to_string(),
            source: "user-chat".to_string(),
            confidence: 0.85,
            valence: 0.2,
            created_at: Utc::now(),
            provenance_hash: "test-hash".to_string(),
        }];

        let summary = distill(&entries);
        assert!(!summary.promotions.is_empty());
        assert!(summary.distilled.contains("distilled"));
    }
}
