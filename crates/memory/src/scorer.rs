use crate::schema::MemoryEntry;

#[derive(Debug, Clone, Copy)]
pub struct PromotionSignals {
    pub repetition_score: f32,
    pub emotional_salience: f32,
    pub user_confirmed_importance: f32,
    pub task_utility: f32,
    /// Bonus for entries that have survived across many days without being
    /// superseded (0.0 – 1.0; full credit at 30 days).
    pub longevity_bonus: f32,
}

pub fn is_core_eligible(entry: &MemoryEntry, signals: PromotionSignals) -> bool {
    if entry.content.trim().is_empty() {
        return false;
    }

    // Weighted formula so that longevity can tip durable observations over the
    // threshold without hard-coding equal importance for every signal.
    let aggregate = signals.repetition_score * 0.25
        + signals.emotional_salience * 0.25
        + signals.user_confirmed_importance * 0.25
        + signals.task_utility * 0.15
        + signals.longevity_bonus * 0.10;

    aggregate >= 0.75 && entry.confidence >= 0.6
}
