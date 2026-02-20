use crate::schema::MemoryEntry;

#[derive(Debug, Clone, Copy)]
pub struct PromotionSignals {
    pub repetition_score: f32,
    pub emotional_salience: f32,
    pub user_confirmed_importance: f32,
    pub task_utility: f32,
}

pub fn is_core_eligible(entry: &MemoryEntry, signals: PromotionSignals) -> bool {
    if entry.content.trim().is_empty() {
        return false;
    }

    let aggregate = (signals.repetition_score
        + signals.emotional_salience
        + signals.user_confirmed_importance
        + signals.task_utility)
        / 4.0;

    aggregate >= 0.75 && entry.confidence >= 0.6
}
