//! Simple heuristic sentiment/valence inference for memory entries.
//!
//! Returns a value in `[-1.0, 1.0]`.  This is intentionally a rough signal;
//! the goal is to give `emotional_salience` a non-zero starting point rather
//! than to replace a full sentiment model.

const POSITIVE_WORDS: &[&str] = &[
    "great", "love", "excited", "happy", "amazing", "solved", "success",
    "excellent", "wonderful", "fantastic", "glad", "pleased", "proud",
    "brilliant", "perfect", "works", "fixed", "done", "achieved", "helpful",
    "thanks", "awesome", "enjoy", "like", "good", "nice", "yes",
];

const NEGATIVE_WORDS: &[&str] = &[
    "frustrated", "confused", "error", "failed", "worried", "stuck",
    "broken", "terrible", "awful", "wrong", "bad", "hate", "annoying",
    "difficult", "struggle", "issue", "bug", "crash", "problem", "not",
    "cannot", "unable", "fail", "loss", "lost", "miss", "missing",
];

/// Infer an emotional valence score for `content` using keyword heuristics.
///
/// Returns a value clamped to `[-1.0, 1.0]`.  Positive content scores > 0,
/// negative/distressing content scores < 0, and neutral prose scores near 0.
pub fn infer_valence(content: &str) -> f32 {
    let lower = content.to_lowercase();
    let mut score: f32 = 0.0;

    for word in POSITIVE_WORDS {
        if lower.contains(word) {
            score += 0.15;
        }
    }

    for word in NEGATIVE_WORDS {
        if lower.contains(word) {
            score -= 0.15;
        }
    }

    // Exclamation marks add +0.05 each, capped at +0.20.
    let exclamations = content.chars().filter(|&c| c == '!').count() as f32;
    score += (exclamations * 0.05).min(0.20);

    // All-caps words (alphabetic chars only, length ≥ 4) add +0.10 each,
    // capped at +0.20 total — they signal emphasis or excitement.
    let mut caps_bonus: f32 = 0.0;
    for word in content.split_whitespace() {
        let alpha_only: String = word.chars().filter(|c| c.is_alphabetic()).collect();
        if alpha_only.len() >= 4 && alpha_only == alpha_only.to_uppercase() {
            caps_bonus += 0.10;
        }
    }
    score += caps_bonus.min(0.20);

    score.clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::infer_valence;

    #[test]
    fn positive_text_scores_positive() {
        let score = infer_valence("This is amazing! I love it, great success!");
        assert!(score > 0.0, "expected positive score, got {score}");
    }

    #[test]
    fn negative_text_scores_negative() {
        let score = infer_valence("I'm so frustrated, this is broken and everything failed");
        assert!(score < 0.0, "expected negative score, got {score}");
    }

    #[test]
    fn neutral_text_scores_near_zero() {
        let score = infer_valence("The user asked about the current project status");
        assert!(
            (-0.1..=0.1).contains(&score),
            "expected near-zero score, got {score}"
        );
    }

    #[test]
    fn score_is_clamped_to_valid_range() {
        let very_positive =
            "amazing fantastic wonderful great love excited happy solved success excellent brilliant";
        let very_negative =
            "frustrated confused error failed worried stuck broken terrible awful wrong bad";
        let pos = infer_valence(very_positive);
        let neg = infer_valence(very_negative);
        assert!(pos <= 1.0 && pos >= -1.0, "positive score {pos} out of [-1,1]");
        assert!(neg <= 1.0 && neg >= -1.0, "negative score {neg} out of [-1,1]");
    }

    #[test]
    fn all_caps_words_add_positive_signal() {
        let score = infer_valence("It WORKS now, totally DONE");
        // Should be positive from WORKS/DONE caps bonus and "works"/"done" keywords
        assert!(score > 0.0, "expected positive score for CAPS emphasis, got {score}");
    }
}
