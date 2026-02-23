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
    "difficult", "struggle", "issue", "bug", "crash", "problem",
    "cannot", "unable", "fail", "loss", "lost", "miss", "missing",
];

/// Infer an emotional valence score for `content` using keyword heuristics.
///
/// Returns a value clamped to `[-1.0, 1.0]`.  Positive content scores > 0,
/// negative/distressing content scores < 0, and neutral prose scores near 0.
///
/// A 2-word lookback window is used to detect negation tokens (`not`, `no`,
/// `never`, `without`) so that phrases like "not a problem" score positively
/// rather than negatively.
pub fn infer_valence(content: &str) -> f32 {
    let lower = content.to_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();

    let mut score: f32 = 0.0;
    for (i, word) in words.iter().enumerate() {
        // Check for a negation token in a 2-word lookback window so that
        // phrases like "not a problem" and "not broken" are both handled.
        let negated = (i > 0
            && matches!(words[i - 1], "not" | "no" | "never" | "without"))
            || (i > 1
                && matches!(words[i - 2], "not" | "no" | "never" | "without"));

        if POSITIVE_WORDS.contains(word) {
            score += if negated { -0.10 } else { 0.15 };
        } else if NEGATIVE_WORDS.contains(word) {
            score += if negated { 0.10 } else { -0.15 };
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
        // A negated positive should NOT score negatively.
        let negated_score = infer_valence("not a problem at all");
        assert!(
            negated_score >= 0.0,
            "negated positive 'not a problem' should score >= 0, got {negated_score}"
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
