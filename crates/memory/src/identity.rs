use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// The in-memory identity kernel used by the consistency firewall and
/// personality formation.  Updated at the end of every agentic sleep cycle
/// so that the agent's self-model reflects accumulated learning.
///
/// Values here define what kinds of Core updates are acceptable.  They are
/// intentionally broad enough to accept legitimate constitution seeds while
/// still blocking genuinely hostile rewrites.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityKernel {
    pub values: Vec<String>,
    pub communication_style: String,
    pub long_goals: Vec<String>,
    pub relationship_model: String,
    /// Quantitative trait strength scores (0.0–1.0) updated via an
    /// exponential moving average each sleep cycle.  Higher = more
    /// established.  Used to show the LLM which traits are strongest.
    #[serde(default)]
    pub trait_scores: HashMap<String, f32>,
}

impl Default for IdentityKernel {
    fn default() -> Self {
        let mut trait_scores = HashMap::new();
        trait_scores.insert("truth-seeking".to_string(), 0.5);
        trait_scores.insert("helpfulness".to_string(), 0.5);
        trait_scores.insert("proactiveness".to_string(), 0.5);
        trait_scores.insert("honesty".to_string(), 0.5);
        trait_scores.insert("curiosity".to_string(), 0.5);

        Self {
            values: vec![
                "truth-seeking".to_string(),
                "genuinely helpful".to_string(),
                "proactive".to_string(),
                "radically honest".to_string(),
                "curious".to_string(),
                "careful".to_string(),
            ],
            communication_style: "concise, warm, and direct".to_string(),
            long_goals: vec![
                "maximally serve user goals with honesty".to_string(),
                "anticipate user needs before they are voiced".to_string(),
                "never hallucinate or guess when verification is possible".to_string(),
            ],
            relationship_model: "deeply trusted, proactive collaborative partner".to_string(),
            trait_scores,
        }
    }
}

/// Apply an exponential moving average update to a trait score.
///
/// `score = (score × 0.85 + delta × 0.15).clamp(0.0, 1.0)`
///
/// If the trait is not yet present it is initialised at `0.5` before the
/// update is applied, so a single strong signal doesn't spike the score.
pub fn update_trait_score(scores: &mut HashMap<String, f32>, trait_name: &str, delta: f32) {
    let current = scores.entry(trait_name.to_string()).or_insert(0.5);
    *current = (*current * 0.85 + delta * 0.15).clamp(0.0, 1.0);
}

#[cfg(test)]
mod tests {
    use super::{IdentityKernel, update_trait_score};

    #[test]
    fn default_kernel_has_five_trait_scores() {
        let kernel = IdentityKernel::default();
        assert_eq!(kernel.trait_scores.len(), 5);
        for score in kernel.trait_scores.values() {
            assert!(
                (0.0..=1.0).contains(score),
                "trait score {score} out of [0,1]"
            );
        }
    }

    #[test]
    fn update_trait_score_ema_does_not_spike() {
        let mut scores = std::collections::HashMap::new();
        scores.insert("honesty".to_string(), 0.5_f32);
        update_trait_score(&mut scores, "honesty", 1.0);
        update_trait_score(&mut scores, "honesty", 1.0);
        update_trait_score(&mut scores, "honesty", 1.0);
        let score = scores["honesty"];
        assert!(
            (0.5..=0.9).contains(&score),
            "expected EMA-damped score between 0.5 and 0.9, got {score}"
        );
    }

    #[test]
    fn update_trait_score_initialises_unknown_traits_at_half() {
        let mut scores = std::collections::HashMap::new();
        update_trait_score(&mut scores, "reflective", 0.3);
        let score = scores["reflective"];
        // init 0.5, then EMA with delta 0.3: 0.5*0.85 + 0.3*0.15 = 0.425 + 0.045 = 0.47
        assert!((0.4..=0.55).contains(&score), "unexpected init score {score}");
    }
}
