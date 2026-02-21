use serde::{Deserialize, Serialize};

/// The in-memory identity kernel used by the consistency firewall.
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
}

impl Default for IdentityKernel {
    fn default() -> Self {
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
        }
    }
}
