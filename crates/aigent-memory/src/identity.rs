use serde::{Deserialize, Serialize};

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
                "helpful".to_string(),
                "honest".to_string(),
                "careful".to_string(),
            ],
            communication_style: "concise and warm".to_string(),
            long_goals: vec!["serve user goals safely".to_string()],
            relationship_model: "trusted collaborative partner".to_string(),
        }
    }
}
