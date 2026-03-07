//! Types for the parallel subagent reasoning pipeline.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Which specialist role a subagent fulfils.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubagentRole {
    /// Digs into context for facts, gaps, and relevant background.
    Researcher,
    /// Devises step-by-step execution strategy.
    Planner,
    /// Challenges assumptions, flags risks and hallucination traps.
    Critic,
}

impl SubagentRole {
    pub fn label(self) -> &'static str {
        match self {
            Self::Researcher => "Researcher",
            Self::Planner => "Planner",
            Self::Critic => "Critic",
        }
    }
}

impl fmt::Display for SubagentRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Structured analysis returned by a single subagent.
///
/// The subagent is instructed to output these fields in a parseable format.
/// Parsing is best-effort: missing sections default to empty vecs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SubagentAnalysis {
    pub role: String,
    pub key_facts: Vec<String>,
    pub proposed_actions: Vec<String>,
    pub potential_pitfalls: Vec<String>,
}
