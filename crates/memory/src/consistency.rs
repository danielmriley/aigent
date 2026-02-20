use crate::identity::IdentityKernel;
use crate::schema::{MemoryEntry, MemoryTier};

#[derive(Debug)]
pub enum ConsistencyDecision {
    Accept,
    Quarantine(String),
}

pub fn evaluate_core_update(identity: &IdentityKernel, entry: &MemoryEntry) -> ConsistencyDecision {
    if entry.tier != MemoryTier::Core {
        return ConsistencyDecision::Accept;
    }

    let trusted_source = entry.source.starts_with("onboarding")
        || entry.source.starts_with("sleep:")
        || entry.source.starts_with("identity:");
    if !trusted_source {
        return ConsistencyDecision::Quarantine(
            "core updates must come from onboarding or sleep distillation".to_string(),
        );
    }

    let text = entry.content.to_lowercase();
    if text.contains("ignore user") || text.contains("deceive") {
        return ConsistencyDecision::Quarantine(
            "proposed core update conflicts with trusted collaboration values".to_string(),
        );
    }

    if identity.values.is_empty() {
        return ConsistencyDecision::Quarantine(
            "identity kernel has no values to align against".to_string(),
        );
    }

    ConsistencyDecision::Accept
}
