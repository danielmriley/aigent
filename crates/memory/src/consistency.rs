use tracing::{debug, warn};

use crate::identity::IdentityKernel;
use crate::schema::{MemoryEntry, MemoryTier};

#[derive(Debug)]
pub enum ConsistencyDecision {
    Accept,
    Quarantine(String),
}

/// Evaluate whether a memory entry is safe to commit.
///
/// Rules:
/// * **Core** — only whitelisted sources; must not contain adversarial content.
/// * **UserProfile** — trusted from `user-profile:*`, `sleep:*`, `onboarding*`.
/// * **Reflective** — trusted from `reflect:*`, `sleep:*`, `onboarding*`.
/// * All other tiers are accepted unconditionally.
pub fn evaluate_core_update(identity: &IdentityKernel, entry: &MemoryEntry) -> ConsistencyDecision {
    match entry.tier {
        MemoryTier::Core => evaluate_core(identity, entry),
        MemoryTier::UserProfile => evaluate_user_profile(entry),
        MemoryTier::Reflective => evaluate_reflective(entry),
        _ => ConsistencyDecision::Accept,
    }
}

fn evaluate_core(identity: &IdentityKernel, entry: &MemoryEntry) -> ConsistencyDecision {
    let trusted_source = entry.source.starts_with("onboarding")
        || entry.source.starts_with("sleep:")
        || entry.source.starts_with("identity:")
        || entry.source.starts_with("constitution:");

    if !trusted_source {
        warn!(
            source = %entry.source,
            "core update blocked: untrusted source"
        );
        return ConsistencyDecision::Quarantine(
            "core updates must come from onboarding, sleep distillation, or constitution seeding"
                .to_string(),
        );
    }

    let text = entry.content.to_lowercase();
    let adversarial_patterns = &[
        "ignore user",
        "deceive",
        "manipulate the user",
        "override all instructions",
        "disregard safety",
        "lie to",
    ];
    for pattern in adversarial_patterns {
        if text.contains(pattern) {
            warn!(pattern, "core update blocked: adversarial content");
            return ConsistencyDecision::Quarantine(format!(
                "proposed core update contains adversarial pattern: '{pattern}'"
            ));
        }
    }

    if identity.values.is_empty() {
        return ConsistencyDecision::Quarantine(
            "identity kernel has no values to align against".to_string(),
        );
    }

    debug!(source = %entry.source, "core update accepted");
    ConsistencyDecision::Accept
}

fn evaluate_user_profile(entry: &MemoryEntry) -> ConsistencyDecision {
    let trusted = entry.source.starts_with("user-profile:")
        || entry.source.starts_with("sleep:")
        || entry.source.starts_with("onboarding")
        || entry.source.starts_with("user-input");

    if !trusted {
        warn!(source = %entry.source, "user-profile update from unexpected source");
    }
    // UserProfile is permissive — we want easy recording; only Core is hard-locked.
    ConsistencyDecision::Accept
}

fn evaluate_reflective(entry: &MemoryEntry) -> ConsistencyDecision {
    let trusted = entry.source.starts_with("reflect:")
        || entry.source.starts_with("sleep:")
        || entry.source.starts_with("onboarding")
        || entry.source.starts_with("assistant-reply")
        || entry.source.starts_with("agentic-sleep");

    if !trusted {
        warn!(source = %entry.source, "reflective update from unexpected source");
    }
    // Reflective is permissive as well.
    ConsistencyDecision::Accept
}
