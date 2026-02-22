//! UserProfile memory helpers.
//!
//! UserProfile entries are stored with sources like `user-profile:preference`,
//! `user-profile:goal`, `user-profile:fact`, etc.  These functions help build
//! a human-readable user profile block for use in LLM prompts.

use std::collections::HashMap;

use crate::schema::{MemoryEntry, MemoryTier};

/// Keys commonly found in user-profile source tags.
pub const SOURCE_PREFERENCE: &str = "user-profile:preference";
pub const SOURCE_GOAL: &str = "user-profile:goal";
pub const SOURCE_FACT: &str = "user-profile:fact";
pub const SOURCE_STYLE: &str = "user-profile:style";
pub const SOURCE_TRIGGER: &str = "user-profile:trigger";

/// Extract all `UserProfile` entries from a slice, newest first.
pub fn user_profile_entries(entries: &[MemoryEntry]) -> Vec<&MemoryEntry> {
    let mut profile: Vec<&MemoryEntry> = entries
        .iter()
        .filter(|entry| entry.tier == MemoryTier::UserProfile)
        .collect();
    profile.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    profile
}

/// Format a compact user-profile block for inclusion in LLM prompts.
///
/// Returns `None` if no profile entries exist.
///
/// Deduplicates by key (the content segment before the first `:`), keeping
/// only the most-recently updated entry for each key so that stale or
/// contradictory entries from older profile updates don't appear alongside
/// newer ones.
pub fn format_user_profile_block(entries: &[MemoryEntry]) -> Option<String> {
    let all_profile = user_profile_entries(entries);
    if all_profile.is_empty() {
        return None;
    }

    // Deduplicate by key: for each leading "key:" prefix, keep the latest entry.
    let mut by_key: HashMap<String, &MemoryEntry> = HashMap::new();
    for entry in &all_profile {
        let key = entry
            .content
            .split(':')
            .next()
            .map(|k| k.trim().to_lowercase())
            .unwrap_or_default();
        by_key
            .entry(key)
            .and_modify(|existing| {
                if entry.created_at > existing.created_at {
                    *existing = entry;
                }
            })
            .or_insert(entry);
    }

    // Group deduplicated entries by source category.
    let mut preferences = Vec::new();
    let mut goals = Vec::new();
    let mut facts = Vec::new();
    let mut style = Vec::new();
    let mut other = Vec::new();

    for entry in by_key.values() {
        let s = entry.source.as_str();
        if s.contains("preference") {
            preferences.push(entry.content.as_str());
        } else if s.contains("goal") {
            goals.push(entry.content.as_str());
        } else if s.contains("fact") {
            facts.push(entry.content.as_str());
        } else if s.contains("style") {
            style.push(entry.content.as_str());
        } else {
            other.push(entry.content.as_str());
        }
    }

    let mut sections = Vec::new();
    if !preferences.is_empty() {
        sections.push(format!("Preferences:\n{}", bullet_list(&preferences)));
    }
    if !goals.is_empty() {
        sections.push(format!("Goals:\n{}", bullet_list(&goals)));
    }
    if !facts.is_empty() {
        sections.push(format!("Known facts:\n{}", bullet_list(&facts)));
    }
    if !style.is_empty() {
        sections.push(format!("Communication style:\n{}", bullet_list(&style)));
    }
    if !other.is_empty() {
        sections.push(format!("Other:\n{}", bullet_list(&other)));
    }

    Some(sections.join("\n"))
}

fn bullet_list(items: &[&str]) -> String {
    items
        .iter()
        .map(|item| format!("  2022 {item}"))
        .collect::<Vec<_>>()
        .join("\n")
}
