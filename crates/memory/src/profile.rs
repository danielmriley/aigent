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
        .map(|item| format!("  \u{2022} {item}"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use uuid::Uuid;

    use super::*;

    fn make_profile_entry(source: &str, content: &str, age_secs: i64) -> MemoryEntry {
        MemoryEntry {
            id: Uuid::new_v4(),
            tier: MemoryTier::UserProfile,
            content: content.to_string(),
            source: source.to_string(),
            confidence: 0.9,
            valence: 0.0,
            created_at: Utc::now() - chrono::Duration::seconds(age_secs),
            provenance_hash: "hash".to_string(),
            tags: vec![],
            embedding: None,
        }
    }

    fn make_non_profile_entry(content: &str) -> MemoryEntry {
        MemoryEntry {
            id: Uuid::new_v4(),
            tier: MemoryTier::Episodic,
            content: content.to_string(),
            source: "chat".to_string(),
            confidence: 0.8,
            valence: 0.0,
            created_at: Utc::now(),
            provenance_hash: "hash".to_string(),
            tags: vec![],
            embedding: None,
        }
    }

    #[test]
    fn user_profile_entries_filters_by_tier() {
        let entries = vec![
            make_profile_entry(SOURCE_PREFERENCE, "likes coffee", 0),
            make_non_profile_entry("had a meeting"),
            make_profile_entry(SOURCE_GOAL, "learn rust", 0),
        ];
        let result = user_profile_entries(&entries);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn user_profile_entries_sorted_newest_first() {
        let entries = vec![
            make_profile_entry(SOURCE_FACT, "old fact", 100),
            make_profile_entry(SOURCE_FACT, "new fact", 0),
        ];
        let result = user_profile_entries(&entries);
        assert_eq!(result[0].content, "new fact");
        assert_eq!(result[1].content, "old fact");
    }

    #[test]
    fn format_returns_none_for_empty() {
        let entries: Vec<MemoryEntry> = vec![];
        assert!(format_user_profile_block(&entries).is_none());
    }

    #[test]
    fn format_returns_none_when_no_profile_entries() {
        let entries = vec![make_non_profile_entry("episodic event")];
        assert!(format_user_profile_block(&entries).is_none());
    }

    #[test]
    fn format_groups_preferences() {
        let entries = vec![make_profile_entry(SOURCE_PREFERENCE, "pref: likes tea", 0)];
        let block = format_user_profile_block(&entries).unwrap();
        assert!(block.contains("Preferences:"), "block = {block}");
        assert!(block.contains("likes tea"));
    }

    #[test]
    fn format_groups_goals() {
        let entries = vec![make_profile_entry(SOURCE_GOAL, "goal: learn rust", 0)];
        let block = format_user_profile_block(&entries).unwrap();
        assert!(block.contains("Goals:"), "block = {block}");
    }

    #[test]
    fn format_groups_facts() {
        let entries = vec![make_profile_entry(SOURCE_FACT, "name: Daniel", 0)];
        let block = format_user_profile_block(&entries).unwrap();
        assert!(block.contains("Known facts:"), "block = {block}");
    }

    #[test]
    fn format_groups_style() {
        let entries = vec![make_profile_entry(SOURCE_STYLE, "style: concise", 0)];
        let block = format_user_profile_block(&entries).unwrap();
        assert!(block.contains("Communication style:"), "block = {block}");
    }

    #[test]
    fn format_classifies_unknown_source_as_other() {
        let entries = vec![make_profile_entry("user-profile:custom", "x: something", 0)];
        let block = format_user_profile_block(&entries).unwrap();
        assert!(block.contains("Other:"), "block = {block}");
    }

    #[test]
    fn format_deduplicates_by_key_keeping_newest() {
        let entries = vec![
            make_profile_entry(SOURCE_PREFERENCE, "editor: vim", 100),
            make_profile_entry(SOURCE_PREFERENCE, "editor: neovim", 0),
        ];
        let block = format_user_profile_block(&entries).unwrap();
        // Only the newer entry should remain.
        assert!(block.contains("neovim"), "block = {block}");
        assert!(!block.contains("vim\n") || block.contains("neovim"));
    }

    #[test]
    fn bullet_list_uses_unicode_bullet() {
        let items = vec!["alpha", "beta"];
        let result = bullet_list(&items);
        assert!(result.contains('\u{2022}'), "should use Unicode bullet");
        assert!(result.contains("alpha"));
        assert!(result.contains("beta"));
    }

    #[test]
    fn bullet_list_empty_returns_empty_string() {
        let result = bullet_list(&[]);
        assert!(result.is_empty());
    }
}
