//! Per-turn micro-profiling: lightweight heuristic extraction of structured
//! `UserProfile` signals directly from user messages.
//!
//! This runs every turn so the agent's profile stays current without waiting
//! for the nightly sleep cycle.

/// Extract `(key, value, category)` triples from a single user message.
///
/// Patterns matched:
/// - `"I prefer X"` / `"I like X"` → `("preference", X, "preference")`
/// - `"my name is X"` → `("name", X, "fact")`
/// - `"I'm working on X"` / `"I am working on X"` → `("current_project", X, "fact")`
/// - `"I use X"` / `"I'm using X"` / `"I am using X"` → `("tooling", X, "preference")`
/// - `"my goal is X"` / `"I want to X"` → `("goal", X, "goal")`
///
/// Only returns matches where the captured value is ≤ 80 characters.
pub fn extract_inline_profile_signals(user_message: &str) -> Vec<(String, String, String)> {
    let mut signals = Vec::new();

    // Helper: case-insensitive find that returns a byte offset valid for the
    // *original* string.  We scan the original string character by character
    // instead of using `to_lowercase()` + byte offsets, which can diverge on
    // multi-byte characters whose lowercase form has a different byte length.
    let find_ci = |haystack: &str, needle: &str| -> Option<usize> {
        let needle_lower: Vec<char> = needle.chars().collect();
        let hay_chars: Vec<(usize, char)> = haystack.char_indices().collect();
        'outer: for (start_idx, &(byte_pos, _)) in hay_chars.iter().enumerate() {
            if start_idx + needle_lower.len() > hay_chars.len() { break; }
            for (j, &nc) in needle_lower.iter().enumerate() {
                let hc = hay_chars[start_idx + j].1;
                // Compare lowercase forms character by character.
                let mut h_lower = hc.to_lowercase();
                let mut n_lower = nc.to_lowercase();
                if h_lower.next() != n_lower.next() { continue 'outer; }
            }
            // Found — return the byte offset in the *original* string.
            return Some(byte_pos);
        }
        None
    };

    // "I prefer X" / "I like X"
    for prefix in &["i prefer ", "i like "] {
        if let Some(pos) = find_ci(user_message, prefix) {
            let rest = &user_message[pos + prefix.len()..];
            let value = extract_phrase(rest);
            if !value.is_empty() && value.len() <= 80 {
                signals.push(("preference".to_string(), value, "preference".to_string()));
            }
        }
    }

    // "my name is X"
    if let Some(pos) = find_ci(user_message, "my name is ") {
        let rest = &user_message[pos + "my name is ".len()..];
        let value = extract_word(rest);
        if !value.is_empty() && value.len() <= 80 {
            signals.push(("name".to_string(), value, "fact".to_string()));
        }
    }

    // "I'm working on X" / "I am working on X"
    for prefix in &["i'm working on ", "i am working on "] {
        if let Some(pos) = find_ci(user_message, prefix) {
            let rest = &user_message[pos + prefix.len()..];
            let value = extract_phrase(rest);
            if !value.is_empty() && value.len() <= 80 {
                signals.push(("current_project".to_string(), value, "fact".to_string()));
            }
        }
    }

    // "I use X" / "I'm using X" / "I am using X"
    for prefix in &["i use ", "i'm using ", "i am using "] {
        if let Some(pos) = find_ci(user_message, prefix) {
            let rest = &user_message[pos + prefix.len()..];
            let value = extract_phrase(rest);
            if !value.is_empty() && value.len() <= 80 {
                signals.push(("tooling".to_string(), value, "preference".to_string()));
            }
        }
    }

    // "my goal is X" / "I want to X"
    for prefix in &["my goal is ", "i want to "] {
        if let Some(pos) = find_ci(user_message, prefix) {
            let rest = &user_message[pos + prefix.len()..];
            let value = extract_phrase(rest);
            if !value.is_empty() && value.len() <= 80 {
                signals.push(("goal".to_string(), value, "goal".to_string()));
            }
        }
    }

    // Deduplicate: if the same key appears more than once, keep only the first.
    let mut seen_keys = std::collections::HashSet::new();
    signals.retain(|(key, _, _)| seen_keys.insert(key.clone()));

    signals
}

/// Extract the phrase up to the first sentence-ending punctuation or newline.
fn extract_phrase(text: &str) -> String {
    let end = text
        .find(['.', '!', '?', '\n', ';'])
        .unwrap_or(text.len());
    text[..end].trim().to_string()
}

/// Extract the first word only (for names etc.).
fn extract_word(text: &str) -> String {
    text.split_whitespace()
        .next()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::extract_inline_profile_signals;

    #[test]
    fn extracts_preference_signal() {
        let signals = extract_inline_profile_signals("I prefer concise answers");
        assert!(
            signals
                .iter()
                .any(|(k, v, c)| k == "preference" && v == "concise answers" && c == "preference"),
            "expected preference signal, got {signals:?}"
        );
    }

    #[test]
    fn extracts_name_signal() {
        let signals = extract_inline_profile_signals("My name is Alice, nice to meet you.");
        assert!(
            signals.iter().any(|(k, v, _)| k == "name" && v == "Alice"),
            "expected name signal, got {signals:?}"
        );
    }

    #[test]
    fn extracts_current_project_signal() {
        let signals =
            extract_inline_profile_signals("I'm working on a Rust web server for my startup");
        assert!(
            signals
                .iter()
                .any(|(k, c, _)| k == "current_project" && c.contains("Rust")),
            "expected current_project signal, got {signals:?}"
        );
    }

    #[test]
    fn extracts_tooling_signal() {
        let signals = extract_inline_profile_signals("I use neovim for most of my editing.");
        assert!(
            signals.iter().any(|(k, v, _)| k == "tooling" && v.contains("neovim")),
            "expected tooling signal, got {signals:?}"
        );
    }

    #[test]
    fn extracts_goal_signal() {
        let signals = extract_inline_profile_signals("My goal is to ship the MVP by Friday");
        assert!(
            signals.iter().any(|(k, v, c)| k == "goal" && v.contains("ship") && c == "goal"),
            "expected goal signal, got {signals:?}"
        );
    }

    #[test]
    fn skips_values_over_80_chars() {
        let long_value = "x".repeat(90);
        let msg = format!("I prefer {long_value}");
        let signals = extract_inline_profile_signals(&msg);
        assert!(
            signals.is_empty() || signals.iter().all(|(_, v, _)| v.len() <= 80),
            "no signal should have value > 80 chars"
        );
    }

    #[test]
    fn deduplicates_same_key_keeps_first() {
        // "I prefer" and "I like" both map to "preference" — only first should survive.
        let signals =
            extract_inline_profile_signals("I prefer dark mode and I like dark mode too");
        let preference_count = signals.iter().filter(|(k, _, _)| k == "preference").count();
        assert_eq!(preference_count, 1, "expected exactly one preference signal");
    }

    #[test]
    fn handles_non_ascii_content() {
        // Should not panic on multi-byte characters around keyword positions.
        let signals = extract_inline_profile_signals("I prefer café-style coding sessions");
        assert!(
            signals.iter().any(|(k, _, _)| k == "preference"),
            "expected preference from non-ASCII input, got {signals:?}"
        );
    }

    #[test]
    fn handles_emoji_in_message() {
        // Emoji = 4-byte chars. Must not panic on byte-offset mismatch.
        let signals = extract_inline_profile_signals("🎉 My name is Daniel 🎉");
        assert!(
            signals.iter().any(|(k, v, _)| k == "name" && v == "Daniel"),
            "expected name signal with emoji, got {signals:?}"
        );
    }

    #[test]
    fn handles_cjk_around_keywords() {
        // CJK characters are 3 bytes each.
        let signals =
            extract_inline_profile_signals("みんな I prefer minimalist design こんにちは");
        assert!(
            signals.iter().any(|(k, _, _)| k == "preference"),
            "expected preference signal with CJK, got {signals:?}"
        );
    }

    #[test]
    fn no_signals_from_empty_message() {
        let signals = extract_inline_profile_signals("");
        assert!(signals.is_empty());
    }

    #[test]
    fn no_signals_from_irrelevant_message() {
        let signals = extract_inline_profile_signals("Hello, how are you doing today?");
        assert!(signals.is_empty(), "should not extract signals from generic greeting, got {signals:?}");
    }
}
