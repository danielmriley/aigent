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
    let lower = user_message.to_lowercase();

    // "I prefer X" / "I like X"
    for prefix in &["i prefer ", "i like "] {
        if let Some(pos) = lower.find(prefix) {
            let rest = &user_message[pos + prefix.len()..];
            let value = extract_phrase(rest);
            if !value.is_empty() && value.len() <= 80 {
                signals.push(("preference".to_string(), value, "preference".to_string()));
            }
        }
    }

    // "my name is X"
    if let Some(pos) = lower.find("my name is ") {
        let rest = &user_message[pos + "my name is ".len()..];
        let value = extract_word(rest);
        if !value.is_empty() && value.len() <= 80 {
            signals.push(("name".to_string(), value, "fact".to_string()));
        }
    }

    // "I'm working on X" / "I am working on X"
    for prefix in &["i'm working on ", "i am working on "] {
        if let Some(pos) = lower.find(prefix) {
            let rest = &user_message[pos + prefix.len()..];
            let value = extract_phrase(rest);
            if !value.is_empty() && value.len() <= 80 {
                signals.push(("current_project".to_string(), value, "fact".to_string()));
            }
        }
    }

    // "I use X" / "I'm using X" / "I am using X"
    for prefix in &["i use ", "i'm using ", "i am using "] {
        if let Some(pos) = lower.find(prefix) {
            let rest = &user_message[pos + prefix.len()..];
            let value = extract_phrase(rest);
            if !value.is_empty() && value.len() <= 80 {
                signals.push(("tooling".to_string(), value, "preference".to_string()));
            }
        }
    }

    // "my goal is X" / "I want to X"
    for prefix in &["my goal is ", "i want to "] {
        if let Some(pos) = lower.find(prefix) {
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
}
