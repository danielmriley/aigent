//! Text-truncation utility shared across the codebase.

/// Truncate `text` to at most `max_chars` characters, appending `…` when cut.
pub fn truncate_for_prompt(text: &str, max_chars: usize) -> String {
    let chars = text.chars().collect::<Vec<_>>();
    if chars.len() <= max_chars {
        return text.to_string();
    }
    let truncated: String = chars.into_iter().take(max_chars).collect();
    format!("{truncated}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_unchanged() {
        assert_eq!(truncate_for_prompt("hello", 10), "hello");
    }

    #[test]
    fn exact_length_unchanged() {
        assert_eq!(truncate_for_prompt("12345", 5), "12345");
    }

    #[test]
    fn long_text_adds_ellipsis() {
        let result = truncate_for_prompt("hello world", 5);
        assert_eq!(result, "hello…");
    }

    #[test]
    fn empty_string() {
        assert_eq!(truncate_for_prompt("", 10), "");
    }

    #[test]
    fn zero_max() {
        assert_eq!(truncate_for_prompt("hello", 0), "…");
    }

    #[test]
    fn multibyte_characters() {
        let text = "café résumé";
        let result = truncate_for_prompt(text, 4);
        assert_eq!(result, "café…");
    }

    #[test]
    fn emoji() {
        let text = "🦀🐍🐹🐿️";
        let result = truncate_for_prompt(text, 2);
        assert_eq!(result, "🦀🐍…");
    }
}
