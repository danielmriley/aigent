//! Lightweight conversational-intent router for the sub-agent gate.
//!
//! [`needs_specialists`] decides whether a user message is substantive enough
//! to warrant parallel specialist pre-analysis (Researcher / Planner / Critic).
//!
//! ## Design rationale
//!
//! The gate intentionally errs toward *running* specialists:
//! - False positive (runs specialists on a simple question): costs a few
//!   seconds of latency.
//! - False negative (skips specialists on a deep question): produces a
//!   shallower captain response.
//!
//! ## Algorithm (three stages)
//!
//! 1. **Clause split + all-social check** — split on sentence-ending
//!    punctuation (`!?.`) and exact-match each clause against a social
//!    phrase list.  If *every* clause is social → skip.  This handles
//!    compound social messages like "Hello! How are you?" that would
//!    otherwise trip the `?` check in Stage 2.
//!
//! 2. **Positive substantive signals** — a question mark or a recognised
//!    task/query word in any *non-social* clause.  Token-level matching
//!    avoids substring false positives.
//!
//! 3. **Word-count floor** — five or more words in non-social clauses
//!    catches verbose requests that lack explicit task keywords.
//!
//! ## Future upgrade path
//!
//! Replace or augment with an embedding semantic router once the memory
//! embedding infrastructure is stable: pre-embed ~30 social vs. ~30 task
//! examples, use cosine similarity at inference time (<5 ms, zero LLM cost).

const SOCIAL: &[&str] = &[
    // Greetings
    "hello",
    "hi",
    "hey",
    "howdy",
    "greetings",
    "good morning",
    "good afternoon",
    "good evening",
    "good night",
    // Acknowledgements / affirmations
    "yes",
    "no",
    "ok",
    "okay",
    "sure",
    "yep",
    "nope",
    "yup",
    "got it",
    "sounds good",
    "sounds great",
    "great",
    "perfect",
    "cool",
    "nice",
    "awesome",
    "alright",
    "right",
    // Thanks
    "thanks",
    "thank you",
    "ty",
    "thx",
    "cheers",
    // Closings
    "bye",
    "goodbye",
    "see you",
    "later",
    "take care",
    "cya",
    // Small talk
    "how are you",
    "how are you doing",
    "how's it going",
    "what's up",
    "sup",
    "yo",
    "i'm doing great",
    "i'm doing well",
    "i'm good",
    "doing well",
    "doing great",
    "not bad",
];

const TASK_WORDS: &[&str] = &[
    // Question words (used in substantive contexts)
    "what", "why", "how", "when", "where", "who", "which", "whose",
    // Analytical verbs
    "explain", "describe", "analyze", "analyse", "compare", "summarize",
    "summarise", "research", "investigate", "evaluate", "assess",
    // Task/action verbs
    "fix", "debug", "implement", "write", "create", "build", "generate",
    "update", "configure", "deploy", "install", "refactor", "optimize",
    "test", "review", "find", "search", "list", "show", "tell",
    // Intent words
    "help", "meaning", "difference", "example", "reason", "cause",
    "plan", "strategy", "approach",
];

/// Returns `true` when `msg` is substantive enough to justify running the
/// parallel specialist team before the captain's turn.
///
/// Returns `false` only for clearly conversational/social messages that add
/// no analytical value (greetings, single-word acks, closings, small talk).
pub fn needs_specialists(msg: &str) -> bool {
    let normalized = msg.trim().to_lowercase();

    // Split into clauses on sentence-ending punctuation so that compound
    // social messages ("Hello! How are you?") are evaluated clause-by-clause
    // rather than as one unrecognised string.
    let clauses: Vec<&str> = normalized
        .split(|c: char| "!?.".contains(c))
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();

    if clauses.is_empty() {
        return false;
    }

    // ── Stage 1: all-social clause check → skip specialists ──────────────
    //
    // If every clause in the message matches a known social phrase the whole
    // message is conversational regardless of punctuation.  This is the fix
    // for "Hello! How are you?" — both clauses are social, so we skip.
    if clauses.iter().all(|c| SOCIAL.contains(c)) {
        return false;
    }

    // ── Stage 2: positive substantive signals → run specialists ──────────
    //
    // At least one clause is non-social.  A question mark in that context
    // is now a reliable signal of genuine inquiry (it can no longer be
    // triggered by "How are you?" alone).  Token-level TASK_WORDS matching
    // avoids substring false hits.
    if normalized.contains('?') {
        return true;
    }

    let has_task_word = normalized
        .split(|c: char| !c.is_alphabetic())
        .any(|token| TASK_WORDS.contains(&token));
    if has_task_word {
        return true;
    }

    // ── Stage 3: word-count floor → catch verbose requests ───────────────
    //
    // Count only words in non-social clauses so "Hello! Can you please do
    // this for me?" doesn't inflate the count with the greeting.
    let substantive_words: usize = clauses
        .iter()
        .filter(|c| !SOCIAL.contains(*c))
        .map(|c| c.split_whitespace().count())
        .sum();

    substantive_words >= 5
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── social — should NOT trigger specialists ───────────────────────────

    #[test]
    fn pure_greeting_skips() {
        assert!(!needs_specialists("Hello!"));
        assert!(!needs_specialists("hi"));
        assert!(!needs_specialists("Hey."));
        assert!(!needs_specialists("Good morning"));
    }

    #[test]
    fn acks_skip() {
        assert!(!needs_specialists("ok"));
        assert!(!needs_specialists("Sure!"));
        assert!(!needs_specialists("Got it."));
        assert!(!needs_specialists("Sounds good!"));
        assert!(!needs_specialists("thanks"));
        assert!(!needs_specialists("Thank you!"));
        assert!(!needs_specialists("Bye"));
    }

    #[test]
    fn compound_social_skips() {
        // The edge case that prompted this fix.
        assert!(!needs_specialists("Hello! How are you?"));
        assert!(!needs_specialists("Hi! How are you doing?"));
        assert!(!needs_specialists("Hey. How's it going?"));
        assert!(!needs_specialists("Good morning! How are you?"));
    }

    // ── substantive — SHOULD trigger specialists ──────────────────────────

    #[test]
    fn deep_question_triggers() {
        assert!(needs_specialists("What is the meaning of life?"));
    }

    #[test]
    fn short_question_triggers() {
        assert!(needs_specialists("Why?"));
        assert!(needs_specialists("How does it work?"));
    }

    #[test]
    fn task_request_triggers() {
        assert!(needs_specialists("Fix the bug in the parser"));
        assert!(needs_specialists("Explain quantum entanglement"));
        assert!(needs_specialists("Write a function that sorts a list"));
    }

    #[test]
    fn greeting_plus_real_question_triggers() {
        // Social opener must NOT suppress specialists when the rest is substantive.
        assert!(needs_specialists("Hi, can you explain quantum entanglement?"));
        assert!(needs_specialists("Hey, what is recursion?"));
        assert!(needs_specialists("Hello! How does the compiler work?"));
    }

    #[test]
    fn long_message_without_keywords_triggers() {
        // Five or more substantive words → probably a real request.
        assert!(needs_specialists("Please run the tests again for me"));
    }

    #[test]
    fn weather_query_triggers() {
        assert!(needs_specialists(
            "I'm doing great! Can you tell me what the weather is going to be like tomorrow?"
        ));
    }
}
