//! `aigent sleep` subcommand implementations.
//!
//! Provides three accelerated-testing tools:
//!
//! - **run**    — Force-trigger the full nightly multi-agent consolidation NOW,
//!   bypassing all time/quiet-window guards.
//! - **status** — Show the current sleep schedule, last-run times, and whether
//!   the daemon is in its configured quiet window.
//! - **seed**   — Inject synthetic episodic memories containing clear preference
//!   language about a topic, then optionally run the sleep cycle immediately.
//!   This is the primary tool for testing opinion formation without waiting
//!   weeks of real conversations.

use anyhow::Result;

use aigent_runtime::{DaemonClient, SeedEntry};

// ── Seed content templates ────────────────────────────────────────────────────
//
// The Identity specialist fires when it sees ≥5 distinct episodic observations
// with "non-zero emotional valence or preference signals (keywords like 'prefer',
// 'like', 'enjoy', 'tend to', 'usually', 'always', 'never')".  All templates
// below contain at least two such keywords so they are reliably detected.

const POSITIVE_TEMPLATES: &[&str] = &[
    "I find myself consistently drawn to {theme} — it feels rewarding and natural to me",
    "When engaging with {theme}, I notice a strong sense of satisfaction and engagement",
    "I tend to prefer {theme} approaches — they align well with how I think and work",
    "There is something about {theme} that I genuinely enjoy; it energises rather than drains me",
    "I have noticed I gravitate toward {theme} repeatedly — this seems to reflect a real preference",
    "Thinking about {theme} feels intuitive and satisfying to me",
    "My experience with {theme} has been consistently positive; I would always seek it out again",
    "I find {theme} compelling in a way that goes beyond mere usefulness",
    "I genuinely enjoy {theme} and return to it by choice when given the option",
    "When I engage with {theme} I feel more capable and focused than usual",
];

const NEGATIVE_TEMPLATES: &[&str] = &[
    "I find {theme} draining and tend to avoid it when I can",
    "When forced to engage with {theme}, I notice frustration and resistance",
    "I distinctly dislike {theme} — it conflicts with how I prefer to work",
    "There is something about {theme} that I find uncomfortable and counterproductive",
    "I have noticed I consistently avoid {theme} — this seems to reflect a genuine dispreference",
    "Working with {theme} always feels aversive; I'd rather approach things differently",
    "My experience with {theme} has been consistently negative; I avoid it when possible",
    "I find {theme} tedious in a way that undermines my effectiveness",
    "I genuinely dislike {theme} and never return to it willingly when given the option",
    "When working with {theme} I feel less capable and more frustrated than usual",
];

const NEUTRAL_TEMPLATES: &[&str] = &[
    "I notice that {theme} comes up repeatedly in my work and interactions",
    "When engaging with {theme}, I observe patterns that seem worth tracking over time",
    "I tend to encounter {theme} frequently — I should develop a clearer stance on it",
    "There is something about {theme} that keeps appearing across different contexts",
    "I have noticed {theme} arising in multiple situations — it seems worth reflecting on",
    "Thinking about {theme} reveals some interesting structural patterns",
    "My experience with {theme} has been mixed — sometimes useful, sometimes counterproductive",
    "I encounter {theme} regularly and have noted its various manifestations",
    "When working with {theme} I observe both benefits and notable drawbacks",
    "{theme} appears often enough that I am starting to form a view about its value",
];

fn templates_for(valence: SeedValence) -> &'static [&'static str] {
    match valence {
        SeedValence::Positive => POSITIVE_TEMPLATES,
        SeedValence::Negative => NEGATIVE_TEMPLATES,
        SeedValence::Neutral  => NEUTRAL_TEMPLATES,
    }
}

/// Valence direction for seeded memories.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum SeedValence {
    /// Positive preference language ("I enjoy", "I prefer", etc.)
    Positive,
    /// Negative aversion language ("I dislike", "I avoid", etc.)
    Negative,
    /// Neutral observation language (no strong valence signal).
    Neutral,
}

// ── Public command handlers ───────────────────────────────────────────────────

/// Force-trigger the full nightly multi-agent sleep cycle.
pub async fn run_sleep_run(socket_path: &str) -> Result<()> {
    let client = DaemonClient::new(socket_path);
    println!("triggering multi-agent sleep cycle…");
    println!("(this may take 30–45 minutes on local Ollama hardware)");
    println!();

    let result = client
        .run_multi_agent_sleep_cycle_with_progress(|msg| {
            println!("  {msg}");
        })
        .await;

    match result {
        Ok(msg) => {
            println!();
            println!("sleep cycle complete: {msg}");
        }
        Err(err) => {
            eprintln!("sleep cycle failed: {err}");
        }
    }
    Ok(())
}

/// Display the current sleep schedule and last-run timing.
pub async fn run_sleep_status(socket_path: &str) -> Result<()> {
    let client = DaemonClient::new(socket_path);
    match client.get_sleep_status().await {
        Ok(s) => {
            println!("── sleep cycle status ───────────────────────────────");
            println!("  auto sleep mode       : {}", s.auto_sleep_mode);
            println!("  passive interval      : {} h", s.passive_interval_hours);
            println!(
                "  last passive sleep    : {}",
                s.last_passive_sleep_at.as_deref().unwrap_or("(never)")
            );
            println!(
                "  last nightly sleep    : {}",
                s.last_nightly_sleep_at.as_deref().unwrap_or("(never)")
            );
            println!(
                "  quiet window          : {:02}:00 – {:02}:00 ({})",
                s.quiet_window_start, s.quiet_window_end, s.timezone
            );
            println!(
                "  in quiet window now   : {}",
                if s.in_quiet_window { "yes" } else { "no" }
            );
            println!();
            println!("tip: run `aigent sleep run` to trigger a cycle immediately");
        }
        Err(err) => eprintln!("failed to fetch sleep status: {err}"),
    }
    Ok(())
}

/// Seed synthetic episodic memories then optionally run the sleep cycle.
pub async fn run_sleep_seed(
    socket_path: &str,
    theme: &str,
    count: usize,
    valence: SeedValence,
    then_run: bool,
) -> Result<()> {
    let templates = templates_for(valence);
    let n = count.min(templates.len()).max(1);

    let entries: Vec<SeedEntry> = templates[..n]
        .iter()
        .map(|tmpl| SeedEntry {
            content: tmpl.replace("{theme}", theme),
            tier: "episodic".to_string(),
            source: "test-seed".to_string(),
        })
        .collect();

    println!("seeding {} episodic memories about \"{}\":", n, theme);
    for entry in &entries {
        println!("  · {}", entry.content);
    }
    println!();

    let client = DaemonClient::new(socket_path);
    match client.seed_memories(entries).await {
        Ok(msg) => println!("{msg}"),
        Err(err) => {
            eprintln!("seed failed: {err}");
            return Ok(());
        }
    }

    if then_run {
        println!();
        run_sleep_run(socket_path).await?;
    } else {
        println!();
        println!("run `aigent sleep run` to process these memories through the sleep pipeline.");
        println!("then check `aigent memory beliefs --tier reflective` for new opinions.");
    }

    Ok(())
}
