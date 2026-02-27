use std::fs;
use std::io;
use std::io::IsTerminal;
use std::io::Write;
use std::path::Path;

use anyhow::{Result, bail};
use chrono::Local;

use aigent_config::AppConfig;
use aigent_memory::{MemoryManager, MemoryTier};
use aigent_memory::event_log::MemoryEventLog;

use crate::CliMemoryLayer;

pub(crate) async fn run_memory_wipe(memory: &mut MemoryManager, layer: CliMemoryLayer, yes: bool) -> Result<()> {
    let targets = layer_to_tiers(layer);
    let total = memory.all().len();
    let target_count = if matches!(layer, CliMemoryLayer::All) {
        total
    } else {
        memory
            .all()
            .iter()
            .filter(|entry| targets.contains(&entry.tier))
            .count()
    };

    println!("⚠️  destructive operation: memory wipe");
    println!("- selected layer: {}", memory_layer_label(layer));
    println!("- targeted entries: {target_count}");
    println!("- total entries: {total}");
    println!(
        "- by tier: episodic={} semantic={} procedural={} core={}",
        memory.entries_by_tier(MemoryTier::Episodic).len(),
        memory.entries_by_tier(MemoryTier::Semantic).len(),
        memory.entries_by_tier(MemoryTier::Procedural).len(),
        memory.entries_by_tier(MemoryTier::Core).len(),
    );

    if target_count == 0 {
        println!("no matching memory entries to wipe");
        return Ok(());
    }

    if !yes {
        if !io::stdin().is_terminal() {
            bail!(
                "refusing destructive wipe in non-interactive mode without --yes (or pass --yes)"
            );
        }

        let expected = format!("WIPE {}", memory_layer_label(layer).to_uppercase());
        print!(
            "This permanently deletes memory from .aigent/memory/events.jsonl. Type '{expected}' to continue: "
        );
        io::stdout().flush()?;

        let mut confirmation = String::new();
        io::stdin().read_line(&mut confirmation)?;
        if confirmation.trim() != expected {
            println!("memory wipe cancelled");
            return Ok(());
        }
    }

    let removed = if matches!(layer, CliMemoryLayer::All) {
        memory.wipe_all().await?
    } else {
        memory.wipe_tiers(&targets).await?
    };

    println!("memory wipe complete: removed {removed} entries");
    println!("remaining entries: {}", memory.all().len());
    Ok(())
}

pub(crate) fn run_memory_stats(memory: &MemoryManager) {
    let stats = memory.stats();
    println!("── memory stats ─────────────────────────────────────");
    println!("  total:        {}", stats.total);
    println!("  core:         {}", stats.core);
    println!("  user_profile: {}", stats.user_profile);
    println!("  reflective:   {}", stats.reflective);
    println!("  semantic:     {}", stats.semantic);
    println!("  procedural:   {}", stats.procedural);
    println!("  episodic:     {}", stats.episodic);

    // ── tool execution stats ──────────────────────────────────────
    {
        let tool_entries = memory.entries_by_tier(MemoryTier::Procedural);
        let tool_execs: Vec<_> = tool_entries
            .iter()
            .filter(|e| e.source.starts_with("tool-use:"))
            .collect();
        let tool_total = tool_execs.len();
        let cutoff = chrono::Utc::now() - chrono::Duration::hours(24);
        let tool_today = tool_execs.iter().filter(|e| e.created_at > cutoff).count();
        println!();
        println!("── tool executions ──────────────────────────────────");
        println!("  today (24h): {tool_today}");
        println!("  all time:    {tool_total}");
        if tool_total > 0 {
            // Count by tool name for a per-tool breakdown.
            let mut by_tool: std::collections::BTreeMap<&str, usize> =
                std::collections::BTreeMap::new();
            for e in &tool_execs {
                let tool_name = e.source.trim_start_matches("tool-use:");
                *by_tool.entry(tool_name).or_insert(0) += 1;
            }
            for (t, n) in &by_tool {
                println!("    {t}: {n}");
            }
        }
    }

    println!();
    println!("── redb index ───────────────────────────────────────");
    match (stats.index_size, stats.index_cache) {
        (Some(size), Some(cache)) => {
            println!("  entries:    {size}");
            println!("  cache cap:  {}", cache.capacity);
            println!("  cache len:  {}", cache.len);
            println!("  hits:       {}", cache.hits);
            println!("  misses:     {}", cache.misses);
            println!("  hit rate:   {:.1}%", cache.hit_rate_pct);
        }
        _ => println!("  (index not enabled — run with daemon to activate)"),
    }

    println!();
    println!("── vault checksums ──────────────────────────────────");
    if stats.vault_files.is_empty() {
        println!("  (vault not configured)");
    } else {
        for f in &stats.vault_files {
            let status = match (f.exists, f.checksum_valid) {
                (false, _) => "MISSING",
                (true, true) => "OK",
                (true, false) => "MODIFIED (human edit detected)",
            };
            println!("  {:<28}  {status}", f.filename);
        }
    }
}

pub(crate) fn run_memory_inspect_core(memory: &MemoryManager, limit: usize) {
    let mut entries = memory.entries_by_tier(MemoryTier::Core);
    entries.sort_by(|left, right| right.created_at.cmp(&left.created_at));

    println!("core memories (latest {limit})");
    for (index, entry) in entries.into_iter().take(limit).enumerate() {
        println!("{}. [{}] {}", index + 1, entry.created_at, entry.content);
    }
}

pub(crate) fn run_memory_promotions(memory: &MemoryManager, limit: usize) {
    let entries = memory.recent_promotions(limit);
    println!("memory promotions (latest {})", entries.len());
    for (index, entry) in entries.into_iter().enumerate() {
        println!(
            "{}. [{}] {:?} {} (source={})",
            index + 1,
            entry.created_at,
            entry.tier,
            entry.content,
            entry.source
        );
    }
}

pub(crate) fn run_memory_export_vault(memory: &MemoryManager, path: &str) -> Result<()> {
    let summary = memory.export_vault(path)?;
    println!("memory vault export complete");
    println!("- root: {}", summary.root);
    println!("- notes: {}", summary.note_count);
    println!("- topics: {}", summary.topic_count);
    println!("- daily notes: {}", summary.daily_note_count);
    Ok(())
}

#[derive(Debug, Clone)]
struct GateCheck {
    name: &'static str,
    passed: bool,
    details: String,
}

pub(crate) async fn run_phase_review_gate(
    config: &mut AppConfig,
    memory: &mut MemoryManager,
    memory_log_path: &Path,
    report_path: Option<&str>,
) -> Result<()> {
    let mut config_changed = false;
    if !config.memory.backend.eq_ignore_ascii_case("eventlog") {
        config.memory.backend = "eventlog".to_string();
        config_changed = true;
    }
    if !config
        .memory
        .auto_sleep_mode
        .eq_ignore_ascii_case("nightly")
    {
        config.memory.auto_sleep_mode = "nightly".to_string();
        config_changed = true;
    }
    if config_changed {
        config.save_to("config/default.toml")?;
    }

    if !Path::new(".aigent/vault/index.md").exists() {
        let _ = memory.export_vault(".aigent/vault")?;
    }

    let has_sleep_marker = memory
        .all()
        .iter()
        .any(|entry| entry.source.starts_with("sleep:"));
    if !has_sleep_marker && !memory.all().is_empty() {
        let _ = memory.run_sleep_cycle().await?;
    }

    let event_log = MemoryEventLog::new(memory_log_path);
    let events = event_log.load().await?;
    let promotions = memory
        .all()
        .iter()
        .filter(|entry| entry.source.starts_with("sleep:"))
        .count();
    let vault_index = Path::new(".aigent/vault/index.md");
    let telegram_token_present = std::env::var("TELEGRAM_BOT_TOKEN")
        .ok()
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false);

    let checks = vec![
        GateCheck {
            name: "config memory backend",
            passed: config.memory.backend.eq_ignore_ascii_case("eventlog"),
            details: format!("backend={}", config.memory.backend),
        },
        GateCheck {
            name: "sleep mode nightly",
            passed: config
                .memory
                .auto_sleep_mode
                .eq_ignore_ascii_case("nightly"),
            details: format!(
                "mode={} window={:02}:00-{:02}:00",
                config.memory.auto_sleep_mode,
                config.memory.night_sleep_start_hour,
                config.memory.night_sleep_end_hour
            ),
        },
        GateCheck {
            name: "memory event log readable",
            passed: memory_log_path.exists(),
            details: format!("path={} events={}", memory_log_path.display(), events.len()),
        },
        GateCheck {
            name: "core identity seeded",
            passed: !memory.entries_by_tier(MemoryTier::Core).is_empty(),
            details: format!(
                "core_entries={}",
                memory.entries_by_tier(MemoryTier::Core).len()
            ),
        },
        GateCheck {
            name: "vault projection exists",
            passed: vault_index.exists(),
            details: format!("index_path={}", vault_index.display()),
        },
        GateCheck {
            name: "sleep promotion evidence",
            passed: promotions > 0,
            details: format!("promotions={promotions}"),
        },
        GateCheck {
            name: "telegram token configured",
            passed: telegram_token_present,
            details: "env=TELEGRAM_BOT_TOKEN".to_string(),
        },
    ];

    let passed = checks.iter().filter(|check| check.passed).count();
    let total = checks.len();

    println!("phase review gate (phase 0-2)");
    println!("- auto-remediation: enabled for fixable checks");
    for check in &checks {
        let marker = if check.passed { "PASS" } else { "FAIL" };
        println!("- [{marker}] {} ({})", check.name, check.details);
    }
    println!("- summary: {passed}/{total} checks passed");

    if let Some(path) = report_path {
        let now = Local::now().to_rfc3339();
        let mut rendered = String::new();
        rendered.push_str("# Aigent Phase Review Gate\n\n");
        rendered.push_str(&format!("Generated: {now}\n\n"));
        rendered.push_str("## Checks\n");
        for check in &checks {
            let marker = if check.passed { "PASS" } else { "FAIL" };
            rendered.push_str(&format!(
                "- [{marker}] {} ({})\n",
                check.name, check.details
            ));
        }
        rendered.push_str(&format!("\n## Summary\n- {passed}/{total} checks passed\n"));
        if passed == total {
            rendered.push_str("- Gate result: PASS\n");
        } else {
            rendered.push_str("- Gate result: FAIL\n");
        }

        let output = Path::new(path);
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(output, rendered)?;
        println!("- report written: {}", output.display());
    }

    if passed != total {
        bail!("phase review gate failed: resolve failing checks before advancing");
    }

    println!("phase review gate passed: ready for phase 3");
    Ok(())
}

pub(crate) fn layer_to_tiers(layer: CliMemoryLayer) -> Vec<MemoryTier> {
    match layer {
        CliMemoryLayer::All => vec![
            MemoryTier::Episodic,
            MemoryTier::Semantic,
            MemoryTier::Procedural,
            MemoryTier::Core,
        ],
        CliMemoryLayer::Episodic => vec![MemoryTier::Episodic],
        CliMemoryLayer::Semantic => vec![MemoryTier::Semantic],
        CliMemoryLayer::Procedural => vec![MemoryTier::Procedural],
        CliMemoryLayer::Core => vec![MemoryTier::Core],
    }
}

pub(crate) fn memory_layer_label(layer: CliMemoryLayer) -> &'static str {
    match layer {
        CliMemoryLayer::All => "all",
        CliMemoryLayer::Episodic => "episodic",
        CliMemoryLayer::Semantic => "semantic",
        CliMemoryLayer::Procedural => "procedural",
        CliMemoryLayer::Core => "core",
    }
}

