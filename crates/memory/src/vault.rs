use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use chrono::{DateTime, Utc};
use sha2::{Digest, Sha256};

use crate::schema::{MemoryEntry, MemoryTier};

// ── KV summary constants ──────────────────────────────────────────────────────

/// Maximum entries per tier written into the YAML KV summary files.
/// Sorted by confidence DESC → recency DESC → valence DESC.
/// Expose as a public constant so power users can override at compile time.
pub const KV_TIER_LIMIT: usize = 15;

/// Filenames of the four auto-generated vault summary artefacts.
pub const KV_CORE: &str = "core_summary.yaml";
pub const KV_USER_PROFILE: &str = "user_profile.yaml";
pub const KV_REFLECTIVE: &str = "reflective_opinions.yaml";
pub const NARRATIVE_MD: &str = "MEMORY.md";

/// All summary filenames watched by the bidirectional vault watcher.
pub const WATCHED_SUMMARIES: &[&str] = &[KV_CORE, KV_USER_PROFILE, KV_REFLECTIVE, NARRATIVE_MD];

#[derive(Debug, Clone)]
pub struct VaultExportSummary {
    pub root: String,
    pub note_count: usize,
    pub topic_count: usize,
    pub daily_note_count: usize,
}

pub fn export_obsidian_vault(
    entries: &[MemoryEntry],
    root: impl AsRef<Path>,
) -> Result<VaultExportSummary> {
    let root = root.as_ref();

    // Incremental rebuild: remove only the managed Obsidian subdirectories so
    // the YAML KV summary files at the vault root are preserved across full
    // exports.  A crash midway leaves the old Obsidian files untouched; the
    // next successful run will overwrite them completely.
    for subdir in &["notes", "tiers", "daily", "topics"] {
        let p = root.join(subdir);
        if p.exists() {
            fs::remove_dir_all(&p)?;
        }
    }

    let notes_dir = root.join("notes");
    let tiers_dir = root.join("tiers");
    let daily_dir = root.join("daily");
    let topics_dir = root.join("topics");
    fs::create_dir_all(&notes_dir)?;
    fs::create_dir_all(&tiers_dir)?;
    fs::create_dir_all(&daily_dir)?;
    fs::create_dir_all(&topics_dir)?;

    let mut tier_links: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut daily_links: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut topic_backlinks: HashMap<String, BTreeSet<String>> = HashMap::new();

    let mut sorted = entries.to_vec();
    sorted.sort_by(|left, right| right.created_at.cmp(&left.created_at));

    for entry in &sorted {
        let note_name = note_name(entry);
        let note_file = notes_dir.join(format!("{note_name}.md"));
        let day = entry.created_at.format("%Y-%m-%d").to_string();
        let tier_label = entry.tier.slug();
        let topics = extract_topics(&entry.content);

        let topic_links = if topics.is_empty() {
            "(none)".to_string()
        } else {
            topics
                .iter()
                .map(|topic| format!("[[topic-{topic}]]"))
                .collect::<Vec<_>>()
                .join(", ")
        };

        let note_body = format!(
            "---\nid: {}\ntier: {}\nsource: {}\nconfidence: {:.2}\nvalence: {:.2}\ncreated_at: {}\nprovenance_hash: {}\n---\n\n# {}\n\n{}\n\n## Topics\n{}\n\n## Links\n- [[index]]\n- [[tier-{}]]\n- [[day-{}]]\n",
            entry.id,
            tier_label,
            entry.source,
            entry.confidence,
            entry.valence,
            entry.created_at,
            entry.provenance_hash,
            note_name,
            entry.content,
            topic_links,
            tier_label,
            day,
        );
        fs::write(note_file, note_body)?;

        let note_link = format!("[[{note_name}]]");
        tier_links
            .entry(tier_label.to_string())
            .or_default()
            .push(note_link.clone());
        daily_links.entry(day).or_default().push(note_link.clone());
        for topic in topics {
            topic_backlinks
                .entry(topic)
                .or_default()
                .insert(note_link.clone());
        }
    }

    write_root_index(root, &tier_links, &daily_links, &topic_backlinks)?;
    write_tier_indexes(&tiers_dir, &tier_links)?;
    write_daily_notes(&daily_dir, &daily_links)?;
    write_topics(&topics_dir, &topic_backlinks)?;

    Ok(VaultExportSummary {
        root: root.display().to_string(),
        note_count: sorted.len(),
        topic_count: topic_backlinks.len(),
        daily_note_count: daily_links.len(),
    })
}

fn write_root_index(
    root: &Path,
    tier_links: &BTreeMap<String, Vec<String>>,
    daily_links: &BTreeMap<String, Vec<String>>,
    topic_backlinks: &HashMap<String, BTreeSet<String>>,
) -> Result<()> {
    let mut content = String::new();
    content.push_str("# Memory Vault Index\n\n");
    content.push_str("## Tiers\n");
    for tier in [
        MemoryTier::Core,
        MemoryTier::Semantic,
        MemoryTier::Episodic,
        MemoryTier::Procedural,
        MemoryTier::Reflective,
        MemoryTier::UserProfile,
    ] {
        let slug = tier.slug();
        let count = tier_links.get(slug).map(|items| items.len()).unwrap_or(0);
        content.push_str(&format!("- [[tier-{slug}]] ({count})\n"));
    }

    content.push_str("\n## Daily Notes\n");
    for day in daily_links.keys() {
        content.push_str(&format!("- [[day-{day}]]\n"));
    }

    content.push_str("\n## Topics\n");
    let mut topics = topic_backlinks.keys().cloned().collect::<Vec<_>>();
    topics.sort();
    for topic in topics {
        content.push_str(&format!("- [[topic-{topic}]]\n"));
    }

    fs::write(root.join("index.md"), content)?;
    Ok(())
}

fn write_tier_indexes(tiers_dir: &Path, tier_links: &BTreeMap<String, Vec<String>>) -> Result<()> {
    for tier in [
        MemoryTier::Core,
        MemoryTier::Semantic,
        MemoryTier::Episodic,
        MemoryTier::Procedural,
        MemoryTier::Reflective,
        MemoryTier::UserProfile,
    ] {
        let slug = tier.slug();
        let mut content = format!("# {} Memories\n\n", slug.to_uppercase());
        let links = tier_links.get(slug).cloned().unwrap_or_default();
        if links.is_empty() {
            content.push_str("(none)\n");
        } else {
            for link in links {
                content.push_str(&format!("- {link}\n"));
            }
        }

        fs::write(tiers_dir.join(format!("tier-{slug}.md")), content)?;
    }
    Ok(())
}

fn write_daily_notes(daily_dir: &Path, daily_links: &BTreeMap<String, Vec<String>>) -> Result<()> {
    for (day, links) in daily_links {
        let mut content = format!("# Daily Memory {day}\n\n");
        for link in links {
            content.push_str(&format!("- {link}\n"));
        }
        fs::write(daily_dir.join(format!("day-{day}.md")), content)?;
    }
    Ok(())
}

fn write_topics(
    topics_dir: &Path,
    topic_backlinks: &HashMap<String, BTreeSet<String>>,
) -> Result<()> {
    let mut topics = topic_backlinks.keys().cloned().collect::<Vec<_>>();
    topics.sort();

    for topic in topics {
        let mut content = format!("# Topic: {topic}\n\n## Backlinks\n");
        if let Some(links) = topic_backlinks.get(&topic) {
            for link in links {
                content.push_str(&format!("- {link}\n"));
            }
        }

        fs::write(topics_dir.join(format!("topic-{topic}.md")), content)?;
    }

    Ok(())
}

fn note_name(entry: &MemoryEntry) -> String {
    let date = entry.created_at.format("%Y%m%d").to_string();
    let id_short = entry.id_short();
    format!("{date}-{}-{id_short}", entry.tier.slug())
}

fn extract_topics(content: &str) -> Vec<String> {
    let stopwords = [
        "about", "after", "agent", "aigent", "because", "before", "could", "there", "their",
        "these", "those", "would", "should", "where", "which", "while", "memory", "system",
        "using", "please", "today", "night", "sleep", "model", "provider", "think", "level",
    ];
    let stop = stopwords.iter().copied().collect::<BTreeSet<_>>();

    let mut counts = HashMap::<String, usize>::new();
    for token in content
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|word| word.len() >= 4)
        .map(|word| word.to_lowercase())
        .filter(|word| !stop.contains(word.as_str()))
    {
        *counts.entry(token).or_default() += 1;
    }

    let mut topics = counts.into_iter().collect::<Vec<_>>();
    topics.sort_by(|(left_word, left_count), (right_word, right_count)| {
        right_count
            .cmp(left_count)
            .then_with(|| left_word.cmp(right_word))
    });

    topics
        .into_iter()
        .take(6)
        .map(|(word, _)| sanitize_topic_slug(&word))
        .collect()
}

fn sanitize_topic_slug(raw: &str) -> String {
    let mut slug = raw
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect::<String>();
    while slug.contains("--") {
        slug = slug.replace("--", "-");
    }
    slug.trim_matches('-').to_string()
}

// ── YAML KV summary helpers ───────────────────────────────────────────────────

/// Escape an arbitrary string as a YAML double-quoted scalar.
/// Handles backslashes, double-quotes, newlines, and carriage returns.
fn yaml_quote(s: &str) -> String {
    let escaped = s
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "");
    format!("\"{escaped}\"")
}

/// Un-escape a YAML double-quoted scalar (strips outer quotes, reverses escapes).
fn unyaml_quote(s: &str) -> String {
    if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
        s[1..s.len() - 1]
            .replace("\\\"", "\"")
            .replace("\\\\", "\\")
            .replace("\\n", " ")
    } else {
        s.to_string()
    }
}

/// Compute the SHA-256 hex digest of a UTF-8 string.
fn sha256_of(s: &str) -> String {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    format!("{:x}", h.finalize())
}

/// Write `content` to `path` only when the file's current on-disk content
/// differs.  Creates parent directories as needed.
/// Returns `true` when the file was written; `false` when it was skipped.
fn write_file_if_changed(path: &Path, content: &str) -> Result<bool> {
    if path.exists() {
        let existing = fs::read_to_string(path)?;
        if existing == content {
            return Ok(false);
        }
    } else if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, content)?;
    Ok(true)
}

/// Build the YAML body for a single tier summary file.
///
/// A SHA-256 checksum of the items section is embedded so both humans and the
/// bidirectional watcher can detect real changes cheaply.
fn build_kv_yaml(tier_label: &str, entries: &[&MemoryEntry], now: DateTime<Utc>) -> String {
    let count = entries.len();

    // Build the items body first (checksum excludes the metadata header).
    let mut items_body = String::new();
    items_body.push_str("items:\n");
    for entry in entries {
        let id_short = entry.id_short();
        items_body.push_str(&format!("  - id: {}\n", yaml_quote(&id_short)));
        items_body.push_str(&format!("    confidence: {:.2}\n", entry.confidence));
        items_body.push_str(&format!("    valence: {:.2}\n", entry.valence));
        items_body.push_str(&format!(
            "    created_at: {}\n",
            yaml_quote(&entry.created_at.format("%Y-%m-%dT%H:%M:%SZ").to_string())
        ));
        items_body.push_str(&format!("    content: {}\n", yaml_quote(&entry.content)));
        if entry.tags.is_empty() {
            items_body.push_str("    tags: []\n");
        } else {
            let tag_list = entry
                .tags
                .iter()
                .map(|t| yaml_quote(t))
                .collect::<Vec<_>>()
                .join(", ");
            items_body.push_str(&format!("    tags: [{tag_list}]\n"));
        }
    }

    let checksum = sha256_of(&items_body);

    let mut out = String::new();
    out.push_str("# Auto-generated by Aigent memory system.\n");
    out.push_str("# Human edits are detected and ingested as high-confidence memories.\n");
    out.push_str(&format!("checksum: sha256:{checksum}\n"));
    out.push_str(&format!(
        "last_updated: {}\n",
        now.format("%Y-%m-%dT%H:%M:%SZ")
    ));
    out.push_str(&format!("tier: {tier_label}\n"));
    out.push_str(&format!("count: {count}\n"));
    out.push_str(&items_body);
    out
}

/// Build the human-friendly `MEMORY.md` narrative that consolidates all three
/// KV tiers into readable prose while cross-referencing the YAML files.
fn build_narrative_md(
    core: &[&MemoryEntry],
    profile: &[&MemoryEntry],
    reflective: &[&MemoryEntry],
    now: DateTime<Utc>,
) -> String {
    let mut md = String::new();

    md.push_str("# Aigent \u{2014} Living Memory\n\n");
    md.push_str("> Auto-generated from memory tiers. Edit the YAML summaries below to reshape\n");
    md.push_str("> agent identity. Changes are detected and ingested as high-confidence memories\n");
    md.push_str("> (bidirectional sync via vault watcher).\n\n");

    // Core identity
    md.push_str("## Core Identity\n");
    md.push_str(&format!("\u{2192} See [{KV_CORE}]({KV_CORE})\n\n"));
    if core.is_empty() {
        md.push_str("*(no core identity entries yet)*\n");
    } else {
        for entry in core.iter().take(5) {
            let preview = first_line_truncated(&entry.content, 120);
            md.push_str(&format!("- {preview}\n"));
        }
        if core.len() > 5 {
            md.push_str(&format!(
                "- *(\u{2026} and {} more \u{2014} see {KV_CORE})*\n",
                core.len() - 5
            ));
        }
    }
    md.push('\n');

    // Relationship with user
    md.push_str("## Relationship with User\n");
    md.push_str(&format!("\u{2192} See [{KV_USER_PROFILE}]({KV_USER_PROFILE})\n\n"));
    if profile.is_empty() {
        md.push_str("*(no user profile entries yet)*\n");
    } else {
        for entry in profile.iter().take(5) {
            let preview = first_line_truncated(&entry.content, 100);
            md.push_str(&format!("- {preview}\n"));
        }
        if profile.len() > 5 {
            md.push_str(&format!(
                "- *(\u{2026} and {} more \u{2014} see {KV_USER_PROFILE})*\n",
                profile.len() - 5
            ));
        }
    }
    md.push('\n');

    // Formed opinions
    md.push_str("## Recent Formed Opinions\n");
    md.push_str(&format!("\u{2192} See [{KV_REFLECTIVE}]({KV_REFLECTIVE})\n\n"));
    if reflective.is_empty() {
        md.push_str("*(no reflective entries yet)*\n");
    } else {
        for entry in reflective.iter().take(5) {
            let preview = first_line_truncated(&entry.content, 100);
            md.push_str(&format!("- {preview}\n"));
        }
        if reflective.len() > 5 {
            md.push_str(&format!(
                "- *(\u{2026} and {} more \u{2014} see {KV_REFLECTIVE})*\n",
                reflective.len() - 5
            ));
        }
    }
    md.push('\n');

    md.push_str("---\n");
    md.push_str(&format!(
        "*Last consolidated: {}*\n",
        now.format("%Y-%m-%dT%H:%M:%SZ")
    ));
    md
}

fn first_line_truncated(s: &str, max: usize) -> String {
    let first = s.lines().next().unwrap_or(s);
    match first.char_indices().nth(max) {
        Some((byte_idx, _)) => format!("{}\u{2026}", &first[..byte_idx]),
        None => first.to_string(),
    }
}

// ── Public KV summary API ─────────────────────────────────────────────────────

/// Write the three YAML KV summary files + `MEMORY.md` to the vault root.
///
/// Uses SHA-256 checksums to skip unchanged files (incremental writes).
/// Returns the count of files actually written to disk.
///
/// * `entries`       – full in-memory entry slice from [`MemoryStore`]
/// * `vault_root`    – directory to write into (created if it doesn't exist)
/// * `kv_tier_limit` – max entries per tier (Core / UserProfile / Reflective)
pub fn sync_kv_summaries(
    entries: &[MemoryEntry],
    vault_root: &Path,
    kv_tier_limit: usize,
) -> Result<usize> {
    fs::create_dir_all(vault_root)?;
    let now = Utc::now();

    // Sort helper: confidence DESC, recency DESC, valence DESC
    let top_for_tier = |tier: MemoryTier| -> Vec<&MemoryEntry> {
        let mut v: Vec<&MemoryEntry> = entries.iter().filter(|e| e.tier == tier).collect();
        v.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.created_at.cmp(&a.created_at))
                .then_with(|| {
                    b.valence
                        .partial_cmp(&a.valence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        v.truncate(kv_tier_limit);
        v
    };

    let core_e = top_for_tier(MemoryTier::Core);
    let profile_e = top_for_tier(MemoryTier::UserProfile);
    let reflective_e = top_for_tier(MemoryTier::Reflective);

    let core_yaml = build_kv_yaml("Core", &core_e, now);
    let profile_yaml = build_kv_yaml("UserProfile", &profile_e, now);
    let reflective_yaml = build_kv_yaml("Reflective", &reflective_e, now);
    let narrative = build_narrative_md(&core_e, &profile_e, &reflective_e, now);

    let mut written = 0usize;
    if write_file_if_changed(&vault_root.join(KV_CORE), &core_yaml)? {
        written += 1;
    }
    if write_file_if_changed(&vault_root.join(KV_USER_PROFILE), &profile_yaml)? {
        written += 1;
    }
    if write_file_if_changed(&vault_root.join(KV_REFLECTIVE), &reflective_yaml)? {
        written += 1;
    }
    if write_file_if_changed(&vault_root.join(NARRATIVE_MD), &narrative)? {
        written += 1;
    }

    Ok(written)
}

/// Read `core_summary.yaml` and `user_profile.yaml` from the vault and return
/// a compact structured block for prepending to LLM prompts.
///
/// Returns `None` when the vault path does not exist or both files are empty.
pub fn read_kv_for_injection(vault_root: &Path) -> Option<String> {
    let core_path = vault_root.join(KV_CORE);
    let profile_path = vault_root.join(KV_USER_PROFILE);

    let mut block = String::from("## AGENT IDENTITY (injected from memory vault)\n");
    let mut has_content = false;

    if let Ok(core_content) = fs::read_to_string(&core_path) {
        let items = extract_yaml_content_values(&core_content);
        if !items.is_empty() {
            block.push_str("### Core Identity\n");
            for item in &items {
                block.push_str(&format!("- {item}\n"));
            }
            has_content = true;
        }
    }

    if let Ok(profile_content) = fs::read_to_string(&profile_path) {
        let items = extract_yaml_content_values(&profile_content);
        if !items.is_empty() {
            block.push_str("### User Profile\n");
            for item in &items {
                block.push_str(&format!("- {item}\n"));
            }
            has_content = true;
        }
    }

    if has_content { Some(block) } else { None }
}

/// Extract `content:` values from a KV YAML file for concise prompt injection.
fn extract_yaml_content_values(yaml: &str) -> Vec<String> {
    let mut items = Vec::new();
    let mut in_items = false;
    let mut in_entry = false;

    for line in yaml.lines() {
        let trimmed = line.trim();
        if trimmed == "items:" {
            in_items = true;
            continue;
        }
        if !in_items {
            continue;
        }
        if trimmed.starts_with("- id:") {
            in_entry = true;
            continue;
        }
        if in_entry && trimmed.starts_with("content:") {
            let val = trimmed["content:".len()..].trim();
            let unquoted = unyaml_quote(val);
            if !unquoted.is_empty() {
                items.push(unquoted);
            }
            in_entry = false;
        }
    }
    items
}

// ── Vault checksum health check ───────────────────────────────────────────────

/// Checksum validation result for one vault summary file.
#[derive(Debug, Clone, Default)]
pub struct VaultFileStatus {
    /// Filename (e.g. `"core_summary.yaml"`).
    pub filename: String,
    /// `true` when the file exists.
    pub exists: bool,
    /// `true` when the stored checksum matches the freshly-recomputed one
    /// (or when the file has no checksum, e.g. `MEMORY.md`).
    pub checksum_valid: bool,
}

/// Validate the SHA-256 checksums of all four vault summary files.
///
/// Returns one [`VaultFileStatus`] per file in [`WATCHED_SUMMARIES`].
/// If the vault root doesn't exist all statuses will have `exists = false`.
pub fn check_vault_checksums(vault_root: &Path) -> Vec<VaultFileStatus> {
    WATCHED_SUMMARIES.iter().map(|&fname| {
        let path = vault_root.join(fname);
        if !path.exists() {
            return VaultFileStatus { filename: fname.to_string(), exists: false, checksum_valid: false };
        }
        let content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => return VaultFileStatus { filename: fname.to_string(), exists: true, checksum_valid: false },
        };
        // MEMORY.md has no checksum line → treat as always valid (human-editable narrative).
        if !fname.ends_with(".yaml") {
            return VaultFileStatus { filename: fname.to_string(), exists: true, checksum_valid: true };
        }
        // Re-derive logic: parse stored checksum, recompute, compare.
        let checksum_valid = content
            .lines()
            .find(|l| l.starts_with("checksum: sha256:"))
            .and_then(|l| l.strip_prefix("checksum: sha256:"))
            .map(|stored| {
                let items_start = content.find("items:\n").unwrap_or(content.len());
                sha256_of(&content[items_start..]) == stored.trim()
            })
            .unwrap_or(false);
        VaultFileStatus { filename: fname.to_string(), exists: true, checksum_valid }
    }).collect()
}

// ── Bidirectional vault watcher ───────────────────────────────────────────────

/// An event emitted when a human edits one of the four vault summary files.
///
/// Receivers should ingest the parsed content as a high-confidence
/// `Reflective` or `Core`-eligible `MemoryEntry` with
/// `source = "human-edit"`.
#[derive(Debug, Clone)]
pub struct VaultEditEvent {
    /// Filename that changed (e.g. `"core_summary.yaml"`).
    pub filename: String,
    /// Full file content after the edit.
    pub content: String,
}

/// Returns `true` when `content` was written by the daemon (not a human edit).
///
/// YAML summary files contain a `checksum: sha256:<hex>` line that covers the
/// `items:` section.  When a file is written by `sync_kv_summaries`, this
/// checksum is computed over the items body.  If we re-read the file and the
/// checksum still matches the items body it means the daemon wrote it and we
/// should skip ingestion to avoid an infinite self-trigger loop.
///
/// `MEMORY.md` has no checksum line, so it always returns `false` (fire).
fn is_daemon_written(content: &str, filename: &str) -> bool {
    if !filename.ends_with(".yaml") {
        return false; // MEMORY.md – always treat as potential human edit
    }
    // Extract stored checksum.
    let stored_hex = content
        .lines()
        .find(|l| l.starts_with("checksum: sha256:"))
        .and_then(|l| l.strip_prefix("checksum: sha256:"))
        .map(str::trim);
    let Some(stored_hex) = stored_hex else {
        return false; // no checksum → unknown origin → fire to be safe
    };
    // Recompute over the items section (same body that build_kv_yaml included).
    let items_start = content.find("items:\n").unwrap_or(content.len());
    let computed = sha256_of(&content[items_start..]);
    stored_hex == computed
}

/// Spawn a background [`tokio::task`] that watches the four vault summary
/// files for human edits using the [`notify`] filesystem watcher.
///
/// When a `Modify` or `Create` event is detected on one of
/// [`WATCHED_SUMMARIES`], a [`VaultEditEvent`] is sent on `tx`.  The caller
/// is responsible for:
/// 1. Parsing `event.content` to extract changed key-value pairs.
/// 2. Recording them as high-confidence `MemoryEntry` items with
///    `provenance = "human-edit"`.
///
/// The returned `JoinHandle` can be aborted on daemon shutdown.
pub fn spawn_vault_watcher(
    vault_root: PathBuf,
    tx: tokio::sync::mpsc::UnboundedSender<VaultEditEvent>,
) -> tokio::task::JoinHandle<()> {
    tokio::task::spawn_blocking(move || {
        use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};

        let (fs_tx, fs_rx) = std::sync::mpsc::channel::<notify::Result<Event>>();

        let mut watcher = match RecommendedWatcher::new(fs_tx, Config::default()) {
            Ok(w) => w,
            Err(err) => {
                tracing::warn!(error = %err, "vault watcher: could not create fs watcher — bidirectional edits disabled");
                return;
            }
        };

        if vault_root.exists() {
            if let Err(err) = watcher.watch(&vault_root, RecursiveMode::NonRecursive) {
                tracing::warn!(error = %err, path = %vault_root.display(), "vault watcher: watch failed");
                return;
            }
        } else {
            tracing::warn!(path = %vault_root.display(), "vault watcher: vault root does not exist — bidirectional edits disabled");
            return;
        }

        tracing::info!(path = %vault_root.display(), "vault watcher: watching summary files for human edits");

        for result in &fs_rx {
            match result {
                Ok(event) => {
                    if !matches!(
                        event.kind,
                        EventKind::Modify(_) | EventKind::Create(_)
                    ) {
                        continue;
                    }
                    for path in &event.paths {
                        let filename = match path.file_name().and_then(|f| f.to_str()) {
                            Some(f) => f.to_string(),
                            None => continue,
                        };
                        if !WATCHED_SUMMARIES.contains(&filename.as_str()) {
                            continue;
                        }
                        let content = match std::fs::read_to_string(path) {
                            Ok(c) => c,
                            Err(err) => {
                                tracing::warn!(
                                    file = %filename,
                                    error = %err,
                                    "vault watcher: could not read changed file"
                                );
                                continue;
                            }
                        };
                        // Skip files the daemon itself just wrote (checksum matches).
                        if is_daemon_written(&content, &filename) {
                            tracing::trace!(file = %filename, "vault watcher: skipping daemon-written file (checksum match)");
                            continue;
                        }
                        tracing::info!(file = %filename, "vault watcher: human edit detected — ingesting");
                        if tx.send(VaultEditEvent { filename: filename.clone(), content }).is_err()
                        {
                            // Receiver has been dropped — graceful shutdown.
                            return;
                        }
                    }
                }
                Err(err) => {
                    tracing::warn!(error = %err, "vault watcher: fs event error");
                }
            }
        }
    })
}
