use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::Path;

use anyhow::Result;

use crate::schema::{MemoryEntry, MemoryTier};

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
    if root.exists() {
        fs::remove_dir_all(root)?;
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
        let tier_label = tier_slug(entry.tier);
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
    ] {
        let slug = tier_slug(tier);
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
    ] {
        let slug = tier_slug(tier);
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
    let id_short = entry.id.to_string().chars().take(8).collect::<String>();
    format!("{date}-{}-{id_short}", tier_slug(entry.tier))
}

fn tier_slug(tier: MemoryTier) -> &'static str {
    match tier {
        MemoryTier::Episodic => "episodic",
        MemoryTier::Semantic => "semantic",
        MemoryTier::Procedural => "procedural",
        MemoryTier::Reflective => "reflective",
        MemoryTier::UserProfile => "user-profile",
        MemoryTier::Core => "core",
    }
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
