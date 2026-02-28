//! File picker overlay — fuzzy file search triggered by `@` in the input.

use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;
use ignore::WalkBuilder;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Clear, Paragraph, Wrap};
use ratatui::Frame;
use std::path::Path;

use crate::layout;
use crate::state::FilePopupState;
use crate::theme::Theme;

/// File picker popup component.
pub struct FilePicker {
    pub state: FilePopupState,
    pub workspace_files: Vec<String>,
}

impl Default for FilePicker {
    fn default() -> Self {
        Self::new()
    }
}

impl FilePicker {
    pub fn new() -> Self {
        Self {
            state: FilePopupState {
                visible: false,
                query: String::new(),
                candidates: Vec::new(),
                selected: 0,
            },
            workspace_files: collect_workspace_files(Path::new(".")),
        }
    }

    /// Refresh the popup based on the current input text.
    pub fn refresh(&mut self, input: &str) {
        let Some(query) = active_at_query(input) else {
            self.state.visible = false;
            self.state.query.clear();
            self.state.candidates.clear();
            self.state.selected = 0;
            return;
        };

        self.state.visible = true;
        self.state.query = query.to_string();

        if query.is_empty() {
            self.state.candidates = self
                .workspace_files
                .iter()
                .take(10)
                .cloned()
                .collect();
            self.state.selected = 0;
            return;
        }

        let matcher = SkimMatcherV2::default();
        let mut scored: Vec<(i64, String)> = self
            .workspace_files
            .iter()
            .filter_map(|path| {
                matcher
                    .fuzzy_match(path, query)
                    .map(|score| (score, path.clone()))
            })
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0));
        self.state.candidates = scored.into_iter().take(10).map(|(_, p)| p).collect();
        self.state.selected = 0;
    }

    pub fn draw(&self, frame: &mut Frame<'_>, theme: &Theme) {
        if !self.state.visible {
            return;
        }
        let area = layout::centered_popup(frame.area(), 70, 45);
        frame.render_widget(Clear, area);

        let mut lines = vec![Line::from(Span::styled(
            format!(" @{}", self.state.query),
            Style::default().fg(theme.accent),
        ))];
        lines.push(Line::from(""));

        if self.state.candidates.is_empty() {
            lines.push(Line::from(Span::styled(
                "  (no matches)",
                Style::default().fg(theme.muted),
            )));
        } else {
            for (idx, item) in self.state.candidates.iter().enumerate() {
                if idx == self.state.selected {
                    lines.push(Line::from(Span::styled(
                        format!("  ▸ {item}"),
                        Style::default()
                            .fg(theme.background)
                            .bg(theme.accent)
                            .add_modifier(Modifier::BOLD),
                    )));
                } else {
                    lines.push(Line::from(Span::styled(
                        format!("    {item}"),
                        Style::default().fg(theme.foreground),
                    )));
                }
            }
        }

        let widget = Paragraph::new(lines)
            .block(
                Block::default()
                    .title(" File Context ")
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(theme.accent)),
            )
            .wrap(Wrap { trim: false });
        frame.render_widget(widget, area);
    }
}

// ── helpers ────────────────────────────────────────────────────

fn collect_workspace_files(root: &Path) -> Vec<String> {
    let mut files = Vec::new();
    for entry in WalkBuilder::new(root)
        .hidden(false)
        .git_ignore(true)
        .build()
    {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
            continue;
        }
        let Ok(rel) = entry.path().strip_prefix(root) else {
            continue;
        };
        files.push(rel.display().to_string());
    }
    files.sort();
    files
}

fn active_at_query(input: &str) -> Option<&str> {
    let token = input.split_whitespace().last()?;
    if !token.starts_with('@') {
        return None;
    }
    Some(token.trim_start_matches('@'))
}

/// Replace the last `@token` in the input with `@path `.
pub fn replace_last_at_token(input: &str, replacement: &str) -> String {
    let end = input.len();
    let start = input
        .char_indices()
        .rev()
        .find(|(_, ch)| ch.is_whitespace())
        .map(|(idx, ch)| idx + ch.len_utf8())
        .unwrap_or(0);

    if !input[start..end].starts_with('@') {
        return input.to_string();
    }

    let mut out = input.to_string();
    out.replace_range(start..end, &format!("@{} ", replacement));
    out
}
