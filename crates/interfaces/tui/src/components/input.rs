//! Input bar — multi-line text area with soft-wrap.

use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::widgets::{Block, BorderType, Borders};
use ratatui::Frame;
use tui_textarea::TextArea;

use crate::theme::Theme;

/// The multi-line input bar at the bottom of the TUI.
pub struct InputBar {
    pub textarea: TextArea<'static>,
    pub wrap_width: usize,
}

impl Default for InputBar {
    fn default() -> Self {
        Self::new()
    }
}

impl InputBar {
    pub fn new() -> Self {
        let mut textarea = TextArea::default();
        textarea.set_cursor_line_style(Default::default());
        Self {
            textarea,
            wrap_width: 60,
        }
    }

    /// Current text content (all lines joined).
    pub fn text(&self) -> String {
        self.textarea.lines().join("\n")
    }

    /// Reset the text area to empty.
    pub fn clear(&mut self) {
        self.textarea = TextArea::default();
        self.textarea.set_cursor_line_style(Default::default());
    }

    /// Apply soft-wrapping to the input text at the given width.
    pub fn apply_soft_wrap(&mut self) {
        let raw = self.text();
        let wrapped = soft_wrap_text(&raw, self.wrap_width);
        if wrapped != raw {
            let cursor = self.textarea.cursor();
            self.textarea = TextArea::default();
            self.textarea.set_cursor_line_style(Default::default());
            self.textarea.insert_str(&wrapped);
            // Try to preserve cursor line position.
            let max_line = self.textarea.lines().len().saturating_sub(1);
            let target_line = cursor.0.min(max_line);
            // Move cursor to approximately the same position.
            for _ in 0..target_line {
                self.textarea.move_cursor(tui_textarea::CursorMove::Down);
            }
        }
    }

    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let mut ta = self.textarea.clone();
        ta.set_block(
            Block::default()
                .title(" Input (Alt+Enter newline) ")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(theme.border)),
        );
        ta.set_style(Style::default().fg(theme.foreground));
        frame.render_widget(&ta, area);
    }
}

// ── soft-wrap helpers ──────────────────────────────────────────

fn soft_wrap_text(input: &str, width: usize) -> String {
    let width = width.max(8);
    input
        .lines()
        .map(|line| soft_wrap_line(line, width))
        .collect::<Vec<_>>()
        .join("\n")
}

fn soft_wrap_line(line: &str, width: usize) -> String {
    if line.chars().count() <= width {
        return line.to_string();
    }

    let mut out = String::new();
    let mut current = String::new();

    for word in line.split_whitespace() {
        let current_len = current.chars().count();
        let word_len = word.chars().count();
        if current_len == 0 {
            if word_len <= width {
                current.push_str(word);
            } else {
                for chunk in split_hard(word, width) {
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str(&chunk);
                }
            }
            continue;
        }

        if current_len + 1 + word_len <= width {
            current.push(' ');
            current.push_str(word);
        } else {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(&current);
            current.clear();

            if word_len <= width {
                current.push_str(word);
            } else {
                for chunk in split_hard(word, width) {
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str(&chunk);
                }
            }
        }
    }

    if !current.is_empty() {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&current);
    }

    if out.is_empty() {
        line.to_string()
    } else {
        out
    }
}

fn split_hard(input: &str, width: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for ch in input.chars() {
        current.push(ch);
        if current.chars().count() >= width {
            chunks.push(current);
            current = String::new();
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}
