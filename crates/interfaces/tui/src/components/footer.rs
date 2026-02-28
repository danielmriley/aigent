//! Footer bar — keybinding hints along the bottom edge.

use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::theme::Theme;

/// Footer component showing keyboard shortcut hints.
pub struct Footer;

impl Default for Footer {
    fn default() -> Self {
        Self::new()
    }
}

impl Footer {
    pub fn new() -> Self {
        Self
    }

    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let hints: Vec<(&str, &str)> = vec![
            ("Enter", "send"),
            ("Alt+Enter", "newline"),
            ("Ctrl+K", "commands"),
            ("Alt+S", "select"),
            ("@", "file"),
            ("Esc", "history"),
        ];

        let mut spans = vec![Span::styled(" ", Style::default())];
        for (i, (key, desc)) in hints.iter().enumerate() {
            if i > 0 {
                spans.push(Span::styled(
                    " │ ",
                    Style::default().fg(theme.border),
                ));
            }
            spans.push(Span::styled(
                (*key).to_string(),
                Style::default().fg(theme.accent),
            ));
            spans.push(Span::styled(
                format!(" {desc}"),
                Style::default().fg(theme.muted),
            ));
        }

        let footer = Paragraph::new(Line::from(spans));
        frame.render_widget(footer, area);
    }
}
