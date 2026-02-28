//! Sidebar panel — session list, conversation stats, keyboard hints.

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Wrap};
use ratatui::Frame;

use crate::state::AppState;
use crate::theme::Theme;

/// Sidebar component with collapsible visibility.
pub struct SidebarPanel {
    pub visible: bool,
}

impl SidebarPanel {
    pub fn new() -> Self {
        Self { visible: true }
    }

    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, state: &AppState, theme: &Theme) {
        let mut lines = Vec::new();

        // ── sessions ────────────────────────────────────────────
        lines.push(Line::from(Span::styled(
            " Sessions",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        )));
        for (idx, session) in state.sessions.iter().enumerate() {
            let marker = if idx == state.current_session {
                " ▸ "
            } else {
                "   "
            };
            let style = if idx == state.current_session {
                Style::default().fg(theme.foreground)
            } else {
                Style::default().fg(theme.muted)
            };
            lines.push(Line::from(Span::styled(
                format!("{marker}{session}"),
                style,
            )));
        }

        lines.push(Line::from(""));

        // ── conversation stats ──────────────────────────────────
        lines.push(Line::from(Span::styled(
            " Conversation",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(Span::styled(
            format!("   messages: {}", state.messages.len()),
            Style::default().fg(theme.foreground),
        )));
        lines.push(Line::from(Span::styled(
            format!(
                "   history:  {}",
                if state.history_mode { "on" } else { "off" }
            ),
            Style::default().fg(theme.foreground),
        )));
        if let Some(sel) = state.selected_message {
            lines.push(Line::from(Span::styled(
                format!("   selected: {}", sel + 1),
                Style::default().fg(theme.foreground),
            )));
        }

        lines.push(Line::from(""));

        // ── keyboard shortcuts ──────────────────────────────────
        lines.push(Line::from(Span::styled(
            " Keys",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        )));
        let shortcuts = [
            ("Ctrl+K", "commands"),
            ("Ctrl+S", "sidebar"),
            ("Esc", "history"),
            ("PgUp/Dn", "scroll"),
            ("Alt+S", "select mode"),
            ("@", "file picker"),
        ];
        for (key, desc) in shortcuts {
            lines.push(Line::from(vec![
                Span::styled(format!("   {key:<9}"), Style::default().fg(theme.accent)),
                Span::styled(desc, Style::default().fg(theme.muted)),
            ]));
        }

        let widget = Paragraph::new(lines)
            .block(
                Block::default()
                    .title(" Sidebar ")
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(theme.border)),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(widget, area);
    }
}
