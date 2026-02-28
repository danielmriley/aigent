//! Status bar — top header with bot name, spinner, and status text.

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::state::AppState;
use crate::theme::Theme;

const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Header / status bar component.
pub struct StatusBar {
    pub spinner_tick: usize,
}

impl Default for StatusBar {
    fn default() -> Self {
        Self::new()
    }
}

impl StatusBar {
    pub fn new() -> Self {
        Self { spinner_tick: 0 }
    }

    pub fn draw(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        state: &AppState,
        theme: &Theme,
        is_thinking: bool,
        is_sleeping: bool,
    ) {
        let status_display = if is_thinking || is_sleeping {
            let f = SPINNER_FRAMES[self.spinner_tick / 2 % SPINNER_FRAMES.len()];
            format!("{f} {}", state.status)
        } else {
            state.status.clone()
        };

        let name_span = Span::styled(
            format!(" {} ", state.bot_name),
            Style::default()
                .fg(theme.background)
                .bg(theme.accent)
                .add_modifier(Modifier::BOLD),
        );
        let sep = Span::styled(" │ ", Style::default().fg(theme.border));

        // ── model badge (if available) ──────────────────────────
        let mut spans = vec![name_span, sep.clone()];

        if let Some(ref model) = state.model_name {
            spans.push(Span::styled(
                format!(" {} ", model),
                Style::default()
                    .fg(theme.background)
                    .bg(theme.info),
            ));
            spans.push(sep.clone());
        }

        spans.push(Span::styled(status_display, Style::default().fg(theme.foreground)));

        // ── token counter ───────────────────────────────────────
        if let Some(total) = state.token_total {
            spans.push(Span::styled(
                format!(" │ tok:{}", total),
                Style::default().fg(theme.muted),
            ));
        }

        let tab_hint = match state.sidebar_tab {
            crate::state::SidebarTab::Sessions => "sessions",
            crate::state::SidebarTab::Context => "context",
        };
        spans.push(Span::styled(
            format!(" │ Ctrl+S sidebar │ Ctrl+Tab {} │ Esc history", tab_hint),
            Style::default().fg(theme.muted),
        ));

        let header = Paragraph::new(Line::from(spans));
        frame.render_widget(header, area);
    }
}
