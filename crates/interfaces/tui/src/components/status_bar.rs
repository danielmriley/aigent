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
        let status_span = Span::styled(status_display, Style::default().fg(theme.foreground));
        let hints = Span::styled(
            " │ Ctrl+S sidebar │ Esc history",
            Style::default().fg(theme.muted),
        );

        let header = Paragraph::new(Line::from(vec![name_span, sep, status_span, hints]));
        frame.render_widget(header, area);
    }
}
