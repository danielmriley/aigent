//! Right-side agent status panel.
//!
//! Displays a compact overview of the running agent's state:
//! ReAct phase, current round, sub-agent status, and resource usage.
//! This panel is rendered in the third column of the advanced 3-column layout.

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::state::AppState;
use crate::theme::Theme;

/// Compact agent-status widget for the right panel.
pub struct AgentPanel;

impl AgentPanel {
    pub fn draw(frame: &mut Frame<'_>, area: Rect, state: &AppState, theme: &Theme) {
        if area.width == 0 {
            return;
        }

        let block = Block::default()
            .title(" Agent ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme.border));

        let phase = state
            .react_phase
            .as_deref()
            .unwrap_or("idle");
        let round = state.react_round.unwrap_or(0);
        let max_rounds = state.react_max_rounds.unwrap_or(0);
        let tokens = state.token_total.unwrap_or(0);
        let tool_count = state.tool_history.len();

        let text = vec![
            Line::from(Span::styled(
                format!("Phase: {phase}"),
                Style::default().fg(theme.accent).add_modifier(Modifier::BOLD),
            )),
            Line::from(format!("Round: {round}/{max_rounds}")),
            Line::from(""),
            Line::from(Span::styled("Resources", Style::default().add_modifier(Modifier::BOLD))),
            Line::from(format!("Tokens: {tokens}")),
            Line::from(format!("Tools:  {tool_count}")),
        ];

        let paragraph = Paragraph::new(text)
            .block(block)
            .wrap(ratatui::widgets::Wrap { trim: true });

        frame.render_widget(paragraph, area);
    }
}
