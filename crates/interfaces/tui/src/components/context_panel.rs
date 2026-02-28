//! Context panel — ReAct state, tool history, and memory peek.
//!
//! This component replaces the sidebar when the user switches to the
//! "Context" tab.  It provides at-a-glance visibility into the agent's
//! internal state: current ReAct phase, recent tool calls, and memory
//! items.

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Wrap};
use ratatui::Frame;

use crate::state::AppState;
use crate::theme::Theme;

/// Context panel component.
pub struct ContextPanel;

impl ContextPanel {
    pub fn new() -> Self {
        Self
    }

    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, state: &AppState, theme: &Theme) {
        let mut lines: Vec<Line<'static>> = Vec::new();

        // ── ReAct phase ─────────────────────────────────────────
        lines.push(Line::from(Span::styled(
            " ReAct",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        )));

        let phase_display = state.react_phase.as_deref().unwrap_or("idle");
        let phase_color = match phase_display {
            "think"    => theme.accent,
            "act"      => theme.warning,
            "observe"  => theme.info,
            "critique" => theme.success,
            "done"     => theme.muted,
            _          => theme.foreground,
        };
        lines.push(Line::from(vec![
            Span::styled("   phase: ", Style::default().fg(theme.foreground)),
            Span::styled(phase_display.to_string(), Style::default().fg(phase_color).add_modifier(Modifier::BOLD)),
        ]));

        if let Some(round) = state.react_round {
            let max = state.react_max_rounds.unwrap_or(0);
            lines.push(Line::from(Span::styled(
                format!("   round: {}/{}", round, max),
                Style::default().fg(theme.foreground),
            )));
        }

        if let Some(ref role) = state.swarm_role {
            lines.push(Line::from(Span::styled(
                format!("   role:  {}", role),
                Style::default().fg(theme.accent),
            )));
        }

        lines.push(Line::from(""));

        // ── Token usage ─────────────────────────────────────────
        lines.push(Line::from(Span::styled(
            " Tokens",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(Span::styled(
            format!("   prompt:  {}", state.token_prompt.unwrap_or(0)),
            Style::default().fg(theme.foreground),
        )));
        lines.push(Line::from(Span::styled(
            format!("   response: {}", state.token_response.unwrap_or(0)),
            Style::default().fg(theme.foreground),
        )));
        if let Some(total) = state.token_total {
            lines.push(Line::from(Span::styled(
                format!("   total:   {}", total),
                Style::default().fg(theme.foreground),
            )));
        }

        lines.push(Line::from(""));

        // ── Recent tool calls ───────────────────────────────────
        lines.push(Line::from(Span::styled(
            " Tool History",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        )));
        if state.tool_history.is_empty() {
            lines.push(Line::from(Span::styled(
                "   (none)",
                Style::default().fg(theme.muted),
            )));
        } else {
            for entry in state.tool_history.iter().rev().take(10) {
                let icon = if entry.success { "✓" } else { "✗" };
                let icon_color = if entry.success { theme.success } else { theme.error };
                lines.push(Line::from(vec![
                    Span::styled(format!("   {} ", icon), Style::default().fg(icon_color)),
                    Span::styled(entry.name.clone(), Style::default().fg(theme.foreground)),
                ]));
            }
        }

        lines.push(Line::from(""));

        // ── Memory peek ─────────────────────────────────────────
        lines.push(Line::from(Span::styled(
            " Memory",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        )));
        if state.memory_peek.is_empty() {
            lines.push(Line::from(Span::styled(
                "   (empty)",
                Style::default().fg(theme.muted),
            )));
        } else {
            for item in state.memory_peek.iter().take(8) {
                let truncated = if item.len() > 40 {
                    format!("{}…", &item[..39])
                } else {
                    item.clone()
                };
                lines.push(Line::from(Span::styled(
                    format!("   • {}", truncated),
                    Style::default().fg(theme.foreground),
                )));
            }
        }

        let widget = Paragraph::new(lines)
            .block(
                Block::default()
                    .title(" Context ")
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(theme.border)),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(widget, area);
    }
}

impl Default for ContextPanel {
    fn default() -> Self {
        Self::new()
    }
}
