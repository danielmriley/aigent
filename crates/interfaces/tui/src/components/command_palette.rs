//! Command palette overlay — slash-command picker.

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Clear, Paragraph};
use ratatui::Frame;

use crate::layout;
use crate::state::CommandPaletteState;
use crate::theme::Theme;

/// Command palette overlay component.
pub struct CommandPalette {
    pub state: CommandPaletteState,
}

impl Default for CommandPalette {
    fn default() -> Self {
        Self::new()
    }
}

impl CommandPalette {
    pub fn new() -> Self {
        Self {
            state: CommandPaletteState {
                visible: false,
                selected: 0,
                commands: vec![
                    "/new",
                    "/switch",
                    "/memory",
                    "/sleep",
                    "/dedup",
                    "/doctor",
                    "/status",
                    "/context",
                    "/tools",
                    "/model show",
                    "/model list",
                    "/exit",
                ],
            },
        }
    }

    pub fn draw(&self, frame: &mut Frame<'_>, theme: &Theme) {
        if !self.state.visible {
            return;
        }
        let area = layout::centered_popup(frame.area(), 50, 40);
        frame.render_widget(Clear, area);

        let mut lines = Vec::new();
        // Title line
        lines.push(Line::from(Span::styled(
            " Select a command:",
            Style::default().fg(theme.muted),
        )));
        lines.push(Line::from(""));

        for (idx, command) in self.state.commands.iter().enumerate() {
            if idx == self.state.selected {
                lines.push(Line::from(Span::styled(
                    format!("  ▸ {command}"),
                    Style::default()
                        .fg(theme.background)
                        .bg(theme.accent)
                        .add_modifier(Modifier::BOLD),
                )));
            } else {
                lines.push(Line::from(Span::styled(
                    format!("    {command}"),
                    Style::default().fg(theme.foreground),
                )));
            }
        }

        let widget = Paragraph::new(lines).block(
            Block::default()
                .title(" Commands ")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(theme.accent)),
        );
        frame.render_widget(widget, area);
    }
}
