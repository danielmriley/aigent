use ratatui::{
    Frame,
    layout::Rect,
    text::Line,
    widgets::{Block, BorderType, Borders, Paragraph, Wrap},
};

use crate::{app::AppState, theme::Theme};

pub fn draw_sidebar(frame: &mut Frame<'_>, area: Rect, state: &AppState, _theme: &Theme) {
    let mut lines = Vec::new();
    lines.push(Line::from("Sessions"));
    for (idx, session) in state.sessions.iter().enumerate() {
        lines.push(Line::from(format!(
            "{} {}",
            if idx == state.current_session {
                ">"
            } else {
                " "
            },
            session
        )));
    }
    lines.push(Line::from(""));
    lines.push(Line::from("Conversation"));
    lines.push(Line::from(format!("- messages: {}", state.messages.len())));
    lines.push(Line::from(format!(
        "- history mode: {}",
        if state.history_mode { "on" } else { "off" }
    )));
    lines.push(Line::from(format!(
        "- selected: {}",
        state
            .selected_message
            .map(|idx| idx + 1)
            .map(|v| v.to_string())
            .unwrap_or_else(|| "none".to_string())
    )));
    lines.push(Line::from(""));
    lines.push(Line::from("Keys"));
    lines.push(Line::from("- Ctrl+K: commands"));
    lines.push(Line::from("- Esc: history"));
    lines.push(Line::from("- PgUp/PgDn: chat scroll"));

    let widget = Paragraph::new(lines)
        .block(
            Block::default()
                .title(" Sidebar ")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded),
        )
        .wrap(Wrap { trim: true });

    frame.render_widget(widget, area);
}
