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
    lines.push(Line::from("Memory Peek"));
    for item in state.memory_peek.iter().take(3) {
        lines.push(Line::from(format!("- {item}")));
    }

    let widget = Paragraph::new(lines)
        .block(
            Block::default()
                .title(" Sidebar ")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded),
        )
        .wrap(Wrap { trim: false });

    frame.render_widget(widget, area);
}
