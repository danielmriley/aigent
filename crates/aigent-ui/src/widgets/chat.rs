use ratatui::{
    Frame,
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph, Wrap},
};

use crate::{app::Message, theme::Theme, widgets::markdown::render_markdown_lines};

pub fn draw_chat(
    frame: &mut Frame<'_>,
    area: Rect,
    messages: &[Message],
    theme: &Theme,
    selected: Option<usize>,
) {
    let mut lines = Vec::<Line<'static>>::new();

    for (idx, message) in messages.iter().enumerate() {
        let is_selected = selected == Some(idx);
        let mut prefix_style = Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD);
        let mut body_style = Style::default().fg(theme.foreground);
        if is_selected {
            body_style = body_style.bg(theme.muted);
            prefix_style = prefix_style.bg(theme.muted);
        }

        if message.role == "user" {
            lines.push(Line::from(vec![
                Span::styled("you> ", prefix_style),
                Span::styled(message.content.clone(), body_style),
            ]));
        } else {
            let rendered = render_markdown_lines(&message.content);
            if let Some(first) = rendered.first() {
                lines.push(Line::from(vec![
                    Span::styled("aigent> ", prefix_style),
                    Span::styled(first.to_string(), body_style),
                ]));
                for extra in rendered.iter().skip(1) {
                    lines.push(extra.clone());
                }
            }
        }
        lines.push(Line::from(""));
    }

    let widget = Paragraph::new(lines)
        .block(
            Block::default()
                .title(" Chat ")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded),
        )
        .wrap(Wrap { trim: false });

    frame.render_widget(widget, area);
}
