use ratatui::{
    Frame,
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph, Wrap},
};

use crate::{app::Message, theme::Theme};

pub fn draw_chat(
    frame: &mut Frame<'_>,
    area: Rect,
    messages: &[Message],
    theme: &Theme,
    selected: Option<usize>,
    assistant_label: &str,
    scroll: usize,
) {
    let lines = build_chat_lines(messages, theme, selected, assistant_label);

    let widget = Paragraph::new(lines)
        .block(
            Block::default()
                .title(" Chat ")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded),
        )
        .scroll((scroll as u16, 0))
        .wrap(Wrap { trim: false });

    frame.render_widget(widget, area);
}

pub fn chat_visual_line_count(
    messages: &[Message],
    theme: &Theme,
    selected: Option<usize>,
    assistant_label: &str,
    content_width: u16,
) -> usize {
    let width = usize::from(content_width).max(1);
    build_chat_lines(messages, theme, selected, assistant_label)
        .into_iter()
        .map(|line| {
            let visible = line.to_string().chars().count().max(1);
            visible.div_ceil(width)
        })
        .sum()
}

pub fn message_visual_line_start(
    messages: &[Message],
    theme: &Theme,
    assistant_label: &str,
    selected_message: usize,
    content_width: u16,
) -> usize {
    let width = usize::from(content_width).max(1);
    let (lines, starts) = build_chat_lines_with_starts(messages, theme, None, assistant_label);
    let Some(logical_start) = starts.get(selected_message).copied() else {
        return 0;
    };

    lines[..logical_start]
        .iter()
        .map(|line| {
            let visible = line.to_string().chars().count().max(1);
            visible.div_ceil(width)
        })
        .sum()
}

fn build_chat_lines(
    messages: &[Message],
    theme: &Theme,
    selected: Option<usize>,
    assistant_label: &str,
) -> Vec<Line<'static>> {
    build_chat_lines_with_starts(messages, theme, selected, assistant_label).0
}

fn build_chat_lines_with_starts(
    messages: &[Message],
    theme: &Theme,
    selected: Option<usize>,
    assistant_label: &str,
) -> (Vec<Line<'static>>, Vec<usize>) {
    let mut lines = Vec::<Line<'static>>::new();
    let mut starts = Vec::<usize>::with_capacity(messages.len());

    for (idx, message) in messages.iter().enumerate() {
        starts.push(lines.len());
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
        } else if let Some(streaming) = message.content.strip_prefix("[stream]") {
            lines.push(Line::from(vec![
                Span::styled(format!("{}> ", assistant_label), prefix_style),
                Span::styled(streaming.to_string(), body_style),
            ]));
        } else if message.role == "⚙" {
            // Tool-call messages get a dedicated dimmed style with an
            // expandable detail section when selected in history mode.
            let tool_style = Style::default().fg(theme.muted);
            let tool_prefix = Style::default()
                .fg(theme.muted)
                .add_modifier(Modifier::DIM);
            // The content may contain a hidden detail block after \0.
            let (summary, detail) = message.content.split_once('\0')
                .map(|(s, d)| (s, Some(d)))
                .unwrap_or((&message.content, None));
            let toggle_hint = if detail.is_some() {
                if is_selected { " ▾" } else { " ▸" }
            } else {
                ""
            };
            lines.push(Line::from(vec![
                Span::styled("  ", tool_prefix),
                Span::styled(format!("{}{}", summary, toggle_hint), tool_style),
            ]));
            // When this message is selected in history mode, show the
            // full tool output underneath as indented detail lines.
            if is_selected {
                if let Some(detail_text) = detail {
                    let detail_style = Style::default().fg(theme.muted).add_modifier(Modifier::DIM);
                    for detail_line in detail_text.lines() {
                        lines.push(Line::from(Span::styled(
                            format!("    {detail_line}"),
                            detail_style,
                        )));
                    }
                }
            }
        } else if message.role == "system" {
            // System / progress messages (e.g. sleep cycle stages).
            let sys_style = Style::default()
                .fg(theme.muted)
                .add_modifier(Modifier::ITALIC);
            lines.push(Line::from(Span::styled(
                format!("  {}", message.content),
                sys_style,
            )));
        } else if message.role != "assistant" {
            // External source (e.g. telegram) — show role as prefix
            let mut ext_prefix = Style::default()
                .fg(theme.muted)
                .add_modifier(Modifier::BOLD);
            if is_selected {
                ext_prefix = ext_prefix.bg(theme.muted);
            }
            lines.push(Line::from(vec![
                Span::styled(format!("{}> ", message.role), ext_prefix),
                Span::styled(message.content.clone(), body_style),
            ]));
        } else {
            let rendered: &[Line<'static>] = message.rendered_md.as_deref().unwrap_or(&[]);
            if let Some(first) = rendered.first() {
                lines.push(Line::from(vec![
                    Span::styled(format!("{}> ", assistant_label), prefix_style),
                    Span::styled(first.to_string(), body_style),
                ]));
                for extra in rendered.iter().skip(1) {
                    lines.push(extra.clone());
                }
            }
        }
        lines.push(Line::from(""));
    }

    (lines, starts)
}
