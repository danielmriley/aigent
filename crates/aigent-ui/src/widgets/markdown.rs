use ratatui::text::{Line, Span};
use termimad::crossterm::style::Color;

pub fn render_markdown_lines(input: &str) -> Vec<Line<'static>> {
    if input.trim().is_empty() {
        return vec![Line::from(Span::raw(""))];
    }

    input
        .lines()
        .map(|line| {
            if line.trim_start().starts_with("#") {
                Line::from(Span::styled(
                    line.to_string(),
                    ratatui::style::Style::default().fg(ratatui::style::Color::Cyan),
                ))
            } else if line.trim_start().starts_with("- ") {
                Line::from(vec![
                    Span::styled(
                        "• ",
                        ratatui::style::Style::default().fg(ratatui::style::Color::Yellow),
                    ),
                    Span::raw(line.trim_start_matches("- ").to_string()),
                ])
            } else {
                let _ = Color::Reset;
                Line::from(Span::raw(line.to_string()))
            }
        })
        .collect()
}
