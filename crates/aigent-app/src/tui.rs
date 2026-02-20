use aigent_config::AppConfig;
use anyhow::Result;
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Margin, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap},
};
use std::io;
use tui_textarea::TextArea;

pub enum Event {
    Tick,
    Key(KeyEvent),
    Paste(String),
    LlmChunk(String),
    LlmDone(Vec<String>),
    Error(String),
}

pub struct App {
    pub transcript: Vec<String>,
    pub textarea: TextArea<'static>,
    pub is_thinking: bool,
    pub is_streaming: bool,
    pub thinking_spinner_tick: usize,
    pub show_sidebar: bool,
    pub auto_follow: bool,
    pub viewport_start_line: usize,
    pub should_quit: bool,
}

impl App {
    pub fn new(initial_transcript: Vec<String>) -> Self {
        Self {
            transcript: initial_transcript,
            textarea: TextArea::default(),
            is_thinking: false,
            is_streaming: false,
            thinking_spinner_tick: 0,
            show_sidebar: false,
            auto_follow: true,
            viewport_start_line: 0,
            should_quit: false,
        }
    }

    pub fn handle_event(&mut self, event: Event) {
        match event {
            Event::Tick => {
                if self.is_thinking {
                    self.thinking_spinner_tick = self.thinking_spinner_tick.wrapping_add(1);
                }
            }
            Event::Key(key) => {
                if key.kind != KeyEventKind::Press {
                    return;
                }
                match key.code {
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        self.should_quit = true;
                    }
                    KeyCode::Char('s') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        self.show_sidebar = !self.show_sidebar;
                    }
                    KeyCode::Esc => {
                        self.should_quit = true;
                    }
                    _ => {
                        self.textarea.input(key);
                    }
                }
            }
            Event::Paste(text) => {
                for ch in text.chars() {
                    self.textarea.insert_char(ch);
                }
            }
            Event::LlmChunk(chunk) => {
                if !self.is_streaming {
                    self.transcript.push(format!("aigent> {chunk}"));
                    self.is_streaming = true;
                } else if let Some(last) = self.transcript.last_mut() {
                    last.push_str(&chunk);
                }

                if self.auto_follow {
                    self.viewport_start_line = usize::MAX;
                }
            }
            Event::LlmDone(messages) => {
                self.is_thinking = false;

                if self.is_streaming {
                    let _ = self.transcript.pop();
                    self.is_streaming = false;
                }

                for msg in messages {
                    self.transcript.push(msg);
                }

                if self.auto_follow {
                    self.viewport_start_line = usize::MAX;
                }
            }
            Event::Error(err) => {
                self.is_thinking = false;
                if self.is_streaming {
                    let _ = self.transcript.pop();
                    self.is_streaming = false;
                }
                self.transcript.push(format!("aigent> error: {err}"));
                if self.auto_follow {
                    self.viewport_start_line = usize::MAX;
                }
            }
        }
    }

    pub fn draw(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
        suggestions: &[&str],
        config: &AppConfig,
    ) -> Result<TranscriptViewport> {
        let size = terminal.size()?;
        let area = Rect::new(0, 0, size.width, size.height);

        let (main_area, sidebar_area) = if self.show_sidebar {
            let split = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Min(20), Constraint::Length(32)])
                .split(area);
            (split[0], Some(split[1]))
        } else {
            (area, None)
        };

        let input_lines = self.textarea.lines().len().max(1) as u16;
        let input_height = input_lines.saturating_add(2);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Min(6),
                Constraint::Length(input_height),
                Constraint::Length(1),
            ])
            .split(main_area);

        let content_width = usize::from(chunks[1].width.saturating_sub(3).max(1));
        let visual_lines = build_visual_lines(&self.transcript, content_width);
        let visible_lines = usize::from(chunks[1].height.saturating_sub(2).max(1));
        let (visible, viewport_start, max_scroll) = viewport_text(
            &visual_lines,
            visible_lines,
            self.viewport_start_line,
            self.auto_follow,
        );

        let spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let status = if self.is_thinking {
            format!("{} thinking...", spinner[self.thinking_spinner_tick % spinner.len()])
        } else {
            "idle".to_string()
        };

        let header = format!(
            "aigent • provider: {} • model: {} • status: {}",
            config.llm.provider,
            config.active_model(),
            status
        );

        let footer = if self.auto_follow || viewport_start >= max_scroll {
            "Ctrl+S: sidebar • PgUp/PgDn: scroll • End: follow • /help: commands".to_string()
        } else {
            format!(
                "Ctrl+S: sidebar • PgUp/PgDn: scroll • End: follow • viewing line {} of {}",
                viewport_start + 1,
                max_scroll + 1
            )
        };

        terminal.draw(|frame| {
            let header_widget = Paragraph::new(header.as_str()).style(Style::default().fg(Color::Gray));
            frame.render_widget(header_widget, chunks[0]);

            let transcript_lines = visible.iter().map(styled_transcript_line).collect::<Vec<_>>();
            let mut transcript_block = Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::DarkGray));

            if self.is_thinking {
                transcript_block = transcript_block.border_style(Style::default().fg(Color::Yellow));
            }

            let transcript_widget = Paragraph::new(transcript_lines)
                .block(transcript_block)
                .wrap(Wrap { trim: false });
            frame.render_widget(transcript_widget, chunks[1]);

            if max_scroll > 0 {
                let scrollbar = Scrollbar::default()
                    .orientation(ScrollbarOrientation::VerticalRight)
                    .begin_symbol(Some("▲"))
                    .end_symbol(Some("▼"));
                let mut state = ScrollbarState::new(max_scroll).position(viewport_start);
                frame.render_stateful_widget(
                    scrollbar,
                    chunks[1].inner(Margin {
                        vertical: 1,
                        horizontal: 0,
                    }),
                    &mut state,
                );
            }

            self.textarea.set_block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .title(" Input (Alt+Enter for newline) "),
            );
            self.textarea.set_style(Style::default().fg(Color::White));
            frame.render_widget(&self.textarea, chunks[2]);

            let hint_widget = Paragraph::new(footer).style(Style::default().fg(Color::DarkGray));
            frame.render_widget(hint_widget, chunks[3]);

            if let Some(sidebar) = sidebar_area {
                let sidebar_lines = vec![
                    Line::from(Span::styled("Status", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
                    Line::from(format!("Provider: {}", config.llm.provider)),
                    Line::from(format!("Model: {}", config.active_model())),
                    Line::from(format!("Thinking: {}", config.agent.thinking_level)),
                    Line::from(""),
                    Line::from(Span::styled("Suggestions", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
                ];

                let mut combined = sidebar_lines;
                if suggestions.is_empty() {
                    combined.push(Line::from("(none)"));
                } else {
                    combined.extend(suggestions.iter().map(|s| Line::from(format!("- {s}"))));
                }

                let sidebar_widget = Paragraph::new(combined)
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .border_type(BorderType::Rounded)
                            .title(" Sidebar "),
                    )
                    .wrap(Wrap { trim: false });
                frame.render_widget(sidebar_widget, sidebar);
            }
        })?;

        Ok(TranscriptViewport {
            start: viewport_start,
            max_scroll,
        })
    }
}

pub struct TranscriptViewport {
    pub start: usize,
    pub max_scroll: usize,
}

fn build_visual_lines(transcript: &[String], width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut out = Vec::new();

    for line in transcript {
        let wrapped = wrap_line(line, width);
        if wrapped.is_empty() {
            out.push(String::new());
        } else {
            out.extend(wrapped);
        }
    }

    out
}

fn wrap_line(input: &str, width: usize) -> Vec<String> {
    if input.is_empty() {
        return vec![String::new()];
    }

    let chars = input.chars().collect::<Vec<_>>();
    let mut lines = Vec::new();
    let mut start = 0;

    while start < chars.len() {
        let end = (start + width).min(chars.len());
        lines.push(chars[start..end].iter().collect::<String>());
        start = end;
    }

    lines
}

fn viewport_text(
    lines: &[String],
    visible_lines: usize,
    requested_start: usize,
    auto_follow: bool,
) -> (Vec<String>, usize, usize) {
    if lines.is_empty() {
        return (vec![String::new()], 0, 0);
    }

    let max_scroll = lines.len().saturating_sub(visible_lines);
    let start = if auto_follow {
        max_scroll
    } else {
        requested_start.min(max_scroll)
    };

    let end = (start + visible_lines).min(lines.len());
    let view = lines[start..end].to_vec();
    (view, start, max_scroll)
}

fn styled_transcript_line(line: &String) -> Line<'static> {
    if let Some(rest) = line.strip_prefix("you> ") {
        return Line::from(vec![
            Span::styled("you> ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::styled(rest.to_string(), Style::default().fg(Color::White)),
        ]);
    }

    if let Some(rest) = line.strip_prefix("aigent> ") {
        return Line::from(vec![
            Span::styled("aigent> ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled(rest.to_string(), Style::default().fg(Color::White)),
        ]);
    }

    Line::from(Span::styled(line.clone(), Style::default().fg(Color::Gray)))
}