use aigent_config::AppConfig;
use anyhow::Result;
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;
use once_cell::sync::Lazy;
use pulldown_cmark::{Event as MdEvent, Options as MdOptions, Parser, Tag, TagEnd};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Margin, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{
        Block, BorderType, Borders, Clear, Paragraph, Scrollbar, ScrollbarOrientation,
        ScrollbarState, Wrap,
    },
};
use std::io;
use syntect::easy::HighlightLines;
use syntect::highlighting::{Theme, ThemeSet};
use syntect::parsing::{SyntaxReference, SyntaxSet};
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
    pub chat_panel: ChatPanel,
    pub input_box: InputBox,
    pub sidebar: Sidebar,
    pub file_popup: FilePopup,
    pub should_quit: bool,
}

pub struct ChatPanel {
    pub transcript: Vec<String>,
    pub is_thinking: bool,
    pub is_streaming: bool,
    pub thinking_spinner_tick: usize,
    pub auto_follow: bool,
    pub viewport_start_line: usize,
    pub history_mode: bool,
    pub selected_message: Option<usize>,
}

pub struct InputBox {
    pub textarea: TextArea<'static>,
}

pub struct Sidebar {
    pub visible: bool,
}

pub struct FilePopup {
    pub all_files: Vec<String>,
    pub visible: bool,
    pub query: String,
    pub candidates: Vec<String>,
    pub selected: usize,
}

impl App {
    pub fn new(initial_transcript: Vec<String>, workspace_files: Vec<String>) -> Self {
        Self {
            chat_panel: ChatPanel {
                transcript: initial_transcript,
                is_thinking: false,
                is_streaming: false,
                thinking_spinner_tick: 0,
                auto_follow: true,
                viewport_start_line: 0,
                history_mode: false,
                selected_message: None,
            },
            input_box: InputBox {
                textarea: TextArea::default(),
            },
            sidebar: Sidebar { visible: false },
            file_popup: FilePopup::new(workspace_files),
            should_quit: false,
        }
    }

    pub fn refresh_file_popup(&mut self) {
        self.file_popup.refresh(&self.input_box.text());
    }

    pub fn handle_file_popup_key(&mut self, key: KeyEvent) -> bool {
        if !self.file_popup.visible {
            return false;
        }

        match key.code {
            KeyCode::Up => {
                self.file_popup.move_up();
                true
            }
            KeyCode::Down | KeyCode::Tab => {
                self.file_popup.move_down();
                true
            }
            KeyCode::Esc => {
                self.file_popup.visible = false;
                true
            }
            KeyCode::Enter => {
                if let Some(choice) = self.file_popup.current_selection() {
                    let updated = replace_last_at_token(&self.input_box.text(), &choice);
                    self.input_box.set_text(&updated);
                }
                self.file_popup.visible = false;
                true
            }
            _ => false,
        }
    }

    pub fn handle_event(&mut self, event: Event) {
        match event {
            Event::Tick => self.chat_panel.handle_event(Event::Tick),
            Event::Key(key) => {
                if key.kind != KeyEventKind::Press {
                    return;
                }
                match key.code {
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        self.should_quit = true;
                    }
                    KeyCode::Char('s') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        self.sidebar.handle_event(Event::Key(key));
                    }
                    KeyCode::Esc => {
                        self.should_quit = true;
                    }
                    _ => self.input_box.handle_event(Event::Key(key)),
                }
            }
            Event::Paste(text) => self.input_box.handle_event(Event::Paste(text)),
            Event::LlmChunk(chunk) => self.chat_panel.handle_event(Event::LlmChunk(chunk)),
            Event::LlmDone(messages) => self.chat_panel.handle_event(Event::LlmDone(messages)),
            Event::Error(err) => self.chat_panel.handle_event(Event::Error(err)),
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

        let (main_area, sidebar_area) = if self.sidebar.visible {
            let split = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Min(20), Constraint::Length(32)])
                .split(area);
            (split[0], Some(split[1]))
        } else {
            (area, None)
        };

        let input_lines = self.input_box.textarea.lines().len().max(1) as u16;
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

        let spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let status = if self.chat_panel.is_thinking {
            format!(
                "{} thinking...",
                spinner[self.chat_panel.thinking_spinner_tick % spinner.len()]
            )
        } else {
            "idle".to_string()
        };

        let header = format!(
            "aigent • provider: {} • model: {} • status: {}",
            config.llm.provider,
            config.active_model(),
            status
        );

        let mut viewport = TranscriptViewport {
            start: 0,
            max_scroll: 0,
        };

        terminal.draw(|frame| {
            let header_widget =
                Paragraph::new(header.as_str()).style(Style::default().fg(Color::Gray));
            frame.render_widget(header_widget, chunks[0]);

            viewport = self.chat_panel.draw(frame, chunks[1]);

            self.input_box.draw(frame, chunks[2]);

            let footer = if self.chat_panel.auto_follow || viewport.start >= viewport.max_scroll {
                "Ctrl+S: sidebar • PgUp/PgDn: scroll • End: follow • /help: commands".to_string()
            } else {
                format!(
                    "Ctrl+S: sidebar • PgUp/PgDn: scroll • End: follow • viewing line {} of {}",
                    viewport.start + 1,
                    viewport.max_scroll + 1
                )
            };

            let hint_widget = Paragraph::new(footer).style(Style::default().fg(Color::DarkGray));
            frame.render_widget(hint_widget, chunks[3]);

            if let Some(sidebar) = sidebar_area {
                self.sidebar.draw(frame, sidebar, suggestions, config);
            }

            if self.file_popup.visible {
                self.file_popup.draw(frame, area);
            }
        })?;

        Ok(viewport)
    }
}

impl ChatPanel {
    pub fn enter_history_mode(&mut self) {
        self.history_mode = true;
        if self.transcript.is_empty() {
            self.selected_message = None;
        } else if self.selected_message.is_none() {
            self.selected_message = Some(self.transcript.len().saturating_sub(1));
        }
    }

    pub fn exit_history_mode(&mut self) {
        self.history_mode = false;
        self.selected_message = None;
    }

    pub fn select_prev_message(&mut self) {
        let Some(current) = self.selected_message else {
            return;
        };
        self.selected_message = Some(current.saturating_sub(1));
    }

    pub fn select_next_message(&mut self) {
        let Some(current) = self.selected_message else {
            return;
        };
        if self.transcript.is_empty() {
            self.selected_message = None;
            return;
        }
        self.selected_message = Some((current + 1).min(self.transcript.len().saturating_sub(1)));
    }

    pub fn selected_transcript_entry(&self) -> Option<&str> {
        let idx = self.selected_message?;
        self.transcript.get(idx).map(String::as_str)
    }

    pub fn handle_event(&mut self, event: Event) {
        match event {
            Event::Tick => {
                if self.is_thinking {
                    self.thinking_spinner_tick = self.thinking_spinner_tick.wrapping_add(1);
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
            _ => {}
        }
    }

    pub fn draw(&mut self, frame: &mut Frame<'_>, area: Rect) -> TranscriptViewport {
        let content_width = usize::from(area.width.saturating_sub(3).max(1));
        let visual_lines = build_visual_lines(
            &self.transcript,
            content_width,
            self.history_mode,
            self.selected_message,
        );
        let visible_lines = usize::from(area.height.saturating_sub(2).max(1));
        let (visible, viewport_start, max_scroll) = viewport_text(
            &visual_lines,
            visible_lines,
            self.viewport_start_line,
            self.auto_follow,
        );

        let transcript_lines = visible
            .iter()
            .map(styled_transcript_line)
            .collect::<Vec<_>>();
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
        frame.render_widget(transcript_widget, area);

        if max_scroll > 0 {
            let scrollbar = Scrollbar::default()
                .orientation(ScrollbarOrientation::VerticalRight)
                .begin_symbol(Some("▲"))
                .end_symbol(Some("▼"));
            let mut state = ScrollbarState::new(max_scroll).position(viewport_start);
            frame.render_stateful_widget(
                scrollbar,
                area.inner(Margin {
                    vertical: 1,
                    horizontal: 0,
                }),
                &mut state,
            );
        }

        TranscriptViewport {
            start: viewport_start,
            max_scroll,
        }
    }
}

impl InputBox {
    pub fn handle_event(&mut self, event: Event) {
        match event {
            Event::Key(key) => {
                if key.kind == KeyEventKind::Press {
                    self.textarea.input(key);
                }
            }
            Event::Paste(text) => {
                for ch in text.chars() {
                    self.textarea.insert_char(ch);
                }
            }
            _ => {}
        }
    }

    pub fn draw(&mut self, frame: &mut Frame<'_>, area: Rect) {
        self.textarea.set_block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Input (Alt+Enter for newline) "),
        );
        self.textarea.set_style(Style::default().fg(Color::White));
        frame.render_widget(&self.textarea, area);
    }

    pub fn text(&self) -> String {
        self.textarea.lines().join("\n")
    }

    pub fn set_text(&mut self, text: &str) {
        self.textarea = TextArea::default();
        self.textarea.insert_str(text);
    }
}

impl Sidebar {
    pub fn handle_event(&mut self, event: Event) {
        if let Event::Key(key) = event {
            if key.kind == KeyEventKind::Press
                && key.code == KeyCode::Char('s')
                && key.modifiers.contains(KeyModifiers::CONTROL)
            {
                self.visible = !self.visible;
            }
        }
    }

    pub fn draw(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        suggestions: &[&str],
        config: &AppConfig,
    ) {
        let sidebar_lines = vec![
            Line::from(Span::styled(
                "Status",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(format!("Provider: {}", config.llm.provider)),
            Line::from(format!("Model: {}", config.active_model())),
            Line::from(format!("Thinking: {}", config.agent.thinking_level)),
            Line::from(""),
            Line::from(Span::styled(
                "Suggestions",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
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
        frame.render_widget(sidebar_widget, area);
    }
}

impl FilePopup {
    pub fn new(all_files: Vec<String>) -> Self {
        Self {
            all_files,
            visible: false,
            query: String::new(),
            candidates: Vec::new(),
            selected: 0,
        }
    }

    pub fn refresh(&mut self, input: &str) {
        let Some(query) = active_at_query(input) else {
            self.visible = false;
            self.query.clear();
            self.candidates.clear();
            self.selected = 0;
            return;
        };

        self.visible = true;
        self.query = query.to_string();

        if self.query.is_empty() {
            self.candidates = self.all_files.iter().take(8).cloned().collect();
            self.selected = 0.min(self.candidates.len().saturating_sub(1));
            return;
        }

        let matcher = SkimMatcherV2::default();
        let mut scored = self
            .all_files
            .iter()
            .filter_map(|path| {
                matcher
                    .fuzzy_match(path, &self.query)
                    .map(|score| (score, path.clone()))
            })
            .collect::<Vec<_>>();
        scored.sort_by(|a, b| b.0.cmp(&a.0));
        self.candidates = scored.into_iter().take(8).map(|(_, path)| path).collect();
        self.selected = 0.min(self.candidates.len().saturating_sub(1));
    }

    pub fn move_up(&mut self) {
        if self.candidates.is_empty() {
            return;
        }
        if self.selected == 0 {
            self.selected = self.candidates.len().saturating_sub(1);
        } else {
            self.selected = self.selected.saturating_sub(1);
        }
    }

    pub fn move_down(&mut self) {
        if self.candidates.is_empty() {
            return;
        }
        self.selected = (self.selected + 1) % self.candidates.len();
    }

    pub fn current_selection(&self) -> Option<String> {
        self.candidates.get(self.selected).cloned()
    }

    pub fn draw(&self, frame: &mut Frame<'_>, screen: Rect) {
        let popup = centered_rect(70, 45, screen);
        frame.render_widget(Clear, popup);

        let mut lines = vec![Line::from(Span::styled(
            format!("query: @{}", self.query),
            Style::default().fg(Color::DarkGray),
        ))];
        lines.push(Line::from(""));

        if self.candidates.is_empty() {
            lines.push(Line::from(Span::styled(
                "(no matches)",
                Style::default().fg(Color::DarkGray),
            )));
        } else {
            for (idx, path) in self.candidates.iter().enumerate() {
                let style = if idx == self.selected {
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Cyan)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::White)
                };
                lines.push(Line::from(Span::styled(path.clone(), style)));
            }
        }

        let widget = Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .title(" File Context (@) "),
            )
            .wrap(Wrap { trim: false });
        frame.render_widget(widget, popup);
    }
}

pub struct TranscriptViewport {
    pub start: usize,
    pub max_scroll: usize,
}

#[derive(Clone)]
struct VisualLine {
    spans: Vec<Span<'static>>,
    selected: bool,
}

static SYNTAX_SET: Lazy<SyntaxSet> = Lazy::new(SyntaxSet::load_defaults_newlines);
static THEME: Lazy<Theme> = Lazy::new(|| {
    let themes = ThemeSet::load_defaults();
    themes
        .themes
        .get("base16-ocean.dark")
        .cloned()
        .unwrap_or_default()
});

fn build_visual_lines(
    transcript: &[String],
    width: usize,
    history_mode: bool,
    selected_message: Option<usize>,
) -> Vec<VisualLine> {
    let _ = width;
    let mut out = Vec::new();

    for (idx, line) in transcript.iter().enumerate() {
        let selected = history_mode && selected_message == Some(idx);
        if let Some(user) = line.strip_prefix("you> ") {
            out.push(VisualLine {
                spans: vec![
                    Span::styled(
                        "you> ",
                        Style::default()
                            .fg(Color::Green)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(user.to_string(), Style::default().fg(Color::White)),
                ],
                selected,
            });
            continue;
        }

        if let Some(assistant) = line.strip_prefix("aigent> ") {
            let mut rendered = render_markdown_lines(assistant);
            if let Some(first) = rendered.first_mut() {
                first.spans.insert(
                    0,
                    Span::styled(
                        "aigent> ",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                );
            }
            rendered
                .iter_mut()
                .for_each(|line| line.selected = selected);
            out.extend(rendered);
            continue;
        }

        out.push(VisualLine {
            spans: vec![Span::styled(line.clone(), Style::default().fg(Color::Gray))],
            selected,
        });
    }

    out
}

fn render_markdown_lines(text: &str) -> Vec<VisualLine> {
    let mut out = Vec::new();
    let mut in_code = false;
    let mut highlighter: Option<HighlightLines<'static>> = None;

    for raw in text.lines() {
        let trimmed = raw.trim_start();
        if trimmed.starts_with("```") {
            if in_code {
                in_code = false;
                highlighter = None;
            } else {
                in_code = true;
                let lang = trimmed.trim_start_matches("```").trim();
                let syntax = syntax_for_lang(lang);
                highlighter = Some(HighlightLines::new(syntax, &THEME));
            }
            out.push(VisualLine {
                spans: vec![Span::styled(
                    raw.to_string(),
                    Style::default().fg(Color::DarkGray),
                )],
                selected: false,
            });
            continue;
        }

        if in_code {
            if let Some(h) = highlighter.as_mut() {
                let ranges = h.highlight_line(raw, &SYNTAX_SET).unwrap_or_default();
                let spans = ranges
                    .into_iter()
                    .map(|(style, segment)| {
                        Span::styled(
                            segment.to_string(),
                            Style::default().fg(Color::Rgb(
                                style.foreground.r,
                                style.foreground.g,
                                style.foreground.b,
                            )),
                        )
                    })
                    .collect::<Vec<_>>();
                out.push(VisualLine {
                    spans,
                    selected: false,
                });
            } else {
                out.push(VisualLine {
                    spans: vec![Span::styled(
                        raw.to_string(),
                        Style::default().fg(Color::Yellow),
                    )],
                    selected: false,
                });
            }
            continue;
        }

        out.push(VisualLine {
            spans: markdown_spans(raw),
            selected: false,
        });
    }

    if out.is_empty() {
        out.push(VisualLine {
            spans: vec![Span::raw("")],
            selected: false,
        });
    }

    out
}

fn syntax_for_lang(lang: &str) -> &'static SyntaxReference {
    if lang.is_empty() {
        return SYNTAX_SET.find_syntax_plain_text();
    }

    SYNTAX_SET
        .find_syntax_by_token(lang)
        .unwrap_or_else(|| SYNTAX_SET.find_syntax_plain_text())
}

fn markdown_spans(line: &str) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let options =
        MdOptions::ENABLE_STRIKETHROUGH | MdOptions::ENABLE_TABLES | MdOptions::ENABLE_TASKLISTS;
    let parser = Parser::new_ext(line, options);

    let mut strong = 0usize;
    let mut emphasis = 0usize;
    let mut heading = false;

    for event in parser {
        match event {
            MdEvent::Start(tag) => match tag {
                Tag::Strong => strong += 1,
                Tag::Emphasis => emphasis += 1,
                Tag::Heading { .. } => heading = true,
                _ => {}
            },
            MdEvent::End(tag_end) => match tag_end {
                TagEnd::Strong => strong = strong.saturating_sub(1),
                TagEnd::Emphasis => emphasis = emphasis.saturating_sub(1),
                TagEnd::Heading { .. } => heading = false,
                _ => {}
            },
            MdEvent::Code(code) => {
                spans.push(Span::styled(
                    code.to_string(),
                    Style::default()
                        .fg(Color::Yellow)
                        .bg(Color::Black)
                        .add_modifier(Modifier::BOLD),
                ));
            }
            MdEvent::Text(text) => {
                let mut style = Style::default().fg(Color::White);
                if heading {
                    style = style.fg(Color::Cyan).add_modifier(Modifier::BOLD);
                }
                if strong > 0 {
                    style = style.add_modifier(Modifier::BOLD);
                }
                if emphasis > 0 {
                    style = style.add_modifier(Modifier::ITALIC);
                }
                spans.push(Span::styled(text.to_string(), style));
            }
            MdEvent::SoftBreak | MdEvent::HardBreak => spans.push(Span::raw(" ")),
            _ => {}
        }
    }

    if spans.is_empty() {
        spans.push(Span::raw(line.to_string()));
    }

    spans
}

fn viewport_text(
    lines: &[VisualLine],
    visible_lines: usize,
    requested_start: usize,
    auto_follow: bool,
) -> (Vec<VisualLine>, usize, usize) {
    if lines.is_empty() {
        return (
            vec![VisualLine {
                spans: vec![Span::raw("")],
                selected: false,
            }],
            0,
            0,
        );
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

fn styled_transcript_line(line: &VisualLine) -> Line<'static> {
    if line.selected {
        let spans = line
            .spans
            .iter()
            .cloned()
            .map(|span| {
                let styled = span.style;
                span.style(
                    styled
                        .bg(Color::DarkGray)
                        .fg(styled.fg.unwrap_or(Color::White)),
                )
            })
            .collect::<Vec<_>>();
        return Line::from(spans);
    }

    Line::from(line.spans.clone())
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

fn active_at_query(input: &str) -> Option<&str> {
    let token = input.split_whitespace().last()?;
    if !token.starts_with('@') {
        return None;
    }
    Some(token.trim_start_matches('@'))
}

fn replace_last_at_token(input: &str, replacement: &str) -> String {
    let end = input.len();
    let start = input
        .char_indices()
        .rev()
        .find(|(_, ch)| ch.is_whitespace())
        .map(|(idx, ch)| idx + ch.len_utf8())
        .unwrap_or(0);

    if !input[start..end].starts_with('@') {
        return input.to_string();
    }

    let mut out = input.to_string();
    out.replace_range(start..end, &format!("@{} ", replacement));
    out
}

pub fn extract_first_code_block(message: &str) -> Option<String> {
    let mut in_code = false;
    let mut buf = Vec::new();

    for line in message.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            if in_code {
                break;
            }
            in_code = true;
            continue;
        }

        if in_code {
            buf.push(line);
        }
    }

    if buf.is_empty() {
        None
    } else {
        Some(buf.join("\n"))
    }
}
