use arboard::Clipboard;
use chrono::Local;
use crossterm::event::{KeyCode, KeyModifiers};
use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;
use ignore::WalkBuilder;
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, Paragraph, Wrap},
};
use std::fs;
use std::path::Path;
use tokio::sync::mpsc;
use tracing::debug;
use tui_textarea::{Input, Key, TextArea};

use aigent_config::AppConfig;
use aigent_daemon::BackendEvent;

use crate::{
    events::AppEvent,
    theme::Theme,
    widgets::{chat::draw_chat, input::draw_input, sidebar::draw_sidebar},
};

#[derive(Debug, Clone)]
pub enum UiCommand {
    Quit,
    Submit(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Focus {
    Sidebar,
    Chat,
    Input,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct AppState {
    pub sessions: Vec<String>,
    pub current_session: usize,
    pub messages: Vec<Message>,
    pub status: String,
    pub memory_peek: Vec<String>,
    pub selected_message: Option<usize>,
    pub history_mode: bool,
}

#[derive(Debug, Clone)]
pub struct FilePopup {
    pub visible: bool,
    pub query: String,
    pub candidates: Vec<String>,
    pub selected: usize,
}

#[derive(Debug, Clone)]
pub struct CommandPalette {
    pub visible: bool,
    pub selected: usize,
    pub commands: Vec<&'static str>,
}

pub struct App {
    pub state: AppState,
    pub textarea: TextArea<'static>,
    pub backend_rx: mpsc::UnboundedReceiver<BackendEvent>,
    pub focus: Focus,
    pub theme: Theme,
    pub scroll: usize,
    pub show_sidebar: bool,
    pub spinner_tick: usize,
    pub pending_stream: String,
    pub workspace_files: Vec<String>,
    pub file_popup: FilePopup,
    pub command_palette: CommandPalette,
}

impl App {
    pub fn new(backend_rx: mpsc::UnboundedReceiver<BackendEvent>, config: &AppConfig) -> Self {
        let mut textarea = TextArea::default();
        textarea.set_cursor_line_style(Default::default());

        Self {
            state: AppState {
                sessions: vec!["default".to_string()],
                current_session: 0,
                messages: Vec::new(),
                status: format!(
                    "model={} provider={}",
                    config.active_model(),
                    config.llm.provider
                ),
                memory_peek: Vec::new(),
                selected_message: None,
                history_mode: false,
            },
            textarea,
            backend_rx,
            focus: Focus::Input,
            theme: Theme::default(),
            scroll: 0,
            show_sidebar: true,
            spinner_tick: 0,
            pending_stream: String::new(),
            workspace_files: collect_workspace_files(Path::new(".")),
            file_popup: FilePopup {
                visible: false,
                query: String::new(),
                candidates: Vec::new(),
                selected: 0,
            },
            command_palette: CommandPalette {
                visible: false,
                selected: 0,
                commands: vec![
                    "/new",
                    "/switch",
                    "/memory",
                    "/doctor",
                    "/status",
                    "/context",
                    "/model show",
                    "/model list",
                    "/exit",
                ],
            },
        }
    }

    pub fn input_text(&self) -> String {
        self.textarea.lines().join("\n")
    }

    pub fn clear_input(&mut self) {
        self.textarea = TextArea::default();
    }

    pub fn push_user_message(&mut self, text: String) {
        self.state.messages.push(Message {
            role: "user".to_string(),
            content: text,
        });
    }

    pub fn push_assistant_message(&mut self, text: String) {
        self.state.messages.push(Message {
            role: "assistant".to_string(),
            content: text,
        });
    }

    pub fn set_status(&mut self, status: impl Into<String>) {
        self.state.status = status.into();
    }

    pub fn update(&mut self, event: AppEvent) -> Option<UiCommand> {
        match event {
            AppEvent::Tick => {
                self.spinner_tick = self.spinner_tick.wrapping_add(1);
                self.refresh_file_popup();
                None
            }
            AppEvent::Resize(_, _) => None,
            AppEvent::Backend(be) => {
                self.apply_backend(be);
                None
            }
            AppEvent::Key(key) => {
                debug!(?key, "ui key event");

                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
                    return Some(UiCommand::Quit);
                }
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('s') {
                    self.show_sidebar = !self.show_sidebar;
                    return None;
                }
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('k') {
                    self.command_palette.visible = !self.command_palette.visible;
                    return None;
                }

                if self.command_palette.visible {
                    match key.code {
                        KeyCode::Esc => self.command_palette.visible = false,
                        KeyCode::Up => {
                            self.command_palette.selected =
                                self.command_palette.selected.saturating_sub(1);
                        }
                        KeyCode::Down | KeyCode::Tab => {
                            self.command_palette.selected = (self.command_palette.selected + 1)
                                .min(self.command_palette.commands.len().saturating_sub(1));
                        }
                        KeyCode::Enter => {
                            if let Some(cmd) = self
                                .command_palette
                                .commands
                                .get(self.command_palette.selected)
                            {
                                self.textarea = TextArea::default();
                                self.textarea.insert_str(cmd);
                                self.command_palette.visible = false;
                            }
                        }
                        _ => {}
                    }
                    return None;
                }

                if self.file_popup.visible {
                    match key.code {
                        KeyCode::Esc => {
                            self.file_popup.visible = false;
                            return None;
                        }
                        KeyCode::Up => {
                            self.file_popup.selected = self.file_popup.selected.saturating_sub(1);
                            return None;
                        }
                        KeyCode::Down | KeyCode::Tab => {
                            self.file_popup.selected = (self.file_popup.selected + 1)
                                .min(self.file_popup.candidates.len().saturating_sub(1));
                            return None;
                        }
                        KeyCode::Enter => {
                            if let Some(path) =
                                self.file_popup.candidates.get(self.file_popup.selected)
                            {
                                let updated = replace_last_at_token(&self.input_text(), path);
                                self.clear_input();
                                self.textarea.insert_str(&updated);
                                self.file_popup.visible = false;
                            }
                            return None;
                        }
                        _ => {}
                    }
                }

                if self.state.history_mode {
                    match key.code {
                        KeyCode::Esc => {
                            self.state.history_mode = false;
                            self.state.selected_message = None;
                        }
                        KeyCode::Up => {
                            let current = self
                                .state
                                .selected_message
                                .unwrap_or_else(|| self.state.messages.len().saturating_sub(1));
                            self.state.selected_message = Some(current.saturating_sub(1));
                        }
                        KeyCode::Down => {
                            let current = self.state.selected_message.unwrap_or(0);
                            self.state.selected_message = Some(
                                (current + 1).min(self.state.messages.len().saturating_sub(1)),
                            );
                        }
                        KeyCode::Char('c') => {
                            if let Some(code) = self.selected_code_block() {
                                match Clipboard::new().and_then(|mut cb| cb.set_text(code)) {
                                    Ok(_) => self.push_assistant_message(
                                        "copied code block to clipboard".to_string(),
                                    ),
                                    Err(err) => self
                                        .push_assistant_message(format!("clipboard error: {err}")),
                                }
                            }
                        }
                        KeyCode::Char('a') => {
                            if let Some(code) = self.selected_code_block() {
                                let dir = Path::new(".aigent").join("snippets");
                                if let Err(err) = fs::create_dir_all(&dir) {
                                    self.push_assistant_message(format!("apply error: {err}"));
                                } else {
                                    let file_name = format!(
                                        "applied_{}.txt",
                                        Local::now().format("%Y%m%d_%H%M%S")
                                    );
                                    let path = dir.join(file_name);
                                    match fs::write(&path, code) {
                                        Ok(_) => self.push_assistant_message(format!(
                                            "applied code block to {}",
                                            path.display()
                                        )),
                                        Err(err) => self
                                            .push_assistant_message(format!("apply error: {err}")),
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                    return None;
                }

                match key.code {
                    KeyCode::Esc => {
                        self.state.history_mode = true;
                        self.state.selected_message =
                            Some(self.state.messages.len().saturating_sub(1));
                        None
                    }
                    KeyCode::Enter if !key.modifiers.contains(KeyModifiers::ALT) => {
                        let text = self.input_text().trim().to_string();
                        if text.is_empty() {
                            return None;
                        }
                        if text == "/exit" {
                            self.push_assistant_message("session closed".to_string());
                            self.clear_input();
                            return Some(UiCommand::Quit);
                        }
                        self.push_user_message(text.clone());
                        self.clear_input();
                        self.state.status = "thinking...".to_string();
                        Some(UiCommand::Submit(text))
                    }
                    _ => {
                        self.textarea.input(Input {
                            key: match key.code {
                                KeyCode::Char(c) => Key::Char(c),
                                KeyCode::Enter => Key::Enter,
                                KeyCode::Backspace => Key::Backspace,
                                KeyCode::Delete => Key::Delete,
                                KeyCode::Left => Key::Left,
                                KeyCode::Right => Key::Right,
                                KeyCode::Up => Key::Up,
                                KeyCode::Down => Key::Down,
                                KeyCode::Tab => Key::Tab,
                                KeyCode::Home => Key::Home,
                                KeyCode::End => Key::End,
                                KeyCode::PageUp => Key::PageUp,
                                KeyCode::PageDown => Key::PageDown,
                                _ => return None,
                            },
                            ctrl: key.modifiers.contains(KeyModifiers::CONTROL),
                            alt: key.modifiers.contains(KeyModifiers::ALT),
                            shift: key.modifiers.contains(KeyModifiers::SHIFT),
                        });
                        self.refresh_file_popup();
                        None
                    }
                }
            }
        }
    }

    pub fn selected_code_block(&self) -> Option<String> {
        let idx = self.state.selected_message?;
        let msg = self.state.messages.get(idx)?;
        if msg.role != "assistant" {
            return None;
        }
        extract_first_code_block(&msg.content)
    }

    fn apply_backend(&mut self, event: BackendEvent) {
        match event {
            BackendEvent::Token(chunk) => {
                self.pending_stream.push_str(&chunk);
                if let Some(last) = self.state.messages.last_mut() {
                    if last.role == "assistant" && last.content.starts_with("[stream]") {
                        last.content = format!("[stream]{}", self.pending_stream);
                        return;
                    }
                }
                self.state.messages.push(Message {
                    role: "assistant".to_string(),
                    content: format!("[stream]{}", self.pending_stream),
                });
            }
            BackendEvent::Thinking => {
                self.state.status = "thinking...".to_string();
            }
            BackendEvent::Done => {
                self.state.status = "idle".to_string();
                if let Some(last) = self.state.messages.last_mut() {
                    if last.role == "assistant" && last.content.starts_with("[stream]") {
                        last.content = self.pending_stream.clone();
                    }
                }
                self.pending_stream.clear();
            }
            BackendEvent::Error(err) => {
                self.state.status = format!("error: {err}");
                self.state.messages.push(Message {
                    role: "assistant".to_string(),
                    content: format!("error: {err}"),
                });
            }
            BackendEvent::MemoryUpdated => {
                self.state.status = "memory updated".to_string();
            }
            BackendEvent::ToolCallStart(info) => {
                self.state.status = format!("tool: {}", info.name);
            }
            BackendEvent::ToolCallEnd(result) => {
                self.state.status = if result.success {
                    "tool complete".to_string()
                } else {
                    "tool failed".to_string()
                };
            }
        }
    }

    pub fn draw(&mut self, frame: &mut Frame<'_>) {
        let outer = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Min(5),
                Constraint::Length(10),
                Constraint::Length(1),
            ])
            .split(frame.area());

        let header = Paragraph::new(Line::from(format!(
            "Aigent • {} • Ctrl+S sidebar • Esc history",
            self.state.status
        )));
        frame.render_widget(header, outer[0]);

        let middle = if self.show_sidebar {
            Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
                .split(outer[1])
        } else {
            Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(100), Constraint::Length(0)])
                .split(outer[1])
        };

        draw_chat(
            frame,
            middle[0],
            &self.state.messages,
            &self.theme,
            if self.state.history_mode {
                self.state.selected_message
            } else {
                None
            },
        );

        if self.show_sidebar {
            draw_sidebar(frame, middle[1], &self.state, &self.theme);
        }

        draw_input(frame, outer[2], &mut self.textarea);

        if self.file_popup.visible {
            let area = centered_popup(frame.area(), 70, 45);
            frame.render_widget(Clear, area);
            let mut lines = vec![Line::from(Span::styled(
                format!("@{}", self.file_popup.query),
                ratatui::style::Style::default().fg(self.theme.muted),
            ))];
            lines.push(Line::from(""));
            if self.file_popup.candidates.is_empty() {
                lines.push(Line::from("(no matches)"));
            } else {
                for (idx, item) in self.file_popup.candidates.iter().enumerate() {
                    let style = if idx == self.file_popup.selected {
                        ratatui::style::Style::default()
                            .fg(self.theme.background)
                            .bg(self.theme.accent)
                    } else {
                        ratatui::style::Style::default().fg(self.theme.foreground)
                    };
                    lines.push(Line::from(Span::styled(item.clone(), style)));
                }
            }
            let widget = Paragraph::new(lines)
                .block(
                    Block::default()
                        .title(" File Context ")
                        .borders(Borders::ALL)
                        .border_type(BorderType::Rounded),
                )
                .wrap(Wrap { trim: false });
            frame.render_widget(widget, area);
        }

        if self.command_palette.visible {
            let area = centered_popup(frame.area(), 60, 40);
            frame.render_widget(Clear, area);
            let lines = self
                .command_palette
                .commands
                .iter()
                .enumerate()
                .map(|(idx, command)| {
                    if idx == self.command_palette.selected {
                        Line::from(Span::styled(
                            (*command).to_string(),
                            ratatui::style::Style::default()
                                .fg(self.theme.background)
                                .bg(self.theme.accent),
                        ))
                    } else {
                        Line::from((*command).to_string())
                    }
                })
                .collect::<Vec<_>>();
            let widget = Paragraph::new(lines).block(
                Block::default()
                    .title(" Commands ")
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded),
            );
            frame.render_widget(widget, area);
        }

        let footer = Paragraph::new(Line::from(
            "Enter send • Alt+Enter newline • Ctrl+K commands • @ file picker • history: Esc/Up/Down",
        ));
        frame.render_widget(footer, outer[3]);
    }

    fn refresh_file_popup(&mut self) {
        let input = self.input_text();
        let Some(query) = active_at_query(&input) else {
            self.file_popup.visible = false;
            self.file_popup.query.clear();
            self.file_popup.candidates.clear();
            self.file_popup.selected = 0;
            return;
        };

        self.file_popup.visible = true;
        self.file_popup.query = query.to_string();
        if query.is_empty() {
            self.file_popup.candidates = self.workspace_files.iter().take(8).cloned().collect();
            self.file_popup.selected = 0;
            return;
        }

        let matcher = SkimMatcherV2::default();
        let mut scored = self
            .workspace_files
            .iter()
            .filter_map(|path| {
                matcher
                    .fuzzy_match(path, query)
                    .map(|score| (score, path.clone()))
            })
            .collect::<Vec<_>>();
        scored.sort_by(|a, b| b.0.cmp(&a.0));
        self.file_popup.candidates = scored.into_iter().take(8).map(|(_, p)| p).collect();
        self.file_popup.selected = 0;
    }
}

fn collect_workspace_files(root: &Path) -> Vec<String> {
    let mut files = Vec::new();
    for entry in WalkBuilder::new(root)
        .hidden(false)
        .git_ignore(true)
        .build()
    {
        let Ok(entry) = entry else {
            continue;
        };
        if !entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
            continue;
        }
        let Ok(rel) = entry.path().strip_prefix(root) else {
            continue;
        };
        files.push(rel.display().to_string());
    }
    files.sort();
    files
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

fn centered_popup(area: ratatui::layout::Rect, w_pct: u16, h_pct: u16) -> ratatui::layout::Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - h_pct) / 2),
            Constraint::Percentage(h_pct),
            Constraint::Percentage((100 - h_pct) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - w_pct) / 2),
            Constraint::Percentage(w_pct),
            Constraint::Percentage((100 - w_pct) / 2),
        ])
        .split(vertical[1])[1]
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
