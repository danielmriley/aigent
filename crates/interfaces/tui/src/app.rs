//! Root application orchestrator — MVU architecture.
//!
//! `App` owns the component tree, shared [`AppState`], and the
//! [`BackendEvent`] receiver.  It preserves the public API expected by
//! the CLI host crate and the terminal event loop in `tui.rs`.

use crossterm::event::{KeyCode, KeyModifiers, MouseEventKind};
use ratatui::Frame;
use tokio::sync::mpsc;
use tui_textarea::{Input, Key};

use aigent_config::AppConfig;
use aigent_runtime::BackendEvent;

use crate::components::chat::ChatPanel;
use crate::components::command_palette::CommandPalette;
use crate::components::file_picker::{replace_last_at_token, FilePicker};
use crate::components::footer::Footer;
use crate::components::input::InputBar;
use crate::components::sidebar::SidebarPanel;
use crate::components::status_bar::StatusBar;
use crate::events::AppEvent;
use crate::layout;
use crate::theme::Theme;
use crate::widgets::markdown::render_markdown_lines;

// Re-export state types so `crate::app::X` paths keep working.
pub use crate::state::{AppState, Focus, Message, UiCommand};

// ── App ────────────────────────────────────────────────────────

pub struct App {
    // ── shared state ───────────────────────────────────────────
    pub state: AppState,
    pub backend_rx: mpsc::UnboundedReceiver<BackendEvent>,
    pub focus: Focus,
    pub theme: Theme,

    // ── streaming / activity ───────────────────────────────────
    pub is_thinking: bool,
    pub is_sleeping: bool,
    pub active_tool: Option<String>,
    pub pending_stream: String,

    // ── components ─────────────────────────────────────────────
    pub chat: ChatPanel,
    pub sidebar: SidebarPanel,
    pub input: InputBar,
    pub status_bar: StatusBar,
    pub footer: Footer,
    pub command_palette: CommandPalette,
    pub file_picker: FilePicker,
}

impl App {
    pub fn new(backend_rx: mpsc::UnboundedReceiver<BackendEvent>, config: &AppConfig) -> Self {
        Self {
            state: AppState {
                bot_name: config.agent.name.clone(),
                sessions: vec!["default".to_string()],
                current_session: 0,
                messages: Vec::new(),
                status: format!(
                    "model={} provider={}",
                    config.active_model(),
                    config.llm.provider,
                ),
                memory_peek: Vec::new(),
                selected_message: None,
                history_mode: false,
            },
            backend_rx,
            focus: Focus::Input,
            theme: Theme::default(),
            is_thinking: false,
            is_sleeping: false,
            active_tool: None,
            pending_stream: String::new(),
            chat: ChatPanel::new(),
            sidebar: SidebarPanel::new(),
            input: InputBar::new(),
            status_bar: StatusBar::new(),
            footer: Footer::new(),
            command_palette: CommandPalette::new(),
            file_picker: FilePicker::new(),
        }
    }

    // ── public helpers (API contract with CLI crate) ───────────

    pub fn input_text(&self) -> String {
        self.input.text()
    }

    pub fn clear_input(&mut self) {
        self.input.clear();
    }

    pub fn push_user_message(&mut self, text: String) {
        self.state.messages.push(Message {
            role: "user".to_string(),
            content: text,
            rendered_md: None,
        });
    }

    pub fn push_assistant_message(&mut self, text: String) {
        self.state.messages.push(Message {
            role: "assistant".to_string(),
            content: text,
            rendered_md: None,
        });
    }

    pub fn set_status(&mut self, status: impl Into<String>) {
        self.state.status = status.into();
    }

    pub fn selected_code_block(&self) -> Option<String> {
        let idx = self.state.selected_message?;
        let msg = self.state.messages.get(idx)?;
        if msg.role != "assistant" {
            return None;
        }
        extract_first_code_block(&msg.content)
    }

    // ── MVU update ─────────────────────────────────────────────

    pub fn update(&mut self, event: AppEvent) -> Option<UiCommand> {
        match event {
            AppEvent::Tick => {
                self.status_bar.spinner_tick =
                    self.status_bar.spinner_tick.wrapping_add(1);
                // Refresh file picker on tick so it reacts to typing.
                self.file_picker.refresh(&self.input.text());
                None
            }
            AppEvent::Backend(be) => {
                self.apply_backend(be);
                None
            }
            AppEvent::Key(key) => self.handle_key(key),
            AppEvent::Mouse(mouse) => {
                match mouse.kind {
                    MouseEventKind::ScrollUp => {
                        self.chat.auto_follow = false;
                        self.chat.scroll = self.chat.scroll.saturating_sub(3);
                    }
                    MouseEventKind::ScrollDown => {
                        self.chat.scroll = self
                            .chat
                            .scroll
                            .saturating_add(3)
                            .min(self.chat.max_scroll);
                        self.chat.auto_follow =
                            self.chat.scroll >= self.chat.max_scroll;
                    }
                    _ => {}
                }
                None
            }
            AppEvent::Resize(_, _) => None,
        }
    }

    fn handle_key(
        &mut self,
        key: crossterm::event::KeyEvent,
    ) -> Option<UiCommand> {
        // ── global shortcuts ───────────────────────────────────
        if key.modifiers.contains(KeyModifiers::CONTROL)
            && key.code == KeyCode::Char('c')
        {
            return Some(UiCommand::Quit);
        }
        if key.modifiers.contains(KeyModifiers::CONTROL)
            && key.code == KeyCode::Char('s')
        {
            self.sidebar.visible = !self.sidebar.visible;
            return None;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL)
            && key.code == KeyCode::Char('k')
        {
            self.command_palette.state.visible =
                !self.command_palette.state.visible;
            return None;
        }

        // ── scroll shortcuts ───────────────────────────────────
        match key.code {
            KeyCode::PageUp => {
                self.chat.auto_follow = false;
                self.chat.scroll = self.chat.scroll.saturating_sub(5);
                return None;
            }
            KeyCode::PageDown => {
                self.chat.scroll = self
                    .chat
                    .scroll
                    .saturating_add(5)
                    .min(self.chat.max_scroll);
                self.chat.auto_follow =
                    self.chat.scroll >= self.chat.max_scroll;
                return None;
            }
            KeyCode::End => {
                self.chat.auto_follow = true;
                self.chat.scroll = self.chat.max_scroll;
                return None;
            }
            _ => {}
        }

        // ── command palette ────────────────────────────────────
        if self.command_palette.state.visible {
            match key.code {
                KeyCode::Esc => {
                    self.command_palette.state.visible = false;
                }
                KeyCode::Up => {
                    self.command_palette.state.selected = self
                        .command_palette
                        .state
                        .selected
                        .saturating_sub(1);
                }
                KeyCode::Down | KeyCode::Tab => {
                    self.command_palette.state.selected =
                        (self.command_palette.state.selected + 1).min(
                            self.command_palette
                                .state
                                .commands
                                .len()
                                .saturating_sub(1),
                        );
                }
                KeyCode::Enter => {
                    if let Some(cmd) = self
                        .command_palette
                        .state
                        .commands
                        .get(self.command_palette.state.selected)
                    {
                        self.input.clear();
                        self.input.textarea.insert_str(cmd);
                        self.command_palette.state.visible = false;
                    }
                }
                _ => {}
            }
            return None;
        }

        // ── file picker ────────────────────────────────────────
        if self.file_picker.state.visible {
            match key.code {
                KeyCode::Esc => {
                    self.file_picker.state.visible = false;
                    return None;
                }
                KeyCode::Up => {
                    self.file_picker.state.selected = self
                        .file_picker
                        .state
                        .selected
                        .saturating_sub(1);
                    return None;
                }
                KeyCode::Down | KeyCode::Tab => {
                    self.file_picker.state.selected =
                        (self.file_picker.state.selected + 1).min(
                            self.file_picker
                                .state
                                .candidates
                                .len()
                                .saturating_sub(1),
                        );
                    return None;
                }
                KeyCode::Enter => {
                    if let Some(path) = self
                        .file_picker
                        .state
                        .candidates
                        .get(self.file_picker.state.selected)
                        .cloned()
                    {
                        let updated =
                            replace_last_at_token(&self.input.text(), &path);
                        self.input.clear();
                        self.input.textarea.insert_str(&updated);
                        self.file_picker.state.visible = false;
                    }
                    return None;
                }
                _ => {} // Fall through to input handling.
            }
        }

        // ── history mode ───────────────────────────────────────
        if self.state.history_mode {
            match key.code {
                KeyCode::Esc => {
                    self.state.history_mode = false;
                    self.state.selected_message = None;
                    self.chat.auto_follow = true;
                }
                KeyCode::Up => {
                    let current = self
                        .state
                        .selected_message
                        .unwrap_or_else(|| {
                            self.state.messages.len().saturating_sub(1)
                        });
                    self.state.selected_message =
                        Some(current.saturating_sub(1));
                    self.chat.auto_follow = false;
                }
                KeyCode::Down => {
                    let current =
                        self.state.selected_message.unwrap_or(0);
                    self.state.selected_message = Some(
                        (current + 1).min(
                            self.state
                                .messages
                                .len()
                                .saturating_sub(1),
                        ),
                    );
                    self.chat.auto_follow = false;
                }
                KeyCode::Char('c') => {
                    if let Some(code) = self.selected_code_block() {
                        match arboard::Clipboard::new()
                            .and_then(|mut cb| cb.set_text(code))
                        {
                            Ok(_) => self.push_assistant_message(
                                "copied code block to clipboard".into(),
                            ),
                            Err(err) => self.push_assistant_message(
                                format!("clipboard error: {err}"),
                            ),
                        }
                    }
                }
                KeyCode::Char('a') => {
                    if let Some(code) = self.selected_code_block() {
                        let dir =
                            std::path::Path::new(".aigent").join("snippets");
                        if let Err(err) = std::fs::create_dir_all(&dir) {
                            self.push_assistant_message(format!(
                                "apply error: {err}"
                            ));
                        } else {
                            let file_name = format!(
                                "applied_{}.txt",
                                chrono::Local::now()
                                    .format("%Y%m%d_%H%M%S")
                            );
                            let path = dir.join(file_name);
                            match std::fs::write(&path, code) {
                                Ok(_) => self.push_assistant_message(
                                    format!(
                                        "applied code block to {}",
                                        path.display()
                                    ),
                                ),
                                Err(err) => self.push_assistant_message(
                                    format!("apply error: {err}"),
                                ),
                            }
                        }
                    }
                }
                _ => {}
            }
            return None;
        }

        // ── normal input mode ──────────────────────────────────
        match key.code {
            KeyCode::Esc => {
                self.state.history_mode = true;
                self.state.selected_message =
                    Some(self.state.messages.len().saturating_sub(1));
                self.chat.auto_follow = false;
                None
            }
            KeyCode::Enter
                if !key.modifiers.contains(KeyModifiers::ALT) =>
            {
                let text = normalize_for_submit(&self.input_text());
                if text.is_empty() {
                    return None;
                }
                if text == "/exit" {
                    self.push_assistant_message("session closed".into());
                    self.clear_input();
                    return Some(UiCommand::Quit);
                }
                self.push_user_message(text.clone());
                self.chat.auto_follow = true;
                self.clear_input();
                self.state.status = "thinking...".to_string();
                Some(UiCommand::Submit(text))
            }
            _ => {
                // Forward to textarea.
                self.input.textarea.input(Input {
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
                // Soft-wrap and refresh file picker after input change.
                self.input.wrap_width = 60; // Will be updated in draw().
                self.input.apply_soft_wrap();
                self.file_picker.refresh(&self.input.text());
                None
            }
        }
    }

    // ── backend event handling ──────────────────────────────────

    fn apply_backend(&mut self, event: BackendEvent) {
        match event {
            BackendEvent::Token(chunk) => {
                self.pending_stream.push_str(&chunk);
                if let Some(last) = self.state.messages.last_mut() {
                    if last.role == "assistant"
                        && last.content.starts_with("[stream]")
                    {
                        last.content =
                            format!("[stream]{}", self.pending_stream);
                        return;
                    }
                }
                self.chat.auto_follow = true;
                self.state.messages.push(Message {
                    role: "assistant".to_string(),
                    content: format!("[stream]{}", self.pending_stream),
                    rendered_md: None,
                });
            }
            BackendEvent::Thinking => {
                self.is_thinking = true;
                self.is_sleeping = false;
                self.state.status = "thinking".to_string();
            }
            BackendEvent::SleepCycleRunning => {
                self.is_sleeping = true;
                self.is_thinking = false;
                self.state.status = "sleep cycle running".to_string();
                self.chat.auto_follow = true;
                self.state.messages.push(Message {
                    role: "system".to_string(),
                    content: "Starting sleep cycle\u{2026}".to_string(),
                    rendered_md: None,
                });
            }
            BackendEvent::SleepProgress(msg) => {
                self.state.status = format!("sleep: {msg}");
                self.chat.auto_follow = true;
                self.state.messages.push(Message {
                    role: "system".to_string(),
                    content: msg,
                    rendered_md: None,
                });
            }
            BackendEvent::Done => {
                self.is_thinking = false;
                self.is_sleeping = false;
                self.state.status = "idle".to_string();
                if let Some(last) = self.state.messages.last_mut() {
                    if last.role == "assistant"
                        && last.content.starts_with("[stream]")
                    {
                        last.content = self.pending_stream.clone();
                        last.rendered_md =
                            Some(render_markdown_lines(&last.content));
                    }
                }
                self.pending_stream.clear();
                self.chat.auto_follow = true;
            }
            BackendEvent::Error(err) => {
                self.is_thinking = false;
                self.is_sleeping = false;
                self.state.status = format!("error: {err}");
                self.state.messages.push(Message {
                    role: "assistant".to_string(),
                    content: format!("error: {err}"),
                    rendered_md: None,
                });
            }
            BackendEvent::MemoryUpdated => {
                self.state.status = "memory updated".to_string();
            }
            BackendEvent::ToolCallStart(info) => {
                self.active_tool = Some(info.name.clone());
                self.state.status = format!("\u{1f527} {}", info.name);
                self.state.messages.push(Message {
                    role: "\u{2699}".to_string(),
                    content: format!("calling {}\u{2026}", info.name),
                    rendered_md: None,
                });
                self.chat.auto_follow = true;
            }
            BackendEvent::ToolCallEnd(result) => {
                self.active_tool = None;
                if let Some(msg) = self
                    .state
                    .messages
                    .iter_mut()
                    .rev()
                    .find(|m| m.role == "\u{2699}")
                {
                    msg.content = if result.success {
                        let snip = if result.output.len() > 80 {
                            format!(
                                "{}\u{2026}",
                                &result.output[..80]
                            )
                        } else {
                            result.output.clone()
                        };
                        format!(
                            "\u{2713} {} \u{2014} {}",
                            result.name, snip
                        )
                    } else {
                        format!("\u{2717} {} failed", result.name)
                    };
                }
                self.state.status = if result.success {
                    "tool complete".to_string()
                } else {
                    "tool failed".to_string()
                };
                self.chat.auto_follow = true;
            }
            BackendEvent::ExternalTurn { source, content } => {
                self.state.messages.push(Message {
                    role: source,
                    content,
                    rendered_md: None,
                });
                self.pending_stream.clear();
                self.chat.auto_follow = true;
            }
            BackendEvent::ReflectionInsight(insight) => {
                let display = if insight.len() > 80 {
                    format!("{}\u{2026}", &insight[..80])
                } else {
                    insight.clone()
                };
                self.state.status = format!("\u{1f4a1} {}", display);
            }
            BackendEvent::BeliefAdded { claim, confidence } => {
                let display = if claim.len() > 70 {
                    format!("{}\u{2026}", &claim[..70])
                } else {
                    claim.clone()
                };
                self.state.status =
                    format!("belief ({:.2}): {}", confidence, display);
            }
            BackendEvent::ProactiveMessage { content } => {
                let rendered = render_markdown_lines(&content);
                self.state.messages.push(Message {
                    role: "aigent".to_string(),
                    content,
                    rendered_md: Some(rendered),
                });
                self.pending_stream.clear();
                self.chat.auto_follow = true;
            }
            BackendEvent::ClearStream => {
                if let Some(last) = self.state.messages.last() {
                    if last.role == "assistant"
                        && last.content.starts_with("[stream]")
                    {
                        self.state.messages.pop();
                    }
                }
                self.pending_stream.clear();
            }
        }
    }

    // ── draw (delegates to components) ─────────────────────────

    pub fn draw(&mut self, frame: &mut Frame<'_>) {
        // Update input wrap width from terminal size.
        self.input.wrap_width =
            usize::from(frame.area().width.saturating_sub(4)).max(8);
        self.input.apply_soft_wrap();

        let input_line_count = self.input.textarea.lines().len().max(1);
        let input_height = (input_line_count as u16 + 2).clamp(3, 8);

        let [header_area, middle_area, input_area, footer_area] =
            layout::root_layout(frame.area(), input_height);
        let [chat_area, sidebar_area] =
            layout::middle_layout(middle_area, self.sidebar.visible);

        // ── status bar ─────────────────────────────────────────
        self.status_bar.draw(
            frame,
            header_area,
            &self.state,
            &self.theme,
            self.is_thinking,
            self.is_sleeping,
        );

        // ── chat ───────────────────────────────────────────────
        self.chat.draw(
            frame,
            chat_area,
            &self.state,
            &self.theme,
            self.is_thinking,
            self.is_sleeping,
            self.status_bar.spinner_tick,
        );

        // ── sidebar ────────────────────────────────────────────
        if self.sidebar.visible {
            self.sidebar.draw(
                frame,
                sidebar_area,
                &self.state,
                &self.theme,
            );
        }

        // ── input ──────────────────────────────────────────────
        self.input.draw(frame, input_area, &self.theme);

        // ── footer ─────────────────────────────────────────────
        self.footer.draw(frame, footer_area, &self.theme);

        // ── overlays (drawn last, on top) ──────────────────────
        self.file_picker.draw(frame, &self.theme);
        self.command_palette.draw(frame, &self.theme);
    }
}

// ── free helpers ───────────────────────────────────────────────

fn normalize_for_submit(input: &str) -> String {
    input.split_whitespace().collect::<Vec<_>>().join(" ")
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
