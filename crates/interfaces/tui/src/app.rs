//! Root application orchestrator — MVU architecture.
//!
//! `App` owns the component tree, shared [`AppState`], and the
//! [`BackendEvent`] receiver.  It preserves the public API expected by
//! the CLI host crate and the terminal event loop in `tui.rs`.

use crossterm::event::{KeyCode, KeyModifiers, MouseEventKind};
use ratatui::Frame;
use tokio::sync::mpsc;
#[cfg(not(feature = "advanced"))]
use tui_textarea::{Input, Key};

use aigent_config::AppConfig;
use aigent_runtime::BackendEvent;

use crate::components::chat::ChatPanel;
use crate::components::command_palette::CommandPalette;
use crate::components::context_panel::ContextPanel;
use crate::components::file_picker::{replace_last_at_token, FilePicker};
use crate::components::footer::Footer;
use crate::components::input::InputBar;
use crate::components::sidebar::SidebarPanel;
use crate::components::status_bar::StatusBar;
use crate::events::AppEvent;
use crate::layout;
use crate::state::SidebarTab;
use crate::theme::Theme;
#[cfg(not(feature = "advanced"))]
use crate::widgets::markdown::render_markdown_lines;

#[cfg(feature = "advanced")]
use crate::components::vim_input::{InputAction, VimInput};
#[cfg(feature = "advanced")]
use crate::widgets::markdown::render_markdown_auto;

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
    pub context_panel: ContextPanel,
    pub input: InputBar,
    pub status_bar: StatusBar,
    pub footer: Footer,
    pub command_palette: CommandPalette,
    pub file_picker: FilePicker,

    #[cfg(feature = "advanced")]
    pub vim_input: VimInput,
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
                react_phase: None,
                react_round: None,
                react_max_rounds: None,
                swarm_role: None,
                token_prompt: None,
                token_response: None,
                token_total: None,
                tool_history: Vec::new(),
                sidebar_tab: SidebarTab::Sessions,
                model_name: Some(config.active_model().to_string()),
                memory_samples: Vec::new(),
                input_history: Vec::new(),
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
            context_panel: ContextPanel::new(),
            input: InputBar::new(),
            status_bar: StatusBar::new(),
            footer: Footer::new(),
            command_palette: CommandPalette::new(),
            file_picker: FilePicker::new(),
            #[cfg(feature = "advanced")]
            vim_input: VimInput::new(),
        }
    }

    // ── public helpers (API contract with CLI crate) ───────────

    pub fn input_text(&self) -> String {
        #[cfg(feature = "advanced")]
        { return self.vim_input.text(); }
        #[cfg(not(feature = "advanced"))]
        { self.input.text() }
    }

    pub fn clear_input(&mut self) {
        #[cfg(feature = "advanced")]
        { self.vim_input.clear(); }
        #[cfg(not(feature = "advanced"))]
        { self.input.clear(); }
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
            && key.code == KeyCode::Tab
        {
            self.state.sidebar_tab = match self.state.sidebar_tab {
                SidebarTab::Sessions => SidebarTab::Context,
                SidebarTab::Context => SidebarTab::Sessions,
            };
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
        // Advanced vim input mode.
        #[cfg(feature = "advanced")]
        {
            match self.vim_input.handle_key(key) {
                InputAction::Submit => {
                    let text = normalize_for_submit(&self.vim_input.text());
                    if text.is_empty() {
                        return None;
                    }
                    if text == "/exit" {
                        self.push_assistant_message("session closed".into());
                        let _ = self.vim_input.submit_and_clear();
                        return Some(UiCommand::Quit);
                    }
                    self.push_user_message(text.clone());
                    self.chat.auto_follow = true;
                    let _ = self.vim_input.submit_and_clear();
                    self.state.status = "thinking...".to_string();
                    return Some(UiCommand::Submit(text));
                }
                InputAction::Consumed => {
                    self.file_picker.refresh(&self.vim_input.text());
                    return None;
                }
                InputAction::Ignored => {
                    return None;
                }
            }
        }

        // Standard tui-textarea input mode.
        #[cfg(not(feature = "advanced"))]
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
                        #[cfg(feature = "advanced")]
                        {
                            last.rendered_md =
                                Some(render_markdown_auto(&last.content));
                        }
                        #[cfg(not(feature = "advanced"))]
                        {
                            last.rendered_md =
                                Some(render_markdown_lines(&last.content));
                        }
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
                let count = self.state.memory_peek.len() as u64;
                self.state.memory_samples.push(count);
                if self.state.memory_samples.len() > 200 {
                    self.state.memory_samples.remove(0);
                }
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
                // Track in tool history for context panel.
                self.state.tool_history.push(crate::state::ToolCallEntry {
                    name: result.name.clone(),
                    success: result.success,
                });
                if let Some(msg) = self
                    .state
                    .messages
                    .iter_mut()
                    .rev()
                    .find(|m| m.role == "\u{2699}")
                {
                    msg.content = if result.success {
                        let snip = if result.output.chars().count() > 80 {
                            format!(
                                "{}\u{2026}",
                                truncate_chars(&result.output, 80)
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
                let display = if insight.chars().count() > 80 {
                    format!("{}\u{2026}", truncate_chars(&insight, 80))
                } else {
                    insight.clone()
                };
                self.state.status = format!("\u{1f4a1} {}", display);
            }
            BackendEvent::BeliefAdded { claim, confidence } => {
                let display = if claim.chars().count() > 70 {
                    format!("{}\u{2026}", truncate_chars(&claim, 70))
                } else {
                    claim.clone()
                };
                self.state.status =
                    format!("belief ({:.2}): {}", confidence, display);
            }
            BackendEvent::ProactiveMessage { content } => {
                #[cfg(feature = "advanced")]
                let rendered = render_markdown_auto(&content);
                #[cfg(not(feature = "advanced"))]
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
            BackendEvent::ReactPhaseChanged { phase, round, max_rounds } => {
                self.state.react_phase = Some(format!("{phase}"));
                self.state.react_round = Some(round);
                self.state.react_max_rounds = Some(max_rounds);
                self.state.status = format!("ReAct {phase} ({}/{})", round + 1, max_rounds);
            }
            BackendEvent::SubAgentSpawned { role, ref task } => {
                let short = if task.chars().count() > 60 {
                    format!("{}\u{2026}", &task.chars().take(60).collect::<String>())
                } else {
                    task.clone()
                };
                self.state.status = format!("sub-agent [{role}]: {short}");
            }
            BackendEvent::SubAgentCompleted { role, success, .. } => {
                let icon = if success { "\u{2713}" } else { "\u{2717}" };
                self.state.status = format!("{icon} sub-agent [{role}] finished");
            }
            BackendEvent::ConfigUpdated { model, provider } => {
                self.state.model_name = Some(model.clone());
                self.state.status = format!("model={model} provider={provider}");
            }
        }
    }

    // ── draw (delegates to components) ─────────────────────────

    pub fn draw(&mut self, frame: &mut Frame<'_>) {
        #[cfg(feature = "advanced")]
        let input_height: u16 = {
            let line_count = self.vim_input.text().lines().count().max(1);
            (line_count as u16 + 2).clamp(3, 8)
        };
        #[cfg(not(feature = "advanced"))]
        let input_height: u16 = {
            self.input.wrap_width =
                usize::from(frame.area().width.saturating_sub(4)).max(8);
            self.input.apply_soft_wrap();
            let input_line_count = self.input.textarea.lines().len().max(1);
            (input_line_count as u16 + 2).clamp(3, 8)
        };

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

        // ── sidebar / context panel ─────────────────────────────
        if self.sidebar.visible {
            match self.state.sidebar_tab {
                SidebarTab::Sessions => {
                    self.sidebar.draw(
                        frame,
                        sidebar_area,
                        &self.state,
                        &self.theme,
                    );
                }
                SidebarTab::Context => {
                    self.context_panel.draw(
                        frame,
                        sidebar_area,
                        &self.state,
                        &self.theme,
                    );
                }
            }
        }

        // ── input ──────────────────────────────────────────────
        #[cfg(feature = "advanced")]
        self.vim_input.draw(frame, input_area, &self.theme);
        #[cfg(not(feature = "advanced"))]
        self.input.draw(frame, input_area, &self.theme);

        // ── footer ─────────────────────────────────────────────
        self.footer.draw(frame, footer_area, &self.theme);

        // ── overlays (drawn last, on top) ──────────────────────
        self.file_picker.draw(frame, &self.theme);
        self.command_palette.draw(frame, &self.theme);
    }
}

// ── free helpers ───────────────────────────────────────────────

/// Truncate string at a char boundary, returning at most `max_chars` characters.
fn truncate_chars(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        Some((byte_idx, _)) => &s[..byte_idx],
        None => s,
    }
}

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
