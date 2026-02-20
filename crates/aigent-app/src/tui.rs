use anyhow::Result;
use crossterm::event::{self, Event as CrosstermEvent, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame, Terminal,
};
use std::{io, time::Duration};
use tokio::sync::mpsc;

pub enum Event {
    Tick,
    Key(KeyEvent),
    Paste(String),
    LlmChunk(String),
    LlmDone(Vec<String>), // The final messages to add to transcript
    Error(String),
}

pub struct App {
    pub transcript: Vec<String>,
    pub input: String,
    pub cursor_pos: usize,
    pub is_thinking: bool,
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
            input: String::new(),
            cursor_pos: 0,
            is_thinking: false,
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
                    KeyCode::Left => {
                        if self.cursor_pos > 0 {
                            self.cursor_pos -= 1;
                        }
                    }
                    KeyCode::Right => {
                        if self.cursor_pos < self.input.chars().count() {
                            self.cursor_pos += 1;
                        }
                    }
                    KeyCode::Backspace => {
                        if self.cursor_pos > 0 {
                            self.cursor_pos -= 1;
                            let byte_idx = self.input.char_indices().nth(self.cursor_pos).map(|(i, _)| i).unwrap_or(self.input.len());
                            self.input.remove(byte_idx);
                        }
                    }
                    KeyCode::Delete => {
                        if self.cursor_pos < self.input.chars().count() {
                            let byte_idx = self.input.char_indices().nth(self.cursor_pos).map(|(i, _)| i).unwrap_or(self.input.len());
                            self.input.remove(byte_idx);
                        }
                    }
                    KeyCode::Char(ch) => {
                        let byte_idx = self.input.char_indices().nth(self.cursor_pos).map(|(i, _)| i).unwrap_or(self.input.len());
                        self.input.insert(byte_idx, ch);
                        self.cursor_pos += 1;
                    }
                    _ => {}
                }
            }
            Event::Paste(text) => {
                let byte_idx = self.input.char_indices().nth(self.cursor_pos).map(|(i, _)| i).unwrap_or(self.input.len());
                self.input.insert_str(byte_idx, &text);
                self.cursor_pos += text.chars().count();
            }
            Event::LlmChunk(chunk) => {
                // For now, just append to the last message or create a new one
                // We'll implement proper streaming later
            }
            Event::LlmDone(messages) => {
                self.is_thinking = false;
                for msg in messages {
                    self.transcript.push(msg);
                }
                if self.auto_follow {
                    // We'll need to calculate max_scroll during draw, but for now we can just set it high
                    self.viewport_start_line = usize::MAX; 
                }
            }
            Event::Error(err) => {
                self.is_thinking = false;
                self.transcript.push(format!("aigent> error: {}", err));
            }
        }
    }
}
