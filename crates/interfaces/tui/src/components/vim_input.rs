//! Vim-mode input bar for the advanced TUI.
//!
//! A native ratatui-compatible input with Normal, Insert, and Visual modes,
//! basic vim motions, fuzzy history search (Ctrl+R), and undo/redo.
//! This replaces tui-textarea when the `advanced` feature is enabled.

use crossterm::event::{KeyCode, KeyModifiers};
use fuzzy_matcher::skim::SkimMatcherV2;
use fuzzy_matcher::FuzzyMatcher;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Wrap};
use ratatui::Frame;

use crate::theme::Theme;

// ── Vim mode ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VimMode {
    Normal,
    Insert,
    Visual,
    /// Ctrl+R fuzzy history search overlay.
    HistorySearch,
}

impl VimMode {
    pub fn label(self) -> &'static str {
        match self {
            VimMode::Normal => "NORMAL",
            VimMode::Insert => "INSERT",
            VimMode::Visual => "VISUAL",
            VimMode::HistorySearch => "SEARCH",
        }
    }

    pub fn color(self) -> Color {
        match self {
            VimMode::Normal => Color::Blue,
            VimMode::Insert => Color::Green,
            VimMode::Visual => Color::Magenta,
            VimMode::HistorySearch => Color::Yellow,
        }
    }
}

// ── Undo ring ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct UndoEntry {
    content: String,
    cursor: usize,
}

// ── VimInput ──────────────────────────────────────────────────

/// Vim-mode input bar — native ratatui widget, no tui-textarea dependency.
pub struct VimInput {
    /// Text buffer (single logical line; newlines encoded as '\n').
    buf: String,
    /// Byte offset of the cursor within `buf`.
    cursor: usize,
    /// Optional visual-mode anchor (byte offset).
    visual_anchor: Option<usize>,
    /// Current vim mode.
    pub mode: VimMode,
    /// Input history (most recent last).
    history: Vec<String>,
    /// Max history entries to keep.
    max_history: usize,
    /// Undo stack.
    undo: Vec<UndoEntry>,
    /// Redo stack.
    redo: Vec<UndoEntry>,
    /// Pending operator key in Normal mode (e.g. 'd' awaiting motion).
    pending_op: Option<char>,
    /// Pending numeric count prefix.
    count: Option<usize>,
    // ── history search state ──────────────────────────────────
    /// Current search query when in HistorySearch mode.
    search_query: String,
    /// Filtered history indices (into `self.history`).
    search_matches: Vec<usize>,
    /// Currently selected match index.
    search_selected: usize,
    /// Fuzzy matcher (reused).
    matcher: SkimMatcherV2,
}

impl Default for VimInput {
    fn default() -> Self {
        Self::new()
    }
}

impl VimInput {
    pub fn new() -> Self {
        Self {
            buf: String::new(),
            cursor: 0,
            visual_anchor: None,
            mode: VimMode::Insert, // start in insert for discoverability
            history: Vec::new(),
            max_history: 500,
            undo: Vec::new(),
            redo: Vec::new(),
            pending_op: None,
            count: None,
            search_query: String::new(),
            search_matches: Vec::new(),
            search_selected: 0,
            matcher: SkimMatcherV2::default(),
        }
    }

    // ── public API ────────────────────────────────────────────

    /// Get the current text content.
    pub fn text(&self) -> String {
        self.buf.clone()
    }

    /// Clear the buffer, push current content to history, and reset to Insert.
    pub fn submit_and_clear(&mut self) -> String {
        let text = std::mem::take(&mut self.buf);
        if !text.trim().is_empty() {
            self.history.push(text.clone());
            if self.history.len() > self.max_history {
                self.history.remove(0);
            }
        }
        self.cursor = 0;
        self.visual_anchor = None;
        self.undo.clear();
        self.redo.clear();
        self.mode = VimMode::Insert;
        text
    }

    /// Force clear without adding to history.
    pub fn clear(&mut self) {
        self.buf.clear();
        self.cursor = 0;
        self.visual_anchor = None;
        self.undo.clear();
        self.redo.clear();
        self.mode = VimMode::Insert;
    }

    /// Insert text at cursor (used by the host to pre-fill).
    pub fn insert_str(&mut self, s: &str) {
        self.save_undo();
        self.buf.insert_str(self.cursor, s);
        self.cursor += s.len();
    }

    // ── key handling ──────────────────────────────────────────

    /// Process a crossterm key event. Returns true if the input was consumed.
    pub fn handle_key(&mut self, key: crossterm::event::KeyEvent) -> InputAction {
        match self.mode {
            VimMode::Insert => self.handle_insert(key),
            VimMode::Normal => self.handle_normal(key),
            VimMode::Visual => self.handle_visual(key),
            VimMode::HistorySearch => self.handle_history_search(key),
        }
    }

    // ── Insert mode ───────────────────────────────────────────

    fn handle_insert(&mut self, key: crossterm::event::KeyEvent) -> InputAction {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);

        match key.code {
            KeyCode::Esc => {
                self.mode = VimMode::Normal;
                // In normal mode cursor doesn't sit past end.
                if self.cursor > 0 && self.cursor >= self.buf.len() {
                    self.cursor = self.prev_char_boundary(self.buf.len());
                }
                InputAction::Consumed
            }
            KeyCode::Enter if alt => {
                // Alt+Enter inserts a literal newline.
                self.save_undo();
                self.buf.insert(self.cursor, '\n');
                self.cursor += 1;
                InputAction::Consumed
            }
            KeyCode::Enter => InputAction::Submit,
            KeyCode::Char('r') if ctrl => {
                // Ctrl+R → enter history search.
                self.enter_history_search();
                InputAction::Consumed
            }
            KeyCode::Char('u') if ctrl => {
                self.perform_undo();
                InputAction::Consumed
            }
            KeyCode::Char(c) => {
                self.save_undo();
                self.buf.insert(self.cursor, c);
                self.cursor += c.len_utf8();
                InputAction::Consumed
            }
            KeyCode::Backspace => {
                if self.cursor > 0 {
                    self.save_undo();
                    let prev = self.prev_char_boundary(self.cursor);
                    self.buf.drain(prev..self.cursor);
                    self.cursor = prev;
                }
                InputAction::Consumed
            }
            KeyCode::Delete => {
                if self.cursor < self.buf.len() {
                    self.save_undo();
                    let next = self.next_char_boundary(self.cursor);
                    self.buf.drain(self.cursor..next);
                }
                InputAction::Consumed
            }
            KeyCode::Left => {
                if self.cursor > 0 {
                    self.cursor = self.prev_char_boundary(self.cursor);
                }
                InputAction::Consumed
            }
            KeyCode::Right => {
                if self.cursor < self.buf.len() {
                    self.cursor = self.next_char_boundary(self.cursor);
                }
                InputAction::Consumed
            }
            KeyCode::Home => {
                self.cursor = self.line_start();
                InputAction::Consumed
            }
            KeyCode::End => {
                self.cursor = self.line_end();
                InputAction::Consumed
            }
            _ => InputAction::Ignored,
        }
    }

    // ── Normal mode ───────────────────────────────────────────

    fn handle_normal(&mut self, key: crossterm::event::KeyEvent) -> InputAction {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);

        // Numeric count prefix.
        if let KeyCode::Char(c @ '1'..='9') = key.code {
            if self.pending_op.is_none() || self.count.is_some() {
                let digit = (c as u32 - '0' as u32) as usize;
                self.count = Some(self.count.unwrap_or(0) * 10 + digit);
                return InputAction::Consumed;
            }
        }
        if let KeyCode::Char('0') = key.code {
            if self.count.is_some() {
                self.count = Some(self.count.unwrap_or(0) * 10);
                return InputAction::Consumed;
            }
        }

        let n = self.count.take().unwrap_or(1);

        match key.code {
            // Mode switches.
            KeyCode::Char('i') if self.pending_op.is_none() => {
                self.mode = VimMode::Insert;
                InputAction::Consumed
            }
            KeyCode::Char('a') if self.pending_op.is_none() => {
                if self.cursor < self.buf.len() {
                    self.cursor = self.next_char_boundary(self.cursor);
                }
                self.mode = VimMode::Insert;
                InputAction::Consumed
            }
            KeyCode::Char('I') if self.pending_op.is_none() => {
                self.cursor = self.line_start();
                self.mode = VimMode::Insert;
                InputAction::Consumed
            }
            KeyCode::Char('A') if self.pending_op.is_none() => {
                self.cursor = self.line_end();
                self.mode = VimMode::Insert;
                InputAction::Consumed
            }
            KeyCode::Char('o') if self.pending_op.is_none() => {
                self.save_undo();
                let end = self.line_end();
                self.buf.insert(end, '\n');
                self.cursor = end + 1;
                self.mode = VimMode::Insert;
                InputAction::Consumed
            }
            KeyCode::Char('O') if self.pending_op.is_none() => {
                self.save_undo();
                let start = self.line_start();
                self.buf.insert(start, '\n');
                self.cursor = start;
                self.mode = VimMode::Insert;
                InputAction::Consumed
            }

            // Visual mode.
            KeyCode::Char('v') if self.pending_op.is_none() => {
                self.visual_anchor = Some(self.cursor);
                self.mode = VimMode::Visual;
                InputAction::Consumed
            }

            // Motions.
            KeyCode::Char('h') | KeyCode::Left => {
                self.motion_left(n);
                InputAction::Consumed
            }
            KeyCode::Char('l') | KeyCode::Right => {
                self.motion_right(n);
                InputAction::Consumed
            }
            KeyCode::Char('j') | KeyCode::Down => {
                self.motion_down(n);
                InputAction::Consumed
            }
            KeyCode::Char('k') | KeyCode::Up => {
                self.motion_up(n);
                InputAction::Consumed
            }
            KeyCode::Char('0') => {
                self.cursor = self.line_start();
                InputAction::Consumed
            }
            KeyCode::Char('$') => {
                self.cursor = self.line_end_inclusive();
                InputAction::Consumed
            }
            KeyCode::Char('^') => {
                let start = self.line_start();
                self.cursor = self.first_non_blank(start);
                InputAction::Consumed
            }
            KeyCode::Char('w') => {
                for _ in 0..n {
                    self.cursor = self.word_forward();
                }
                InputAction::Consumed
            }
            KeyCode::Char('b') => {
                for _ in 0..n {
                    self.cursor = self.word_backward();
                }
                InputAction::Consumed
            }
            KeyCode::Char('e') => {
                for _ in 0..n {
                    self.cursor = self.word_end_forward();
                }
                InputAction::Consumed
            }
            KeyCode::Char('G') => {
                self.cursor = self.buf.len().saturating_sub(1).min(self.buf.len());
                InputAction::Consumed
            }
            KeyCode::Char('g') => {
                // gg → go to start.
                self.cursor = 0;
                InputAction::Consumed
            }

            // Delete (operator or single-char).
            KeyCode::Char('x') if self.pending_op.is_none() => {
                self.save_undo();
                for _ in 0..n {
                    if self.cursor < self.buf.len() {
                        let next = self.next_char_boundary(self.cursor);
                        self.buf.drain(self.cursor..next);
                    }
                }
                self.clamp_cursor_normal();
                InputAction::Consumed
            }
            KeyCode::Char('d') if self.pending_op.is_none() => {
                self.pending_op = Some('d');
                InputAction::Consumed
            }
            KeyCode::Char('d') if self.pending_op == Some('d') => {
                // dd — delete current line.
                self.save_undo();
                let start = self.line_start();
                let end = self.line_end();
                // Include trailing newline if present.
                let end = if end < self.buf.len() { end + 1 } else { end };
                // Include leading newline if we're not at the start.
                let start = if start > 0 && self.buf.as_bytes().get(start.wrapping_sub(1)) == Some(&b'\n') {
                    start - 1
                } else {
                    start
                };
                self.buf.drain(start..end);
                self.cursor = start.min(self.buf.len());
                self.clamp_cursor_normal();
                self.pending_op = None;
                InputAction::Consumed
            }
            KeyCode::Char('D') if self.pending_op.is_none() => {
                // Delete from cursor to end of line.
                self.save_undo();
                let end = self.line_end();
                self.buf.drain(self.cursor..end);
                self.clamp_cursor_normal();
                InputAction::Consumed
            }

            // Change.
            KeyCode::Char('c') if self.pending_op.is_none() => {
                self.pending_op = Some('c');
                InputAction::Consumed
            }
            KeyCode::Char('c') if self.pending_op == Some('c') => {
                // cc — clear line, enter insert.
                self.save_undo();
                let start = self.line_start();
                let end = self.line_end();
                self.buf.drain(start..end);
                self.cursor = start;
                self.pending_op = None;
                self.mode = VimMode::Insert;
                InputAction::Consumed
            }
            KeyCode::Char('C') if self.pending_op.is_none() => {
                self.save_undo();
                let end = self.line_end();
                self.buf.drain(self.cursor..end);
                self.mode = VimMode::Insert;
                InputAction::Consumed
            }

            // Undo/Redo.
            KeyCode::Char('u') => {
                self.perform_undo();
                InputAction::Consumed
            }
            KeyCode::Char('r') if ctrl => {
                self.perform_redo();
                InputAction::Consumed
            }

            // History search (Ctrl+R in normal mode too).
            KeyCode::Char('/') => {
                self.enter_history_search();
                InputAction::Consumed
            }

            // Replace single char.
            KeyCode::Char('r') if !ctrl && self.pending_op.is_none() => {
                self.pending_op = Some('r');
                InputAction::Consumed
            }

            // Paste (p/P) - from last deleted content (simplified: just redo).
            KeyCode::Char('p') if self.pending_op.is_none() => {
                self.perform_redo();
                InputAction::Consumed
            }

            // Escape clears pending ops.
            KeyCode::Esc => {
                self.pending_op = None;
                InputAction::Consumed
            }

            _ => {
                // If we have a pending 'r' (replace), the next char replaces under cursor.
                if self.pending_op == Some('r') {
                    if let KeyCode::Char(c) = key.code {
                        if self.cursor < self.buf.len() {
                            self.save_undo();
                            let next = self.next_char_boundary(self.cursor);
                            self.buf.replace_range(self.cursor..next, &c.to_string());
                        }
                        self.pending_op = None;
                        return InputAction::Consumed;
                    }
                }
                self.pending_op = None;
                InputAction::Ignored
            }
        }
    }

    // ── Visual mode ───────────────────────────────────────────

    fn handle_visual(&mut self, key: crossterm::event::KeyEvent) -> InputAction {
        let n = self.count.take().unwrap_or(1);

        match key.code {
            KeyCode::Esc => {
                self.mode = VimMode::Normal;
                self.visual_anchor = None;
                InputAction::Consumed
            }
            // Motions (same as normal, cursor moves but anchor stays).
            KeyCode::Char('h') | KeyCode::Left => { self.motion_left(n); InputAction::Consumed }
            KeyCode::Char('l') | KeyCode::Right => { self.motion_right(n); InputAction::Consumed }
            KeyCode::Char('j') | KeyCode::Down => { self.motion_down(n); InputAction::Consumed }
            KeyCode::Char('k') | KeyCode::Up => { self.motion_up(n); InputAction::Consumed }
            KeyCode::Char('w') => { for _ in 0..n { self.cursor = self.word_forward(); } InputAction::Consumed }
            KeyCode::Char('b') => { for _ in 0..n { self.cursor = self.word_backward(); } InputAction::Consumed }
            KeyCode::Char('0') => { self.cursor = self.line_start(); InputAction::Consumed }
            KeyCode::Char('$') => { self.cursor = self.line_end_inclusive(); InputAction::Consumed }

            // Delete selection.
            KeyCode::Char('d') | KeyCode::Char('x') => {
                self.save_undo();
                let anchor = self.visual_anchor.unwrap_or(self.cursor);
                let (start, end) = if anchor <= self.cursor {
                    (anchor, self.next_char_boundary(self.cursor))
                } else {
                    (self.cursor, self.next_char_boundary(anchor))
                };
                let end = end.min(self.buf.len());
                self.buf.drain(start..end);
                self.cursor = start.min(self.buf.len());
                self.visual_anchor = None;
                self.mode = VimMode::Normal;
                self.clamp_cursor_normal();
                InputAction::Consumed
            }

            // Change selection → delete + enter insert.
            KeyCode::Char('c') => {
                self.save_undo();
                let anchor = self.visual_anchor.unwrap_or(self.cursor);
                let (start, end) = if anchor <= self.cursor {
                    (anchor, self.next_char_boundary(self.cursor))
                } else {
                    (self.cursor, self.next_char_boundary(anchor))
                };
                let end = end.min(self.buf.len());
                self.buf.drain(start..end);
                self.cursor = start;
                self.visual_anchor = None;
                self.mode = VimMode::Insert;
                InputAction::Consumed
            }

            // Yank (copy) — just exit visual.
            KeyCode::Char('y') => {
                self.visual_anchor = None;
                self.mode = VimMode::Normal;
                InputAction::Consumed
            }

            _ => InputAction::Ignored,
        }
    }

    // ── History search mode ───────────────────────────────────

    fn enter_history_search(&mut self) {
        self.mode = VimMode::HistorySearch;
        self.search_query.clear();
        self.search_matches = (0..self.history.len()).rev().collect();
        self.search_selected = 0;
    }

    fn refresh_search_matches(&mut self) {
        if self.search_query.is_empty() {
            self.search_matches = (0..self.history.len()).rev().collect();
        } else {
            let mut scored: Vec<(usize, i64)> = self
                .history
                .iter()
                .enumerate()
                .filter_map(|(i, h)| {
                    self.matcher
                        .fuzzy_match(h, &self.search_query)
                        .map(|score| (i, score))
                })
                .collect();
            scored.sort_by(|a, b| b.1.cmp(&a.1));
            self.search_matches = scored.into_iter().map(|(i, _)| i).collect();
        }
        self.search_selected = 0;
    }

    fn handle_history_search(&mut self, key: crossterm::event::KeyEvent) -> InputAction {
        match key.code {
            KeyCode::Esc => {
                self.mode = VimMode::Insert;
                InputAction::Consumed
            }
            KeyCode::Enter => {
                // Accept selected history entry.
                if let Some(&idx) = self.search_matches.get(self.search_selected) {
                    if let Some(entry) = self.history.get(idx) {
                        self.buf = entry.clone();
                        self.cursor = self.buf.len();
                    }
                }
                self.mode = VimMode::Insert;
                InputAction::Consumed
            }
            KeyCode::Up | KeyCode::BackTab => {
                if !self.search_matches.is_empty() {
                    self.search_selected =
                        (self.search_selected + 1).min(self.search_matches.len() - 1);
                }
                InputAction::Consumed
            }
            KeyCode::Down | KeyCode::Tab => {
                self.search_selected = self.search_selected.saturating_sub(1);
                InputAction::Consumed
            }
            KeyCode::Backspace => {
                self.search_query.pop();
                self.refresh_search_matches();
                InputAction::Consumed
            }
            KeyCode::Char(c) => {
                self.search_query.push(c);
                self.refresh_search_matches();
                InputAction::Consumed
            }
            _ => InputAction::Consumed,
        }
    }

    // ── Motion helpers ────────────────────────────────────────

    fn motion_left(&mut self, n: usize) {
        for _ in 0..n {
            if self.cursor > 0 {
                self.cursor = self.prev_char_boundary(self.cursor);
            }
        }
    }

    fn motion_right(&mut self, n: usize) {
        for _ in 0..n {
            if self.cursor < self.buf.len() {
                self.cursor = self.next_char_boundary(self.cursor);
            }
        }
    }

    fn motion_down(&mut self, n: usize) {
        for _ in 0..n {
            let col = self.cursor - self.line_start();
            let end = self.line_end();
            if end < self.buf.len() {
                let next_start = end + 1;
                let next_end = self.line_end_at(next_start);
                self.cursor = (next_start + col).min(next_end);
            }
        }
    }

    fn motion_up(&mut self, n: usize) {
        for _ in 0..n {
            let col = self.cursor - self.line_start();
            let start = self.line_start();
            if start > 0 {
                let prev_end = start - 1; // the '\n' before this line
                let prev_start = self.line_start_at(prev_end);
                self.cursor = (prev_start + col).min(prev_end);
            }
        }
    }

    fn word_forward(&self) -> usize {
        let bytes = self.buf.as_bytes();
        let mut pos = self.cursor;
        let len = bytes.len();
        if pos >= len { return pos; }

        // Skip current word chars.
        while pos < len && !bytes[pos].is_ascii_whitespace() { pos += 1; }
        // Skip whitespace.
        while pos < len && bytes[pos].is_ascii_whitespace() { pos += 1; }
        pos
    }

    fn word_backward(&self) -> usize {
        let bytes = self.buf.as_bytes();
        let mut pos = self.cursor;
        if pos == 0 { return 0; }
        pos -= 1;

        // Skip whitespace.
        while pos > 0 && bytes[pos].is_ascii_whitespace() { pos -= 1; }
        // Skip word chars backward.
        while pos > 0 && !bytes[pos - 1].is_ascii_whitespace() { pos -= 1; }
        pos
    }

    fn word_end_forward(&self) -> usize {
        let bytes = self.buf.as_bytes();
        let mut pos = self.cursor;
        let len = bytes.len();
        if pos >= len { return pos; }
        pos += 1;
        if pos >= len { return len.saturating_sub(1); }

        // Skip whitespace.
        while pos < len && bytes[pos].is_ascii_whitespace() { pos += 1; }
        // Skip to end of word.
        while pos < len && !bytes[pos].is_ascii_whitespace() { pos += 1; }
        pos.saturating_sub(1)
    }

    fn line_start(&self) -> usize {
        self.line_start_at(self.cursor)
    }

    fn line_start_at(&self, pos: usize) -> usize {
        let pos = pos.min(self.buf.len());
        self.buf[..pos].rfind('\n').map(|i| i + 1).unwrap_or(0)
    }

    fn line_end(&self) -> usize {
        self.line_end_at(self.cursor)
    }

    fn line_end_at(&self, pos: usize) -> usize {
        let pos = pos.min(self.buf.len());
        self.buf[pos..].find('\n').map(|i| pos + i).unwrap_or(self.buf.len())
    }

    fn line_end_inclusive(&self) -> usize {
        let end = self.line_end();
        if end > 0 { end.saturating_sub(1).max(self.line_start()) } else { 0 }
    }

    fn first_non_blank(&self, from: usize) -> usize {
        let end = self.line_end();
        let mut pos = from;
        while pos < end {
            if !self.buf.as_bytes()[pos].is_ascii_whitespace() {
                return pos;
            }
            pos += 1;
        }
        from
    }

    fn prev_char_boundary(&self, pos: usize) -> usize {
        let mut p = pos.saturating_sub(1);
        while p > 0 && !self.buf.is_char_boundary(p) { p -= 1; }
        p
    }

    fn next_char_boundary(&self, pos: usize) -> usize {
        let mut p = pos + 1;
        while p < self.buf.len() && !self.buf.is_char_boundary(p) { p += 1; }
        p.min(self.buf.len())
    }

    fn clamp_cursor_normal(&mut self) {
        if self.buf.is_empty() {
            self.cursor = 0;
        } else if self.cursor >= self.buf.len() {
            self.cursor = self.prev_char_boundary(self.buf.len());
        }
    }

    // ── Undo / Redo ───────────────────────────────────────────

    fn save_undo(&mut self) {
        // Only push if state actually differs from last entry.
        if let Some(last) = self.undo.last() {
            if last.content == self.buf && last.cursor == self.cursor {
                return;
            }
        }
        self.undo.push(UndoEntry {
            content: self.buf.clone(),
            cursor: self.cursor,
        });
        self.redo.clear();
        // Cap undo stack.
        if self.undo.len() > 200 {
            self.undo.remove(0);
        }
    }

    fn perform_undo(&mut self) {
        if let Some(entry) = self.undo.pop() {
            self.redo.push(UndoEntry {
                content: self.buf.clone(),
                cursor: self.cursor,
            });
            self.buf = entry.content;
            self.cursor = entry.cursor.min(self.buf.len());
        }
    }

    fn perform_redo(&mut self) {
        if let Some(entry) = self.redo.pop() {
            self.undo.push(UndoEntry {
                content: self.buf.clone(),
                cursor: self.cursor,
            });
            self.buf = entry.content;
            self.cursor = entry.cursor.min(self.buf.len());
        }
    }

    // ── Rendering ─────────────────────────────────────────────

    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let mode_span = Span::styled(
            format!(" {} ", self.mode.label()),
            Style::default()
                .fg(Color::Black)
                .bg(self.mode.color())
                .add_modifier(Modifier::BOLD),
        );

        let title = if self.mode == VimMode::HistorySearch {
            format!(
                " Search: {} ({}/{}) ",
                self.search_query,
                if self.search_matches.is_empty() { 0 } else { self.search_selected + 1 },
                self.search_matches.len(),
            )
        } else {
            " Input (Esc→Normal, i→Insert, Ctrl+R→Search) ".to_string()
        };

        let block = Block::default()
            .title(Line::from(vec![mode_span, Span::raw(title)]))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(theme.border));

        // Build display lines with cursor highlight.
        let display_text = if self.mode == VimMode::HistorySearch {
            // Show matched history entries.
            self.render_history_search(area.width.saturating_sub(2) as usize, theme)
        } else {
            self.render_buffer(theme)
        };

        let widget = Paragraph::new(display_text)
            .block(block)
            .wrap(Wrap { trim: false });

        frame.render_widget(widget, area);
    }

    fn render_buffer(&self, theme: &Theme) -> Vec<Line<'static>> {
        let visual_range = self.visual_anchor.map(|anchor| {
            if anchor <= self.cursor {
                (anchor, self.cursor)
            } else {
                (self.cursor, anchor)
            }
        });

        let mut lines = Vec::new();
        for (line_idx, line_text) in self.buf.split('\n').enumerate() {
            let line_byte_start = self.buf.split('\n')
                .take(line_idx)
                .map(|l| l.len() + 1)
                .sum::<usize>();

            let mut spans = Vec::new();
            for (char_offset, ch) in line_text.char_indices() {
                let byte_pos = line_byte_start + char_offset;
                let is_cursor = byte_pos == self.cursor;
                let is_visual = visual_range
                    .map(|(s, e)| byte_pos >= s && byte_pos <= e)
                    .unwrap_or(false);

                let style = if is_cursor {
                    Style::default().fg(Color::Black).bg(theme.accent)
                } else if is_visual {
                    Style::default().fg(Color::Black).bg(Color::LightYellow)
                } else {
                    Style::default().fg(theme.foreground)
                };

                spans.push(Span::styled(ch.to_string(), style));
            }

            // Show cursor at end of line if cursor is there.
            let line_end_byte = line_byte_start + line_text.len();
            if self.cursor == line_end_byte && self.mode == VimMode::Insert {
                spans.push(Span::styled(
                    " ",
                    Style::default().fg(Color::Black).bg(theme.accent),
                ));
            }

            if spans.is_empty() && self.cursor == line_byte_start {
                spans.push(Span::styled(
                    " ",
                    Style::default().fg(Color::Black).bg(theme.accent),
                ));
            }

            lines.push(Line::from(spans));
        }

        if lines.is_empty() {
            lines.push(Line::from(Span::styled(
                " ",
                Style::default().fg(Color::Black).bg(theme.accent),
            )));
        }

        lines
    }

    fn render_history_search(&self, _width: usize, theme: &Theme) -> Vec<Line<'static>> {
        let mut lines = Vec::new();
        let visible_count = 8;
        let start = self.search_selected.saturating_sub(visible_count / 2);

        for (display_idx, &hist_idx) in self.search_matches.iter().skip(start).take(visible_count).enumerate() {
            let actual_idx = start + display_idx;
            if let Some(entry) = self.history.get(hist_idx) {
                let is_selected = actual_idx == self.search_selected;
                let prefix = if is_selected { "▸ " } else { "  " };
                let display: String = entry.chars().take(60).collect();
                let style = if is_selected {
                    Style::default().fg(theme.accent).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(theme.foreground)
                };
                lines.push(Line::from(Span::styled(format!("{prefix}{display}"), style)));
            }
        }

        if lines.is_empty() {
            lines.push(Line::from(Span::styled(
                "  (no matches)",
                Style::default().fg(theme.muted),
            )));
        }

        lines
    }
}

// ── InputAction ───────────────────────────────────────────────

/// Result of processing a key event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputAction {
    /// Key was consumed by the vim input.
    Consumed,
    /// Enter was pressed in insert mode — submit the input.
    Submit,
    /// Key was not handled (pass to parent).
    Ignored,
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyEvent;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::empty())
    }

    fn key_ctrl(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::CONTROL)
    }

    #[test]
    fn insert_mode_typing() {
        let mut vi = VimInput::new();
        assert_eq!(vi.mode, VimMode::Insert);
        vi.handle_key(key(KeyCode::Char('h')));
        vi.handle_key(key(KeyCode::Char('i')));
        assert_eq!(vi.text(), "hi");
    }

    #[test]
    fn escape_to_normal() {
        let mut vi = VimInput::new();
        vi.handle_key(key(KeyCode::Char('a')));
        vi.handle_key(key(KeyCode::Esc));
        assert_eq!(vi.mode, VimMode::Normal);
    }

    #[test]
    fn normal_i_to_insert() {
        let mut vi = VimInput::new();
        vi.handle_key(key(KeyCode::Esc)); // → Normal
        vi.handle_key(key(KeyCode::Char('i'))); // → Insert
        assert_eq!(vi.mode, VimMode::Insert);
    }

    #[test]
    fn normal_dd_deletes_line() {
        let mut vi = VimInput::new();
        vi.insert_str("hello\nworld");
        vi.cursor = 0;
        vi.mode = VimMode::Normal;
        vi.handle_key(key(KeyCode::Char('d')));
        vi.handle_key(key(KeyCode::Char('d')));
        assert_eq!(vi.text(), "world");
    }

    #[test]
    fn undo_redo() {
        let mut vi = VimInput::new();
        vi.handle_key(key(KeyCode::Char('a')));
        vi.handle_key(key(KeyCode::Char('b')));
        assert_eq!(vi.text(), "ab");
        // Undo last char.
        vi.handle_key(key_ctrl(KeyCode::Char('u')));
        assert_eq!(vi.text(), "a");
    }

    #[test]
    fn submit_clears_and_adds_history() {
        let mut vi = VimInput::new();
        vi.handle_key(key(KeyCode::Char('x')));
        let text = vi.submit_and_clear();
        assert_eq!(text, "x");
        assert!(vi.text().is_empty());
        assert_eq!(vi.history.len(), 1);
    }

    #[test]
    fn history_search_enters_and_exits() {
        let mut vi = VimInput::new();
        vi.history.push("hello world".to_string());
        vi.history.push("goodbye".to_string());
        vi.handle_key(key_ctrl(KeyCode::Char('r')));
        assert_eq!(vi.mode, VimMode::HistorySearch);
        assert_eq!(vi.search_matches.len(), 2);
        vi.handle_key(key(KeyCode::Esc));
        assert_eq!(vi.mode, VimMode::Insert);
    }

    #[test]
    fn word_forward_motion() {
        let mut vi = VimInput::new();
        vi.insert_str("hello world foo");
        vi.cursor = 0;
        vi.mode = VimMode::Normal;
        vi.handle_key(key(KeyCode::Char('w')));
        assert_eq!(vi.cursor, 6); // "world"
    }

    #[test]
    fn visual_mode_delete() {
        let mut vi = VimInput::new();
        vi.insert_str("abcdef");
        vi.cursor = 1; // 'b'
        vi.mode = VimMode::Normal;
        vi.handle_key(key(KeyCode::Char('v'))); // visual
        assert_eq!(vi.mode, VimMode::Visual);
        vi.handle_key(key(KeyCode::Char('l'))); // extend to 'c'
        vi.handle_key(key(KeyCode::Char('l'))); // extend to 'd'
        vi.handle_key(key(KeyCode::Char('d'))); // delete selection
        assert_eq!(vi.text(), "aef");
        assert_eq!(vi.mode, VimMode::Normal);
    }
}
