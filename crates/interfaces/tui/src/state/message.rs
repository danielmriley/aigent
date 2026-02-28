//! Core data types for the TUI state.
//!
//! These types are pure data — no UI handles, no channels, no rendering
//! logic.  They form the **Model** in the Elm-inspired MVU architecture.

/// Commands the TUI sends back to the host (CLI) layer.
#[derive(Debug, Clone)]
pub enum UiCommand {
    Quit,
    Submit(String),
}

/// Which region of the TUI currently owns keyboard focus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Focus {
    Sidebar,
    Chat,
    Input,
}

/// A single message in the conversation history.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
    /// Pre-rendered markdown lines (populated when `BackendEvent::Done`
    /// finalises a streamed assistant reply).
    pub rendered_md: Option<Vec<ratatui::text::Line<'static>>>,
}

/// Observable application state (pure data, no UI handles).
#[derive(Debug, Clone)]
pub struct AppState {
    pub bot_name: String,
    pub sessions: Vec<String>,
    pub current_session: usize,
    pub messages: Vec<Message>,
    pub status: String,
    pub memory_peek: Vec<String>,
    pub selected_message: Option<usize>,
    pub history_mode: bool,
}

/// State for the file-picker popup.
#[derive(Debug, Clone)]
pub struct FilePopupState {
    pub visible: bool,
    pub query: String,
    pub candidates: Vec<String>,
    pub selected: usize,
}

/// State for the command-palette overlay.
#[derive(Debug, Clone)]
pub struct CommandPaletteState {
    pub visible: bool,
    pub selected: usize,
    pub commands: Vec<&'static str>,
}
