//! Shared TUI state types.
//!
//! Re-exports the pure-data structs that every component and the root
//! orchestrator depend on.

mod message;

pub use message::{
    AppState, CommandPaletteState, FilePopupState, Focus, Message, UiCommand,
};
