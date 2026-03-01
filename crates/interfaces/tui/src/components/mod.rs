//! Component-based UI architecture.
//!
//! Each component owns its local state and provides `draw()` for rendering.
//! The root `App` orchestrator dispatches events and delegates rendering.

pub mod chat;
pub mod command_palette;
pub mod context_panel;
pub mod file_picker;
pub mod footer;
pub mod input;
pub mod sidebar;
pub mod status_bar;
#[cfg(feature = "advanced")]
pub mod vim_input;
