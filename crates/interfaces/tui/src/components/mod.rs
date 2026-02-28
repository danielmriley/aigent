//! Component-based UI architecture.
//!
//! Each component implements the [`Component`] trait which follows the
//! Elm-inspired Model-View-Update pattern:
//!
//! - **update**: handle an [`AppEvent`], optionally returning [`Action`]s
//! - **draw**: render into a [`ratatui::Frame`] area

pub mod chat;
pub mod command_palette;
pub mod file_picker;
pub mod footer;
pub mod input;
pub mod sidebar;
pub mod status_bar;

use ratatui::layout::Rect;
use ratatui::Frame;

use crate::action::Action;
use crate::events::AppEvent;
use crate::state::AppState;
use crate::theme::Theme;

/// Trait implemented by every visual component.
pub trait Component {
    /// Process an event, returning zero or more actions to be dispatched
    /// by the root orchestrator.
    fn update(&mut self, event: &AppEvent, state: &AppState) -> Vec<Action>;

    /// Render the component into the given area.
    fn draw(&self, frame: &mut Frame<'_>, area: Rect, state: &AppState, theme: &Theme);
}
