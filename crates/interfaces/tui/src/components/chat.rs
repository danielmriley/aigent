//! Chat panel — displays the conversation history.
//!
//! Responsible for rendering messages, handling scroll, and (later)
//! virtualized line rendering for large histories.

use ratatui::layout::Rect;
use ratatui::Frame;

use crate::action::Action;
use crate::components::Component;
use crate::events::AppEvent;
use crate::state::AppState;
use crate::theme::Theme;

/// Scroll and viewport state for the chat panel.
pub struct ChatPanel {
    pub scroll: usize,
    pub max_scroll: usize,
    pub auto_follow: bool,
}

impl ChatPanel {
    pub fn new() -> Self {
        Self {
            scroll: 0,
            max_scroll: 0,
            auto_follow: true,
        }
    }
}

impl Component for ChatPanel {
    fn update(&mut self, _event: &AppEvent, _state: &AppState) -> Vec<Action> {
        // Full implementation in Step 2.
        Vec::new()
    }

    fn draw(&self, _frame: &mut Frame<'_>, _area: Rect, _state: &AppState, _theme: &Theme) {
        // Full implementation in Step 2.
    }
}
