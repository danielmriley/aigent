//! Footer bar — keybinding hints along the bottom edge.

use ratatui::layout::Rect;
use ratatui::Frame;

use crate::action::Action;
use crate::components::Component;
use crate::events::AppEvent;
use crate::state::AppState;
use crate::theme::Theme;

/// Footer component showing keyboard shortcut hints.
pub struct Footer;

impl Footer {
    pub fn new() -> Self {
        Self
    }
}

impl Component for Footer {
    fn update(&mut self, _event: &AppEvent, _state: &AppState) -> Vec<Action> {
        Vec::new()
    }

    fn draw(&self, _frame: &mut Frame<'_>, _area: Rect, _state: &AppState, _theme: &Theme) {
        // Full implementation in Step 2.
    }
}
