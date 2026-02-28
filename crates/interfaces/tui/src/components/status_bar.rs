//! Status bar — top header showing bot name, spinner, and status text.

use ratatui::layout::Rect;
use ratatui::Frame;

use crate::action::Action;
use crate::components::Component;
use crate::events::AppEvent;
use crate::state::AppState;
use crate::theme::Theme;

/// Header / status bar component.
pub struct StatusBar {
    pub spinner_tick: usize,
}

impl StatusBar {
    pub fn new() -> Self {
        Self { spinner_tick: 0 }
    }
}

impl Component for StatusBar {
    fn update(&mut self, _event: &AppEvent, _state: &AppState) -> Vec<Action> {
        Vec::new()
    }

    fn draw(&self, _frame: &mut Frame<'_>, _area: Rect, _state: &AppState, _theme: &Theme) {
        // Full implementation in Step 2.
    }
}
