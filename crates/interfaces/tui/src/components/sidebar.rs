//! Sidebar panel — session list, conversation stats, keyboard shortcut hints.

use ratatui::layout::Rect;
use ratatui::Frame;

use crate::action::Action;
use crate::components::Component;
use crate::events::AppEvent;
use crate::state::AppState;
use crate::theme::Theme;

/// Sidebar component with collapsible visibility.
pub struct SidebarPanel {
    pub visible: bool,
}

impl SidebarPanel {
    pub fn new() -> Self {
        Self { visible: true }
    }
}

impl Component for SidebarPanel {
    fn update(&mut self, _event: &AppEvent, _state: &AppState) -> Vec<Action> {
        Vec::new()
    }

    fn draw(&self, _frame: &mut Frame<'_>, _area: Rect, _state: &AppState, _theme: &Theme) {
        // Full implementation in Step 2.
    }
}
