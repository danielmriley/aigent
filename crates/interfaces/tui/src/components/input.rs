//! Input bar — multi-line text area with soft-wrap and `@` file trigger.

use ratatui::layout::Rect;
use ratatui::Frame;
use tui_textarea::TextArea;

use crate::action::Action;
use crate::components::Component;
use crate::events::AppEvent;
use crate::state::AppState;
use crate::theme::Theme;

/// The multi-line input bar at the bottom of the TUI.
pub struct InputBar {
    pub textarea: TextArea<'static>,
    pub wrap_width: usize,
}

impl InputBar {
    pub fn new() -> Self {
        let mut textarea = TextArea::default();
        textarea.set_cursor_line_style(Default::default());
        Self {
            textarea,
            wrap_width: 60,
        }
    }

    /// Current text content (all lines joined).
    pub fn text(&self) -> String {
        self.textarea.lines().join("\n")
    }

    /// Reset the text area to empty.
    pub fn clear(&mut self) {
        self.textarea = TextArea::default();
        self.textarea.set_cursor_line_style(Default::default());
    }
}

impl Component for InputBar {
    fn update(&mut self, _event: &AppEvent, _state: &AppState) -> Vec<Action> {
        Vec::new()
    }

    fn draw(&self, _frame: &mut Frame<'_>, _area: Rect, _state: &AppState, _theme: &Theme) {
        // Full implementation in Step 2.
    }
}
