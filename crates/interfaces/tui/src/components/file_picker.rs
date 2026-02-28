//! File picker overlay — fuzzy file search triggered by `@` in the input.

use ratatui::layout::Rect;
use ratatui::Frame;

use crate::action::Action;
use crate::components::Component;
use crate::events::AppEvent;
use crate::state::{AppState, FilePopupState};
use crate::theme::Theme;

/// File picker popup component.
pub struct FilePicker {
    pub state: FilePopupState,
    pub workspace_files: Vec<String>,
}

impl FilePicker {
    pub fn new() -> Self {
        Self {
            state: FilePopupState {
                visible: false,
                query: String::new(),
                candidates: Vec::new(),
                selected: 0,
            },
            workspace_files: Vec::new(),
        }
    }
}

impl Component for FilePicker {
    fn update(&mut self, _event: &AppEvent, _state: &AppState) -> Vec<Action> {
        Vec::new()
    }

    fn draw(&self, _frame: &mut Frame<'_>, _area: Rect, _state: &AppState, _theme: &Theme) {
        // Full implementation in Step 5.
    }
}
