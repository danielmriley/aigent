//! Command palette overlay — fuzzy-filtered slash-command picker.

use ratatui::layout::Rect;
use ratatui::Frame;

use crate::action::Action;
use crate::components::Component;
use crate::events::AppEvent;
use crate::state::{AppState, CommandPaletteState};
use crate::theme::Theme;

/// Command palette overlay component.
pub struct CommandPalette {
    pub state: CommandPaletteState,
}

impl CommandPalette {
    pub fn new() -> Self {
        Self {
            state: CommandPaletteState {
                visible: false,
                selected: 0,
                commands: vec![
                    "/new",
                    "/switch",
                    "/memory",
                    "/sleep",
                    "/dedup",
                    "/doctor",
                    "/status",
                    "/context",
                    "/tools",
                    "/model show",
                    "/model list",
                    "/exit",
                ],
            },
        }
    }
}

impl Component for CommandPalette {
    fn update(&mut self, _event: &AppEvent, _state: &AppState) -> Vec<Action> {
        Vec::new()
    }

    fn draw(&self, _frame: &mut Frame<'_>, _area: Rect, _state: &AppState, _theme: &Theme) {
        // Full implementation in Step 5.
    }
}
