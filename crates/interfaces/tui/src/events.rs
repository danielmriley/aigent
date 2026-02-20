use crossterm::event::{KeyEvent, MouseEvent};

use aigent_runtime::BackendEvent;

#[derive(Debug, Clone)]
pub enum AppEvent {
    Key(KeyEvent),
    Mouse(MouseEvent),
    Backend(BackendEvent),
    Tick,
    Resize(u16, u16),
}
