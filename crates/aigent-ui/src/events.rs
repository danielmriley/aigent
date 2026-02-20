use crossterm::event::KeyEvent;

use aigent_daemon::BackendEvent;

#[derive(Debug, Clone)]
pub enum AppEvent {
    Key(KeyEvent),
    Backend(BackendEvent),
    Tick,
    Resize(u16, u16),
}
