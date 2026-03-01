pub mod action;
pub mod app;
pub mod components;
pub mod events;
pub mod layout;
pub mod onboard;
pub mod state;
pub mod theme;
pub mod tui;
pub mod widgets;

pub use app::{App, AppState, Focus, UiCommand};
pub use events::AppEvent;

pub mod agent_panel;

/// Feature-gated advanced TUI with three-column layout, live agent status panel,
/// and richer interactive widgets.
///
/// Enabled via the `advanced` feature flag on `aigent-ui`.
/// Falls back to the standard `App` when the feature is not enabled.
#[cfg(feature = "advanced")]
pub type AdvancedTui = App;

#[cfg(not(feature = "advanced"))]
pub type AdvancedTui = App;
