//! Onboarding and configuration wizard.

pub mod state;
mod ui;
mod wizard;

pub use state::AvailableModels;
pub use wizard::{run_onboarding, run_configuration};
