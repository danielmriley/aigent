use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Command {
    Memory,
    Core,
    Sleep,
    Correct,
    Pin,
    Forget,
    Status,
    Help,
}
