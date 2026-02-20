#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
