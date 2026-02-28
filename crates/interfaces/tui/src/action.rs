//! Internal actions for the MVU event loop.
//!
//! Actions are the "messages" in the Elm architecture: they describe
//! *what happened* and let the root `App::update` decide how to mutate
//! state and which side-effects to trigger.

/// An internal action produced by a component or the root App.
#[derive(Debug, Clone)]
pub enum Action {
    /// No-op.
    Noop,

    /// User submitted text from the input bar.
    SubmitInput(String),

    /// Toggle sidebar visibility.
    ToggleSidebar,

    /// Toggle command palette visibility.
    ToggleCommandPalette,

    /// Enter history / message-browsing mode.
    EnterHistoryMode,

    /// Exit history mode.
    ExitHistoryMode,

    /// Navigate selection in history mode (delta).
    HistoryNavigate(i32),

    /// Scroll chat by delta lines (negative = up).
    ScrollChat(i32),

    /// Lock/unlock auto-follow.
    SetAutoFollow(bool),

    /// Copy the selected code block to the clipboard.
    CopyCodeBlock,

    /// Save the selected code block to a snippet file.
    SaveCodeBlock,

    /// Insert a command string into the input bar.
    InsertCommand(String),

    /// Open / close the file picker popup.
    SetFilePickerVisible(bool),

    /// Select a file from the picker and splice it into the input.
    AcceptFilePick(String),

    /// Request quit.
    Quit,
}
