//! Root layout computation.
//!
//! Provides deterministic layout functions that convert a terminal area
//! into named regions.  All layout decisions live here so components
//! never compute their own positions.

use ratatui::layout::{Constraint, Direction, Layout, Rect};

/// The four-row root regions: header, middle, input, footer.
pub fn root_layout(area: Rect, input_height: u16) -> [Rect; 4] {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(5),
            Constraint::Length(input_height),
            Constraint::Length(1),
        ])
        .split(area);
    [chunks[0], chunks[1], chunks[2], chunks[3]]
}

/// Split the middle area into chat and sidebar columns.
pub fn middle_layout(area: Rect, show_sidebar: bool) -> [Rect; 2] {
    let chunks = if show_sidebar {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
            .split(area)
    } else {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(100), Constraint::Length(0)])
            .split(area)
    };
    [chunks[0], chunks[1]]
}


/// Split the middle area into three columns: sidebar (25%), chat (60%), agent panel (15%).
/// Falls back to two-column layout when the terminal is narrow (<120 cols).
pub fn advanced_middle_layout(area: Rect, show_sidebar: bool) -> [Rect; 3] {
    let wide_enough = area.width >= 120;
    if show_sidebar && wide_enough {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(25),
                Constraint::Percentage(60),
                Constraint::Percentage(15),
            ])
            .split(area);
        [chunks[0], chunks[1], chunks[2]]
    } else if show_sidebar {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(30),
                Constraint::Percentage(70),
                Constraint::Length(0),
            ])
            .split(area);
        [chunks[0], chunks[1], chunks[2]]
    } else {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Length(0),
                Constraint::Percentage(100),
                Constraint::Length(0),
            ])
            .split(area);
        [chunks[0], chunks[1], chunks[2]]
    }
}
/// Compute a centered popup area as a percentage of the parent.
pub fn centered_popup(area: Rect, w_pct: u16, h_pct: u16) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - h_pct) / 2),
            Constraint::Percentage(h_pct),
            Constraint::Percentage((100 - h_pct) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - w_pct) / 2),
            Constraint::Percentage(w_pct),
            Constraint::Percentage((100 - w_pct) / 2),
        ])
        .split(vertical[1])[1]
}
