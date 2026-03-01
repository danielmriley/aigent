//! Live statistics charts for the advanced TUI.
//!
//! Provides `draw_tool_success_chart` (BarChart of tool success %) and
//! `draw_memory_sparkline` (Sparkline of memory-item growth).
//! Both use ratatui's built-in chart widgets — no extra dependencies.

use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::widgets::{BarChart, Block, BorderType, Borders, Sparkline};
use ratatui::Frame;

use crate::state::{AppState, ToolCallEntry};
use crate::theme::Theme;

/// Render a bar chart showing per-tool success rate.
///
/// Groups `tool_history` by name, computes success percentage, and draws
/// a horizontal `BarChart`.  If there are no tool calls, renders an
/// empty block with a placeholder.
pub fn draw_tool_success_chart(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &AppState,
    theme: &Theme,
) {
    let block = Block::default()
        .title(" Tool Success Rate ")
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme.border));

    if state.tool_history.is_empty() {
        frame.render_widget(block, area);
        return;
    }

    // Aggregate per tool → (successes, total).
    let agg = aggregate_tools(&state.tool_history);

    // Build bar data.  Label = short tool name, value = success %.
    let bars: Vec<(String, u64)> = agg
        .iter()
        .map(|(name, ok, total)| {
            let pct = if *total == 0 {
                0
            } else {
                (*ok as u64 * 100) / *total as u64
            };
            // Truncate long names.
            let short: String = name.chars().take(10).collect();
            (short, pct)
        })
        .collect();

    // BarChart expects &[(&str, u64)] — keep owned data alive.
    let bar_refs: Vec<(&str, u64)> =
        bars.iter().map(|(s, v)| (s.as_str(), *v)).collect();

    let chart = BarChart::default()
        .block(block)
        .data(&bar_refs)
        .bar_width(3)
        .bar_gap(1)
        .bar_style(Style::default().fg(theme.success))
        .value_style(Style::default().fg(theme.foreground));

    frame.render_widget(chart, area);
}

/// Render a sparkline of memory growth over time.
///
/// The data points come from `state.memory_samples` — a rolling
/// window of memory-item counts captured on each `MemoryUpdated`
/// backend event.  If empty, draws a placeholder.
pub fn draw_memory_sparkline(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &AppState,
    theme: &Theme,
) {
    let block = Block::default()
        .title(" Memory Growth ")
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme.border));

    if state.memory_samples.is_empty() {
        frame.render_widget(block, area);
        return;
    }

    let spark = Sparkline::default()
        .block(block)
        .data(&state.memory_samples)
        .style(Style::default().fg(Color::Rgb(137, 180, 250))); // accent-ish

    frame.render_widget(spark, area);
}

// ── helpers ────────────────────────────────────────────────────

/// Aggregate tool calls → Vec<(name, successes, total)>.
fn aggregate_tools(history: &[ToolCallEntry]) -> Vec<(String, usize, usize)> {
    use std::collections::HashMap;
    let mut map: HashMap<&str, (usize, usize)> = HashMap::new();
    for entry in history {
        let (ok, total) = map.entry(&entry.name).or_insert((0, 0));
        *total += 1;
        if entry.success {
            *ok += 1;
        }
    }
    let mut v: Vec<(String, usize, usize)> = map
        .into_iter()
        .map(|(name, (ok, total))| (name.to_string(), ok, total))
        .collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_empty() {
        let result = aggregate_tools(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn aggregate_mixed() {
        let history = vec![
            ToolCallEntry { name: "run-shell".to_string(), success: true },
            ToolCallEntry { name: "run-shell".to_string(), success: false },
            ToolCallEntry { name: "read-file".to_string(), success: true },
        ];
        let agg = aggregate_tools(&history);
        assert_eq!(agg.len(), 2);
        // read-file first (alphabetical).
        assert_eq!(agg[0], ("read-file".to_string(), 1, 1));
        assert_eq!(agg[1], ("run-shell".to_string(), 1, 2));
    }
}
