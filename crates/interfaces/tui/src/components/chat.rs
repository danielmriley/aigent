//! Chat panel — conversation history with themed messages, markdown,
//! tool-call blocks, scroll, and history selection.

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Wrap};
use ratatui::Frame;

use crate::state::{AppState, Message};
use crate::theme::Theme;

const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Scroll and viewport state for the chat panel.
pub struct ChatPanel {
    pub scroll: usize,
    pub max_scroll: usize,
    pub auto_follow: bool,
}

impl ChatPanel {
    pub fn new() -> Self {
        Self {
            scroll: 0,
            max_scroll: 0,
            auto_follow: true,
        }
    }

    /// Render the chat panel.
    pub fn draw(
        &mut self,
        frame: &mut Frame<'_>,
        area: Rect,
        state: &AppState,
        theme: &Theme,
        is_thinking: bool,
        is_sleeping: bool,
        spinner_tick: usize,
    ) {
        let selected = if state.history_mode {
            state.selected_message
        } else {
            None
        };

        let (chat_lines, msg_starts) =
            build_chat_lines(&state.messages, theme, selected, &state.bot_name);

        // ── scroll bookkeeping ──────────────────────────────────
        let chat_body_height = area.height.saturating_sub(2) as usize;
        let content_width = area.width.saturating_sub(2) as usize;
        let visual_lines: usize = chat_lines
            .iter()
            .map(|line| {
                let chars = line.to_string().chars().count().max(1);
                chars.div_ceil(content_width.max(1))
            })
            .sum();

        self.max_scroll = visual_lines.saturating_sub(chat_body_height);

        // In history mode, scroll to selected message.
        if state.history_mode {
            if let Some(sel_idx) = state.selected_message {
                if let Some(&logical_start) = msg_starts.get(sel_idx) {
                    let visual_start: usize = chat_lines[..logical_start]
                        .iter()
                        .map(|line| {
                            let chars = line.to_string().chars().count().max(1);
                            chars.div_ceil(content_width.max(1))
                        })
                        .sum();
                    self.scroll = visual_start.min(self.max_scroll);
                }
            }
        }

        if self.auto_follow {
            self.scroll = self.max_scroll;
        } else {
            self.scroll = self.scroll.min(self.max_scroll);
        }

        // ── title ───────────────────────────────────────────────
        let title = if is_thinking || is_sleeping {
            let f = SPINNER_FRAMES[spinner_tick / 2 % SPINNER_FRAMES.len()];
            format!(" Chat {f} ")
        } else {
            " Chat ".to_string()
        };

        let chat_widget = Paragraph::new(chat_lines)
            .block(
                Block::default()
                    .title(title)
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(theme.border)),
            )
            .scroll((self.scroll as u16, 0))
            .wrap(Wrap { trim: false });
        frame.render_widget(chat_widget, area);
    }
}

/// Build all chat lines and record the logical-line index where each
/// message starts (for history-mode scroll-to-selection).
fn build_chat_lines(
    messages: &[Message],
    theme: &Theme,
    selected: Option<usize>,
    assistant_label: &str,
) -> (Vec<Line<'static>>, Vec<usize>) {
    let mut lines = Vec::<Line<'static>>::new();
    let mut starts = Vec::<usize>::with_capacity(messages.len());

    for (idx, msg) in messages.iter().enumerate() {
        starts.push(lines.len());
        let is_selected = selected == Some(idx);

        let mut prefix_style = Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD);
        let mut body_style = Style::default().fg(theme.foreground);
        if is_selected {
            body_style = body_style.bg(theme.user_bubble);
            prefix_style = prefix_style.bg(theme.user_bubble);
        }

        // ── streaming ───────────────────────────────────────────
        if let Some(streaming) = msg.content.strip_prefix("[stream]") {
            lines.push(Line::from(vec![
                Span::styled(format!("{assistant_label}> "), prefix_style),
                Span::styled(streaming.to_string(), body_style),
            ]));
        }
        // ── user message ────────────────────────────────────────
        else if msg.role == "user" {
            lines.push(Line::from(vec![
                Span::styled("you> ", prefix_style),
                Span::styled(msg.content.clone(), body_style),
            ]));
        }
        // ── tool call ⚙ ─────────────────────────────────────────
        else if msg.role == "\u{2699}" {
            let tool_style = Style::default()
                .fg(theme.muted)
                .add_modifier(Modifier::DIM);
            let (summary, detail) = msg
                .content
                .split_once('\0')
                .map(|(s, d)| (s, Some(d)))
                .unwrap_or((&msg.content, None));
            let toggle = if detail.is_some() {
                if is_selected { " ▾" } else { " ▸" }
            } else {
                ""
            };
            lines.push(Line::from(vec![
                Span::styled("  ", tool_style),
                Span::styled(format!("{summary}{toggle}"), tool_style),
            ]));
            if is_selected {
                if let Some(detail_text) = detail {
                    let detail_style = Style::default()
                        .fg(theme.muted)
                        .add_modifier(Modifier::DIM);
                    for detail_line in detail_text.lines() {
                        lines.push(Line::from(Span::styled(
                            format!("    {detail_line}"),
                            detail_style,
                        )));
                    }
                }
            }
        }
        // ── system message ──────────────────────────────────────
        else if msg.role == "system" {
            let sys_style = Style::default()
                .fg(theme.muted)
                .add_modifier(Modifier::ITALIC);
            lines.push(Line::from(Span::styled(
                format!("  {}", msg.content),
                sys_style,
            )));
        }
        // ── assistant / aigent ──────────────────────────────────
        else if msg.role == "assistant" || msg.role == "aigent" {
            if let Some(ref rendered) = msg.rendered_md {
                if let Some(first) = rendered.first() {
                    lines.push(Line::from(vec![
                        Span::styled(format!("{assistant_label}> "), prefix_style),
                        Span::styled(first.to_string(), body_style),
                    ]));
                    for extra in rendered.iter().skip(1) {
                        lines.push(extra.clone());
                    }
                }
            } else {
                lines.push(Line::from(vec![
                    Span::styled(format!("{assistant_label}> "), prefix_style),
                    Span::styled(msg.content.clone(), body_style),
                ]));
            }
        }
        // ── external source ─────────────────────────────────────
        else {
            let mut ext_prefix = Style::default()
                .fg(theme.muted)
                .add_modifier(Modifier::BOLD);
            if is_selected {
                ext_prefix = ext_prefix.bg(theme.user_bubble);
            }
            lines.push(Line::from(vec![
                Span::styled(format!("{}> ", msg.role), ext_prefix),
                Span::styled(msg.content.clone(), body_style),
            ]));
        }

        lines.push(Line::from(""));
    }

    (lines, starts)
}
