use ratatui::{
    Frame,
    layout::Rect,
    style::Style,
    widgets::{Block, BorderType, Borders},
};
use tui_textarea::TextArea;

pub fn draw_input(frame: &mut Frame<'_>, area: Rect, textarea: &mut TextArea<'static>) {
    textarea.set_block(
        Block::default()
            .title(" Input (Alt+Enter newline) ")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded),
    );
    textarea.set_style(Style::default());
    frame.render_widget(&*textarea, area);
}
