use ratatui::widgets::Block;
use ratatui::text::Line;
fn test() {
    let b = Block::default().title(Line::from("test").centered());
}
