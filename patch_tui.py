import re

with open('crates/aigent-app/src/tui.rs', 'r') as f:
    content = f.read()

# Add import
content = content.replace('use tokio::sync::mpsc;', 'use tokio::sync::mpsc;\nuse tui_textarea::TextArea;')

# Update App struct
content = re.sub(r'pub input: String,\n\s*pub cursor_pos: usize,', 'pub textarea: TextArea<\'static>,', content)

# Update App::new
content = re.sub(r'input: String::new\(\),\n\s*cursor_pos: 0,', 'textarea: TextArea::default(),', content)

# Update App::handle_event
handle_event_old = """            Event::Key(key) => {
                if key.kind != KeyEventKind::Press {
                    return;
                }
                match key.code {
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        self.should_quit = true;
                    }
                    KeyCode::Char('s') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        self.show_sidebar = !self.show_sidebar;
                    }
                    KeyCode::Esc => {
                        self.should_quit = true;
                    }
                    KeyCode::Left => {
                        if self.cursor_pos > 0 {
                            self.cursor_pos -= 1;
                        }
                    }
                    KeyCode::Right => {
                        if self.cursor_pos < self.input.chars().count() {
                            self.cursor_pos += 1;
                        }
                    }
                    KeyCode::Backspace => {
                        if self.cursor_pos > 0 {
                            self.cursor_pos -= 1;
                            let byte_idx = self.input.char_indices().nth(self.cursor_pos).map(|(i, _)| i).unwrap_or(self.input.len());
                            self.input.remove(byte_idx);
                        }
                    }
                    KeyCode::Delete => {
                        if self.cursor_pos < self.input.chars().count() {
                            let byte_idx = self.input.char_indices().nth(self.cursor_pos).map(|(i, _)| i).unwrap_or(self.input.len());
                            self.input.remove(byte_idx);
                        }
                    }
                    KeyCode::Char(ch) => {
                        let byte_idx = self.input.char_indices().nth(self.cursor_pos).map(|(i, _)| i).unwrap_or(self.input.len());
                        self.input.insert(byte_idx, ch);
                        self.cursor_pos += 1;
                    }
                    _ => {}
                }
            }
            Event::Paste(text) => {
                let byte_idx = self.input.char_indices().nth(self.cursor_pos).map(|(i, _)| i).unwrap_or(self.input.len());
                self.input.insert_str(byte_idx, &text);
                self.cursor_pos += text.chars().count();
            }"""

handle_event_new = """            Event::Key(key) => {
                if key.kind != KeyEventKind::Press {
                    return;
                }
                match key.code {
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        self.should_quit = true;
                    }
                    KeyCode::Char('s') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        self.show_sidebar = !self.show_sidebar;
                    }
                    KeyCode::Esc => {
                        self.should_quit = true;
                    }
                    _ => {
                        self.textarea.input(key);
                    }
                }
            }
            Event::Paste(text) => {
                // tui-textarea doesn't have a direct paste method that takes a string easily without simulating keypresses,
                // but we can insert a string by inserting chars.
                for ch in text.chars() {
                    self.textarea.insert_char(ch);
                }
            }"""

content = content.replace(handle_event_old, handle_event_new)

# Update App::draw
# 1. input_lines
content = re.sub(r'let input_lines = self\.input\.split\(\'\\n\'\)\.count\(\)\.max\(1\) as u16;', 'let input_lines = self.textarea.lines().len().max(1) as u16;', content)

# 2. input_widget and cursor
draw_old = """            let input_widget = Paragraph::new(self.input.as_str())
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_type(ratatui::widgets::BorderType::Rounded)
                        .title(" Input (Alt+Enter for newline) "),
                )
                .style(Style::default().fg(Color::White));
            frame.render_widget(input_widget, chunks[2]);

            let input_width = usize::from(chunks[2].width.saturating_sub(2));
            if input_width > 0 {
                let text_before_cursor =
                    self.input.chars().take(self.cursor_pos).collect::<String>();
                let lines_before_cursor = text_before_cursor.split('\\n').collect::<Vec<_>>();
                let cursor_y_offset = lines_before_cursor.len().saturating_sub(1) as u16;
                let last_line_len =
                    lines_before_cursor.last().unwrap_or(&"").chars().count() as u16;

                let cursor_x =
                    chunks[2].x + 1 + last_line_len.min(chunks[2].width.saturating_sub(3));
                let cursor_y =
                    chunks[2].y + 1 + cursor_y_offset.min(chunks[2].height.saturating_sub(3));

                frame.set_cursor_position((cursor_x, cursor_y));
            }"""

draw_new = """            self.textarea.set_block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(ratatui::widgets::BorderType::Rounded)
                    .title(" Input (Alt+Enter for newline) "),
            );
            self.textarea.set_style(Style::default().fg(Color::White));
            frame.render_widget(&self.textarea, chunks[2]);"""

content = content.replace(draw_old, draw_new)

with open('crates/aigent-app/src/tui.rs', 'w') as f:
    f.write(content)
