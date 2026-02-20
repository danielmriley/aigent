import re

with open('crates/aigent-app/src/tui.rs', 'r') as f:
    content = f.read()

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
                for ch in text.chars() {
                    self.textarea.insert_char(ch);
                }
            }"""

content = content.replace(handle_event_old, handle_event_new)

with open('crates/aigent-app/src/tui.rs', 'w') as f:
    f.write(content)
