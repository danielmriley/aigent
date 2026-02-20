import re

with open('crates/aigent-app/src/tui.rs', 'r') as f:
    content = f.read()

# Find the start and end of handle_event
start_idx = content.find('    pub fn handle_event(&mut self, event: Event) {')
end_idx = content.find('    }\n}\n\npub struct TranscriptViewport {')

if start_idx != -1 and end_idx != -1:
    new_handle_event = """    pub fn handle_event(&mut self, event: Event) {
        match event {
            Event::Tick => {
                if self.is_thinking {
                    self.thinking_spinner_tick = self.thinking_spinner_tick.wrapping_add(1);
                }
            }
            Event::Key(key) => {
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
            }
            Event::LlmChunk(_chunk) => {
                // For now, just append to the last message or create a new one
                // We'll implement proper streaming later
            }
            Event::LlmDone(messages) => {
                self.is_thinking = false;
                for msg in messages {
                    self.transcript.push(msg);
                }
                if self.auto_follow {
                    // We'll need to calculate max_scroll during draw, but for now we can just set it high
                    self.viewport_start_line = usize::MAX;
                }
            }
            Event::Error(err) => {
                self.is_thinking = false;
                self.transcript.push(format!("aigent> error: {}", err));
            }
        }
"""
    content = content[:start_idx] + new_handle_event + content[end_idx:]
    with open('crates/aigent-app/src/tui.rs', 'w') as f:
        f.write(content)
    print("Patched handle_event")
else:
    print("Could not find handle_event")
