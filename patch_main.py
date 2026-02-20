import re

with open('crates/aigent-app/src/main.rs', 'r') as f:
    content = f.read()

# Replace `app.input` with `app.textarea.lines().join("\n")` in command_suggestions
content = re.sub(r'command_suggestions\(&app\.input\)', 'command_suggestions(&app.textarea.lines().join("\\n"))', content)

# Replace Tab handling
tab_old = """                            KeyCode::Tab => {
                                if let Some(suggestion) = command_suggestions(&app.input).first() {
                                    app.input = (*suggestion).to_string();
                                    app.cursor_pos = app.input.chars().count();
                                }
                            }"""
tab_new = """                            KeyCode::Tab => {
                                let input = app.textarea.lines().join("\\n");
                                if let Some(suggestion) = command_suggestions(&input).first() {
                                    app.textarea = tui_textarea::TextArea::default();
                                    app.textarea.insert_str(*suggestion);
                                }
                            }"""
content = content.replace(tab_old, tab_new)

# Replace Enter handling
enter_old = """                            KeyCode::Enter => {
                                if key.modifiers.contains(KeyModifiers::ALT) {
                                    let byte_idx = app.input.char_indices().nth(app.cursor_pos).map(|(i, _)| i).unwrap_or(app.input.len());
                                    app.input.insert(byte_idx, '\\n');
                                    app.cursor_pos += 1;
                                    continue;
                                }

                                let line = app.input.trim().to_string();
                                app.input.clear();
                                app.cursor_pos = 0;
                                if line.is_empty() {
                                    continue;
                                }"""
enter_new = """                            KeyCode::Enter => {
                                if key.modifiers.contains(KeyModifiers::ALT) {
                                    app.textarea.insert_newline();
                                    continue;
                                }

                                let line = app.textarea.lines().join("\\n").trim().to_string();
                                app.textarea = tui_textarea::TextArea::default();
                                if line.is_empty() {
                                    continue;
                                }"""
content = content.replace(enter_old, enter_new)

with open('crates/aigent-app/src/main.rs', 'w') as f:
    f.write(content)
