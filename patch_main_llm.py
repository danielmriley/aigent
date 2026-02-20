import re

with open('crates/aigent-app/src/main.rs', 'r') as f:
    content = f.read()

pattern = re.compile(r'(\s*// Spawn LLM task\s*let _tx_llm = tx\.clone\(\);\s*// We need to clone runtime config.*?if outcome\.exit_requested \{\s*app\.should_quit = true;\s*\})', re.DOTALL)

replacement = """                                // Spawn LLM task
                                let tx_llm = tx.clone();
                                // We need to clone runtime config and memory for the background task, 
                                // but we can't easily do that since they are mutable references.
                                // For now, we'll block the main loop on the LLM call, but we'll use tokio::select!
                                // to keep processing Tick events for the spinner.
                                
                                let config_clone = runtime.config.clone();
                                let mut outcome_future = Box::pin(handle_interactive_input(
                                    runtime,
                                    memory,
                                    &line,
                                    &mut turn_count,
                                    &mut recent_turns,
                                ));

                                let outcome = loop {
                                    tokio::select! {
                                        res = &mut outcome_future => {
                                            break res;
                                        }
                                        Some(inner_event) = rx.recv() => {
                                            app.handle_event(inner_event);
                                            let suggestions = command_suggestions(&app.textarea.lines().join("\\n"));
                                            app.draw(&mut terminal, &suggestions, &config_clone)?;
                                        }
                                    }
                                };

                                match outcome {
                                    Ok(outcome) => {
                                        let _ = tx_llm.try_send(tui::Event::LlmDone(outcome.messages));
                                        if outcome.exit_requested {
                                            app.should_quit = true;
                                        }
                                    }
                                    Err(e) => {
                                        let _ = tx_llm.try_send(tui::Event::Error(e.to_string()));
                                    }
                                }"""

content, count = pattern.subn(replacement, content)
print(f"Replaced {count} occurrences")

with open('crates/aigent-app/src/main.rs', 'w') as f:
    f.write(content)
