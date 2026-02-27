//! TUI rendering: onboarding wizard drawing and event loop.

use std::io;
use std::path::{Path, PathBuf};

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Flex, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Gauge, Paragraph, Wrap};

use aigent_config::AppConfig;
use crate::theme::Theme;

use std::time::Duration;

use super::state::*;
use super::wizard;

pub fn center(area: Rect, horizontal: Constraint, vertical: Constraint) -> Rect {
    let [area] = Layout::horizontal([horizontal])
        .flex(Flex::Center)
        .areas(area);
    let [area] = Layout::vertical([vertical]).flex(Flex::Center).areas(area);
    area
}

pub(super) fn run_onboarding_tui(
    config: &mut AppConfig,
    mode: SetupMode,
    models: AvailableModels,
) -> Result<()> {
    let mut draft = OnboardingDraft::from_config(config, models);

    let mut stdout = io::stdout();
    enable_raw_mode()?;
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let mut step = if mode == SetupMode::Configuration {
        WizardStep::ConfigMenu
    } else {
        WizardStep::Welcome
    };
    let mut input = draft.value_for_step(step);
    let mut error_message: Option<String> = None;

    let result = (|| -> Result<()> {
        loop {
            draw_onboarding(
                &mut terminal,
                step,
                &draft,
                &input,
                error_message.as_deref(),
                mode,
            )?;

            if !event::poll(Duration::from_millis(120))? {
                continue;
            }

            let Event::Key(key) = event::read()? else {
                continue;
            };
            if key.kind != KeyEventKind::Press {
                continue;
            }

            match key.code {
                KeyCode::Esc => {
                    draft.apply_partial(config)?;
                    if draft.provider == "openrouter" && !draft.openrouter_key.trim().is_empty() {
                        wizard::upsert_env_value(
                            Path::new(".env"),
                            "OPENROUTER_API_KEY",
                            draft.openrouter_key.trim(),
                        )?;
                    }
                    if draft.telegram_enabled && !draft.telegram_token.trim().is_empty() {
                        wizard::upsert_env_value(
                            Path::new(".env"),
                            "TELEGRAM_BOT_TOKEN",
                            draft.telegram_token.trim(),
                        )?;
                    }
                    if !draft.brave_api_key.trim().is_empty() {
                        wizard::upsert_env_value(
                            Path::new(".env"),
                            "BRAVE_API_KEY",
                            draft.brave_api_key.trim(),
                        )?;
                    }
                    return Ok(());
                }
                KeyCode::Backspace => {
                    if step.is_text_step() {
                        input.pop();
                    }
                }
                KeyCode::Left => {
                    step = prev_step(step, &draft, mode);
                    input = draft.value_for_step(step);
                    error_message = None;
                }
                KeyCode::Up => {
                    draft.choice_prev(step);
                    input = draft.value_for_step(step);
                }
                KeyCode::Down => {
                    draft.choice_next(step);
                    input = draft.value_for_step(step);
                }
                KeyCode::Char('b') if !step.is_text_step() => {
                    step = prev_step(step, &draft, mode);
                    input = draft.value_for_step(step);
                    error_message = None;
                }
                KeyCode::Char('t') if !step.is_text_step() => {
                    let tested = match step {
                        WizardStep::Model | WizardStep::OpenRouterKey => {
                            Some(wizard::test_llm_connection_for_draft(&draft).map(|ok| {
                                format!("Connected ✓ {ok}")
                            }))
                        }
                        WizardStep::TelegramToken => Some(wizard::test_telegram_connection(&draft.telegram_token).map(|_| {
                            "Connected ✓ Telegram token looks valid".to_string()
                        })),
                        _ => None,
                    };

                    if let Some(result) = tested {
                        error_message = Some(match result {
                            Ok(msg) => msg,
                            Err(err) => format!("Connection failed: {err}"),
                        });
                    }
                }
                KeyCode::Char(ch) => {
                    if step.is_text_step() {
                        input.push(ch);
                    }
                }
                KeyCode::Enter => {
                    if step == WizardStep::Summary {
                        if let Err(err) = draft.apply(config) {
                            error_message = Some(err.to_string());
                            continue;
                        }
                        if draft.provider == "openrouter" && !draft.openrouter_key.trim().is_empty()
                        {
                            wizard::upsert_env_value(
                                Path::new(".env"),
                                "OPENROUTER_API_KEY",
                                draft.openrouter_key.trim(),
                            )?;
                        }
                        if draft.telegram_enabled && !draft.telegram_token.trim().is_empty() {
                            wizard::upsert_env_value(
                                Path::new(".env"),
                                "TELEGRAM_BOT_TOKEN",
                                draft.telegram_token.trim(),
                            )?;
                        }
                        if !draft.brave_api_key.trim().is_empty() {
                            wizard::upsert_env_value(
                                Path::new(".env"),
                                "BRAVE_API_KEY",
                                draft.brave_api_key.trim(),
                            )?;
                        }
                        break;
                    }

                    if step == WizardStep::Welcome || step == WizardStep::ConfigMenu {
                        step = next_step(step, &draft, mode);
                        input = draft.value_for_step(step);
                        error_message = None;
                        continue;
                    }

                    if let Err(err) = draft.commit_step(step, &input) {
                        error_message = Some(err.to_string());
                        continue;
                    }
                    step = next_step(step, &draft, mode);
                    if step == WizardStep::ConfigMenu && mode == SetupMode::Configuration {
                        let _ = draft.apply_partial(config);
                    }
                    input = draft.value_for_step(step);
                    error_message = None;
                }
                KeyCode::Tab => {
                    draft.choice_next(step);
                    input = draft.value_for_step(step);
                    error_message = None;
                }
                _ => {}
            }
        }

        Ok(())
    })();

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    result
}

fn draw_onboarding(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    step: WizardStep,
    draft: &OnboardingDraft,
    input: &str,
    error_message: Option<&str>,
    mode: SetupMode,
) -> Result<()> {
    let (title, description, input_display) = step_content(step, draft, input);
    let theme = Theme::default();
    let (step_index, step_total) = step_progress(step, mode, draft);
    let progress = if step_total == 0 {
        0.0
    } else {
        step_index as f64 / step_total as f64
    };

    terminal.draw(|frame| {
        let area = frame.area();

        let wizard_area = center(area, Constraint::Length(92), Constraint::Length(26));

        let header_title = if mode == SetupMode::Onboarding {
            " Aigent Onboarding "
        } else {
            " Aigent Configuration "
        };

        // Draw background block
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::DarkGray))
            .title_top(
                ratatui::text::Line::from(Span::styled(
                    header_title,
                    Style::default()
                        .fg(theme.accent)
                        .add_modifier(Modifier::BOLD),
                ))
                .centered(),
            )
            .title_bottom(
                ratatui::text::Line::from(Span::styled(
                    format!(" Step {step_index}/{step_total} "),
                    Style::default().fg(Color::DarkGray),
                ))
                .right_aligned(),
            );

        frame.render_widget(block.clone(), wizard_area);

        let inner_area = wizard_area.inner(ratatui::layout::Margin {
            vertical: 1,
            horizontal: 2,
        });

        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(24), Constraint::Min(40)])
            .split(inner_area);

        let sidebar_block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(theme.muted))
            .title(" Steps ");
        let sidebar_inner = sidebar_block.inner(columns[0]);
        frame.render_widget(sidebar_block, columns[0]);

        let labels = active_steps(mode, draft)
            .into_iter()
            .map(|candidate| {
                let label = match candidate {
                    WizardStep::ConfigMenu => "Configuration Menu",
                    WizardStep::Welcome => "Welcome",
                    WizardStep::UserName => "User Name",
                    WizardStep::BotName => "Bot Name",
                    WizardStep::DataDirectory => "Data Directory",
                    WizardStep::Provider => "Provider",
                    WizardStep::Model => "Model",
                    WizardStep::OpenRouterKey => "OpenRouter Key",
                    WizardStep::Thinking => "Thinking",
                    WizardStep::NightSleepStart => "Sleep Start",
                    WizardStep::NightSleepEnd => "Sleep End",
                    WizardStep::Safety => "Safety",
                    WizardStep::ApprovalMode => "Approval Mode",
                    WizardStep::ApiKeys => "API Keys",
                    WizardStep::Messaging => "Messaging",
                    WizardStep::TelegramToken => "Telegram Token",
                    WizardStep::Summary => "Summary",
                };

                if candidate == step {
                    Line::from(Span::styled(
                        format!("> {label}"),
                        Style::default()
                            .fg(theme.accent)
                            .add_modifier(Modifier::BOLD),
                    ))
                } else {
                    Line::from(Span::styled(
                        format!("  {label}"),
                        Style::default().fg(theme.foreground),
                    ))
                }
            })
            .collect::<Vec<_>>();
        frame.render_widget(Paragraph::new(labels), sidebar_inner);

        let content_area = columns[1];

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // progress
                Constraint::Length(2), // Title
                Constraint::Min(3),    // Description
                Constraint::Length(5), // Input box
                Constraint::Length(1), // Spacer
                Constraint::Length(2), // Footer/Error
            ])
            .split(content_area);

        let gauge = Gauge::default()
            .gauge_style(Style::default().fg(theme.accent).bg(theme.background))
            .ratio(progress)
            .label(format!("Step {step_index} of {step_total}"));
        frame.render_widget(gauge, chunks[0]);

        // Title
        let title_widget = Paragraph::new(title)
            .style(
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )
            .alignment(ratatui::layout::Alignment::Center);
        frame.render_widget(title_widget, chunks[1]);

        // Description
        let desc_widget = Paragraph::new(description)
            .style(Style::default().fg(Color::Gray))
            .wrap(Wrap { trim: true })
            .alignment(ratatui::layout::Alignment::Center);
        frame.render_widget(desc_widget, chunks[2]);

        // Input box
        if step != WizardStep::Welcome && step != WizardStep::Summary {
            let input_block = Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(theme.accent));
            let input_widget = Paragraph::new(input_display)
                .block(input_block)
                .style(Style::default().fg(Color::White))
                .wrap(Wrap { trim: false })
                .alignment(ratatui::layout::Alignment::Center);

            // Center the input box horizontally
            let input_area = center(chunks[3], Constraint::Percentage(80), Constraint::Length(5));
            frame.render_widget(input_widget, input_area);
        } else if step == WizardStep::Summary {
            // For summary, we use the input box area for more description
            let summary_widget = Paragraph::new(input_display)
                .style(Style::default().fg(Color::White))
                .wrap(Wrap { trim: true })
                .alignment(ratatui::layout::Alignment::Left);

            let summary_area = chunks[2].union(chunks[3]).union(chunks[4]);
            frame.render_widget(summary_widget, summary_area);
        }

        // Footer
        let footer_text = if let Some(message) = error_message {
            let is_success = message.starts_with("Connected ✓");
            if is_success {
                Line::from(Span::styled(
                    message,
                    Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
                ))
            } else {
                Line::from(vec![
                    Span::styled(
                        "Error: ",
                        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(message, Style::default().fg(Color::LightRed)),
                ])
            }
        } else {
            Line::from(Span::styled(
                "Enter=next • Left/b=back • t=test connection • Esc=save draft & exit",
                Style::default().fg(Color::DarkGray),
            ))
        };

        let footer_widget =
            Paragraph::new(footer_text).alignment(ratatui::layout::Alignment::Center);
        frame.render_widget(footer_widget, chunks[5]);
    })?;

    Ok(())
}

fn format_choice_list<'a>(options: &[&'a str], current: &str) -> ratatui::text::Text<'a> {
    if options.is_empty() {
        return ratatui::text::Text::from(vec![
            Line::from(""),
            Line::from(Span::styled(
                "No options available",
                Style::default().fg(Color::Red),
            )),
            Line::from(""),
        ]);
    }

    let idx = options.iter().position(|&o| o == current).unwrap_or(0);
    let prev_idx = if idx == 0 { options.len() - 1 } else { idx - 1 };
    let next_idx = (idx + 1) % options.len();

    let mut lines = Vec::new();

    if options.len() == 1 {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled(
                "  > ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                options[0],
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                " <  ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        lines.push(Line::from(""));
    } else if options.len() == 2 {
        lines.push(Line::from(vec![
            Span::styled("    ", Style::default().fg(Color::DarkGray)),
            Span::styled(options[prev_idx], Style::default().fg(Color::DarkGray)),
            Span::styled("    ", Style::default().fg(Color::DarkGray)),
        ]));
        lines.push(Line::from(vec![
            Span::styled(
                "  > ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                options[idx],
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                " <  ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        lines.push(Line::from(""));
    } else {
        lines.push(Line::from(vec![
            Span::styled("    ", Style::default().fg(Color::DarkGray)),
            Span::styled(options[prev_idx], Style::default().fg(Color::DarkGray)),
            Span::styled("    ", Style::default().fg(Color::DarkGray)),
        ]));
        lines.push(Line::from(vec![
            Span::styled(
                "  > ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                options[idx],
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                " <  ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::styled("    ", Style::default().fg(Color::DarkGray)),
            Span::styled(options[next_idx], Style::default().fg(Color::DarkGray)),
            Span::styled("    ", Style::default().fg(Color::DarkGray)),
        ]));
    }

    ratatui::text::Text::from(lines)
}

fn step_content<'a>(
    step: WizardStep,
    draft: &'a OnboardingDraft,
    input: &'a str,
) -> (&'static str, String, ratatui::text::Text<'a>) {
    match step {
        WizardStep::ConfigMenu => {
            let options = ConfigSection::all()
                .into_iter()
                .map(ConfigSection::label)
                .collect::<Vec<_>>();
            (
                "Configuration Menu",
                "Select a settings area, then press Enter.".to_string(),
                format_choice_list(&options, draft.config_section.label()),
            )
        }
        WizardStep::Welcome => (
            "Welcome to Aigent",
            "This wizard configures identity, memory policy, and safety.\n\n\
             Aigent ships with 8 built-in tools (web_search, read_file, write_file, \n\
             list_directory, run_shell, calendar_add_event, remind_me, draft_email) \n\
             running in WASM isolation by default (seccomp/sandbox active on Linux & macOS).\n\
             You'll choose an approval mode to control how much autonomy the agent has.\n\n\
             Press Enter to begin.".to_string(),
            ratatui::text::Text::from(""),
        ),
        WizardStep::BotName => (
            "Bot Identity",
            "Choose a name for your bot. This is how it will identify itself.".to_string(),
            ratatui::text::Text::from(format!("\n{}\n", input)),
        ),
        WizardStep::UserName => (
            "User Identity",
            "What is your name? (the agent stores this in Core memory).".to_string(),
            ratatui::text::Text::from(format!("\n{}\n", input)),
        ),
        WizardStep::DataDirectory => (
            "Data Directory",
            "Where Aigent stores memory files, event logs, vault exports, and config (default: aigent-data).".to_string(),
            ratatui::text::Text::from(format!("\n{}\n", input)),
        ),
        WizardStep::Provider => (
            "LLM Provider",
            "Choose the LLM provider (Up/Down/Tab to cycle):\n\nOllama = local-first\nOpenRouter = hosted API".to_string(),
            format_choice_list(&["ollama", "openrouter"], &draft.provider),
        ),
        WizardStep::Model => {
            let is_openrouter = draft.provider.eq_ignore_ascii_case("openrouter");
            let models = if is_openrouter {
                &draft.available_models.openrouter
            } else {
                &draft.available_models.ollama
            };
            let current = if is_openrouter {
                &draft.openrouter_model
            } else {
                &draft.ollama_model
            };
            let model_refs: Vec<&str> = models.iter().map(|s| s.as_str()).collect();
            (
                "Model Selection",
                format!("Set the model to use for provider {}:", draft.provider),
                format_choice_list(&model_refs, current),
            )
        }
        WizardStep::OpenRouterKey => (
            "OpenRouter API Key",
            "Optional OpenRouter API key (saved to .env if set). Leave blank to keep existing.".to_string(),
            ratatui::text::Text::from(format!("\n{}\n", wizard::mask_secret(input))),
        ),
        WizardStep::Thinking => (
            "Thinking Level",
            "Choose default thinking depth (Up/Down/Tab to cycle):".to_string(),
            format_choice_list(&["low", "balanced", "deep"], &draft.thinking_level),
        ),
        WizardStep::NightSleepStart => (
            "Night Sleep Start",
            "Local hour (0-23) when the nightly sleep window starts:".to_string(),
            ratatui::text::Text::from(format!("\n{}\n", input)),
        ),
        WizardStep::NightSleepEnd => (
            "Night Sleep End",
            "Local hour (0-23) when the nightly sleep window ends:".to_string(),
            ratatui::text::Text::from(format!("\n{}\n", input)),
        ),
        WizardStep::Safety => (
            "Safety Profile",
            "Choose safety profile (Up/Down/Tab to cycle):\n\nstrict: deny shell/wasm by default\npower-user: allow shell/wasm with approvals".to_string(),
            format_choice_list(&["strict", "power-user"], &draft.safety_profile),
        ),
        WizardStep::ApprovalMode => (
            "Tool Approval Mode",
            "How should the agent ask for approval before running tools?\n\nsafer     — ask for every tool call\nbalanced  — read-only tools free; writes/shell require approval (default)\nautonomous — never ask; all gated tools auto-approved within workspace".to_string(),
            format_choice_list(
                &["safer", "balanced", "autonomous"],
                &draft.approval_mode,
            ),
        ),
        WizardStep::ApiKeys => (
            "API Keys (optional)",
            "Brave Search API key for high-quality web search results.\nLeave blank to use the free DuckDuckGo fallback instead.\nKey is saved to config (never logged).".to_string(),
            ratatui::text::Text::from(format!("\n{}\n", wizard::mask_secret(input))),
        ),
        WizardStep::Messaging => (
            "Messaging Integration",
            "Toggle Telegram integration (Up/Down/Tab to cycle):".to_string(),
            format_choice_list(&["enabled", "disabled"], if draft.telegram_enabled { "enabled" } else { "disabled" }),
        ),
        WizardStep::TelegramToken => (
            "Telegram Token",
            "Optional Telegram bot token (saved to .env if set). Leave blank to keep existing.".to_string(),
            ratatui::text::Text::from(format!("\n{}\n", wizard::mask_secret(input))),
        ),
        WizardStep::Summary => {
            let docker_ok = wizard::command_exists("docker");
            let memory_path = PathBuf::from(&draft.workspace_path)
                .join(".aigent")
                .join("memory");
            let sandbox_status = if aigent_exec::sandbox::is_active() { "active (platform sandbox compiled in)" } else { "inactive (rebuild with sandbox feature)" };
            (
                "Setup Summary",
                String::new(),
                ratatui::text::Text::from(format!(
                    "User: {}\nBot: {}\nProvider: {}\nModel: {}\nThinking: {}\nData directory: {}\nSleep mode: nightly ({}:00-{}:00 local)\nSafety: {}\nApproval mode: {}\nBrave Search key: {}\nTelegram: {}\nSandbox: {}\nDocker available: {}\nMemory path: {}\n\nMemory contract:\n- Event-log memory is canonical\n- Nightly sleep distills and promotes important memories\n- Core memories shape personality over time\n\nPress Enter to finalize setup.",
                    draft.user_name,
                    draft.bot_name,
                    draft.provider,
                    if draft.provider.eq_ignore_ascii_case("openrouter") {
                        &draft.openrouter_model
                    } else {
                        &draft.ollama_model
                    },
                    draft.thinking_level,
                    draft.workspace_path,
                    draft.night_sleep_start_hour,
                    draft.night_sleep_end_hour,
                    draft.safety_profile,
                    draft.approval_mode,
                    if draft.brave_api_key.trim().is_empty() { "not set (DuckDuckGo fallback)" } else { "configured" },
                    if draft.telegram_enabled {
                        "enabled"
                    } else {
                        "disabled"
                    },
                    sandbox_status,
                    if docker_ok { "yes" } else { "no" },
                    memory_path.display()
                ) + "\n\nBuilt-in tools (8): web_search, read_file, write_file, list_directory, \nrun_shell, calendar_add_event, remind_me, draft_email"),
            )
        }
    }
}

