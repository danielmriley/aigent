use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Result, bail};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Flex, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Wrap};

use aigent_config::AppConfig;

pub fn center(area: Rect, horizontal: Constraint, vertical: Constraint) -> Rect {
    let [area] = Layout::horizontal([horizontal])
        .flex(Flex::Center)
        .areas(area);
    let [area] = Layout::vertical([vertical]).flex(Flex::Center).areas(area);
    area
}

#[derive(Debug, Clone, Default)]
pub struct AvailableModels {
    pub ollama: Vec<String>,
    pub openrouter: Vec<String>,
}

pub fn run_onboarding(config: &mut AppConfig, models: AvailableModels) -> Result<()> {
    run_setup_wizard(config, SetupMode::Onboarding, models)
}

pub fn run_configuration(config: &mut AppConfig, models: AvailableModels) -> Result<()> {
    run_setup_wizard(config, SetupMode::Configuration, models)
}

fn run_setup_wizard(
    config: &mut AppConfig,
    mode: SetupMode,
    models: AvailableModels,
) -> Result<()> {
    if io::stdin().is_terminal() && io::stdout().is_terminal() {
        return run_onboarding_tui(config, mode, models);
    }

    run_onboarding_prompt(config, mode, models)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SetupMode {
    Onboarding,
    Configuration,
}

fn run_onboarding_prompt(
    config: &mut AppConfig,
    mode: SetupMode,
    _models: AvailableModels,
) -> Result<()> {
    if mode == SetupMode::Onboarding {
        println!("onboarding");
    } else {
        println!("configuration wizard");
    }

    let bot_name = prompt(
        "Choose a name for your bot",
        if config.agent.name.trim().is_empty() {
            "Aigent"
        } else {
            &config.agent.name
        },
    )?;
    config.agent.name = bot_name;

    let provider = prompt(
        "LLM provider (ollama/openrouter)",
        if config.llm.provider.trim().is_empty() {
            "ollama"
        } else {
            &config.llm.provider
        },
    )?;
    let provider = normalize_provider(&provider)?;
    if provider == "openrouter" {
        config.llm.provider = "openrouter".to_string();
        let model = prompt("OpenRouter model", &config.llm.openrouter_model)?;
        config.llm.openrouter_model = model;

        let key = prompt_optional("OpenRouter API key (leave blank to keep existing)")?;
        if let Some(key) = key {
            upsert_env_value(Path::new(".env"), "OPENROUTER_API_KEY", &key)?;
        }
    } else {
        config.llm.provider = "ollama".to_string();
        let model = prompt("Ollama model", &config.llm.ollama_model)?;
        config.llm.ollama_model = model;
    }

    let thinking_level = prompt(
        "Thinking level (low/balanced/deep)",
        &config.agent.thinking_level,
    )?;
    config.agent.thinking_level = normalize_thinking_level(&thinking_level)?;

    let workspace = prompt("Workspace path", &config.agent.workspace_path)?;
    config.agent.workspace_path = workspace;

    let sleep_start = prompt(
        "Night sleep start hour (0-23)",
        &config.memory.night_sleep_start_hour.to_string(),
    )?;
    config.memory.night_sleep_start_hour = parse_hour(&sleep_start)?;
    let sleep_end = prompt(
        "Night sleep end hour (0-23)",
        &config.memory.night_sleep_end_hour.to_string(),
    )?;
    config.memory.night_sleep_end_hour = parse_hour(&sleep_end)?;

    let safety_profile = prompt("Safety profile (strict/power-user)", "strict")?;
    let safety_profile = normalize_safety_profile(&safety_profile)?;
    apply_safety_profile(config, &safety_profile);

    let telegram_enabled = prompt_bool(
        "Enable Telegram integration (yes/no)",
        config.integrations.telegram_enabled,
    )?;
    config.integrations.telegram_enabled = telegram_enabled;
    if telegram_enabled {
        let token = prompt_optional("Telegram bot token (leave blank to keep existing)")?;
        if let Some(token) = token {
            upsert_env_value(Path::new(".env"), "TELEGRAM_BOT_TOKEN", &token)?;
        }
    }

    let workspace = resolve_workspace_path(&config.agent.workspace_path)?;
    apply_memory_contract(config, &workspace)?;
    config.onboarding.completed = true;

    Ok(())
}

fn run_onboarding_tui(
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

    let mut step = WizardStep::Welcome;
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
                KeyCode::Esc => bail!("onboarding cancelled"),
                KeyCode::Backspace => {
                    if step.is_text_step() {
                        input.pop();
                    }
                }
                KeyCode::Up => {
                    draft.choice_prev(step);
                    input = draft.value_for_step(step);
                }
                KeyCode::Down => {
                    draft.choice_next(step);
                    input = draft.value_for_step(step);
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
                            upsert_env_value(
                                Path::new(".env"),
                                "OPENROUTER_API_KEY",
                                draft.openrouter_key.trim(),
                            )?;
                        }
                        if draft.telegram_enabled && !draft.telegram_token.trim().is_empty() {
                            upsert_env_value(
                                Path::new(".env"),
                                "TELEGRAM_BOT_TOKEN",
                                draft.telegram_token.trim(),
                            )?;
                        }
                        break;
                    }

                    if step == WizardStep::Welcome {
                        step = next_step(step, &draft);
                        input = draft.value_for_step(step);
                        error_message = None;
                        continue;
                    }

                    if let Err(err) = draft.commit_step(step, &input) {
                        error_message = Some(err.to_string());
                        continue;
                    }
                    step = next_step(step, &draft);
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WizardStep {
    Welcome,
    BotName,
    Workspace,
    Provider,
    Model,
    OpenRouterKey,
    Thinking,
    NightSleepStart,
    NightSleepEnd,
    Safety,
    Messaging,
    TelegramToken,
    Summary,
}

impl WizardStep {
    fn is_text_step(self) -> bool {
        matches!(
            self,
            WizardStep::BotName
                | WizardStep::Workspace
                | WizardStep::OpenRouterKey
                | WizardStep::NightSleepStart
                | WizardStep::NightSleepEnd
                | WizardStep::TelegramToken
        )
    }
}

#[derive(Debug, Clone)]
struct OnboardingDraft {
    bot_name: String,
    workspace_path: String,
    provider: String,
    ollama_model: String,
    openrouter_model: String,
    openrouter_key: String,
    thinking_level: String,
    night_sleep_start_hour: u8,
    night_sleep_end_hour: u8,
    safety_profile: String,
    telegram_enabled: bool,
    telegram_token: String,
    available_models: AvailableModels,
}

impl OnboardingDraft {
    fn from_config(config: &AppConfig, models: AvailableModels) -> Self {
        Self {
            bot_name: config.agent.name.clone(),
            workspace_path: config.agent.workspace_path.clone(),
            provider: config.llm.provider.clone(),
            ollama_model: config.llm.ollama_model.clone(),
            openrouter_model: config.llm.openrouter_model.clone(),
            openrouter_key: String::new(),
            thinking_level: config.agent.thinking_level.clone(),
            night_sleep_start_hour: config.memory.night_sleep_start_hour,
            night_sleep_end_hour: config.memory.night_sleep_end_hour,
            safety_profile: if config.safety.allow_shell || config.safety.allow_wasm {
                "power-user".to_string()
            } else {
                "strict".to_string()
            },
            telegram_enabled: config.integrations.telegram_enabled,
            telegram_token: String::new(),
            available_models: models,
        }
    }

    fn value_for_step(&self, step: WizardStep) -> String {
        match step {
            WizardStep::Welcome => String::new(),
            WizardStep::BotName => self.bot_name.clone(),
            WizardStep::Workspace => self.workspace_path.clone(),
            WizardStep::Provider => self.provider.clone(),
            WizardStep::Model => {
                if self.provider.eq_ignore_ascii_case("openrouter") {
                    self.openrouter_model.clone()
                } else {
                    self.ollama_model.clone()
                }
            }
            WizardStep::OpenRouterKey => self.openrouter_key.clone(),
            WizardStep::Thinking => self.thinking_level.clone(),
            WizardStep::NightSleepStart => self.night_sleep_start_hour.to_string(),
            WizardStep::NightSleepEnd => self.night_sleep_end_hour.to_string(),
            WizardStep::Safety => self.safety_profile.clone(),
            WizardStep::Messaging => {
                if self.telegram_enabled {
                    "enabled".to_string()
                } else {
                    "disabled".to_string()
                }
            }
            WizardStep::TelegramToken => self.telegram_token.clone(),
            WizardStep::Summary => String::new(),
        }
    }

    fn commit_step(&mut self, step: WizardStep, input: &str) -> Result<()> {
        let value = input.trim();
        match step {
            WizardStep::BotName => {
                if value.is_empty() {
                    bail!("bot name cannot be empty");
                }
                self.bot_name = value.to_string();
            }
            WizardStep::Workspace => {
                if value.is_empty() {
                    bail!("workspace path cannot be empty");
                }
                self.workspace_path = value.to_string();
            }
            WizardStep::Model => {
                if value.is_empty() {
                    bail!("model cannot be empty");
                }
                if self.provider.eq_ignore_ascii_case("openrouter") {
                    self.openrouter_model = value.to_string();
                } else {
                    self.ollama_model = value.to_string();
                }
            }
            WizardStep::OpenRouterKey => {
                self.openrouter_key = value.to_string();
            }
            WizardStep::NightSleepStart => {
                self.night_sleep_start_hour = parse_hour(value)?;
            }
            WizardStep::NightSleepEnd => {
                self.night_sleep_end_hour = parse_hour(value)?;
            }
            WizardStep::TelegramToken => {
                self.telegram_token = value.to_string();
            }
            _ => {}
        }

        Ok(())
    }

    fn choice_next(&mut self, step: WizardStep) {
        match step {
            WizardStep::Provider => {
                self.provider = if self.provider.eq_ignore_ascii_case("openrouter") {
                    "ollama".to_string()
                } else {
                    "openrouter".to_string()
                }
            }
            WizardStep::Model => {
                let is_openrouter = self.provider.eq_ignore_ascii_case("openrouter");
                let models = if is_openrouter {
                    &self.available_models.openrouter
                } else {
                    &self.available_models.ollama
                };

                if models.is_empty() {
                    return;
                }

                let current = if is_openrouter {
                    &self.openrouter_model
                } else {
                    &self.ollama_model
                };

                let idx = models.iter().position(|m| m == current).unwrap_or(0);
                let next_idx = (idx + 1) % models.len();
                let next_model = models[next_idx].clone();

                if is_openrouter {
                    self.openrouter_model = next_model;
                } else {
                    self.ollama_model = next_model;
                }
            }
            WizardStep::Thinking => {
                self.thinking_level = match self.thinking_level.to_lowercase().as_str() {
                    "low" => "balanced".to_string(),
                    "balanced" => "deep".to_string(),
                    _ => "low".to_string(),
                }
            }
            WizardStep::Safety => {
                self.safety_profile = if self.safety_profile.eq_ignore_ascii_case("power-user") {
                    "strict".to_string()
                } else {
                    "power-user".to_string()
                }
            }
            WizardStep::Messaging => {
                self.telegram_enabled = !self.telegram_enabled;
            }
            _ => {}
        }
    }

    fn choice_prev(&mut self, step: WizardStep) {
        match step {
            WizardStep::Model => {
                let is_openrouter = self.provider.eq_ignore_ascii_case("openrouter");
                let models = if is_openrouter {
                    &self.available_models.openrouter
                } else {
                    &self.available_models.ollama
                };

                if models.is_empty() {
                    return;
                }

                let current = if is_openrouter {
                    &self.openrouter_model
                } else {
                    &self.ollama_model
                };

                let idx = models.iter().position(|m| m == current).unwrap_or(0);
                let prev_idx = if idx == 0 { models.len() - 1 } else { idx - 1 };
                let prev_model = models[prev_idx].clone();

                if is_openrouter {
                    self.openrouter_model = prev_model;
                } else {
                    self.ollama_model = prev_model;
                }
            }
            WizardStep::Thinking => {
                self.thinking_level = match self.thinking_level.to_lowercase().as_str() {
                    "low" => "deep".to_string(),
                    "balanced" => "low".to_string(),
                    _ => "balanced".to_string(),
                }
            }
            _ => self.choice_next(step),
        }
    }

    fn apply(&self, config: &mut AppConfig) -> Result<()> {
        config.agent.name = self.bot_name.clone();
        config.agent.workspace_path = self.workspace_path.clone();
        config.llm.provider = normalize_provider(&self.provider)?;
        config.llm.ollama_model = self.ollama_model.clone();
        config.llm.openrouter_model = self.openrouter_model.clone();
        config.agent.thinking_level = normalize_thinking_level(&self.thinking_level)?;
        config.memory.night_sleep_start_hour = self.night_sleep_start_hour;
        config.memory.night_sleep_end_hour = self.night_sleep_end_hour;
        let safety_profile = normalize_safety_profile(&self.safety_profile)?;
        apply_safety_profile(config, &safety_profile);
        config.integrations.telegram_enabled = self.telegram_enabled;

        let workspace = resolve_workspace_path(&config.agent.workspace_path)?;
        apply_memory_contract(config, &workspace)?;
        config.onboarding.completed = true;

        Ok(())
    }
}

fn next_step(current: WizardStep, draft: &OnboardingDraft) -> WizardStep {
    match current {
        WizardStep::Welcome => WizardStep::BotName,
        WizardStep::BotName => WizardStep::Workspace,
        WizardStep::Workspace => WizardStep::Provider,
        WizardStep::Provider => WizardStep::Model,
        WizardStep::Model => {
            if draft.provider.eq_ignore_ascii_case("openrouter") {
                WizardStep::OpenRouterKey
            } else {
                WizardStep::Thinking
            }
        }
        WizardStep::OpenRouterKey => WizardStep::Thinking,
        WizardStep::Thinking => WizardStep::NightSleepStart,
        WizardStep::NightSleepStart => WizardStep::NightSleepEnd,
        WizardStep::NightSleepEnd => WizardStep::Safety,
        WizardStep::Safety => WizardStep::Messaging,
        WizardStep::Messaging => {
            if draft.telegram_enabled {
                WizardStep::TelegramToken
            } else {
                WizardStep::Summary
            }
        }
        WizardStep::TelegramToken => WizardStep::Summary,
        WizardStep::Summary => WizardStep::Summary,
    }
}

fn step_number(step: WizardStep) -> usize {
    match step {
        WizardStep::Welcome => 1,
        WizardStep::BotName => 2,
        WizardStep::Workspace => 3,
        WizardStep::Provider => 4,
        WizardStep::Model => 5,
        WizardStep::OpenRouterKey => 6,
        WizardStep::Thinking => 7,
        WizardStep::NightSleepStart => 8,
        WizardStep::NightSleepEnd => 9,
        WizardStep::Safety => 10,
        WizardStep::Messaging => 11,
        WizardStep::TelegramToken => 12,
        WizardStep::Summary => 13,
    }
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

    terminal.draw(|frame| {
        let area = frame.area();

        // Center the wizard box
        let wizard_area = center(area, Constraint::Length(70), Constraint::Length(24));

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
                        .fg(Color::LightBlue)
                        .add_modifier(Modifier::BOLD),
                ))
                .centered(),
            )
            .title_bottom(
                ratatui::text::Line::from(Span::styled(
                    format!(" Step {}/13 ", step_number(step)),
                    Style::default().fg(Color::DarkGray),
                ))
                .right_aligned(),
            );

        frame.render_widget(block.clone(), wizard_area);

        // Inner area
        let inner_area = wizard_area.inner(ratatui::layout::Margin {
            vertical: 1,
            horizontal: 2,
        });

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2), // Title
                Constraint::Min(3),    // Description
                Constraint::Length(5), // Input box
                Constraint::Length(1), // Spacer
                Constraint::Length(2), // Footer/Error
            ])
            .split(inner_area);

        // Title
        let title_widget = Paragraph::new(title)
            .style(
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )
            .alignment(ratatui::layout::Alignment::Center);
        frame.render_widget(title_widget, chunks[0]);

        // Description
        let desc_widget = Paragraph::new(description)
            .style(Style::default().fg(Color::Gray))
            .wrap(Wrap { trim: true })
            .alignment(ratatui::layout::Alignment::Center);
        frame.render_widget(desc_widget, chunks[1]);

        // Input box
        if step != WizardStep::Welcome && step != WizardStep::Summary {
            let input_block = Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Cyan));
            let input_widget = Paragraph::new(input_display)
                .block(input_block)
                .style(Style::default().fg(Color::White))
                .wrap(Wrap { trim: false })
                .alignment(ratatui::layout::Alignment::Center);

            // Center the input box horizontally
            let input_area = center(chunks[2], Constraint::Percentage(80), Constraint::Length(5));
            frame.render_widget(input_widget, input_area);
        } else if step == WizardStep::Summary {
            // For summary, we use the input box area for more description
            let summary_widget = Paragraph::new(input_display)
                .style(Style::default().fg(Color::White))
                .wrap(Wrap { trim: true })
                .alignment(ratatui::layout::Alignment::Left);

            let summary_area = chunks[1].union(chunks[2]).union(chunks[3]);
            frame.render_widget(summary_widget, summary_area);
        }

        // Footer
        let footer_text = if let Some(message) = error_message {
            Line::from(vec![
                Span::styled(
                    "Error: ",
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                ),
                Span::styled(message, Style::default().fg(Color::LightRed)),
            ])
        } else {
            Line::from(Span::styled(
                "Enter=continue • Up/Down/Tab=cycle • Esc=cancel",
                Style::default().fg(Color::DarkGray),
            ))
        };

        let footer_widget =
            Paragraph::new(footer_text).alignment(ratatui::layout::Alignment::Center);
        frame.render_widget(footer_widget, chunks[4]);
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
        WizardStep::Welcome => (
            "Welcome to Aigent",
            "This wizard configures identity, memory policy, and safety.\n\nPress Enter to begin.".to_string(),
            ratatui::text::Text::from(""),
        ),
        WizardStep::BotName => (
            "Bot Identity",
            "Choose a name for your bot. This is how it will identify itself.".to_string(),
            ratatui::text::Text::from(format!("\n{}\n", input)),
        ),
        WizardStep::Workspace => (
            "Workspace Path",
            "Set the workspace path for operations. This is where the bot will store its memory and configuration.".to_string(),
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
            ratatui::text::Text::from(format!("\n{}\n", input)),
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
        WizardStep::Messaging => (
            "Messaging Integration",
            "Toggle Telegram integration (Up/Down/Tab to cycle):".to_string(),
            format_choice_list(&["enabled", "disabled"], if draft.telegram_enabled { "enabled" } else { "disabled" }),
        ),
        WizardStep::TelegramToken => (
            "Telegram Token",
            "Optional Telegram bot token (saved to .env if set). Leave blank to keep existing.".to_string(),
            ratatui::text::Text::from(format!("\n{}\n", input)),
        ),
        WizardStep::Summary => {
            let docker_ok = command_exists("docker");
            let memory_path = PathBuf::from(&draft.workspace_path)
                .join(".aigent")
                .join("memory");
            (
                "Memory Contract & Summary",
                String::new(),
                ratatui::text::Text::from(format!(
                    "Bot: {}\nProvider: {}\nModel: {}\nThinking: {}\nWorkspace: {}\nSleep mode: nightly ({}:00-{}:00 local)\nSafety: {}\nTelegram: {}\nDocker available: {}\nMemory path: {}\n\nMemory contract:\n- Event-log memory is canonical\n- Nightly sleep distills and promotes important memories\n- Core memories shape personality over time\n\nPress Enter to finalize setup.",
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
                    if draft.telegram_enabled {
                        "enabled"
                    } else {
                        "disabled"
                    },
                    if docker_ok { "yes" } else { "no" },
                    memory_path.display()
                )),
            )
        }
    }
}

fn command_exists(command: &str) -> bool {
    std::process::Command::new(command)
        .arg("--version")
        .output()
        .is_ok()
}

fn parse_hour(raw: &str) -> Result<u8> {
    let value: u8 = raw
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("hour must be an integer between 0 and 23"))?;
    if value > 23 {
        bail!("hour must be between 0 and 23");
    }
    Ok(value)
}

fn apply_safety_profile(config: &mut AppConfig, profile: &str) {
    if profile.eq_ignore_ascii_case("power-user") {
        config.safety.approval_required = true;
        config.safety.allow_shell = true;
        config.safety.allow_wasm = true;
    } else {
        config.safety.approval_required = true;
        config.safety.allow_shell = false;
        config.safety.allow_wasm = false;
    }
}

fn normalize_provider(raw: &str) -> Result<String> {
    if raw.eq_ignore_ascii_case("openrouter") {
        Ok("openrouter".to_string())
    } else if raw.eq_ignore_ascii_case("ollama") {
        Ok("ollama".to_string())
    } else {
        bail!("provider must be one of: ollama, openrouter");
    }
}

fn normalize_thinking_level(raw: &str) -> Result<String> {
    if raw.eq_ignore_ascii_case("low") {
        Ok("low".to_string())
    } else if raw.eq_ignore_ascii_case("balanced") {
        Ok("balanced".to_string())
    } else if raw.eq_ignore_ascii_case("deep") {
        Ok("deep".to_string())
    } else {
        bail!("thinking level must be one of: low, balanced, deep");
    }
}

fn normalize_safety_profile(raw: &str) -> Result<String> {
    if raw.eq_ignore_ascii_case("strict") {
        Ok("strict".to_string())
    } else if raw.eq_ignore_ascii_case("power-user") || raw.eq_ignore_ascii_case("poweruser") {
        Ok("power-user".to_string())
    } else {
        bail!("safety profile must be one of: strict, power-user");
    }
}

fn resolve_workspace_path(raw: &str) -> Result<PathBuf> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        bail!("workspace path cannot be empty");
    }

    let home = std::env::var_os("HOME").map(PathBuf::from);
    let path = if trimmed == "~" {
        home.ok_or_else(|| anyhow::anyhow!("failed to resolve home directory"))?
    } else if let Some(rest) = trimmed.strip_prefix("~/") {
        home.ok_or_else(|| anyhow::anyhow!("failed to resolve home directory"))?
            .join(rest)
    } else {
        PathBuf::from(trimmed)
    };

    if !path.exists() {
        bail!("workspace path does not exist: {}", path.display());
    }
    if !path.is_dir() {
        bail!("workspace path must be a directory: {}", path.display());
    }

    Ok(path)
}

fn apply_memory_contract(config: &mut AppConfig, workspace: &Path) -> Result<()> {
    config.memory.backend = "eventlog".to_string();
    config.memory.auto_sleep_mode = "nightly".to_string();
    config.memory.core_rewrite_requires_approval = true;

    let memory_dir = workspace.join(".aigent").join("memory");
    fs::create_dir_all(&memory_dir)?;
    Ok(())
}

fn prompt(label: &str, default_value: &str) -> Result<String> {
    print!("{} [{}]: ", label, default_value);
    io::stdout().flush()?;

    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(default_value.to_string());
    }

    Ok(trimmed.to_string())
}

fn prompt_optional(label: &str) -> Result<Option<String>> {
    print!("{}: ", label);
    io::stdout().flush()?;

    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    Ok(Some(trimmed.to_string()))
}

fn prompt_bool(label: &str, default_value: bool) -> Result<bool> {
    let default_label = if default_value { "yes" } else { "no" };
    let value = prompt(label, default_label)?;
    parse_bool_like(&value)
}

fn parse_bool_like(raw: &str) -> Result<bool> {
    let normalized = raw.trim().to_lowercase();
    match normalized.as_str() {
        "y" | "yes" | "true" | "1" | "enabled" | "on" => Ok(true),
        "n" | "no" | "false" | "0" | "disabled" | "off" => Ok(false),
        _ => bail!("expected yes/no"),
    }
}

fn upsert_env_value(path: &Path, key: &str, value: &str) -> Result<()> {
    let existing = fs::read_to_string(path).unwrap_or_default();
    let mut updated = Vec::new();
    let mut replaced = false;

    for line in existing.lines() {
        if line.trim_start().starts_with(&format!("{key}=")) {
            updated.push(format!("{key}={value}"));
            replaced = true;
        } else {
            updated.push(line.to_string());
        }
    }

    if !replaced {
        updated.push(format!("{key}={value}"));
    }

    let content = format!("{}\n", updated.join("\n"));
    fs::write(path, content)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        normalize_provider, normalize_safety_profile, normalize_thinking_level, parse_bool_like,
        parse_hour,
    };

    #[test]
    fn normalizes_provider_values() {
        assert_eq!(normalize_provider("OLLAMA").unwrap(), "ollama");
        assert_eq!(normalize_provider("openrouter").unwrap(), "openrouter");
        assert!(normalize_provider("something-else").is_err());
    }

    #[test]
    fn normalizes_thinking_values() {
        assert_eq!(normalize_thinking_level("LOW").unwrap(), "low");
        assert_eq!(normalize_thinking_level("balanced").unwrap(), "balanced");
        assert_eq!(normalize_thinking_level("deep").unwrap(), "deep");
        assert!(normalize_thinking_level("medium").is_err());
    }

    #[test]
    fn normalizes_safety_profile_values() {
        assert_eq!(normalize_safety_profile("strict").unwrap(), "strict");
        assert_eq!(normalize_safety_profile("poweruser").unwrap(), "power-user");
        assert!(normalize_safety_profile("unsafe").is_err());
    }

    #[test]
    fn validates_sleep_hour_bounds() {
        assert_eq!(parse_hour("0").unwrap(), 0);
        assert_eq!(parse_hour("23").unwrap(), 23);
        assert!(parse_hour("24").is_err());
        assert!(parse_hour("-1").is_err());
    }

    #[test]
    fn parses_boolean_prompt_values() {
        assert!(parse_bool_like("yes").unwrap());
        assert!(parse_bool_like("enabled").unwrap());
        assert!(!parse_bool_like("no").unwrap());
        assert!(!parse_bool_like("off").unwrap());
        assert!(parse_bool_like("maybe").is_err());
    }
}
