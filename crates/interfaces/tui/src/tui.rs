use std::future::Future;
use std::io;
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event as CrosstermEvent, KeyEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use tokio::sync::mpsc;
use tracing::debug;

use crate::app::App;
use crate::app::UiCommand;
use crate::events::AppEvent;

pub fn banner() -> &'static str {
    r#"
    _    _                  _
   / \  (_) __ _  ___ _ __ | |_
  / _ \ | |/ _` |/ _ \ '_ \| __|
 / ___ \| | (_| |  __/ | | | |_
/_/   \_\_|\__, |\___|_| |_|\__|
           |___/

Aigent is online
"#
}

pub async fn run_app(app: &mut App) -> Result<()> {
    run_app_with(app, |_| async { Ok(()) }).await
}

pub async fn run_app_with<F, Fut>(app: &mut App, mut on_command: F) -> Result<()>
where
    F: FnMut(UiCommand) -> Fut,
    Fut: Future<Output = Result<()>>,
{
    let mut stdout = io::stdout();
    enable_raw_mode()?;
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;
    let mut selection_mode = false;
    let mut previous_status: Option<String> = None;

    let result = async {
        loop {
            if !selection_mode {
                terminal.draw(|f| app.draw(f))?;
            }

            tokio::select! {
                backend = app.backend_rx.recv() => {
                    if let Some(backend) = backend {
                        let _ = app.update(AppEvent::Backend(backend));
                    }
                }
                _ = tokio::time::sleep(Duration::from_millis(50)) => {
                    let _ = app.update(AppEvent::Tick);
                }
                key_event = async {
                    if event::poll(Duration::from_millis(10)).unwrap_or(false) {
                        event::read().ok()
                    } else {
                        None
                    }
                } => {
                    if let Some(CrosstermEvent::Key(key)) = key_event {
                        let force_quit = key.code == crossterm::event::KeyCode::Char('c')
                            && key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL)
                            && key.kind == KeyEventKind::Press;

                        if force_quit {
                            if selection_mode {
                                execute!(terminal.backend_mut(), EnableMouseCapture)?;
                                if let Some(status) = previous_status.take() {
                                    app.set_status(status);
                                }
                                selection_mode = false;
                            }
                            break;
                        }

                        let select_shortcut = key.code == crossterm::event::KeyCode::Char('s')
                            && (key.modifiers.contains(crossterm::event::KeyModifiers::ALT)
                                || (key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL)
                                    && key.modifiers.contains(crossterm::event::KeyModifiers::SHIFT)))
                            && key.kind == KeyEventKind::Press;

                        if selection_mode {
                            if select_shortcut || matches!(key.code, crossterm::event::KeyCode::Esc | crossterm::event::KeyCode::Enter) {
                                execute!(terminal.backend_mut(), EnableMouseCapture)?;
                                if let Some(status) = previous_status.take() {
                                    app.set_status(status);
                                }
                                selection_mode = false;
                            }
                            continue;
                        }

                        if select_shortcut
                        {
                            previous_status = Some(app.state.status.clone());
                            app.set_status("selection mode: drag to highlight, copy with terminal shortcut, Alt+S/Esc/Enter to return");
                            execute!(terminal.backend_mut(), DisableMouseCapture)?;
                            selection_mode = true;
                            continue;
                        }

                        if key.kind != KeyEventKind::Press {
                            continue;
                        }
                        if let Some(command) = app.update(AppEvent::Key(key)) {
                            if matches!(command, UiCommand::Quit) {
                                break;
                            }
                            on_command(command).await?;
                        }
                    } else if let Some(CrosstermEvent::Mouse(mouse)) = key_event {
                        let _ = app.update(AppEvent::Mouse(mouse));
                    } else if let Some(CrosstermEvent::Resize(w, h)) = key_event {
                        let _ = app.update(AppEvent::Resize(w, h));
                    }
                }
            }
        }
        Ok(()) as Result<()>
    }
    .await;

    debug!("restoring terminal state");
    if selection_mode {
        execute!(terminal.backend_mut(), EnableMouseCapture)?;
    }
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    result
}

pub fn create_backend_channel() -> (
    mpsc::UnboundedSender<aigent_runtime::BackendEvent>,
    mpsc::UnboundedReceiver<aigent_runtime::BackendEvent>,
) {
    mpsc::unbounded_channel()
}
