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

/// Read crossterm events using a non-blocking channel so we never call
/// synchronous `event::poll()` inside the async `tokio::select!` loop.
/// This prevents a 10ms block on every iteration and lets the spinner
/// tick truly in parallel with rapid backend events.
fn spawn_crossterm_reader() -> mpsc::UnboundedReceiver<CrosstermEvent> {
    let (tx, rx) = mpsc::unbounded_channel();
    std::thread::spawn(move || {
        while let Ok(ev) = event::read() {
            if tx.send(ev).is_err() {
                break;
            }
        }
    });
    rx
}

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
    let mut tick_interval = tokio::time::interval(Duration::from_millis(50));
    // First tick fires immediately; skip it so we don't double-draw on entry.
    tick_interval.tick().await;
    // Read crossterm events on a dedicated OS thread so we never block the
    // async runtime with synchronous `event::poll()`.  This guarantees the
    // spinner tick and backend branches get fair scheduling.
    let mut term_rx = spawn_crossterm_reader();

    let result = async {
        loop {
            if !selection_mode {
                terminal.draw(|f| app.draw(f))?;
            }

            tokio::select! {
                // Drain all pending backend events before anything else so
                // the UI stays responsive during rapid streaming.
                backend = app.backend_rx.recv() => {
                    if let Some(backend) = backend {
                        let _ = app.update(AppEvent::Backend(backend));
                    }
                }
                _ = tick_interval.tick() => {
                    let _ = app.update(AppEvent::Tick);
                }
                key_event = term_rx.recv() => {
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
