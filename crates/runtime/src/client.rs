use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Result, bail};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::sync::mpsc;
use tracing::warn;

use crate::{ClientCommand, DaemonStatus, ServerEvent};

#[derive(Debug, Clone)]
pub struct DaemonClient {
    socket_path: PathBuf,
}

impl DaemonClient {
    pub fn new(socket_path: impl AsRef<Path>) -> Self {
        Self {
            socket_path: socket_path.as_ref().to_path_buf(),
        }
    }

    pub async fn connect_with_backoff(&self, max_attempts: usize) -> Result<()> {
        let mut delay = Duration::from_millis(100);
        for attempt in 0..max_attempts.max(1) {
            match UnixStream::connect(&self.socket_path).await {
                Ok(_) => return Ok(()),
                Err(err) => {
                    if attempt + 1 == max_attempts.max(1) {
                        return Err(err.into());
                    }
                    warn!(attempt, ?err, "daemon connect failed; retrying");
                    tokio::time::sleep(delay).await;
                    delay = (delay * 2).min(Duration::from_secs(2));
                }
            }
        }
        Ok(())
    }

    pub async fn stream_submit(
        &self,
        user: String,
        source: impl Into<String>,
        tx: mpsc::UnboundedSender<crate::BackendEvent>,
    ) -> Result<()> {
        self.stream_command(
            ClientCommand::SubmitTurn {
                user,
                source: source.into(),
            },
            tx,
        )
        .await
    }

    /// Subscribe to broadcast events from external sources (Telegram, etc.)
    /// Runs until the connection drops or the channel closes.
    pub async fn subscribe(&self, tx: mpsc::UnboundedSender<crate::BackendEvent>) -> Result<()> {
        let stream = UnixStream::connect(&self.socket_path).await?;
        let (read_half, mut write_half) = stream.into_split();

        let request = serde_json::to_string(&ClientCommand::Subscribe)?;
        write_half.write_all(request.as_bytes()).await?;
        write_half.write_all(b"\n").await?;
        write_half.flush().await?;

        let mut reader = BufReader::new(read_half);
        let mut line = String::new();
        loop {
            line.clear();
            let bytes = reader.read_line(&mut line).await?;
            if bytes == 0 {
                break;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let event: ServerEvent = match serde_json::from_str(trimmed) {
                Ok(e) => e,
                Err(err) => {
                    warn!("subscribe: bad json: {err}");
                    continue;
                }
            };
            if let ServerEvent::Backend(be) = event {
                if tx.send(be).is_err() {
                    break; // TUI has dropped the receiver
                }
            }
        }
        Ok(())
    }

    pub async fn stream_command(
        &self,
        command: ClientCommand,
        tx: mpsc::UnboundedSender<crate::BackendEvent>,
    ) -> Result<()> {
        let stream = UnixStream::connect(&self.socket_path).await?;
        let (read_half, mut write_half) = stream.into_split();

        let request = serde_json::to_string(&command)?;
        write_half.write_all(request.as_bytes()).await?;
        write_half.write_all(b"\n").await?;
        write_half.flush().await?;

        let mut reader = BufReader::new(read_half);
        let mut line = String::new();
        let mut saw_terminal = false;
        loop {
            line.clear();
            let bytes = reader.read_line(&mut line).await?;
            if bytes == 0 {
                break;
            }
            let event: ServerEvent = serde_json::from_str(line.trim())?;
            if let ServerEvent::Backend(be) = event {
                let done = matches!(
                    be,
                    crate::BackendEvent::Done | crate::BackendEvent::Error(_)
                );
                let _ = tx.send(be);
                if done {
                    saw_terminal = true;
                    break;
                }
            }
        }

        if !saw_terminal {
            bail!(
                "daemon stream ended before terminal event (Done/Error); check daemon logs"
            );
        }

        Ok(())
    }

    pub async fn get_status(&self) -> Result<DaemonStatus> {
        let events = self.request_events(ClientCommand::GetStatus).await?;
        for event in events {
            if let ServerEvent::Status(status) = event {
                return Ok(status);
            }
        }
        bail!("daemon status response missing")
    }

    pub async fn get_memory_peek(&self, limit: usize) -> Result<Vec<String>> {
        let events = self
            .request_events(ClientCommand::GetMemoryPeek { limit })
            .await?;
        for event in events {
            if let ServerEvent::MemoryPeek(peek) = event {
                return Ok(peek);
            }
        }
        Ok(Vec::new())
    }

    pub async fn graceful_shutdown(&self) -> Result<()> {
        let _ = self.request_events(ClientCommand::Shutdown).await?;
        Ok(())
    }

    pub async fn reload_config(&self) -> Result<()> {
        let events = self.request_events(ClientCommand::ReloadConfig).await?;
        for event in events {
            if let ServerEvent::Ack(_) = event {
                return Ok(());
            }
        }
        bail!("daemon reload config response missing")
    }

    pub async fn run_sleep_cycle(&self) -> Result<String> {
        let events = self
            .request_events(ClientCommand::RunSleepCycle)
            .await?;
        for event in events {
            if let ServerEvent::Ack(msg) = event {
                return Ok(msg);
            }
        }
        bail!("daemon sleep cycle response missing")
    }

    pub async fn list_tools(&self) -> Result<Vec<aigent_tools::ToolSpec>> {
        let events = self.request_events(ClientCommand::ListTools).await?;
        for event in events {
            if let ServerEvent::ToolList(specs) = event {
                return Ok(specs);
            }
        }
        Ok(Vec::new())
    }

    pub async fn execute_tool(
        &self,
        name: &str,
        args: std::collections::HashMap<String, String>,
    ) -> Result<(bool, String)> {
        let events = self
            .request_events(ClientCommand::ExecuteTool {
                name: name.to_string(),
                args,
            })
            .await?;
        for event in events {
            if let ServerEvent::ToolResult { success, output } = event {
                return Ok((success, output));
            }
        }
        bail!("daemon tool execution response missing")
    }

    async fn request_events(&self, command: ClientCommand) -> Result<Vec<ServerEvent>> {
        let stream = UnixStream::connect(&self.socket_path).await?;
        let (read_half, mut write_half) = stream.into_split();

        let request = serde_json::to_string(&command)?;
        write_half.write_all(request.as_bytes()).await?;
        write_half.write_all(b"\n").await?;
        write_half.flush().await?;

        let mut reader = BufReader::new(read_half);
        let mut line = String::new();
        let mut events = Vec::new();
        loop {
            line.clear();
            let bytes = reader.read_line(&mut line).await?;
            if bytes == 0 {
                break;
            }
            let event: ServerEvent = serde_json::from_str(line.trim())?;
            let done = matches!(
                event,
                ServerEvent::Ack(_)
                    | ServerEvent::Status(_)
                    | ServerEvent::MemoryPeek(_)
                    | ServerEvent::ToolList(_)
                    | ServerEvent::ToolResult { .. }
            );
            events.push(event);
            if done {
                break;
            }
        }

        Ok(events)
    }
}
