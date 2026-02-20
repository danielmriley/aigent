import re

with open('crates/aigent-app/src/main.rs', 'r') as f:
    content = f.read()

# Update handle_interactive_input signature
sig_old = """async fn handle_interactive_input(
    runtime: &mut AgentRuntime,
    memory: &mut MemoryManager,
    line: &str,
    turn_count: &mut usize,
    recent_turns: &mut VecDeque<ConversationTurn>,
) -> Result<InputOutcome> {"""

sig_new = """async fn handle_interactive_input(
    runtime: &mut AgentRuntime,
    memory: &mut MemoryManager,
    line: &str,
    turn_count: &mut usize,
    recent_turns: &mut VecDeque<ConversationTurn>,
    tx: Option<tokio::sync::mpsc::Sender<String>>,
) -> Result<InputOutcome> {"""

content = content.replace(sig_old, sig_new)

# Update respond_and_remember call
call_old = """    let mut messages = Vec::new();
    let recent_context = recent_turns.iter().cloned().collect::<Vec<_>>();
    let reply = runtime
        .respond_and_remember(memory, line, &recent_context)
        .await?;
    messages.push(format!("aigent> {reply}"));"""

call_new = """    let mut messages = Vec::new();
    let recent_context = recent_turns.iter().cloned().collect::<Vec<_>>();
    let reply = if let Some(tx) = tx {
        runtime
            .respond_and_remember_stream(memory, line, &recent_context, tx)
            .await?
    } else {
        runtime
            .respond_and_remember(memory, line, &recent_context)
            .await?
    };
    messages.push(format!("aigent> {reply}"));"""

content = content.replace(call_old, call_new)

# Update handle_interactive_input call in run_interactive_line_session
line_call_old = """        let outcome =
            handle_interactive_input(runtime, memory, line, &mut turn_count, &mut recent_turns)
                .await?;"""

line_call_new = """        let outcome =
            handle_interactive_input(runtime, memory, line, &mut turn_count, &mut recent_turns, None)
                .await?;"""

content = content.replace(line_call_old, line_call_new)

# Update handle_interactive_input call in run_interactive_session
tui_call_old = """                                let mut outcome_future = Box::pin(handle_interactive_input(
                                    runtime,
                                    memory,
                                    &line,
                                    &mut turn_count,
                                    &mut recent_turns,
                                ));"""

tui_call_new = """                                let (tx_chunk, mut rx_chunk) = tokio::sync::mpsc::channel(100);
                                let tx_llm_chunk = tx.clone();
                                tokio::spawn(async move {
                                    while let Some(chunk) = rx_chunk.recv().await {
                                        let _ = tx_llm_chunk.send(tui::Event::LlmChunk(chunk)).await;
                                    }
                                });

                                let mut outcome_future = Box::pin(handle_interactive_input(
                                    runtime,
                                    memory,
                                    &line,
                                    &mut turn_count,
                                    &mut recent_turns,
                                    Some(tx_chunk),
                                ));"""

content = content.replace(tui_call_old, tui_call_new)

with open('crates/aigent-app/src/main.rs', 'w') as f:
    f.write(content)
