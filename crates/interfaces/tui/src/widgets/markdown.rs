use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};
use std::sync::OnceLock;
use syntect::easy::HighlightLines;
use syntect::highlighting::{self, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::LinesWithEndings;

// ── lazy singletons for syntect ────────────────────────────────
fn syntax_set() -> &'static SyntaxSet {
    static SS: OnceLock<SyntaxSet> = OnceLock::new();
    SS.get_or_init(SyntaxSet::load_defaults_newlines)
}

fn highlight_theme() -> &'static highlighting::Theme {
    static TH: OnceLock<highlighting::Theme> = OnceLock::new();
    TH.get_or_init(|| {
        let ts = ThemeSet::load_defaults();
        ts.themes["base16-ocean.dark"].clone()
    })
}

// ── public entry point ─────────────────────────────────────────
pub fn render_markdown_lines(input: &str) -> Vec<Line<'static>> {
    if input.trim().is_empty() {
        return vec![Line::from(Span::raw(""))];
    }

    let mut out: Vec<Line<'static>> = Vec::new();
    let mut in_code_block = false;
    let mut code_lang = String::new();
    let mut code_buf = String::new();

    for raw_line in input.lines() {
        // fenced code-block boundaries
        if raw_line.trim_start().starts_with("```") {
            if in_code_block {
                // closing fence – flush highlighted block
                out.extend(highlight_code_block(&code_buf, &code_lang));
                code_buf.clear();
                code_lang.clear();
                in_code_block = false;
            } else {
                // opening fence
                code_lang = raw_line.trim_start().trim_start_matches('`').to_string();
                in_code_block = true;
            }
            continue;
        }

        if in_code_block {
            code_buf.push_str(raw_line);
            code_buf.push('\n');
            continue;
        }

        out.push(render_inline(raw_line));
    }

    // unclosed code block – render what we have
    if !code_buf.is_empty() {
        out.extend(highlight_code_block(&code_buf, &code_lang));
    }

    out
}

// ── inline markdown spans ──────────────────────────────────────
fn render_inline(line: &str) -> Line<'static> {
    let trimmed = line.trim_start();

    // headings
    if let Some(rest) = trimmed.strip_prefix("### ") {
        return Line::from(Span::styled(
            format!("   {rest}"),
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ));
    }
    if let Some(rest) = trimmed.strip_prefix("## ") {
        return Line::from(Span::styled(
            format!("  {rest}"),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
        ));
    }
    if let Some(rest) = trimmed.strip_prefix("# ") {
        return Line::from(Span::styled(
            rest.to_string(),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
        ));
    }

    // unordered list items  (-, *, +)
    if trimmed.starts_with("- ")
        || trimmed.starts_with("* ")
        || trimmed.starts_with("+ ")
    {
        let body = &trimmed[2..];
        return Line::from(vec![
            Span::styled("  • ", Style::default().fg(Color::Yellow)),
            Span::raw(body.to_string()),
        ]);
    }

    // ordered list items (1. 2. etc.)
    if let Some(idx) = trimmed.find(". ") {
        if idx <= 3 && trimmed[..idx].chars().all(|c| c.is_ascii_digit()) {
            let num = &trimmed[..idx];
            let body = &trimmed[idx + 2..];
            return Line::from(vec![
                Span::styled(
                    format!("  {num}. "),
                    Style::default().fg(Color::Yellow),
                ),
                Span::raw(body.to_string()),
            ]);
        }
    }

    // blockquote
    if let Some(rest) = trimmed.strip_prefix("> ") {
        return Line::from(vec![
            Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
            Span::styled(rest.to_string(), Style::default().fg(Color::Gray)),
        ]);
    }

    // horizontal rule
    if trimmed == "---" || trimmed == "***" || trimmed == "___" {
        return Line::from(Span::styled(
            "────────────────────────────",
            Style::default().fg(Color::DarkGray),
        ));
    }

    // inline formatting: **bold**, *italic*, `code`
    parse_inline_spans(line)
}

/// Parse inline **bold**, *italic*, and `code` spans.
fn parse_inline_spans(text: &str) -> Line<'static> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;
    let mut plain = String::new();

    while i < len {
        // inline code
        if chars[i] == '`' {
            if !plain.is_empty() {
                spans.push(Span::raw(std::mem::take(&mut plain)));
            }
            let start = i + 1;
            i = start;
            while i < len && chars[i] != '`' {
                i += 1;
            }
            let code: String = chars[start..i].iter().collect();
            spans.push(Span::styled(
                code,
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ));
            if i < len {
                i += 1;
            }
            continue;
        }

        // bold **text**
        if i + 1 < len && chars[i] == '*' && chars[i + 1] == '*' {
            if !plain.is_empty() {
                spans.push(Span::raw(std::mem::take(&mut plain)));
            }
            let start = i + 2;
            i = start;
            while i + 1 < len && !(chars[i] == '*' && chars[i + 1] == '*') {
                i += 1;
            }
            let bold: String = chars[start..i].iter().collect();
            spans.push(Span::styled(
                bold,
                Style::default().add_modifier(Modifier::BOLD),
            ));
            if i + 1 < len {
                i += 2;
            }
            continue;
        }

        // italic *text*
        if chars[i] == '*' {
            if !plain.is_empty() {
                spans.push(Span::raw(std::mem::take(&mut plain)));
            }
            let start = i + 1;
            i = start;
            while i < len && chars[i] != '*' {
                i += 1;
            }
            let italic: String = chars[start..i].iter().collect();
            spans.push(Span::styled(
                italic,
                Style::default().add_modifier(Modifier::ITALIC),
            ));
            if i < len {
                i += 1;
            }
            continue;
        }

        plain.push(chars[i]);
        i += 1;
    }

    if !plain.is_empty() {
        spans.push(Span::raw(plain));
    }

    Line::from(spans)
}

// ── syntect-powered code highlighting ──────────────────────────
fn highlight_code_block(code: &str, lang: &str) -> Vec<Line<'static>> {
    let ss = syntax_set();
    let syntax = if lang.is_empty() {
        ss.find_syntax_plain_text()
    } else {
        ss.find_syntax_by_token(lang)
            .unwrap_or_else(|| ss.find_syntax_plain_text())
    };

    let theme = highlight_theme();
    let mut h = HighlightLines::new(syntax, theme);
    let mut lines: Vec<Line<'static>> = Vec::new();

    // top border
    lines.push(Line::from(Span::styled(
        format!("╭─ {lang} ─────────────────────────"),
        Style::default().fg(Color::DarkGray),
    )));

    for src_line in LinesWithEndings::from(code) {
        let Ok(ranges) = h.highlight_line(src_line, ss) else {
            // fallback: plain dimmed text
            lines.push(Line::from(vec![
                Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
                Span::raw(src_line.trim_end_matches('\n').to_string()),
            ]));
            continue;
        };

        let mut spans: Vec<Span<'static>> = Vec::with_capacity(ranges.len() + 1);
        spans.push(Span::styled(" │ ", Style::default().fg(Color::DarkGray)));

        for (style, fragment) in ranges {
            let fg = syntect_to_ratatui_color(style.foreground);
            let mut restyle = Style::default().fg(fg);
            if style.font_style.contains(highlighting::FontStyle::BOLD) {
                restyle = restyle.add_modifier(Modifier::BOLD);
            }
            if style.font_style.contains(highlighting::FontStyle::ITALIC) {
                restyle = restyle.add_modifier(Modifier::ITALIC);
            }
            spans.push(Span::styled(
                fragment.trim_end_matches('\n').to_string(),
                restyle,
            ));
        }
        lines.push(Line::from(spans));
    }

    // bottom border
    lines.push(Line::from(Span::styled(
        "╰───────────────────────────────",
        Style::default().fg(Color::DarkGray),
    )));

    lines
}

fn syntect_to_ratatui_color(c: highlighting::Color) -> Color {
    Color::Rgb(c.r, c.g, c.b)
}
