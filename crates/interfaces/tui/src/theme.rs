use ratatui::style::Color;

#[derive(Debug, Clone)]
pub struct Theme {
    pub background: Color,
    pub foreground: Color,
    pub accent: Color,
    pub muted: Color,
    pub user_bubble: Color,
    pub assistant_bubble: Color,
}

#[derive(Debug, Clone, Copy)]
pub enum ThemeName {
    CatppuccinMocha,
    TokyoNight,
    Nord,
}

impl ThemeName {
    /// Parse a config string into a [`ThemeName`].  Falls back to
    /// `CatppuccinMocha` for unrecognised values.
    pub fn from_config(s: &str) -> Self {
        match s.to_lowercase().replace('_', "-").as_str() {
            "tokyo-night" | "tokyonight" => Self::TokyoNight,
            "nord" => Self::Nord,
            _ => Self::CatppuccinMocha,
        }
    }
}

impl Theme {
    pub fn from_name(name: ThemeName) -> Self {
        match name {
            ThemeName::CatppuccinMocha => Self {
                background: Color::Rgb(30, 30, 46),
                foreground: Color::Rgb(205, 214, 244),
                accent: Color::Rgb(137, 180, 250),
                muted: Color::Rgb(108, 112, 134),
                user_bubble: Color::Rgb(49, 50, 68),
                assistant_bubble: Color::Rgb(24, 24, 37),
            },
            ThemeName::TokyoNight => Self {
                background: Color::Rgb(26, 27, 38),
                foreground: Color::Rgb(192, 202, 245),
                accent: Color::Rgb(122, 162, 247),
                muted: Color::Rgb(86, 95, 137),
                user_bubble: Color::Rgb(41, 46, 66),
                assistant_bubble: Color::Rgb(31, 35, 53),
            },
            ThemeName::Nord => Self {
                background: Color::Rgb(46, 52, 64),
                foreground: Color::Rgb(236, 239, 244),
                accent: Color::Rgb(136, 192, 208),
                muted: Color::Rgb(129, 161, 193),
                user_bubble: Color::Rgb(59, 66, 82),
                assistant_bubble: Color::Rgb(67, 76, 94),
            },
        }
    }

    /// Build a theme from a config string (e.g. `"tokyo-night"`).
    pub fn from_config(s: &str) -> Self {
        Self::from_name(ThemeName::from_config(s))
    }
}

impl Default for Theme {
    fn default() -> Self {
        Self::from_name(ThemeName::CatppuccinMocha)
    }
}
