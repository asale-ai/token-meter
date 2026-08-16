//! Model families and the prompt framing each one adds.
//!
//! A prompt is never billed as the bare concatenation of its text. Every
//! provider wraps each message in role markers and prepends a fixed preamble,
//! and those envelopes are billed like any other token. The constants here are
//! the published (OpenAI) or widely reproduced (Claude, Gemini) figures for that
//! overhead — small, but on a conversation of many short turns they are the
//! difference between a 5% error and a 30% one.

/// A family of models that share a tokenizer and a prompt layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Family {
    /// Anthropic Claude.
    Claude,
    /// OpenAI GPT / o-series / Codex.
    Gpt,
    /// Google Gemini.
    Gemini,
    /// Anything unrecognised: framing is assumed to look like the common case.
    #[default]
    Other,
}

impl Family {
    /// Classify a model id.
    ///
    /// Matching is substring-based and deliberately loose: ids arrive with
    /// vendor prefixes (`anthropic/claude-sonnet-5`), deployment suffixes
    /// (`gpt-5.6-sol`) and date stamps, and getting the family right matters far
    /// more than rejecting an id that is nearly right.
    #[must_use]
    pub fn from_model(model: &str) -> Self {
        let m = model.to_ascii_lowercase();
        if m.contains("claude") {
            Family::Claude
        } else if m.contains("gemini") || m.contains("gemma") {
            Family::Gemini
        } else if m.starts_with("gpt")
            || m.contains("/gpt")
            || m.starts_with("o1")
            || m.starts_with("o3")
            || m.starts_with("o4")
            || m.contains("codex")
        {
            Family::Gpt
        } else {
            Family::Other
        }
    }

    /// Per-message framing: role tags and separators.
    ///
    /// GPT's chat format spends ~4 tokens per message
    /// (`<|im_start|>role\n…<|im_end|>`); Claude's `Human:`/`Assistant:` framing
    /// and Gemini's role wrappers are comparable and slightly lighter.
    #[must_use]
    pub fn per_message_overhead(self) -> i64 {
        match self {
            Family::Gpt => 4,
            Family::Claude | Family::Gemini | Family::Other => 3,
        }
    }

    /// Fixed conversation overhead: the system preamble slot plus the reply
    /// primer the provider appends before generation starts.
    #[must_use]
    pub fn base_overhead(self) -> i64 {
        match self {
            Family::Claude => 5,
            Family::Gpt | Family::Gemini | Family::Other => 3,
        }
    }

    /// Whether tool definitions are serialized as a TypeScript namespace.
    ///
    /// This is an OpenAI-family implementation detail (see [`crate::tools`]) and
    /// applying it to Claude or Gemini would be a guess dressed up as a
    /// measurement — those two get the JSON-shaped estimate instead.
    #[must_use]
    pub fn tools_as_typescript(self) -> bool {
        matches!(self, Family::Gpt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn families_survive_vendor_prefixes_and_suffixes() {
        assert_eq!(Family::from_model("claude-sonnet-5"), Family::Claude);
        assert_eq!(Family::from_model("anthropic/claude-opus-4-20250514"), Family::Claude);
        assert_eq!(Family::from_model("gpt-5.6-sol"), Family::Gpt);
        assert_eq!(Family::from_model("openai/gpt-4o-mini"), Family::Gpt);
        assert_eq!(Family::from_model("o3-mini"), Family::Gpt);
        assert_eq!(Family::from_model("gpt-5-codex"), Family::Gpt);
        assert_eq!(Family::from_model("gemini-3-pro"), Family::Gemini);
        assert_eq!(Family::from_model("llama-4-70b"), Family::Other);
    }

    #[test]
    fn only_the_gpt_family_serializes_tools_as_typescript() {
        assert!(Family::Gpt.tools_as_typescript());
        assert!(!Family::Claude.tools_as_typescript());
        assert!(!Family::Gemini.tools_as_typescript());
    }
}
