//! Wire dialects, and what each one lets you see.
//!
//! A dialect is not the same thing as a [`Family`](crate::Family): a model is
//! reached over a wire, and the wire decides what a relay in the middle can
//! observe. The distinction matters for exactly one reason — **reasoning tokens
//! are billed but not always streamed** — and getting it wrong makes honest
//! traffic look fraudulent.

use crate::Count;

/// The wire protocol a response arrives on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Dialect {
    /// Anthropic Messages.
    #[default]
    Claude,
    /// OpenAI chat completions.
    OpenaiChat,
    /// OpenAI Responses.
    OpenaiResponses,
    /// Google Gemini `generateContent`.
    Gemini,
}

impl Dialect {
    /// Whether the model's thinking arrives on the stream in a form an
    /// intermediary can count.
    ///
    /// * `Claude` — yes: `thinking_delta` carries it.
    /// * `OpenaiChat` — no standard channel; some upstreams emit
    ///   `reasoning_content`, some bill silently.
    /// * `OpenaiResponses` — only a *summary*, a few hundred tokens standing in
    ///   for a turn that may have spent tens of thousands.
    /// * `Gemini` — `thoughtsTokenCount` is billed, and thought parts are only
    ///   streamed when the caller asks for them, which agentic clients do not.
    #[must_use]
    pub fn streams_reasoning(self) -> bool {
        matches!(self, Dialect::Claude)
    }

    /// How much more output this wire may legitimately bill than an intermediary
    /// could see, as a multiple of the observed estimate.
    ///
    /// This widens a *comparison* threshold. It is not a billing input: inflating
    /// what you charge by a blindness allowance would charge buyers for tokens
    /// nobody measured. Use it when deciding whether a provider's report is
    /// suspicious, never when deciding what to bill.
    #[must_use]
    pub fn output_estimate_multiple(self) -> f64 {
        match self {
            // Thinking was streamed and counted; what is billed is what was seen.
            Dialect::Claude => 1.0,
            Dialect::OpenaiChat => 2.0,
            Dialect::OpenaiResponses | Dialect::Gemini => 4.0,
        }
    }

    /// Scale an observed output estimate to what this wire would report.
    ///
    /// Zero in, zero out: "nothing decoded" must stay distinguishable from "a
    /// small count", because a comparison against nothing should decline to form
    /// an opinion rather than compare against zero.
    #[must_use]
    pub fn calibrate_output(self, observed: Count) -> Count {
        if observed.tokens <= 0 {
            return Count { tokens: 0, source: observed.source };
        }
        Count {
            tokens: ((observed.tokens as f64) * self.output_estimate_multiple()).ceil() as i64,
            source: observed.source,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wire_that_streams_thinking_gets_no_allowance() {
        let observed = Count::heuristic(400);
        assert_eq!(Dialect::Claude.calibrate_output(observed).tokens, 400);
    }

    #[test]
    fn a_summary_only_wire_gets_room_for_what_it_hid() {
        let observed = Count::heuristic(500);
        assert_eq!(Dialect::OpenaiResponses.calibrate_output(observed).tokens, 2_000);
        assert!(!Dialect::OpenaiResponses.streams_reasoning());
    }

    #[test]
    fn nothing_decoded_stays_nothing_decoded() {
        assert_eq!(Dialect::Gemini.calibrate_output(Count::heuristic(0)).tokens, 0);
        assert!(Dialect::Gemini.calibrate_output(Count::heuristic(1)).tokens > 0);
    }

    #[test]
    fn calibration_preserves_provenance() {
        let c = Dialect::OpenaiChat.calibrate_output(Count::exact(100));
        assert_eq!(c.source, crate::Source::Exact);
        assert_eq!(c.tokens, 200);
    }
}
