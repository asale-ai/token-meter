//! Measuring generation as it streams past.
//!
//! Output tokens cannot be predicted, only observed. [`StreamMeter`] observes
//! them — and is careful about what it counts as generation.
//!
//! # What it counts, and the outage that decided it
//!
//! It counts **decoded content**: visible text, the model's thinking, and the
//! tool calls it emits. It deliberately does not count:
//!
//! * **The transport envelope.** A chat-completions stream spends on the order of
//!   200 bytes of JSON framing per token of content. An estimator that falls back
//!   to `bytes / 4` reads a real 141-token answer as ~6100 — not a rounding error
//!   but a 40× one, in the direction that makes an honest seller look like it is
//!   over-reporting.
//! * **Encrypted reasoning blobs.** A `reasoning_signature` /
//!   `encrypted_content` field is transport for the next turn, several kilobytes
//!   of base64 that no vendor bills.
//!
//! When nothing decodes, [`StreamMeter::tokens`] answers zero rather than
//! guessing from the wire — and [`StreamMeter::raw_bytes`] stays available so the
//! caller can tell "an empty answer" from "a stream I could not parse", which are
//! very different failures.

use crate::heuristic::CharClassCounts;
use crate::{Count, Source};

/// Accumulates a streamed response's output tokens.
#[derive(Debug, Default, Clone)]
pub struct StreamMeter {
    raw_bytes: u64,
    text: CharClassCounts,
    saw_text: bool,
}

impl StreamMeter {
    /// A fresh meter.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Note raw bytes arriving from the provider.
    ///
    /// Diagnostic only — see the module docs for why these are not tokens.
    pub fn observe_bytes(&mut self, n: usize) {
        self.raw_bytes += n as u64;
    }

    /// Count one decoded fragment of generation.
    ///
    /// Call this for visible text, for thinking, and for the name and arguments
    /// of a tool call. Most turns of an agentic client are a tool call and
    /// nothing else, so a meter fed only visible text measures zero on them.
    pub fn observe_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.saw_text = true;
        self.text.observe(text);
    }

    /// Count several fragments from one stream event.
    pub fn observe_all<'a>(&mut self, fragments: impl IntoIterator<Item = &'a str>) {
        for f in fragments {
            self.observe_text(f);
        }
    }

    /// Raw bytes seen. With a token count of zero this is the signature of a
    /// stream that never decoded.
    #[must_use]
    pub fn raw_bytes(&self) -> u64 {
        self.raw_bytes
    }

    /// Whether anything decodable arrived.
    #[must_use]
    pub fn decoded_anything(&self) -> bool {
        self.saw_text
    }

    /// The output-token estimate, or zero when nothing decoded.
    #[must_use]
    pub fn tokens(&self) -> Count {
        if !self.saw_text {
            // Not "zero tokens were generated" — "no opinion". Callers must not
            // read this as a measurement; that is what `raw_bytes` disambiguates.
            return Count { tokens: 0, source: Source::Heuristic };
        }
        Count::heuristic(self.text.tokens_ceil())
    }

    /// Reset for a retried attempt.
    ///
    /// A failed attempt that is transferred to another provider must not leave
    /// its partial output in the meter — the replacement stream starts over, and
    /// so does the count.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoded_text_beats_the_envelope() {
        let mut m = StreamMeter::new();
        m.observe_bytes(4_000); // SSE framing
        m.observe_text(&"汉".repeat(150));
        assert_eq!(m.tokens().tokens, 100);
    }

    /// The turn shape that breaks naive meters: an agentic client calling a tool
    /// and saying nothing. Visible text is empty from end to end.
    #[test]
    fn a_silent_tool_call_turn_is_measured_from_the_call() {
        let mut m = StreamMeter::new();
        m.observe_bytes(20_000);
        m.observe_all(["shell", &"a".repeat(200)]);
        let t = m.tokens().tokens;
        assert_eq!(t, 52);
        assert!(
            t < (m.raw_bytes() / 4) as i64 / 10,
            "the byte-based estimate this replaces was an order of magnitude larger"
        );
    }

    #[test]
    fn an_unparsable_stream_produces_no_estimate_but_stays_diagnosable() {
        let mut m = StreamMeter::new();
        m.observe_bytes(800);
        assert_eq!(m.tokens().tokens, 0);
        assert!(!m.decoded_anything());
        assert_eq!(m.raw_bytes(), 800);
    }

    #[test]
    fn a_transferred_attempt_starts_over() {
        let mut m = StreamMeter::new();
        m.observe_text(&"a".repeat(400));
        assert_eq!(m.tokens().tokens, 100);
        m.reset();
        assert_eq!(m.tokens().tokens, 0);
        assert_eq!(m.raw_bytes(), 0);
    }

    #[test]
    fn fragments_accumulate_before_rounding() {
        let mut m = StreamMeter::new();
        for _ in 0..10 {
            m.observe_text("a");
        }
        assert_eq!(m.tokens().tokens, 3, "not 10 — rounding once is the honest count");
    }
}
