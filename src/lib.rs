//! Cross-provider LLM token metering.
//!
//! One API for counting what Claude, the GPT family and Gemini will bill for a
//! prompt: text, replayed thinking, tool definitions, tool calls, images and
//! documents — with the per-message framing each provider adds on top.
//!
//! # What this crate will and will not tell you
//!
//! **Every count carries its provenance.** [`Count::source`] says whether a
//! number came from a real tokenizer ([`Source::Exact`]), from the character-class
//! heuristic ([`Source::Heuristic`]), or from the provider's own counting endpoint
//! ([`Source::Remote`]). Callers that feed these numbers into a threshold — a
//! spend guard, a fraud check — need that distinction, because an exact count and
//! a ±10% estimate do not deserve the same tolerance. Nothing here silently
//! upgrades a guess into a fact.
//!
//! **Output tokens cannot be predicted.** Nothing can tell you how long an answer
//! will be before the model writes it. [`StreamMeter`] counts generation as it
//! streams past, which is a measurement after the fact, not a forecast — and on
//! wires that bill reasoning without streaming it (OpenAI Responses, Gemini) even
//! that measurement is structurally short of what the vendor charges.
//!
//! **Cache hits are not a property of your prompt.** Whether a prefix is served
//! from cache depends on provider-side state — TTLs, prefix identity, server
//! load — that no local computation can see. What this crate computes is the
//! *cacheable* extent of a prompt; turning that into a read/write split needs
//! history, which you supply through [`PrefixSeen`]. See the [`cache`] module for
//! what that prediction is worth.
//!
//! **This crate ships no price table.** It will do the arithmetic —
//! [`Rates`] turns a [`Usage`] into a [`Cost`], with long-context tiers — but the
//! rates are yours to supply. Prices change per model, per region, per contract
//! and per day, and a library that also claims to know yours is a library that
//! will quietly be wrong about money.
//!
//! # Estimate, measure, compare
//!
//! The three halves of the job, and the reason they live together:
//!
//! * [`Prompt::count`] estimates a request before it is sent.
//! * [`Usage::from_response`] reads what the provider says it actually billed,
//!   out of whichever shape its dialect uses — including the two traps that
//!   silently corrupt the numbers (see [`usage`]).
//! * [`compare`] holds one against the other, with thresholds scaled to how the
//!   estimate was produced, and catches the cache-split misreport that no
//!   total-based check can see.
//!
//! # Example
//!
//! ```
//! use token_meter::{Prompt, Message, Content};
//!
//! let msgs = [Message::user([Content::Text("Explain BPE in one line.")])];
//! let count = Prompt::new("claude-sonnet-5")
//!     .system("You are terse.")
//!     .messages(&msgs)
//!     .count();
//!
//! assert!(count.tokens > 0);
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod cache;
pub mod compare;
pub mod dialect;
pub mod family;
pub mod heuristic;
pub mod image;
pub mod pricing;
pub mod prompt;
pub mod stream;
pub mod tokenizer;
pub mod tools;
pub mod usage;

pub use cache::{CacheSplit, PrefixSeen};
pub use compare::{Deviation, Policy, SplitCheck};
pub use dialect::Dialect;
pub use family::Family;
pub use image::ImageDims;
pub use pricing::{Cost, RateCard, Rates};
pub use prompt::{Content, Image, Message, Prompt, Role, Tool};
pub use stream::StreamMeter;
pub use tokenizer::{Heuristic, RemoteCounter, Tokenizer};
pub use usage::Usage;

/// Where a token count came from.
///
/// Ordered by how much you should trust it, weakest first, so `max`/`min` over a
/// set of counts does the obvious thing: a total assembled from one heuristic
/// part and one exact part is only as good as its weakest input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub enum Source {
    /// The character-class estimate. Order-of-magnitude correct, routinely ±10%
    /// on mixed scripts, and worst on short inputs where framing dominates.
    #[default]
    Heuristic,
    /// A real tokenizer for this model family ran over the text.
    ///
    /// Exact for the text itself. Framing overhead is still a documented
    /// constant rather than a vendor guarantee, so a whole-prompt count marked
    /// `Exact` is exact to within a few tokens of envelope, not to the token.
    Exact,
    /// The provider's own counting endpoint answered.
    ///
    /// The only count that is authoritative by construction — it is the same
    /// code path that will bill you.
    Remote,
}

impl Source {
    /// Combine two provenances: a total is only as trustworthy as its weakest
    /// component.
    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        self.min(other)
    }

    /// Whether this count can be compared against a provider's report with a
    /// tight tolerance. `false` for [`Source::Heuristic`].
    #[must_use]
    pub fn is_precise(self) -> bool {
        !matches!(self, Source::Heuristic)
    }
}

/// A token count and where it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Count {
    /// The number of tokens. Never negative.
    pub tokens: i64,
    /// How this number was arrived at.
    pub source: Source,
}

impl Count {
    /// A count from a real tokenizer.
    #[must_use]
    pub fn exact(tokens: i64) -> Self {
        Self { tokens: tokens.max(0), source: Source::Exact }
    }

    /// A count from the character-class heuristic.
    #[must_use]
    pub fn heuristic(tokens: i64) -> Self {
        Self { tokens: tokens.max(0), source: Source::Heuristic }
    }

    /// A count reported by the provider's own endpoint.
    #[must_use]
    pub fn remote(tokens: i64) -> Self {
        Self { tokens: tokens.max(0), source: Source::Remote }
    }

    /// Add two counts, keeping the weaker provenance.
    ///
    /// Named `merge` rather than `add` because the provenance side of it is not
    /// addition: two exact counts stay exact, but one estimate anywhere in the
    /// sum makes the total an estimate.
    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        Self { tokens: self.tokens + other.tokens, source: self.source.merge(other.source) }
    }

    /// Add a plain token delta without weakening the provenance.
    ///
    /// For framing constants and other additions that are not themselves
    /// measurements — adding 3 tokens of role envelope to an exact text count
    /// leaves it exact.
    #[must_use]
    pub fn plus(self, tokens: i64) -> Self {
        Self { tokens: (self.tokens + tokens).max(0), source: self.source }
    }
}

impl std::ops::Add for Count {
    type Output = Count;
    fn add(self, rhs: Count) -> Count {
        Count::merge(self, rhs)
    }
}

impl std::iter::Sum for Count {
    fn sum<I: Iterator<Item = Count>>(iter: I) -> Count {
        iter.fold(Count { tokens: 0, source: Source::Remote }, Count::merge)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provenance_degrades_to_the_weakest_input() {
        let mixed = Count::exact(100).merge(Count::heuristic(50));
        assert_eq!(mixed.tokens, 150);
        assert_eq!(mixed.source, Source::Heuristic, "one guess taints the total");

        let both_exact = Count::exact(100).merge(Count::remote(50));
        assert_eq!(both_exact.source, Source::Exact, "remote is stronger, exact wins as the floor");
    }

    #[test]
    fn framing_constants_do_not_weaken_a_count() {
        let c = Count::exact(100).plus(3);
        assert_eq!(c.tokens, 103);
        assert_eq!(c.source, Source::Exact);
    }

    #[test]
    fn an_empty_sum_is_not_a_guess() {
        let empty: Count = std::iter::empty().sum();
        assert_eq!(empty.tokens, 0);
        assert_eq!(empty.source, Source::Remote, "nothing measured, nothing estimated");
    }

    #[test]
    fn counts_never_go_negative() {
        assert_eq!(Count::exact(-5).tokens, 0);
        assert_eq!(Count::heuristic(10).plus(-100).tokens, 0);
    }
}
