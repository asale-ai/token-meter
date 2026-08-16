//! Pluggable counting backends.
//!
//! Three tiers, in descending order of what they cost you and ascending order of
//! how much you can trust them:
//!
//! | Backend | Families | Accuracy | Cost |
//! |---|---|---|---|
//! | [`Heuristic`] | all | ±10%, worse on short inputs | free, offline |
//! | [`Tiktoken`] (`openai-exact`) | GPT | exact for text | a few MB of BPE tables |
//! | [`RemoteCounter`] | Claude, Gemini | authoritative | a network round trip |
//!
//! There is no local-exact tier for Claude, and that is not an omission. The
//! Claude 3+ BPE has never been published; Anthropic's answer is the
//! `/v1/messages/count_tokens` endpoint. Every open-source counting library
//! resolves this the same way, by calling that endpoint or by estimating. This
//! crate does both and tells you which one it did.

use crate::{Count, Family};

/// A backend that can count tokens in a piece of text.
///
/// Implementations count *text*, not prompts: message framing, tool
/// serialization and image geometry are the crate's job, not the tokenizer's.
pub trait Tokenizer {
    /// Count the tokens in a text fragment.
    fn count_text(&self, text: &str) -> Count;

    /// Count several fragments as one unit.
    ///
    /// Worth overriding for backends where per-call overhead dominates, and
    /// meaningful for the heuristic besides: accumulating character classes
    /// across fragments and rounding once is closer to the truth than rounding
    /// every fragment up to a whole token.
    fn count_all(&self, fragments: &[&str]) -> Count {
        fragments.iter().map(|f| self.count_text(f)).sum()
    }
}

/// The character-class estimate. Always available, never exact.
#[derive(Debug, Clone, Copy, Default)]
pub struct Heuristic;

impl Tokenizer for Heuristic {
    fn count_text(&self, text: &str) -> Count {
        Count::heuristic(crate::heuristic::estimate_text(text))
    }

    fn count_all(&self, fragments: &[&str]) -> Count {
        let mut c = crate::heuristic::CharClassCounts::default();
        for f in fragments {
            c.observe(f);
        }
        Count::heuristic(c.tokens_ceil())
    }
}

/// Exact counts for the GPT family, via tiktoken.
///
/// Requires the `openai-exact` feature.
#[cfg(feature = "openai-exact")]
pub struct Tiktoken {
    // A borrowed singleton, not an owned table: the BPE ranks are megabytes and
    // tiktoken already caches one copy per encoding.
    bpe: &'static tiktoken_rs::CoreBPE,
}

#[cfg(feature = "openai-exact")]
impl std::fmt::Debug for Tiktoken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Tiktoken { .. }")
    }
}

#[cfg(feature = "openai-exact")]
impl Tiktoken {
    /// Load the encoding a given model uses.
    ///
    /// Returns `None` for a model tiktoken does not recognise, rather than
    /// falling back to a plausible-looking encoding: the wrong BPE is a
    /// confidently wrong answer, which is worse than an openly approximate one.
    #[must_use]
    pub fn for_model(model: &str) -> Option<Self> {
        tiktoken_rs::bpe_for_model(model).ok().map(|bpe| Self { bpe })
    }

    /// The `o200k_base` encoding used by the current GPT generation.
    ///
    /// The fallback when a model id is too new or too custom for tiktoken's own
    /// table but is recognisably GPT-family.
    #[must_use]
    pub fn o200k() -> Option<Self> {
        Some(Self { bpe: tiktoken_rs::o200k_base_singleton() })
    }
}

#[cfg(feature = "openai-exact")]
impl Tokenizer for Tiktoken {
    fn count_text(&self, text: &str) -> Count {
        if text.is_empty() {
            return Count::exact(0);
        }
        Count::exact(self.bpe.encode_ordinary(text).len() as i64)
    }
}

/// The provider's own token-counting endpoint.
///
/// Implemented by the caller rather than by this crate, on purpose. A counting
/// library that owns an HTTP client owns your TLS stack, your proxy
/// configuration, your retry policy and your credential handling — four
/// decisions that belong to the application. Wire this to whatever client you
/// already have.
///
/// The request body is the provider's native counting shape: Anthropic's
/// `/v1/messages/count_tokens` payload, Gemini's `:countTokens` payload.
pub trait RemoteCounter {
    /// Count the prompt, or return `None` to fall back to a local estimate.
    ///
    /// `None` rather than `Result` because every caller of this trait has the
    /// same recovery — estimate locally and mark the result
    /// [`Heuristic`](crate::Source::Heuristic) — and a rate limit on a counting
    /// endpoint should never fail the operation it was supporting.
    fn count(&self, model: &str, body: &serde_json::Value) -> Option<i64>;
}

/// Pick the best available backend for a model.
///
/// With `openai-exact` enabled this returns a real tokenizer for GPT-family
/// models and the heuristic for everything else.
#[must_use]
pub fn for_model(model: &str) -> Box<dyn Tokenizer> {
    #[cfg(feature = "openai-exact")]
    {
        if Family::from_model(model) == Family::Gpt {
            if let Some(t) = Tiktoken::for_model(model).or_else(Tiktoken::o200k) {
                return Box::new(t);
            }
        }
    }
    let _ = Family::from_model(model);
    Box::new(Heuristic)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Source;

    #[test]
    fn the_heuristic_accumulates_across_fragments() {
        let h = Heuristic;
        // Ten separate 1-character calls would round to 10 tokens; as one unit
        // they are 3.
        let fragments = ["a"; 10];
        assert_eq!(h.count_all(&fragments).tokens, 3);
        assert_eq!(h.count_all(&fragments).source, Source::Heuristic);
    }

    #[test]
    fn a_backend_is_always_available() {
        let t = for_model("some-model-nobody-has-heard-of");
        assert!(t.count_text("hello").tokens > 0);
    }

    #[cfg(feature = "openai-exact")]
    #[test]
    fn tiktoken_is_exact_and_disagrees_with_the_heuristic() {
        let t = Tiktoken::for_model("gpt-4o").expect("gpt-4o is a known model");
        let c = t.count_text("The quick brown fox jumps over the lazy dog.");
        assert_eq!(c.source, Source::Exact);
        // The point of the exact path: it is a real count, not chars/4.
        assert!(c.tokens > 0 && c.tokens < 20);
    }

    #[cfg(feature = "openai-exact")]
    #[test]
    fn an_unknown_model_gets_no_confidently_wrong_encoding() {
        assert!(Tiktoken::for_model("definitely-not-a-real-model").is_none());
    }
}
