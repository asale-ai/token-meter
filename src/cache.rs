//! Predicting the prompt-cache split.
//!
//! # Read this before using it
//!
//! **A cache hit is not a property of your prompt.** It depends on provider-side
//! state — whether an identical prefix was sent recently enough, whether the TTL
//! has lapsed, whether the server kept the entry at all under load. No local
//! computation can observe any of that. What this module produces is a
//! *prediction*, and every [`CacheSplit`] it returns is marked
//! [`Source::Heuristic`] no matter how confident the inputs look.
//!
//! **Never bill from this.** The provider reports the real split in its usage
//! response, and that report is the only defensible input to money. This module
//! is for the jobs a prediction is actually good for:
//!
//! * sizing a spend estimate before a request goes out,
//! * sanity-checking a counterparty's reported split — the check that matters,
//!   because `cache_write` is typically 1.25× the input rate and `cache_read`
//!   0.1×, a 12.5× spread that a total-only comparison cannot see,
//! * capacity and cost modelling over historical traffic.
//!
//! # What you have to supply
//!
//! History. [`PrefixSeen`] is the seam: this crate computes *what could be
//! cached*, you answer *whether this prefix was seen recently on this lane*.
//! Backing it with a shared, TTL-expiring store (Redis with an expiry matching
//! the provider's cache window is the obvious fit) keeps the prediction honest
//! across processes; an in-memory map is fine for a single node.
//!
//! # Accuracy, by provider
//!
//! * **Claude** — best case. Breakpoints are explicit, so the cacheable extent is
//!   known rather than inferred, and the remaining question is only whether the
//!   prefix is still live.
//! * **OpenAI** — automatic prefix caching, 1024-token minimum, 128-token
//!   granularity, and explicitly not guaranteed. Writes are not billed
//!   separately, so only the read side is ever predicted.
//! * **Gemini** — implicit caching with no guarantee at all. Treat the output as
//!   an upper bound rather than a forecast.
//!
//! One structural bias worth knowing: if the upstream account is also used
//! outside your view, its cache will be warmer than your history suggests. The
//! error is therefore one-sided — this **under**-predicts hits — which is the
//! safe direction for a fraud check (a seller claiming more `cache_read` than you
//! predicted is unremarkable; claiming more `cache_write` is the finding).

use crate::{Family, Source};
use std::hash::{Hash, Hasher};

/// A prompt's predicted split across the billed token types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CacheSplit {
    /// Prompt tokens billed at the ordinary input rate.
    pub input: i64,
    /// Prompt tokens predicted to be served from cache.
    pub cache_read: i64,
    /// Prompt tokens predicted to be written into the cache.
    pub cache_write: i64,
    /// Always [`Source::Heuristic`]. A prediction is never a measurement.
    pub source: Source,
}

impl CacheSplit {
    /// The whole prompt, none of it cached.
    #[must_use]
    pub fn uncached(total: i64) -> Self {
        Self { input: total.max(0), cache_read: 0, cache_write: 0, source: Source::Heuristic }
    }

    /// Total prompt tokens across all three buckets.
    #[must_use]
    pub fn total(&self) -> i64 {
        self.input + self.cache_read + self.cache_write
    }
}

/// Whether a prefix has been seen recently on a given lane.
///
/// "Lane" is whatever isolates one upstream cache from another — an account, an
/// API key, a device. Two requests only share a cache if they share a lane, so
/// getting this key wrong is the difference between a useful prediction and
/// noise.
///
/// Implementations are expected to expire entries at the provider's cache TTL
/// (five minutes is the common default). This crate does not track time; letting
/// the store expire entries is both simpler and correct across processes.
pub trait PrefixSeen {
    /// Has this prefix been sent on this lane recently enough to still be cached?
    fn seen(&self, lane: &str, prefix: u64) -> bool;

    /// Record that this prefix has now been sent.
    fn record(&self, lane: &str, prefix: u64);
}

/// Hash a cacheable prefix into a key for [`PrefixSeen`].
///
/// Not cryptographic — this identifies a prefix, it does not authenticate one.
/// Hash the *exact* text that will be sent: providers match prefixes byte for
/// byte, so a prompt that differs by a timestamp in its system preamble is a
/// different prefix and will not hit.
#[must_use]
pub fn prefix_hash(prefix: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    prefix.hash(&mut h);
    h.finish()
}

/// The smallest prompt a provider will cache at all, in tokens.
///
/// Below this the provider does not cache, so a prediction of a hit would be
/// wrong by construction.
#[must_use]
pub fn min_cacheable(family: Family, model: &str) -> i64 {
    let m = model.to_ascii_lowercase();
    match family {
        // Anthropic's floor doubles on the small models.
        Family::Claude if m.contains("haiku") => 2_048,
        Family::Claude => 1_024,
        Family::Gpt => 1_024,
        // Gemini's floor is higher on the pro tier.
        Family::Gemini if m.contains("pro") => 2_048,
        Family::Gemini => 1_024,
        Family::Other => 1_024,
    }
}

/// Granularity at which a provider counts cached tokens.
///
/// OpenAI credits caching in 128-token blocks; the others report the exact
/// cached extent.
#[must_use]
pub fn block_size(family: Family) -> i64 {
    match family {
        Family::Gpt => 128,
        _ => 1,
    }
}

/// Whether a family bills cache writes as their own token type.
///
/// Anthropic charges a premium to write an entry. OpenAI and Gemini populate
/// their caches as a side effect of an ordinary request and bill nothing extra,
/// so predicting a write for them would invent a charge that does not exist.
#[must_use]
pub fn bills_cache_writes(family: Family) -> bool {
    matches!(family, Family::Claude)
}

/// Predict how a prompt's tokens will be split.
///
/// * `total` — the whole prompt, as counted by [`Prompt::count`](crate::Prompt::count).
/// * `cacheable` — tokens up to the last cache breakpoint. For Claude this is
///   known from `cache_control`; for the others it is the stable prefix of the
///   prompt (system message and tool definitions, typically).
/// * `lane` / `prefix` — cache identity, see [`PrefixSeen`].
///
/// This does not record the prefix. Call [`PrefixSeen::record`] yourself once the
/// request is actually sent, so a prompt that never left does not poison the
/// history.
#[must_use]
pub fn predict(
    family: Family,
    model: &str,
    total: i64,
    cacheable: i64,
    lane: &str,
    prefix: u64,
    seen: &dyn PrefixSeen,
) -> CacheSplit {
    predict_seen(family, model, total, cacheable, seen.seen(lane, prefix))
}

/// [`predict`] for callers that already know whether the prefix is warm.
///
/// [`PrefixSeen`] is a synchronous trait, and the natural place to keep prefix
/// history is a shared store reached over the network. Rather than force an
/// async trait on every caller, look the answer up however your stack prefers
/// and pass the boolean in.
#[must_use]
pub fn predict_seen(
    family: Family,
    model: &str,
    total: i64,
    cacheable: i64,
    seen: bool,
) -> CacheSplit {
    let total = total.max(0);
    let cacheable = cacheable.clamp(0, total);

    if cacheable < min_cacheable(family, model) {
        return CacheSplit::uncached(total);
    }

    // Round down to what the provider will actually credit.
    let block = block_size(family);
    let cacheable = (cacheable / block) * block;
    if cacheable == 0 {
        return CacheSplit::uncached(total);
    }

    if seen {
        CacheSplit {
            input: total - cacheable,
            cache_read: cacheable,
            cache_write: 0,
            source: Source::Heuristic,
        }
    } else if bills_cache_writes(family) {
        CacheSplit {
            input: total - cacheable,
            cache_read: 0,
            cache_write: cacheable,
            source: Source::Heuristic,
        }
    } else {
        // A cold cache on a provider that does not bill writes: the whole prompt
        // is ordinary input, and the entry gets populated for free.
        CacheSplit::uncached(total)
    }
}

/// An in-memory [`PrefixSeen`] with no expiry, for tests and single-shot tools.
///
/// Not for production: without a TTL it will claim a hit for a prefix whose cache
/// entry lapsed hours ago.
#[derive(Debug, Default)]
pub struct InMemorySeen {
    seen: std::sync::Mutex<std::collections::HashSet<(String, u64)>>,
}

impl PrefixSeen for InMemorySeen {
    fn seen(&self, lane: &str, prefix: u64) -> bool {
        self.seen.lock().is_ok_and(|s| s.contains(&(lane.to_string(), prefix)))
    }

    fn record(&self, lane: &str, prefix: u64) {
        if let Ok(mut s) = self.seen.lock() {
            s.insert((lane.to_string(), prefix));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Cold;
    impl PrefixSeen for Cold {
        fn seen(&self, _: &str, _: u64) -> bool {
            false
        }
        fn record(&self, _: &str, _: u64) {}
    }

    struct Warm;
    impl PrefixSeen for Warm {
        fn seen(&self, _: &str, _: u64) -> bool {
            true
        }
        fn record(&self, _: &str, _: u64) {}
    }

    #[test]
    fn a_warm_claude_prefix_is_read_not_written() {
        let s = predict(Family::Claude, "claude-sonnet-5", 10_000, 8_000, "lane", 1, &Warm);
        assert_eq!(s.cache_read, 8_000);
        assert_eq!(s.cache_write, 0);
        assert_eq!(s.input, 2_000);
        assert_eq!(s.total(), 10_000, "nothing invented, nothing lost");
    }

    #[test]
    fn a_cold_claude_prefix_is_a_billed_write() {
        let s = predict(Family::Claude, "claude-sonnet-5", 10_000, 8_000, "lane", 1, &Cold);
        assert_eq!(s.cache_write, 8_000);
        assert_eq!(s.cache_read, 0);
        assert_eq!(s.total(), 10_000);
    }

    #[test]
    fn openai_never_predicts_a_billed_write() {
        let cold = predict(Family::Gpt, "gpt-5", 10_000, 8_000, "lane", 1, &Cold);
        assert_eq!(cold.cache_write, 0, "populating an OpenAI cache is free");
        assert_eq!(cold.input, 10_000);

        // 8 000 is not a whole number of 128-token blocks, so the credit lands
        // on the block below it — see `openai_credits_whole_blocks_only`.
        let warm = predict(Family::Gpt, "gpt-5", 10_000, 8_000, "lane", 1, &Warm);
        assert_eq!(warm.cache_read, 7_936);
        assert_eq!(warm.cache_write, 0);
    }

    #[test]
    fn openai_credits_whole_blocks_only() {
        // 8 100 cacheable tokens round down to 63 blocks of 128 = 8 064.
        let s = predict(Family::Gpt, "gpt-5", 10_000, 8_100, "lane", 1, &Warm);
        assert_eq!(s.cache_read, 8_064);
        assert_eq!(s.total(), 10_000);
    }

    #[test]
    fn a_prompt_below_the_floor_is_never_cached() {
        let s = predict(Family::Claude, "claude-sonnet-5", 900, 900, "lane", 1, &Warm);
        assert_eq!(s, CacheSplit::uncached(900));

        // Haiku's floor is twice as high, so the same prompt that caches on
        // Sonnet does not cache here.
        let sonnet = predict(Family::Claude, "claude-sonnet-5", 4_000, 1_500, "l", 1, &Warm);
        let haiku = predict(Family::Claude, "claude-haiku-4-5", 4_000, 1_500, "l", 1, &Warm);
        assert_eq!(sonnet.cache_read, 1_500);
        assert_eq!(haiku.cache_read, 0);
    }

    #[test]
    fn the_cacheable_extent_cannot_exceed_the_prompt() {
        let s = predict(Family::Claude, "claude-sonnet-5", 5_000, 99_999, "lane", 1, &Warm);
        assert_eq!(s.cache_read, 5_000);
        assert_eq!(s.input, 0);
        assert_eq!(s.total(), 5_000);
    }

    #[test]
    fn a_prediction_never_claims_to_be_a_measurement() {
        let s = predict(Family::Claude, "claude-sonnet-5", 10_000, 8_000, "lane", 1, &Warm);
        assert_eq!(s.source, Source::Heuristic);
        assert!(!s.source.is_precise());
    }

    #[test]
    fn lanes_do_not_share_a_cache() {
        let seen = InMemorySeen::default();
        let h = prefix_hash("a long stable system preamble");
        seen.record("device-a", h);

        assert!(seen.seen("device-a", h));
        assert!(!seen.seen("device-b", h), "another account's cache is not yours");
    }

    #[test]
    fn a_prefix_that_differs_at_all_is_a_different_prefix() {
        assert_ne!(prefix_hash("system preamble"), prefix_hash("system preamble "));
    }
}
