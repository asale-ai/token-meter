//! Turning tokens into money.
//!
//! Deliberately narrow: this module multiplies counts by rates you supply. It
//! ships no price table, because rates change per model, per region, per
//! contract and per day, and a counting library that also claims to know your
//! prices is a library that will quietly be wrong about money. Load rates from
//! wherever you already keep them.
//!
//! # Units
//!
//! Rates are integers: **minor currency units per million tokens**. Pick your
//! own minor unit — micro-USD, cents, wei — and stay in it. Integers rather than
//! floats because money that has been through a float is money that no longer
//! reconciles; intermediate arithmetic is done in `i128` so a large request at a
//! high rate cannot overflow.
//!
//! ```
//! use token_meter::pricing::Rates;
//! use token_meter::Usage;
//!
//! // $3/M input, $15/M output, in micro-USD.
//! let rates = Rates::per_million(3_000_000, 15_000_000, 300_000, 3_750_000);
//! let usage = Usage { input: 1_000, output: 500, ..Default::default() };
//!
//! assert_eq!(rates.cost(&usage).total, 3_000 + 7_500);
//! ```
//!
//! # Tiers
//!
//! Long-context requests are billed at a higher rate past a threshold — Claude
//! doubles above 200k prompt tokens — and the switch is *all-or-nothing*: every
//! token of a qualifying request is billed at the higher tier, not just the
//! tokens past the line. [`RateCard`] models that.

use crate::{Count, Usage};

/// Per-token rates, in minor currency units per million tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rates {
    /// Ordinary prompt tokens.
    pub input: i64,
    /// Generated tokens.
    pub output: i64,
    /// Prompt tokens served from cache. Typically ~0.1× input.
    pub cache_read: i64,
    /// Prompt tokens written to cache. Typically ~1.25× input.
    pub cache_write: i64,
}

/// What a request cost, broken out by what drove it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Cost {
    /// Cost of ordinary prompt tokens.
    pub input: i64,
    /// Cost of generated tokens.
    pub output: i64,
    /// Cost of cache reads.
    pub cache_read: i64,
    /// Cost of cache writes.
    pub cache_write: i64,
    /// Sum of the four.
    pub total: i64,
}

impl Cost {
    /// The prompt side alone.
    #[must_use]
    pub fn prompt(&self) -> i64 {
        self.input + self.cache_read + self.cache_write
    }

    /// What this request would have cost with no caching at all.
    ///
    /// The denominator for "how much did caching save us" — and the number to
    /// watch, because a cache that keeps missing costs *more* than no cache at
    /// all on providers that bill writes.
    #[must_use]
    pub fn without_cache(usage: &Usage, rates: &Rates) -> i64 {
        mul(usage.prompt_total(), rates.input) + mul(usage.output, rates.output)
    }
}

/// Multiply tokens by a per-million rate, in `i128` and back.
fn mul(tokens: i64, rate_per_million: i64) -> i64 {
    if tokens <= 0 || rate_per_million == 0 {
        return 0;
    }
    ((tokens as i128 * rate_per_million as i128) / 1_000_000) as i64
}

impl Rates {
    /// Rates given directly in minor units per million tokens.
    #[must_use]
    pub fn per_million(input: i64, output: i64, cache_read: i64, cache_write: i64) -> Self {
        Self { input, output, cache_read, cache_write }
    }

    /// Rates given in minor units per thousand tokens.
    ///
    /// The convention a lot of existing price tables use.
    #[must_use]
    pub fn per_1k(input: i64, output: i64, cache_read: i64, cache_write: i64) -> Self {
        Self {
            input: input * 1_000,
            output: output * 1_000,
            cache_read: cache_read * 1_000,
            cache_write: cache_write * 1_000,
        }
    }

    /// Rates given as a currency amount per single token, the convention public
    /// price sheets use (`3e-6` USD/token).
    ///
    /// `minor_units_per_unit` scales the currency into your integer unit —
    /// 1_000_000 for micro-USD, 100 for cents. Rounded to the nearest integer.
    #[must_use]
    pub fn per_token(
        input: f64,
        output: f64,
        cache_read: f64,
        cache_write: f64,
        minor_units_per_unit: f64,
    ) -> Self {
        let scale = |v: f64| (v * minor_units_per_unit * 1_000_000.0).round() as i64;
        Self {
            input: scale(input),
            output: scale(output),
            cache_read: scale(cache_read),
            cache_write: scale(cache_write),
        }
    }

    /// Cost of a reported usage.
    #[must_use]
    pub fn cost(&self, usage: &Usage) -> Cost {
        let u = usage.non_negative();
        let input = mul(u.input, self.input);
        let output = mul(u.output, self.output);
        let cache_read = mul(u.cache_read, self.cache_read);
        let cache_write = mul(u.cache_write, self.cache_write);
        Cost { input, output, cache_read, cache_write, total: input + output + cache_read + cache_write }
    }

    /// Cost of a request that has not happened yet.
    ///
    /// `prompt` is a counted prompt; `expected_output` is your own forecast,
    /// because nothing can measure it in advance — a max-tokens ceiling is the
    /// usual choice, which makes this an upper bound rather than a prediction.
    #[must_use]
    pub fn estimate(&self, prompt: Count, expected_output: i64) -> Cost {
        self.cost(&Usage {
            input: prompt.tokens,
            output: expected_output,
            ..Default::default()
        })
    }

    /// Cost of a predicted cache split.
    ///
    /// The reason to bother predicting one: on a warm prefix this can be a
    /// fraction of the uncached price, and on a cold one it is *more* expensive
    /// than not caching at all.
    #[must_use]
    pub fn estimate_split(&self, split: &crate::CacheSplit, expected_output: i64) -> Cost {
        self.cost(&Usage {
            input: split.input,
            output: expected_output,
            cache_read: split.cache_read,
            cache_write: split.cache_write,
            reasoning: 0,
        })
    }
}

/// A rate that applies above a prompt-size threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tier {
    /// The tier applies when prompt tokens exceed this.
    pub above_prompt_tokens: i64,
    /// The rates that then apply — to the whole request, not just the excess.
    pub rates: Rates,
}

/// Base rates plus any long-context tiers.
#[derive(Debug, Clone, Default)]
pub struct RateCard {
    /// Rates for a request below every threshold.
    pub base: Rates,
    /// Higher tiers, applied all-or-nothing by prompt size.
    pub tiers: Vec<Tier>,
}

impl RateCard {
    /// A card with no tiers.
    #[must_use]
    pub fn flat(base: Rates) -> Self {
        Self { base, tiers: Vec::new() }
    }

    /// Add a tier.
    #[must_use]
    pub fn with_tier(mut self, above_prompt_tokens: i64, rates: Rates) -> Self {
        self.tiers.push(Tier { above_prompt_tokens, rates });
        self
    }

    /// The rates that apply to a prompt of this size.
    ///
    /// The highest threshold the prompt clears wins, so tiers may be declared in
    /// any order.
    #[must_use]
    pub fn rates_for(&self, prompt_tokens: i64) -> Rates {
        self.tiers
            .iter()
            .filter(|t| prompt_tokens > t.above_prompt_tokens)
            .max_by_key(|t| t.above_prompt_tokens)
            .map_or(self.base, |t| t.rates)
    }

    /// Cost of a reported usage, at whichever tier its prompt size selects.
    #[must_use]
    pub fn cost(&self, usage: &Usage) -> Cost {
        self.rates_for(usage.prompt_total()).cost(usage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn claude_sonnet() -> Rates {
        // $3 / $15 / $0.30 / $3.75 per million, in micro-USD.
        Rates::per_million(3_000_000, 15_000_000, 300_000, 3_750_000)
    }

    #[test]
    fn each_token_type_is_priced_on_its_own_line() {
        let usage = Usage {
            input: 1_000,
            output: 500,
            cache_read: 10_000,
            cache_write: 2_000,
            reasoning: 0,
        };
        let c = claude_sonnet().cost(&usage);
        assert_eq!(c.input, 3_000);
        assert_eq!(c.output, 7_500);
        assert_eq!(c.cache_read, 3_000);
        assert_eq!(c.cache_write, 7_500);
        assert_eq!(c.total, 21_000);
        assert_eq!(c.prompt(), 13_500);
    }

    /// The spread that makes the read/write split worth getting right: the same
    /// 10 000 prompt tokens cost 12.5× more as a write than as a read.
    #[test]
    fn a_cache_write_costs_over_ten_times_a_cache_read() {
        let r = claude_sonnet();
        let read = r.cost(&Usage { cache_read: 10_000, ..Default::default() }).total;
        let write = r.cost(&Usage { cache_write: 10_000, ..Default::default() }).total;
        assert_eq!(write / read, 12);
    }

    #[test]
    fn caching_can_cost_more_than_not_caching() {
        let r = claude_sonnet();
        // A cold prefix: everything written, nothing read.
        let cold = Usage { cache_write: 100_000, output: 100, ..Default::default() };
        assert!(
            r.cost(&cold).total > Cost::without_cache(&cold, &r),
            "a cache that keeps missing is a surcharge, not a saving"
        );

        // A warm one pays for itself many times over.
        let warm = Usage { cache_read: 100_000, output: 100, ..Default::default() };
        assert!(r.cost(&warm).total < Cost::without_cache(&warm, &r) / 5);
    }

    #[test]
    fn rates_convert_from_the_conventions_price_sheets_use() {
        // $3/M expressed per-token is 3e-6; in micro-USD that is 3 000 000/M.
        let from_sheet = Rates::per_token(3e-6, 15e-6, 0.3e-6, 3.75e-6, 1_000_000.0);
        assert_eq!(from_sheet, claude_sonnet());

        // Per-1k tables scale by a thousand.
        assert_eq!(Rates::per_1k(3_000, 15_000, 300, 3_750), claude_sonnet());
    }

    #[test]
    fn a_long_prompt_reprices_the_whole_request() {
        let card = RateCard::flat(claude_sonnet())
            .with_tier(200_000, Rates::per_million(6_000_000, 22_500_000, 600_000, 7_500_000));

        let short = Usage { input: 1_000, output: 1_000, ..Default::default() };
        let long = Usage { input: 250_000, output: 1_000, ..Default::default() };

        assert_eq!(card.cost(&short).output, 15_000, "base output rate");
        assert_eq!(
            card.cost(&long).output,
            22_500,
            "the whole request reprices, output included — not just the tokens past 200k"
        );
    }

    #[test]
    fn tiers_may_be_declared_in_any_order() {
        let hi = Rates::per_million(9, 9, 9, 9);
        let mid = Rates::per_million(5, 5, 5, 5);
        let card = RateCard::flat(Rates::per_million(1, 1, 1, 1))
            .with_tier(1_000_000, hi)
            .with_tier(200_000, mid);

        assert_eq!(card.rates_for(1_000).input, 1);
        assert_eq!(card.rates_for(500_000).input, 5);
        assert_eq!(card.rates_for(2_000_000).input, 9);
    }

    #[test]
    fn an_estimate_is_an_upper_bound_from_a_ceiling() {
        let r = claude_sonnet();
        let c = r.estimate(Count::heuristic(10_000), 4_096);
        assert_eq!(c.input, 30_000);
        assert_eq!(c.output, 61_440);
    }

    #[test]
    fn a_predicted_split_prices_cheaper_than_a_cold_one() {
        let r = claude_sonnet();
        let warm = crate::CacheSplit {
            input: 1_000,
            cache_read: 50_000,
            cache_write: 0,
            source: crate::Source::Heuristic,
        };
        let cold = crate::CacheSplit {
            input: 1_000,
            cache_read: 0,
            cache_write: 50_000,
            source: crate::Source::Heuristic,
        };
        assert!(r.estimate_split(&warm, 500).total < r.estimate_split(&cold, 500).total);
    }

    #[test]
    fn a_huge_request_at_a_high_rate_does_not_overflow() {
        let r = Rates::per_million(i64::MAX / 1_000_000, 0, 0, 0);
        let c = r.cost(&Usage { input: 1_000_000, ..Default::default() });
        assert!(c.total > 0, "i128 intermediate keeps this in range");
    }

    #[test]
    fn zero_rates_and_zero_usage_cost_nothing() {
        assert_eq!(Rates::default().cost(&Usage { input: 10_000, ..Default::default() }).total, 0);
        assert_eq!(claude_sonnet().cost(&Usage::default()).total, 0);
    }
}
