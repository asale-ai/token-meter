//! Comparing what you counted against what you were billed.
//!
//! The point of counting a prompt yourself is to have something to check the
//! invoice against. This module is that check, and most of it is about not
//! crying wolf.
//!
//! # Three ways a naive comparison goes wrong
//!
//! **Relative error alone is meaningless on small requests.** A 22-token
//! estimate against a real 46-token prompt is off by 109% and off by 24 tokens.
//! Flag on the ratio alone and every honest short request becomes an incident.
//! [`Policy::floor_tokens`] requires an absolute gap as well.
//!
//! **The tolerance depends on how you counted.** A tiktoken count and a
//! character-class estimate do not deserve the same threshold.
//! [`Policy::for_source`] picks one from the [`Source`] the count carries, which
//! is the whole reason counts carry it.
//!
//! **Under-reporting is usually not a finding.** If you are checking a bill,
//! being charged *less* than you counted is not a problem to escalate. If you
//! are checking a counterparty who gets *paid* from the report, it is the
//! over-report that mints money and the under-report that costs them. Both are
//! one-sided; [`Direction`] says which side you care about.
//!
//! # Cache splits deserve their own check
//!
//! A total-only comparison cannot see the most profitable misreport available:
//! moving prompt tokens from `cache_read` to `cache_write` leaves every total
//! identical and multiplies the bill by more than ten. [`check_split`] is the
//! comparison that catches it.

use crate::pricing::Rates;
use crate::{CacheSplit, Count, Source, Usage};

/// Which direction of disagreement is worth reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Direction {
    /// Only a report that exceeds what you counted. For checking a party that is
    /// paid from the number.
    #[default]
    OverOnly,
    /// Only a report that falls short of what you counted.
    UnderOnly,
    /// Any disagreement in either direction.
    Both,
}

impl Direction {
    fn admits(self, over: i64) -> bool {
        match self {
            Direction::OverOnly => over > 0,
            Direction::UnderOnly => over < 0,
            Direction::Both => true,
        }
    }
}

/// When a gap counts as a finding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Policy {
    /// Relative gap above which a finding is possible, e.g. `0.25` for 25%.
    pub tau: f64,
    /// Absolute token gap that must *also* be cleared.
    pub floor_tokens: i64,
    /// Which direction to report.
    pub direction: Direction,
}

impl Default for Policy {
    fn default() -> Self {
        Self { tau: 0.25, floor_tokens: 200, direction: Direction::OverOnly }
    }
}

impl Policy {
    /// A policy scaled to how the estimate was produced.
    ///
    /// An exact count can be held to a few percent — the remaining error is
    /// framing overhead, not tokenization. A heuristic cannot: its own error
    /// routinely reaches the tens of percent, and a tight threshold against it
    /// reports the estimator rather than the provider.
    #[must_use]
    pub fn for_source(source: Source) -> Self {
        match source {
            Source::Remote => Self { tau: 0.02, floor_tokens: 50, ..Self::default() },
            Source::Exact => Self { tau: 0.10, floor_tokens: 100, ..Self::default() },
            Source::Heuristic => Self::default(),
        }
    }

    /// Report disagreement in both directions.
    #[must_use]
    pub fn both_ways(mut self) -> Self {
        self.direction = Direction::Both;
        self
    }

    /// Whether a given gap clears this policy.
    #[must_use]
    pub fn clears(&self, over: i64, baseline: i64) -> bool {
        if baseline <= 0 || !self.direction.admits(over) {
            return false;
        }
        let ratio = (over.abs() as f64) / (baseline as f64);
        ratio > self.tau && over.abs() > self.floor_tokens
    }
}

/// The outcome of comparing an estimate against a report.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Deviation {
    /// Relative gap on the prompt side. Zero when there was no estimate to
    /// compare against — "no opinion", not "agreement".
    pub input: f64,
    /// Relative gap on the output side.
    pub output: f64,
    /// The larger of the two.
    pub max: f64,
    /// Signed token gap on the prompt side; positive means the report exceeds
    /// the estimate.
    pub over_input: i64,
    /// Signed token gap on the output side.
    pub over_output: i64,
    /// Whether this clears the policy.
    pub flagged: bool,
}

/// Compare a counted prompt and an observed output against a reported usage.
///
/// `estimated_input` is what you counted before sending; `estimated_output` is
/// what you observed on the stream, or zero if you could not observe it. A side
/// with no estimate is not checked at all rather than compared against zero —
/// the difference between "I measured something else" and "I measured nothing".
#[must_use]
pub fn compare(
    estimated_input: Count,
    estimated_output: Count,
    reported: &Usage,
    policy: Policy,
) -> Deviation {
    let r = reported.non_negative();

    // Cached tokens were still part of the prompt that was sent, so the whole
    // prompt accounting is what the request-side estimate is compared against.
    let over_input = r.prompt_total() - estimated_input.tokens;
    let input = if estimated_input.tokens > 0 {
        (over_input.abs() as f64) / (estimated_input.tokens as f64)
    } else {
        0.0
    };

    let over_output = r.output - estimated_output.tokens;
    let output = if estimated_output.tokens > 0 {
        (over_output.abs() as f64) / (estimated_output.tokens as f64)
    } else {
        0.0
    };

    let flagged = policy.clears(over_input, estimated_input.tokens)
        || policy.clears(over_output, estimated_output.tokens);

    Deviation { input, output, max: input.max(output), over_input, over_output, flagged }
}

/// The outcome of checking a reported cache split against a predicted one.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SplitCheck {
    /// Tokens reported as cache writes beyond what was predicted.
    pub excess_write: i64,
    /// What that excess costs, in your rate's minor units.
    ///
    /// The number worth alerting on: the token counts may match exactly while
    /// this runs into real money.
    pub excess_cost: i64,
    /// The reported split cost this much more than the predicted one.
    pub cost_over: i64,
    /// Whether the excess is large enough to be worth acting on.
    pub flagged: bool,
}

/// Check a reported cache split against what was predicted for the same prompt.
///
/// This exists because the totals cannot catch it. Reporting 10 000 prompt
/// tokens as `cache_write` rather than `cache_read` leaves `prompt_total`
/// unchanged, sails past any total-based bound, and bills over ten times as
/// much.
///
/// `min_cost` is the smallest discrepancy worth a finding, in the same minor
/// units as `rates` — a threshold in money rather than tokens, because that is
/// what the check is actually about.
///
/// Remember what the prediction is worth: it under-predicts hits whenever the
/// upstream account is also used outside your view. Excess *writes* are
/// therefore the signal; excess reads are unremarkable.
#[must_use]
pub fn check_split(
    predicted: &CacheSplit,
    reported: &Usage,
    rates: &Rates,
    min_cost: i64,
) -> SplitCheck {
    let r = reported.non_negative();
    let excess_write = (r.cache_write - predicted.cache_write).max(0);
    let excess_cost = rates
        .cost(&Usage { cache_write: excess_write, ..Default::default() })
        .total;

    let predicted_cost = rates.estimate_split(predicted, r.output).total;
    let reported_cost = rates.cost(&r).total;

    SplitCheck {
        excess_write,
        excess_cost,
        cost_over: reported_cost - predicted_cost,
        flagged: excess_cost > min_cost,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rates() -> Rates {
        Rates::per_million(3_000_000, 15_000_000, 300_000, 3_750_000)
    }

    #[test]
    fn a_large_over_report_is_flagged() {
        let d = compare(
            Count::heuristic(1_000),
            Count::heuristic(200),
            &Usage { input: 1_000, output: 1_000, ..Default::default() },
            Policy::default(),
        );
        assert!(d.flagged);
        assert_eq!(d.over_output, 800);
    }

    #[test]
    fn a_small_honest_gap_is_not() {
        // The real shape: 22/5 estimated, 46/10 actually billed. Both ratios blow
        // past tau while the absolute miss is tens of tokens.
        let d = compare(
            Count::heuristic(22),
            Count::heuristic(5),
            &Usage { input: 46, output: 10, ..Default::default() },
            Policy::default(),
        );
        assert!(d.max > 0.25, "the ratio is still reported honestly");
        assert!(!d.flagged, "but a gap of tens of tokens is estimator error");
    }

    #[test]
    fn under_reporting_is_ignored_by_default_and_visible_on_request() {
        let reported = Usage { input: 59_665, output: 141, ..Default::default() };
        let est_in = Count::heuristic(57_319);
        let est_out = Count::heuristic(6_124);

        let d = compare(est_in, est_out, &reported, Policy::default());
        assert!(d.output > 0.97, "the gap is reported either way");
        assert!(!d.flagged, "claiming less than you counted mints nothing");

        let both = compare(est_in, est_out, &reported, Policy::default().both_ways());
        assert!(both.flagged, "and is available when you are checking your own bill");
    }

    #[test]
    fn a_side_with_no_estimate_is_not_checked() {
        let d = compare(
            Count::heuristic(1_000),
            Count::heuristic(0), // nothing decoded
            &Usage { input: 1_000, output: 5_000, ..Default::default() },
            Policy::default(),
        );
        assert_eq!(d.output, 0.0);
        assert!(!d.flagged);
    }

    #[test]
    fn cached_tokens_count_toward_the_prompt_side() {
        // A prompt served almost entirely from cache is not a deviation.
        let d = compare(
            Count::heuristic(1_000),
            Count::heuristic(100),
            &Usage { input: 100, cache_read: 900, output: 100, ..Default::default() },
            Policy::default(),
        );
        assert!(!d.flagged);
        assert_eq!(d.over_input, 0);
    }

    #[test]
    fn an_exact_count_earns_a_tighter_threshold() {
        let reported = Usage { input: 1_150, output: 100, ..Default::default() };
        // 15% over a 1 000-token prompt: within heuristic noise, not within
        // tiktoken's.
        assert!(!compare(
            Count::heuristic(1_000),
            Count::heuristic(100),
            &reported,
            Policy::for_source(Source::Heuristic)
        )
        .flagged);
        assert!(compare(
            Count::exact(1_000),
            Count::exact(100),
            &reported,
            Policy::for_source(Source::Exact)
        )
        .flagged);
    }

    /// The misreport a total-based check cannot see.
    #[test]
    fn moving_reads_to_writes_is_caught_although_every_total_matches() {
        let predicted = CacheSplit {
            input: 1_000,
            cache_read: 50_000,
            cache_write: 0,
            source: Source::Heuristic,
        };
        // Same prompt total, same output — every aggregate agrees.
        let honest = Usage { input: 1_000, cache_read: 50_000, output: 500, ..Default::default() };
        let liar = Usage { input: 1_000, cache_write: 50_000, output: 500, ..Default::default() };
        assert_eq!(honest.prompt_total(), liar.prompt_total());

        let r = rates();
        assert!(!check_split(&predicted, &honest, &r, 1_000).flagged);

        let caught = check_split(&predicted, &liar, &r, 1_000);
        assert!(caught.flagged);
        assert_eq!(caught.excess_write, 50_000);
        assert_eq!(caught.excess_cost, 187_500, "12.5x the read price on the same tokens");
    }

    #[test]
    fn extra_cache_reads_are_not_a_finding() {
        // The prediction under-counts hits whenever the upstream is warmer than
        // our history knows, so a report of *more* reads is expected.
        let predicted =
            CacheSplit { input: 51_000, cache_read: 0, cache_write: 0, source: Source::Heuristic };
        let reported = Usage { input: 1_000, cache_read: 50_000, output: 500, ..Default::default() };
        let c = check_split(&predicted, &reported, &rates(), 1_000);
        assert!(!c.flagged);
        assert!(c.cost_over < 0, "it cost less than predicted, which is the good direction");
    }

    #[test]
    fn a_trivial_discrepancy_stays_below_the_money_threshold() {
        let predicted =
            CacheSplit { input: 1_000, cache_read: 100, cache_write: 0, source: Source::Heuristic };
        let reported = Usage { input: 1_000, cache_write: 120, output: 10, ..Default::default() };
        assert!(!check_split(&predicted, &reported, &rates(), 1_000).flagged);
    }
}
