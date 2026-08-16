//! The character-class token estimate.
//!
//! When no tokenizer is available — always for Claude, whose BPE has never been
//! published, and for the GPT family without the `openai-exact` feature — text is
//! counted by character class rather than by byte or by word.
//!
//! The reason is that "4 characters per token" is only true of Latin text. On
//! claude/gpt-family BPEs, CJK averages closer to 1.5 characters per token, so a
//! flat divisor under-counts a Chinese prompt by roughly 2.5×. Segmenting by
//! script and weighting each class separately keeps the error in the tens of
//! percent instead of the hundreds.
//!
//! This is an order-of-magnitude instrument, documented and reproducible. It is
//! not a tokenizer and does not pretend to be one: everything it produces is
//! marked [`Source::Heuristic`](crate::Source::Heuristic).

/// Average characters per token for Latin/ASCII text.
pub const CHARS_PER_TOKEN_ASCII: f64 = 4.0;
/// Average characters per token for CJK scripts.
pub const CHARS_PER_TOKEN_CJK: f64 = 1.5;
/// Average characters per token for other non-ASCII scripts (Cyrillic, Arabic,
/// Devanagari, emoji): denser than Latin, sparser than CJK.
pub const CHARS_PER_TOKEN_OTHER: f64 = 2.5;

/// Whether a character belongs to a CJK-family script.
///
/// Hangul and kana are included: they share the property that matters here,
/// which is a much denser characters-per-token ratio than Latin.
#[must_use]
pub fn is_cjk(c: char) -> bool {
    matches!(u32::from(c),
        0x2E80..=0x2EFF        // CJK radicals
        | 0x3000..=0x303F      // CJK punctuation
        | 0x3040..=0x30FF      // hiragana + katakana
        | 0x3100..=0x312F      // bopomofo
        | 0x3130..=0x318F      // hangul compatibility jamo
        | 0x31F0..=0x31FF      // katakana extensions
        | 0x3400..=0x4DBF      // CJK extension A
        | 0x4E00..=0x9FFF      // CJK unified ideographs
        | 0xAC00..=0xD7AF      // hangul syllables
        | 0xF900..=0xFAFF      // CJK compatibility ideographs
        | 0xFF00..=0xFFEF      // full-width / half-width forms
        | 0x20000..=0x2FA1F    // CJK extensions B..F
    )
}

/// Running character-class tallies.
///
/// Accumulating counts rather than tokens means a long stream is weighted once
/// at the end instead of rounding up on every fragment — the difference between
/// counting a 10 000-chunk stream and counting 10 000 one-token chunks.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CharClassCounts {
    /// ASCII characters seen.
    pub ascii: u64,
    /// CJK-family characters seen.
    pub cjk: u64,
    /// Everything else.
    pub other: u64,
}

impl CharClassCounts {
    /// Tally one fragment of text.
    pub fn observe(&mut self, text: &str) {
        for c in text.chars() {
            if c.is_ascii() {
                self.ascii += 1;
            } else if is_cjk(c) {
                self.cjk += 1;
            } else {
                self.other += 1;
            }
        }
    }

    /// Whether anything has been counted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ascii == 0 && self.cjk == 0 && self.other == 0
    }

    /// The weighted token estimate over everything tallied so far.
    #[must_use]
    pub fn tokens(&self) -> f64 {
        self.ascii as f64 / CHARS_PER_TOKEN_ASCII
            + self.cjk as f64 / CHARS_PER_TOKEN_CJK
            + self.other as f64 / CHARS_PER_TOKEN_OTHER
    }

    /// The weighted estimate, rounded up. Non-empty input is at least 1 token.
    #[must_use]
    pub fn tokens_ceil(&self) -> i64 {
        if self.is_empty() {
            return 0;
        }
        (self.tokens().ceil() as i64).max(1)
    }
}

/// Estimate the tokens in a text fragment by character class.
///
/// Empty input is 0; anything non-empty is at least 1.
#[must_use]
pub fn estimate_text(text: &str) -> i64 {
    if text.is_empty() {
        return 0;
    }
    let mut c = CharClassCounts::default();
    c.observe(text);
    c.tokens_ceil()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latin_text_is_about_four_characters_per_token() {
        assert_eq!(estimate_text(&"a".repeat(400)), 100);
    }

    #[test]
    fn cjk_is_counted_denser_than_latin() {
        // The whole reason this is not a flat divisor: 300 Chinese characters
        // are ~200 tokens, not the 75 a `/4` rule would report.
        assert_eq!(estimate_text(&"汉".repeat(300)), 200);
        assert!(estimate_text(&"汉".repeat(300)) > estimate_text(&"a".repeat(300)));
    }

    #[test]
    fn kana_and_hangul_count_as_cjk() {
        assert_eq!(estimate_text(&"あ".repeat(150)), 100);
        assert_eq!(estimate_text(&"한".repeat(150)), 100);
    }

    #[test]
    fn scripts_are_weighted_separately_within_one_string() {
        // 40 ASCII (10 tokens) + 30 CJK (20 tokens).
        let s = format!("{}{}", "a".repeat(40), "字".repeat(30));
        assert_eq!(estimate_text(&s), 30);
    }

    #[test]
    fn other_scripts_sit_between_the_two() {
        let cyrillic = estimate_text(&"я".repeat(100));
        assert_eq!(cyrillic, 40);
        assert!(cyrillic > estimate_text(&"a".repeat(100)));
        assert!(cyrillic < estimate_text(&"字".repeat(100)));
    }

    #[test]
    fn empty_is_zero_but_anything_at_all_is_one() {
        assert_eq!(estimate_text(""), 0);
        assert_eq!(estimate_text("a"), 1);
        assert_eq!(estimate_text(" "), 1);
    }

    #[test]
    fn accumulating_beats_rounding_each_fragment() {
        // Ten 1-character fragments: rounding each up gives 10, accumulating
        // gives 3. The second is the honest answer.
        let mut c = CharClassCounts::default();
        for _ in 0..10 {
            c.observe("a");
        }
        assert_eq!(c.tokens_ceil(), 3);
    }
}
