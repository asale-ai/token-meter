//! Real usage, as the provider reported it.
//!
//! Counting a prompt is an estimate. This module is the other half: reading what
//! the provider says it actually billed, out of whichever shape its dialect
//! happens to use. That report is the only defensible input to money — estimates
//! are for guards and comparisons.
//!
//! # The two traps
//!
//! **Prompt totals versus prompt remainders.** Anthropic's `input_tokens`
//! *excludes* cached tokens; OpenAI's `prompt_tokens` and Gemini's
//! `promptTokenCount` *include* them. Mapping both straight onto the same field
//! bills the cached portion twice on the dialects that report a total — and
//! cached reads are priced at a tenth of ordinary input, so the error is not
//! small. [`Usage::merge_object`] decides which convention it is looking at from
//! the detail keys present, not from whether a cached count happened to be
//! non-zero.
//!
//! **Gemini's thinking is not in its candidates.** `candidatesTokenCount` counts
//! the visible answer and nothing else; `thoughtsTokenCount` is a sibling field,
//! and Google bills it at the output rate. Reading only the former under-reports
//! every reasoning turn. Where `totalTokenCount` is present it is used to confirm
//! the two are really disjoint before adding them.

use serde_json::Value;

/// Tokens a provider reported for one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Usage {
    /// Prompt tokens billed at the ordinary input rate, **excluding** anything
    /// served from or written to cache.
    pub input: i64,
    /// Generated tokens, including reasoning.
    pub output: i64,
    /// Prompt tokens served from cache.
    pub cache_read: i64,
    /// Prompt tokens written into the cache.
    pub cache_write: i64,
    /// The reasoning portion of [`Usage::output`].
    ///
    /// Informational: a subset of `output`, not an addition to it. Kept because
    /// some providers price reasoning separately and every provider's support
    /// tickets ask about it.
    pub reasoning: i64,
}

impl Usage {
    /// Every prompt-side token, however it was billed.
    #[must_use]
    pub fn prompt_total(&self) -> i64 {
        self.input + self.cache_read + self.cache_write
    }

    /// Prompt plus generation.
    #[must_use]
    pub fn total(&self) -> i64 {
        self.prompt_total() + self.output
    }

    /// Whether anything at all was reported.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total() == 0
    }

    /// Clamp every field to zero or above.
    #[must_use]
    pub fn non_negative(self) -> Self {
        Self {
            input: self.input.max(0),
            output: self.output.max(0),
            cache_read: self.cache_read.max(0),
            cache_write: self.cache_write.max(0),
            reasoning: self.reasoning.max(0),
        }
    }

    /// Read usage out of a provider response body or stream frame.
    ///
    /// Accepts every shape the three dialects use, at every nesting level they
    /// use it: a bare `usage` object, Anthropic's `message.usage`, the Responses
    /// API's `response.usage`, and Gemini's `usageMetadata`.
    #[must_use]
    pub fn from_response(v: &Value) -> Self {
        let mut u = Self::default();
        u.merge_response(v);
        u
    }

    /// Read an OpenAI- or Anthropic-shaped `usage` object directly.
    ///
    /// For callers that have already navigated to the object — a translator
    /// handed the `usage` field rather than the whole frame.
    #[must_use]
    pub fn from_object(u: &Value) -> Self {
        let mut usage = Self::default();
        usage.merge_object(u);
        usage
    }

    /// Read a Gemini `usageMetadata` object directly.
    #[must_use]
    pub fn from_gemini_metadata(u: &Value) -> Self {
        let mut usage = Self::default();
        usage.merge_gemini(u);
        usage
    }

    /// Merge a frame into a running total.
    ///
    /// Streaming reports usage across several frames — Anthropic sends the
    /// prompt side in `message_start` and the output side in `message_delta` —
    /// so a single-shot read of one frame loses half the numbers.
    pub fn merge_response(&mut self, v: &Value) {
        // A bare usage object at the top level: buffered responses of every
        // dialect, plus Anthropic's `message_delta`.
        if let Some(u) = v.get("usage") {
            self.merge_object(u);
        }
        // Anthropic `message_start`: the only frame carrying the prompt side.
        if let Some(u) = v.get("message").and_then(|m| m.get("usage")) {
            self.merge_object(u);
        }
        // Responses API: usage arrives only in `response.completed`, nested one
        // level down, and is null until then.
        if let Some(u) = v.get("response").and_then(|r| r.get("usage")).filter(|u| !u.is_null()) {
            self.merge_object(u);
        }
        // Gemini spells every field its own way.
        if let Some(u) = v.get("usageMetadata") {
            self.merge_gemini(u);
        }
    }

    /// Merge one OpenAI/Anthropic-shaped usage object.
    pub fn merge_object(&mut self, u: &Value) {
        let n = |k: &str| u.get(k).and_then(Value::as_i64);

        if let Some(o) = n("output_tokens").or_else(|| n("completion_tokens")) {
            self.output = o.max(0);
        }
        // Reasoning, where the dialect breaks it out. A subset of output.
        if let Some(r) = ["completion_tokens_details", "output_tokens_details"]
            .iter()
            .find_map(|k| u.get(k).and_then(|d| d.get("reasoning_tokens")).and_then(Value::as_i64))
        {
            self.reasoning = r.max(0);
        }

        // Anthropic's cache counts sit beside the prompt count and map across
        // untouched.
        if let Some(r) = n("cache_read_input_tokens") {
            self.cache_read = r.max(0);
        }
        if let Some(w) = n("cache_creation_input_tokens") {
            self.cache_write = w.max(0);
        }

        let Some(prompt) = n("input_tokens").or_else(|| n("prompt_tokens")) else {
            return;
        };

        // Which convention is this? Decided by the detail keys the OpenAI
        // dialects always emit, not by whether a cached count happened to be
        // non-zero — a request that cached nothing still has to leave an
        // Anthropic frame's cache fields alone.
        let reports_prompt_total = u.get("prompt_tokens").is_some()
            || u.get("prompt_tokens_details").is_some()
            || u.get("input_tokens_details").is_some();

        if !reports_prompt_total {
            self.input = prompt.max(0);
            return;
        }

        // Both halves must come from this same object. Taking the total from one
        // frame and the cached share from another bills the cached tokens in
        // full on whichever frame omits the detail.
        let cached = ["prompt_tokens_details", "input_tokens_details"]
            .iter()
            .find_map(|k| u.get(k).and_then(|d| d.get("cached_tokens")).and_then(Value::as_i64))
            .unwrap_or(0)
            .max(0);
        self.input = (prompt - cached).max(0);
        self.cache_read = cached;
    }

    /// Merge a Gemini `usageMetadata` object.
    pub fn merge_gemini(&mut self, u: &Value) {
        let n = |k: &str| u.get(k).and_then(Value::as_i64).unwrap_or(0).max(0);

        let candidates = n("candidatesTokenCount");
        let thoughts = n("thoughtsTokenCount");
        let prompt = n("promptTokenCount");
        let cached = n("cachedContentTokenCount");
        let total = u.get("totalTokenCount").and_then(Value::as_i64).unwrap_or(0);

        // Thinking is billed at the output rate and is *not* inside
        // `candidatesTokenCount`. Where the response states a total, use it to
        // confirm the two are disjoint before adding — a future revision that
        // folds thoughts into candidates would otherwise be double-counted.
        self.output = if thoughts > 0 && total > 0 && prompt + candidates == total {
            candidates
        } else {
            candidates + thoughts
        };
        self.reasoning = thoughts;

        if u.get("promptTokenCount").is_some() {
            // `promptTokenCount` is a total and includes the cached share.
            self.input = (prompt - cached).max(0);
            self.cache_read = cached;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn anthropic_input_excludes_cache_and_is_taken_as_is() {
        let u = Usage::from_response(&json!({
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 900,
                "cache_creation_input_tokens": 200
            }
        }));
        assert_eq!(u.input, 100, "Anthropic already excludes cached tokens");
        assert_eq!(u.cache_read, 900);
        assert_eq!(u.cache_write, 200);
        assert_eq!(u.prompt_total(), 1_200);
    }

    /// The expensive mistake: OpenAI's `prompt_tokens` is a total. Mapping it
    /// straight to `input` bills the cached share twice — once at the full rate
    /// and once at the cached rate.
    #[test]
    fn openai_prompt_total_has_its_cached_share_removed() {
        let u = Usage::from_response(&json!({
            "usage": {
                "prompt_tokens": 1_000,
                "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 900}
            }
        }));
        assert_eq!(u.input, 100, "1000 total − 900 cached");
        assert_eq!(u.cache_read, 900);
        assert_eq!(u.prompt_total(), 1_000, "and the total is preserved exactly");
    }

    #[test]
    fn an_openai_frame_that_cached_nothing_is_still_a_total() {
        let u = Usage::from_response(&json!({
            "usage": {
                "prompt_tokens": 500,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 0}
            }
        }));
        assert_eq!(u.input, 500);
        assert_eq!(u.cache_read, 0);
    }

    /// A zero cached count must not make an Anthropic frame look like an OpenAI
    /// one, or the convention would flip based on traffic rather than dialect.
    #[test]
    fn a_cold_anthropic_frame_keeps_the_anthropic_convention() {
        let u = Usage::from_response(&json!({
            "usage": {"input_tokens": 500, "output_tokens": 20}
        }));
        assert_eq!(u.input, 500);
        assert_eq!(u.cache_read, 0);
    }

    #[test]
    fn responses_usage_is_nested_and_null_until_the_end() {
        let mut u = Usage::default();
        u.merge_response(&json!({"type": "response.created", "response": {"usage": null}}));
        assert!(u.is_empty(), "a null usage is not a report of zero");

        u.merge_response(&json!({
            "type": "response.completed",
            "response": {"usage": {
                "input_tokens": 300,
                "output_tokens": 40,
                "input_tokens_details": {"cached_tokens": 128},
                "output_tokens_details": {"reasoning_tokens": 25}
            }}
        }));
        assert_eq!(u.input, 172);
        assert_eq!(u.cache_read, 128);
        assert_eq!(u.output, 40);
        assert_eq!(u.reasoning, 25, "a subset of output, not an addition");
    }

    /// Gemini bills thinking at the output rate and reports it *outside*
    /// `candidatesTokenCount`. Reading only candidates under-reports every
    /// reasoning turn.
    #[test]
    fn gemini_thinking_is_added_to_the_visible_answer() {
        let u = Usage::from_response(&json!({
            "usageMetadata": {
                "promptTokenCount": 1_000,
                "candidatesTokenCount": 200,
                "thoughtsTokenCount": 1_500,
                "totalTokenCount": 2_700
            }
        }));
        assert_eq!(u.output, 1_700, "200 visible + 1500 thinking");
        assert_eq!(u.reasoning, 1_500);
        assert_eq!(u.input, 1_000);
    }

    /// Defensive: if a future revision folds thoughts into candidates, the
    /// declared total says so and the two must not be added.
    #[test]
    fn gemini_thinking_is_not_double_counted_when_already_included() {
        let u = Usage::from_response(&json!({
            "usageMetadata": {
                "promptTokenCount": 1_000,
                "candidatesTokenCount": 1_700,
                "thoughtsTokenCount": 1_500,
                "totalTokenCount": 2_700
            }
        }));
        assert_eq!(u.output, 1_700, "the total proves candidates already contains thoughts");
        assert_eq!(u.reasoning, 1_500);
    }

    #[test]
    fn gemini_prompt_count_has_its_cached_share_removed() {
        let u = Usage::from_response(&json!({
            "usageMetadata": {
                "promptTokenCount": 5_000,
                "cachedContentTokenCount": 4_000,
                "candidatesTokenCount": 100
            }
        }));
        assert_eq!(u.input, 1_000);
        assert_eq!(u.cache_read, 4_000);
        assert_eq!(u.prompt_total(), 5_000);
    }

    /// Anthropic streams the prompt side in `message_start` and the output side
    /// in `message_delta`. Reading one frame gets you half the bill.
    #[test]
    fn a_streamed_report_accumulates_across_frames() {
        let mut u = Usage::default();
        u.merge_response(&json!({
            "type": "message_start",
            "message": {"usage": {
                "input_tokens": 100,
                "cache_read_input_tokens": 900,
                "output_tokens": 1
            }}
        }));
        u.merge_response(&json!({
            "type": "message_delta",
            "usage": {"output_tokens": 250}
        }));
        assert_eq!(u.input, 100);
        assert_eq!(u.cache_read, 900);
        assert_eq!(u.output, 250, "the delta's count replaces the placeholder");
    }

    #[test]
    fn an_unrelated_frame_reports_nothing() {
        let u = Usage::from_response(&json!({"type": "content_block_delta", "delta": {"text": "hi"}}));
        assert!(u.is_empty());
    }

    #[test]
    fn negative_counts_are_refused() {
        let u = Usage::from_response(&json!({"usage": {"input_tokens": -5, "output_tokens": -1}}));
        assert_eq!(u.non_negative(), Usage::default());
    }
}
