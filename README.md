# token-meter

English | [简体中文](README.zh-CN.md)

Cross-provider LLM token metering and cost accounting for Rust, covering
**Claude**, the **GPT family** and **Gemini**.

Estimate a prompt before you send it. Read what the provider says it actually
billed. Price either one. Hold them against each other.

Every count carries its own provenance, so you always know whether you are
holding a measurement or an estimate.

```rust
use token_meter::{Prompt, Message, Content, Source};

let msgs = [Message::user([Content::Text("Explain BPE in one line.")])];
let count = Prompt::new("claude-sonnet-5")
    .system("You are terse.")
    .messages(&msgs)
    .count();

println!("{} tokens ({:?})", count.tokens, count.source);
```

## Why provenance

A token count that does not say where it came from cannot be used safely. An
exact tiktoken count and a ±10% character-class estimate deserve very different
tolerances when you compare them against a provider's invoice, and code that
treats them alike either over-trusts the estimate or wastes the exact count.

```rust
if count.source.is_precise() {
    // tiktoken or the provider's own endpoint — compare tightly
} else {
    // an estimate — leave room
}
```

`Source` is ordered weakest-first, and totals degrade to their weakest input: one
estimated part anywhere in a prompt makes the whole total an estimate.

## What it counts

| | Claude | GPT family | Gemini |
|---|---|---|---|
| Text | heuristic | **exact** (`openai-exact`) | heuristic |
| Tool definitions | JSON estimate | **TypeScript declaration** | JSON estimate |
| Tool calls & results | ✓ | ✓ | ✓ |
| Replayed thinking | ✓ | ✓ | ✓ |
| Images | area ÷ 750 | 512px tiles | 768px tiles |
| Documents | by size | by size | by size |
| Message framing | ✓ | ✓ | ✓ |

**Tool definitions are not counted as JSON.** The GPT family rewrites your tool
list into a TypeScript namespace declaration before it reaches the model, and
that declaration is what gets billed:

```text
namespace functions {

// Get the weather in a location
type get_weather = (_: {
// The city and state
location: string,
unit?: "celsius" | "fahrenheit",
}) => any;

} // namespace functions
```

Counting the original JSON over-charges the largest fixed component of an agentic
prompt, on every single turn. `token_meter::tools::format_definitions` renders
the real thing so you can see, diff and test it.

**Images are counted from their own headers.** PNG, JPEG, GIF and WebP
dimensions are parsed directly — no image decoding, no dependencies — because a
1024×1024 screenshot is over a thousand Claude tokens, not the flat 85 that a
naive estimator assumes. An unreadable header falls back to the flat rate rather
than inventing a number.

## Reading what you were actually billed

```rust
use token_meter::Usage;

let usage = Usage::from_response(&frame);   // any dialect, any nesting
```

Two traps live here, and both silently corrupt the numbers:

**Prompt totals versus prompt remainders.** Anthropic's `input_tokens`
*excludes* cached tokens. OpenAI's `prompt_tokens` and Gemini's
`promptTokenCount` *include* them. Map both onto the same field and you bill the
cached portion twice — at a tenth of the rate, but on the largest part of an
agentic prompt. The convention is decided from the detail keys present, not from
whether a cached count happened to be non-zero.

**Gemini's thinking is not in its candidates.** `candidatesTokenCount` counts
the visible answer alone; `thoughtsTokenCount` is a sibling field billed at the
output rate. Reading only the former under-reports every reasoning turn,
sometimes by most of it. Where `totalTokenCount` is present it confirms the two
are disjoint before they are added.

Streaming works the same way — `merge_response` folds frames into a running
total, because Anthropic sends the prompt side in `message_start` and the output
side in `message_delta`.

## Pricing

```rust
use token_meter::{Rates, RateCard};

// $3/M in, $15/M out, $0.30/M cache read, $3.75/M cache write — in micro-USD.
let rates = Rates::per_million(3_000_000, 15_000_000, 300_000, 3_750_000);
let cost = rates.cost(&usage);
```

Rates are integers in minor currency units per million tokens, with `i128`
intermediates — money that has been through a float is money that no longer
reconciles. Constructors exist for the conventions price sheets use
(`per_1k`, `per_token`).

`RateCard` adds long-context tiers, which are **all-or-nothing**: a request past
the threshold reprices entirely, output included, not just the tokens beyond the
line.

`Cost::without_cache` gives you the counterfactual, which is worth watching in
both directions — a cache that keeps missing costs *more* than no cache at all
on providers that bill writes.

## Comparing

```rust
use token_meter::{compare, Policy, Source};

let dev = compare(estimated_input, observed_output, &reported, Policy::for_source(estimated_input.source));
```

`Policy::for_source` scales the threshold to how you counted — a tiktoken count
can be held to a few percent, a character-class estimate cannot. Findings are
one-sided by default (`Direction::OverOnly`), and require an absolute token gap
as well as a ratio, because relative error alone is meaningless on short
requests and turns every honest one into an incident.

`check_split` is the comparison the totals cannot do. Reporting prompt tokens as
`cache_write` rather than `cache_read` leaves every aggregate identical and
multiplies the bill by more than ten:

```rust
let finding = check_split(&predicted, &reported, &rates, min_cost);
```

It thresholds on **money**, not tokens, because that is what the check is
actually about.

## What it will not do

**Predict output tokens.** Nothing can tell you how long an answer will be before
the model writes it. `StreamMeter` measures generation as it streams past — a
measurement after the fact, not a forecast. On wires that bill reasoning without
streaming it (OpenAI Responses, Gemini), even that is structurally short of what
the vendor charges; `Dialect::output_estimate_multiple` tells you by how much,
for use in comparisons and never in billing.

**Predict cache hits on its own.** Whether a prefix is served from cache depends
on provider-side state no local computation can see. This crate computes the
*cacheable extent* of a prompt — `Prompt::count_prefix` and
`Prompt::prefix_fingerprint` — and you supply the history, either through the
`PrefixSeen` trait or by passing the answer straight to `predict_seen` if your
lookup is async. The resulting `CacheSplit` is always marked as a prediction.

Back it with a TTL-expiring store (Redis, at the provider's cache window) and it
is useful for spend estimates and for sanity-checking a counterparty's reported
split — where `cache_write` at 1.25× and `cache_read` at 0.1× make a 12.5× spread
that a total-only comparison cannot see. Never bill from it, and measure its
error against real settlements before wiring it to anything that penalises
anyone: it under-predicts hits by construction, because an upstream account is
usually warmer than any one caller's history knows.

**Ship a price table.** It does the arithmetic; the rates are yours to supply.
Prices change per model, per region, per contract and per day, and a library that
also claims to know yours is one that will quietly be wrong about money.

**Invent a Claude tokenizer.** The Claude 3+ BPE has never been published.
Anthropic's answer is `/v1/messages/count_tokens`, and this crate's answer is the
`RemoteCounter` trait — you wire it to your own HTTP client, it falls back to the
local estimate on any failure, and the result says which path answered.

## Features

```toml
[dependencies]
token-meter = { version = "0.1", features = ["openai-exact"] }
```

- **default** — no dependencies beyond `serde_json`. Everything is estimated.
- **`openai-exact`** — pulls in `tiktoken-rs` for real GPT-family counts. A few
  megabytes of BPE tables, borrowed as a singleton rather than cloned.

## License

Apache-2.0
