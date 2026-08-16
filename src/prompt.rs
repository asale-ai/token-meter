//! The prompt model everything else counts.
//!
//! Deliberately provider-neutral and deliberately borrowed: a caller relaying
//! traffic already holds the request in memory, and a metering pass should not
//! double it. Nothing here owns a `String`.

use crate::image::{ImageDims, UNKNOWN_IMAGE_TOKENS};
use crate::tokenizer::Tokenizer;
use crate::{Count, Family, Heuristic, RemoteCounter, Source};
use serde_json::{json, Value};

/// Who a message is from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    /// Instructions outside the conversation.
    System,
    /// The human turn.
    User,
    /// The model's own turn, replayed on the next request.
    Assistant,
    /// The result of a tool the model called.
    Tool,
}

impl Role {
    /// The wire spelling, for building provider payloads.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// A tool the model may call.
#[derive(Debug, Clone, Copy)]
pub struct Tool<'a> {
    /// Function name, as declared to the provider.
    pub name: &'a str,
    /// Human description. Billed — it goes into the prompt verbatim.
    pub description: &'a str,
    /// JSON Schema for the arguments.
    pub schema: &'a Value,
}

/// An image, in whatever form the caller happens to have it.
#[derive(Debug, Clone, Copy)]
pub enum Image<'a> {
    /// Raw file bytes. Only the header is read.
    Bytes(&'a [u8]),
    /// Base64, with or without a `data:` URL prefix.
    Base64(&'a str),
    /// Dimensions the caller already knows.
    Dims(ImageDims),
}

impl Image<'_> {
    /// Resolve to dimensions, if the header can be read.
    #[must_use]
    pub fn dims(&self) -> Option<ImageDims> {
        match self {
            Image::Bytes(b) => ImageDims::parse(b),
            Image::Base64(s) => ImageDims::parse_base64(s),
            Image::Dims(d) => Some(*d),
        }
    }

    /// Tokens this image costs the given family, falling back to the flat rate
    /// when the header is unreadable.
    #[must_use]
    pub fn tokens(&self, family: Family) -> i64 {
        self.dims().map_or(UNKNOWN_IMAGE_TOKENS, |d| d.tokens_for(family))
    }
}

/// One piece of a message.
#[derive(Debug, Clone, Copy)]
pub enum Content<'a> {
    /// Visible text.
    Text(&'a str),
    /// The model's thinking, replayed on a later turn.
    ///
    /// Billed as prompt when it is sent back — on a reasoning-heavy turn it can
    /// outweigh the visible text, so leaving it out is not a rounding error.
    /// Omit it if your relay strips thinking before forwarding.
    Reasoning(&'a str),
    /// A tool call the model made. `input` is the arguments, already serialized.
    ToolUse {
        /// Function name.
        name: &'a str,
        /// Serialized arguments.
        input: &'a str,
    },
    /// What a tool returned.
    ToolResult(&'a str),
    /// An image.
    Image(Image<'a>),
    /// A document — a PDF, mostly — given by its decoded size in bytes.
    ///
    /// Counted from length alone, because the real figure depends on extracted
    /// page content this crate does not decode. Deliberately conservative.
    Document {
        /// Decoded size in bytes.
        bytes: usize,
    },
}

/// A message in the conversation.
#[derive(Debug, Clone)]
pub struct Message<'a> {
    /// Who it is from.
    pub role: Role,
    /// Its parts.
    pub content: Vec<Content<'a>>,
}

impl<'a> Message<'a> {
    /// A message with an explicit role.
    pub fn new(role: Role, content: impl IntoIterator<Item = Content<'a>>) -> Self {
        Self { role, content: content.into_iter().collect() }
    }

    /// A user turn.
    pub fn user(content: impl IntoIterator<Item = Content<'a>>) -> Self {
        Self::new(Role::User, content)
    }

    /// An assistant turn.
    pub fn assistant(content: impl IntoIterator<Item = Content<'a>>) -> Self {
        Self::new(Role::Assistant, content)
    }

    /// A tool-result turn.
    pub fn tool(content: impl IntoIterator<Item = Content<'a>>) -> Self {
        Self::new(Role::Tool, content)
    }
}

/// A prompt, ready to be counted.
#[derive(Debug, Clone)]
pub struct Prompt<'a> {
    model: &'a str,
    family: Family,
    system: &'a str,
    messages: &'a [Message<'a>],
    tools: &'a [Tool<'a>],
}

const NO_MESSAGES: &[Message<'static>] = &[];
const NO_TOOLS: &[Tool<'static>] = &[];

impl<'a> Prompt<'a> {
    /// Start a prompt for a model.
    #[must_use]
    pub fn new(model: &'a str) -> Self {
        Self {
            model,
            family: Family::from_model(model),
            system: "",
            messages: NO_MESSAGES,
            tools: NO_TOOLS,
        }
    }

    /// Override the family inferred from the model id.
    ///
    /// For gateways that route a caller-facing model name to a different
    /// upstream, where the id says nothing useful about the tokenizer.
    #[must_use]
    pub fn family(mut self, family: Family) -> Self {
        self.family = family;
        self
    }

    /// Set the system prompt.
    #[must_use]
    pub fn system(mut self, system: &'a str) -> Self {
        self.system = system;
        self
    }

    /// Set the conversation.
    #[must_use]
    pub fn messages(mut self, messages: &'a [Message<'a>]) -> Self {
        self.messages = messages;
        self
    }

    /// Set the tool list.
    #[must_use]
    pub fn tools(mut self, tools: &'a [Tool<'a>]) -> Self {
        self.tools = tools;
        self
    }

    /// The model id this prompt targets.
    #[must_use]
    pub fn model_id(&self) -> &'a str {
        self.model
    }

    /// The family being counted for.
    #[must_use]
    pub fn model_family(&self) -> Family {
        self.family
    }

    /// Count the prompt with the best backend available for this model.
    #[must_use]
    pub fn count(&self) -> Count {
        let tk = crate::tokenizer::for_model(self.model);
        self.count_with(tk.as_ref())
    }

    /// Count the prompt with a specific backend.
    #[must_use]
    pub fn count_with(&self, tk: &dyn Tokenizer) -> Count {
        let per_msg = self.family.per_message_overhead();
        let mut total = Count { tokens: self.family.base_overhead(), source: Source::Remote };

        if !self.system.is_empty() {
            total = total.merge(tk.count_text(self.system)).plus(per_msg);
        }

        for m in self.messages {
            total = total.plus(per_msg);
            for c in &m.content {
                total = total.merge(self.count_content(c, tk));
            }
        }

        total.merge(crate::tools::count(self.tools, self.family, !self.system.is_empty(), tk))
    }

    /// Count only the leading portion of the prompt: system, tools, and the
    /// first `messages` turns.
    ///
    /// This is what a prompt cache can hold. Caching works on prefixes, so the
    /// cacheable extent of a request is always "everything up to some point" —
    /// the last explicit breakpoint on Anthropic, the stable head of the prompt
    /// elsewhere. Counting to that point is the input [`crate::cache::predict`]
    /// needs.
    #[must_use]
    pub fn count_prefix(&self, messages: usize) -> Count {
        let tk = crate::tokenizer::for_model(self.model);
        self.count_prefix_with(messages, tk.as_ref())
    }

    /// [`Prompt::count_prefix`] with a specific backend.
    #[must_use]
    pub fn count_prefix_with(&self, messages: usize, tk: &dyn Tokenizer) -> Count {
        let n = messages.min(self.messages.len());
        let head = Prompt { messages: &self.messages[..n], ..self.clone() };
        head.count_with(tk)
    }

    /// A fingerprint of that same leading portion, for [`crate::PrefixSeen`].
    ///
    /// Providers match cached prefixes byte for byte, so this hashes the content
    /// itself rather than the token count — two different prompts of identical
    /// length are not the same prefix, and a system preamble carrying a
    /// timestamp is a different prefix on every request.
    #[must_use]
    pub fn prefix_fingerprint(&self, messages: usize) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();

        // The model is part of the identity: the same text sent to a different
        // model is a different cache entry.
        self.model.hash(&mut h);
        self.system.hash(&mut h);
        for t in self.tools {
            t.name.hash(&mut h);
            t.description.hash(&mut h);
            t.schema.to_string().hash(&mut h);
        }
        for m in self.messages.iter().take(messages) {
            m.role.as_str().hash(&mut h);
            for c in &m.content {
                match c {
                    Content::Text(t) | Content::Reasoning(t) | Content::ToolResult(t) => {
                        t.hash(&mut h);
                    }
                    Content::ToolUse { name, input } => {
                        name.hash(&mut h);
                        input.hash(&mut h);
                    }
                    // Bytes are not hashed: a multi-megabyte payload would cost
                    // more to fingerprint than the prediction is worth, and its
                    // dimensions plus position identify it well enough here.
                    Content::Image(img) => {
                        let d = img.dims();
                        d.map(|d| (d.width, d.height)).hash(&mut h);
                    }
                    Content::Document { bytes } => bytes.hash(&mut h),
                }
            }
        }
        h.finish()
    }

    /// Count one content part.
    fn count_content(&self, c: &Content<'a>, tk: &dyn Tokenizer) -> Count {
        match c {
            Content::Text(t) | Content::Reasoning(t) | Content::ToolResult(t) => tk.count_text(t),
            Content::ToolUse { name, input } => tk.count_all(&[name, input]),
            // Geometry is arithmetic on a header the crate read itself, not an
            // estimate of text — but it is a documented formula rather than the
            // provider's own count, so it never claims to be exact.
            Content::Image(img) => Count::heuristic(img.tokens(self.family)),
            Content::Document { bytes } => Count::heuristic((*bytes / 4) as i64),
        }
    }

    /// Count via the provider's own endpoint, falling back to a local estimate.
    ///
    /// The fallback is the point: a rate limit or a network blip on a counting
    /// endpoint must never fail the operation that was only trying to measure
    /// itself. Check [`Count::source`] to see which path answered.
    #[must_use]
    pub fn count_via(&self, remote: &dyn RemoteCounter) -> Count {
        match remote.count(self.model, &self.remote_body()) {
            Some(n) => Count::remote(n),
            None => self.count(),
        }
    }

    /// Build an Anthropic-shaped `count_tokens` request body for this prompt.
    ///
    /// Text-bearing parts only: images and documents are referenced by the
    /// caller's own payload, which this neutral model does not retain. Use it
    /// with [`RemoteCounter`], or as a starting point for another dialect.
    #[must_use]
    pub fn remote_body(&self) -> Value {
        let messages: Vec<Value> = self
            .messages
            .iter()
            .map(|m| {
                let blocks: Vec<Value> = m
                    .content
                    .iter()
                    .filter_map(|c| match c {
                        Content::Text(t) | Content::Reasoning(t) | Content::ToolResult(t) => {
                            Some(json!({"type": "text", "text": t}))
                        }
                        Content::ToolUse { name, input } => {
                            Some(json!({"type": "text", "text": format!("{name}{input}")}))
                        }
                        Content::Image(_) | Content::Document { .. } => None,
                    })
                    .collect();
                json!({"role": m.role.as_str(), "content": blocks})
            })
            .collect();

        let mut body = json!({"model": self.model, "messages": messages});
        if !self.system.is_empty() {
            body["system"] = json!(self.system);
        }
        if !self.tools.is_empty() {
            let tools: Vec<Value> = self
                .tools
                .iter()
                .map(|t| json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.schema,
                }))
                .collect();
            body["tools"] = json!(tools);
        }
        body
    }
}

impl Default for Prompt<'_> {
    fn default() -> Self {
        Self::new("")
    }
}

/// Count a bare string for a model, with no conversation framing.
///
/// For sizing a single blob — a document, a retrieved chunk — rather than a
/// request.
#[must_use]
pub fn count_text(model: &str, text: &str) -> Count {
    crate::tokenizer::for_model(model).count_text(text)
}

/// The heuristic backend, for callers that want to force it.
#[must_use]
pub fn heuristic() -> Heuristic {
    Heuristic
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn framing_is_counted_on_top_of_the_text() {
        // 80 ASCII of system (20 tokens), two 40-char messages (10 each).
        let (a, b) = ("a".repeat(40), "b".repeat(40));
        let msgs = [
            Message::user([Content::Text(&a)]),
            Message::assistant([Content::Text(&b)]),
        ];
        let sys = "You are helpful.".repeat(5);
        let n = Prompt::new("gpt-5").system(&sys).messages(&msgs).count_with(&Heuristic);
        // base 3 + system (20 + 4) + msg (4 + 10) + msg (4 + 10) = 55
        assert_eq!(n.tokens, 55);
        assert_eq!(n.source, Source::Heuristic);
    }

    #[test]
    fn claude_and_gpt_frame_differently() {
        let msgs = [Message::user([Content::Text("hello")])];
        let gpt = Prompt::new("gpt-5").messages(&msgs).count_with(&Heuristic).tokens;
        let claude = Prompt::new("claude-sonnet-5").messages(&msgs).count_with(&Heuristic).tokens;
        // Claude: base 5 + 3 framing; GPT: base 3 + 4 framing. Same text.
        assert_eq!(claude - gpt, 1);
    }

    #[test]
    fn an_empty_prompt_is_just_the_preamble() {
        let n = Prompt::new("claude-sonnet-5").count_with(&Heuristic);
        assert_eq!(n.tokens, 5);
        assert!(n.source.is_precise(), "a constant is not an estimate");
    }

    #[test]
    fn replayed_thinking_is_prompt_and_is_counted() {
        let thinking = "t".repeat(400);
        let with = [Message::assistant([
            Content::Text("ok"),
            Content::Reasoning(&thinking),
        ])];
        let without = [Message::assistant([Content::Text("ok")])];
        let a = Prompt::new("claude-sonnet-5").messages(&with).count_with(&Heuristic).tokens;
        let b = Prompt::new("claude-sonnet-5").messages(&without).count_with(&Heuristic).tokens;
        assert_eq!(a - b, 100);
    }

    #[test]
    fn a_tool_call_counts_its_name_and_arguments() {
        let args = "a".repeat(200);
        let msgs = [Message::assistant([Content::ToolUse {
            name: "shell",
            input: &args,
        }])];
        let n = Prompt::new("gpt-5").messages(&msgs).count_with(&Heuristic).tokens;
        // 50 tokens of arguments + 2 of name, plus base 3 and framing 4.
        assert_eq!(n, 59);
    }

    #[test]
    fn an_image_is_counted_from_its_real_size() {
        let big = Image::Dims(ImageDims { width: 1024, height: 1024 });
        let msgs = [Message::user([Content::Image(big)])];
        let n = Prompt::new("claude-sonnet-5").messages(&msgs).count_with(&Heuristic).tokens;
        assert!(n > 1_000, "a full-page screenshot is not 85 tokens, got {n}");
    }

    #[test]
    fn an_unreadable_image_falls_back_to_the_flat_rate() {
        let msgs = [Message::user([Content::Image(Image::Bytes(b"garbage"))])];
        let n = Prompt::new("claude-sonnet-5").messages(&msgs).count_with(&Heuristic).tokens;
        assert_eq!(n, 5 + 3 + UNKNOWN_IMAGE_TOKENS);
    }

    #[test]
    fn tools_are_counted_and_change_the_total() {
        let schema = json!({
            "type": "object",
            "properties": {"path": {"type": "string", "description": "file path"}},
            "required": ["path"]
        });
        let tools = [Tool { name: "read_file", description: "Read a file", schema: &schema }];
        let msgs = [Message::user([Content::Text("hi")])];
        let with = Prompt::new("gpt-5").messages(&msgs).tools(&tools).count_with(&Heuristic);
        let without = Prompt::new("gpt-5").messages(&msgs).count_with(&Heuristic);
        assert!(with.tokens > without.tokens);
    }

    #[test]
    fn a_prefix_counts_less_than_the_whole_and_grows_with_it() {
        let a = "a".repeat(400);
        let msgs = [
            Message::user([Content::Text(&a)]),
            Message::assistant([Content::Text(&a)]),
            Message::user([Content::Text(&a)]),
        ];
        let p = Prompt::new("claude-sonnet-5").system("sys").messages(&msgs);

        let head = p.count_prefix_with(1, &Heuristic).tokens;
        let two = p.count_prefix_with(2, &Heuristic).tokens;
        let all = p.count_with(&Heuristic).tokens;

        assert!(head < two && two < all);
        assert_eq!(p.count_prefix_with(3, &Heuristic).tokens, all);
        assert_eq!(p.count_prefix_with(99, &Heuristic).tokens, all, "saturates");
    }

    #[test]
    fn a_fingerprint_tracks_content_not_length() {
        let (a, b) = ("a".repeat(400), "b".repeat(400));
        let one = [Message::user([Content::Text(&a)])];
        let other = [Message::user([Content::Text(&b)])];

        let f1 = Prompt::new("claude-sonnet-5").messages(&one).prefix_fingerprint(1);
        let f2 = Prompt::new("claude-sonnet-5").messages(&other).prefix_fingerprint(1);
        assert_ne!(f1, f2, "same length, different bytes, different prefix");

        // The model is part of cache identity.
        assert_ne!(
            Prompt::new("claude-sonnet-5").messages(&one).prefix_fingerprint(1),
            Prompt::new("claude-opus-4").messages(&one).prefix_fingerprint(1)
        );
        // And the extent is: a longer prefix is a different prefix.
        let two = [Message::user([Content::Text(&a)]), Message::user([Content::Text(&a)])];
        assert_ne!(
            Prompt::new("claude-sonnet-5").messages(&two).prefix_fingerprint(1),
            Prompt::new("claude-sonnet-5").messages(&two).prefix_fingerprint(2)
        );
    }

    #[test]
    fn the_remote_body_is_shaped_for_anthropic() {
        let msgs = [Message::user([Content::Text("hello")])];
        let body = Prompt::new("claude-sonnet-5").system("be terse").messages(&msgs).remote_body();
        assert_eq!(body["model"], "claude-sonnet-5");
        assert_eq!(body["system"], "be terse");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"][0]["text"], "hello");
    }

    #[test]
    fn a_remote_failure_degrades_to_a_local_estimate() {
        struct Down;
        impl RemoteCounter for Down {
            fn count(&self, _: &str, _: &Value) -> Option<i64> {
                None
            }
        }
        let text = "a".repeat(400);
        let msgs = [Message::user([Content::Text(&text)])];
        let n = Prompt::new("claude-sonnet-5").messages(&msgs).count_via(&Down);
        assert_eq!(n.source, Source::Heuristic, "and says so");
        assert!(n.tokens > 100);
    }

    #[test]
    fn a_remote_answer_is_marked_authoritative() {
        struct Up;
        impl RemoteCounter for Up {
            fn count(&self, _: &str, _: &Value) -> Option<i64> {
                Some(4_242)
            }
        }
        let n = Prompt::new("claude-sonnet-5").count_via(&Up);
        assert_eq!(n, Count { tokens: 4_242, source: Source::Remote });
    }
}
