//! Tool definitions, as the provider actually bills them.
//!
//! Tool schemas are JSON on the wire, and the obvious way to count them — run
//! the tokenizer over `serde_json::to_string(&schema)` — is wrong, in a
//! direction that matters. The GPT family does not put your JSON in the prompt.
//! It rewrites the whole tool list as a **TypeScript namespace declaration**:
//!
//! ```text
//! namespace functions {
//!
//! // Get the weather in a location
//! type get_weather = (_: {
//! // The city and state
//! location: string,
//! unit?: "celsius" | "fahrenheit",
//! }) => any;
//!
//! } // namespace functions
//! ```
//!
//! JSON spends tokens on quotes, braces and the words `"type"`, `"properties"`,
//! `"required"`; the declaration above spends them on the names and descriptions
//! that survive. Counting the JSON therefore over-estimates, and it does so on
//! the most stable, largest part of an agentic client's prompt — a coding agent
//! ships thousands of tokens of tool definitions on *every* turn.
//!
//! The transformation is reproduced here from the behaviour the community has
//! reverse-engineered against the API's own reported counts. It is not a
//! published specification, so [`count`] never returns [`Source::Exact`] on the
//! strength of this shape alone — it inherits whatever the tokenizer gives it,
//! and the fixed overheads below are best-effort constants.
//!
//! Claude and Gemini publish nothing equivalent and their serialization is not
//! TypeScript-shaped, so [`count`] falls back to measuring the JSON for them
//! (see [`Family::tools_as_typescript`]).

use crate::prompt::Tool;
use crate::{Count, Family, Tokenizer};
use serde_json::Value;

/// Tokens added once for the namespace wrapper around the tool list.
pub const DEFINITION_OVERHEAD: i64 = 9;

/// Tokens deducted when a system message is present alongside tools: the two
/// share a preamble that would otherwise be counted twice.
pub const SYSTEM_DEDUCTION: i64 = 4;

/// Render a tool list the way the GPT family serializes it into the prompt.
///
/// Exposed because it is worth being able to see, diff and test the exact string
/// that is being counted rather than trusting a number that came out of it.
#[must_use]
pub fn format_definitions(tools: &[Tool<'_>]) -> String {
    let mut lines: Vec<String> = vec!["namespace functions {".into(), String::new()];

    for tool in tools {
        if !tool.description.is_empty() {
            lines.push(format!("// {}", tool.description));
        }

        let properties = tool.schema.get("properties").and_then(Value::as_object);
        match properties {
            Some(props) if !props.is_empty() => {
                lines.push(format!("type {} = (_: {{", tool.name));
                let body = format_object_properties(tool.schema, 0);
                if !body.is_empty() {
                    lines.push(body);
                }
                lines.push("}) => any;".into());
            }
            // No parameters at all: a zero-argument declaration, not an empty
            // object literal. The provider elides the braces and so must this.
            _ => lines.push(format!("type {} = () => any;", tool.name)),
        }
        lines.push(String::new());
    }

    lines.push("} // namespace functions".into());
    lines.join("\n")
}

/// Render one object's properties as TypeScript members.
fn format_object_properties(obj: &Value, indent: usize) -> String {
    let Some(props) = obj.get("properties").and_then(Value::as_object) else {
        return String::new();
    };
    let required: Vec<&str> = obj
        .get("required")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();

    let pad = " ".repeat(indent);
    let mut lines: Vec<String> = Vec::with_capacity(props.len());

    for (name, param) in props {
        // Descriptions survive only at the top two levels. Deeper than that the
        // provider drops them, and counting them would inflate every deeply
        // nested schema.
        if indent < 2 {
            if let Some(d) = param.get("description").and_then(Value::as_str) {
                if !d.is_empty() {
                    lines.push(format!("{pad}// {d}"));
                }
            }
        }
        let optional = if required.iter().any(|r| r == name) { "" } else { "?" };
        lines.push(format!("{pad}{name}{optional}: {},", format_type(param, indent)));
    }

    lines.join("\n")
}

/// Render one JSON-Schema type as a TypeScript type expression.
fn format_type(param: &Value, indent: usize) -> String {
    let ty = param.get("type").and_then(Value::as_str).unwrap_or("");
    let enum_values = param.get("enum").and_then(Value::as_array);

    match ty {
        "string" => match enum_values {
            // Quoted and escaped, exactly as a TS string-literal union.
            Some(vals) => join_union(vals.iter().map(|v| match v {
                Value::String(s) => serde_json::to_string(s).unwrap_or_else(|_| format!("\"{s}\"")),
                other => other.to_string(),
            })),
            None => "string".into(),
        },
        "integer" | "number" => match enum_values {
            // Numeric literals are bare — no quotes.
            Some(vals) => join_union(vals.iter().map(ToString::to_string)),
            None => "number".into(),
        },
        "boolean" => "boolean".into(),
        "null" => "null".into(),
        "array" => match param.get("items") {
            Some(items) => format!("{}[]", format_type(items, indent)),
            None => "any[]".into(),
        },
        "object" => {
            let inner = format_object_properties(param, indent + 2);
            let closing = " ".repeat(indent);
            format!("{{\n{inner}\n{closing}}}")
        }
        _ => "any".into(),
    }
}

fn join_union<I: Iterator<Item = String>>(values: I) -> String {
    let joined: Vec<String> = values.collect();
    if joined.is_empty() {
        return "string".into();
    }
    joined.join(" | ")
}

/// Count what a tool list adds to a prompt.
///
/// `has_system` drives the [`SYSTEM_DEDUCTION`]: with a system message present
/// the tool preamble overlaps it, and the provider bills the overlap once.
#[must_use]
pub fn count(tools: &[Tool<'_>], family: Family, has_system: bool, tk: &dyn Tokenizer) -> Count {
    if tools.is_empty() {
        return Count { tokens: 0, source: crate::Source::Remote };
    }

    if family.tools_as_typescript() {
        let rendered = format_definitions(tools);
        let mut c = tk.count_text(&rendered).plus(DEFINITION_OVERHEAD);
        if has_system {
            c = c.plus(-SYSTEM_DEDUCTION);
        }
        return c;
    }

    // Claude and Gemini: no published serialization to reproduce, so measure the
    // parts that certainly reach the prompt — name, description and the schema
    // itself — and accept the JSON punctuation as slack.
    tools
        .iter()
        .map(|t| {
            let schema = t.schema.to_string();
            tk.count_all(&[t.name, t.description, &schema])
                .plus(family.per_message_overhead())
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Heuristic;
    use serde_json::json;

    fn weather_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        })
    }

    #[test]
    fn renders_the_typescript_declaration() {
        let schema = weather_schema();
        let tools = [Tool { name: "get_weather", description: "Get the weather", schema: &schema }];
        let out = format_definitions(&tools);

        assert!(out.starts_with("namespace functions {\n"));
        assert!(out.ends_with("} // namespace functions"));
        assert!(out.contains("// Get the weather"));
        assert!(out.contains("type get_weather = (_: {"));
        assert!(out.contains("// The city and state"));
        assert!(out.contains("location: string,"), "required params carry no `?`");
        assert!(out.contains(r#"unit?: "celsius" | "fahrenheit","#), "optional + enum union");
        assert!(out.contains("}) => any;"));
    }

    #[test]
    fn a_parameterless_tool_is_a_zero_argument_declaration() {
        let empty = json!({"type": "object", "properties": {}});
        let tools = [Tool { name: "ping", description: "", schema: &empty }];
        let out = format_definitions(&tools);
        assert!(out.contains("type ping = () => any;"));
        assert!(!out.contains("(_: {"), "no empty object literal");
    }

    #[test]
    fn numeric_enums_are_unquoted_and_string_enums_are_quoted() {
        let schema = json!({
            "type": "object",
            "properties": {
                "level": {"type": "integer", "enum": [1, 2, 3]},
                "mode": {"type": "string", "enum": ["fast", "slow"]}
            },
            "required": ["level", "mode"]
        });
        let tools = [Tool { name: "t", description: "", schema: &schema }];
        let out = format_definitions(&tools);
        assert!(out.contains("level: 1 | 2 | 3,"));
        assert!(out.contains(r#"mode: "fast" | "slow","#));
    }

    #[test]
    fn nested_objects_indent_and_drop_deep_descriptions() {
        let schema = json!({
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "description": "kept at depth 0",
                    "properties": {
                        "inner": {"type": "string", "description": "dropped below depth 2"}
                    }
                }
            }
        });
        let tools = [Tool { name: "t", description: "", schema: &schema }];
        let out = format_definitions(&tools);
        assert!(out.contains("// kept at depth 0"));
        assert!(!out.contains("dropped below depth 2"), "deep descriptions are not billed");
        assert!(out.contains("  inner?: string,"), "nested members are indented");
    }

    #[test]
    fn arrays_render_as_element_type_suffixed() {
        let schema = json!({
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
                "loose": {"type": "array"}
            }
        });
        let tools = [Tool { name: "t", description: "", schema: &schema }];
        let out = format_definitions(&tools);
        assert!(out.contains("tags?: string[],"));
        assert!(out.contains("loose?: any[],"));
    }

    /// The reason this module exists.
    ///
    /// On a single tiny tool the two approaches land within a few tokens of each
    /// other — the namespace wrapper costs about what the JSON punctuation
    /// saves. The gap opens at the scale an agentic client actually ships:
    /// several tools, each with several described parameters, where JSON repeats
    /// `{"type":"string","description":...}` per parameter and the declaration
    /// spends `: string,`.
    #[test]
    fn the_typescript_shape_undercuts_json_at_agentic_scale() {
        let read = json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file to read"},
                "offset": {"type": "integer", "description": "Line number to start from"},
                "limit": {"type": "integer", "description": "How many lines to read"}
            },
            "required": ["path"]
        });
        let edit = json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file to edit"},
                "old": {"type": "string", "description": "Exact text to replace"},
                "new": {"type": "string", "description": "Replacement text"},
                "all": {"type": "boolean", "description": "Replace every occurrence"}
            },
            "required": ["path", "old", "new"]
        });
        let shell = json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command line to run"},
                "timeout": {"type": "integer", "description": "Milliseconds before giving up"},
                "mode": {"type": "string", "enum": ["foreground", "background"]}
            },
            "required": ["command"]
        });
        let tools = [
            Tool { name: "read_file", description: "Read a file from disk", schema: &read },
            Tool { name: "edit_file", description: "Replace text in a file", schema: &edit },
            Tool { name: "run_shell", description: "Run a shell command", schema: &shell },
        ];
        let tk = Heuristic;

        let ts = count(&tools, Family::Gpt, false, &tk).tokens;
        // What measuring the JSON costs — the approach this module replaces.
        let as_json: i64 = tools
            .iter()
            .map(|t| {
                crate::heuristic::estimate_text(t.name)
                    + crate::heuristic::estimate_text(t.description)
                    + crate::heuristic::estimate_text(&t.schema.to_string())
            })
            .sum();

        assert!(
            ts < as_json,
            "TS declaration {ts} should undercut the JSON estimate {as_json}"
        );
    }

    #[test]
    fn a_system_message_deducts_the_shared_preamble() {
        let schema = weather_schema();
        let tools = [Tool { name: "get_weather", description: "d", schema: &schema }];
        let tk = Heuristic;
        let without = count(&tools, Family::Gpt, false, &tk).tokens;
        let with = count(&tools, Family::Gpt, true, &tk).tokens;
        assert_eq!(without - with, SYSTEM_DEDUCTION);
    }

    #[test]
    fn non_gpt_families_do_not_get_the_typescript_treatment() {
        let schema = weather_schema();
        let tools = [Tool { name: "get_weather", description: "d", schema: &schema }];
        let tk = Heuristic;
        let gpt = count(&tools, Family::Gpt, false, &tk).tokens;
        let claude = count(&tools, Family::Claude, false, &tk).tokens;
        assert_ne!(gpt, claude);
    }

    #[test]
    fn no_tools_costs_nothing_and_taints_nothing() {
        let c = count(&[], Family::Gpt, true, &Heuristic);
        assert_eq!(c.tokens, 0);
        assert!(c.source.is_precise(), "an absent tool list is not an estimate");
    }
}
