//! Translation layer between the Anthropic Messages API (what Claude Code
//! speaks) and the OpenAI-compatible chat-completions API (what
//! gpu-server-hard/llama-server speaks, and what loop-guard's existing
//! `/v1/chat/completions` route already forwards for opencode).
//!
//! This lets Claude Code point straight at loop-guard's `/v1/messages` and
//! get the same backend model - and the same <think>-loop protection - that
//! opencode already gets, without adopting a heavier general-purpose
//! gateway (LiteLLM, claude-code-router) just for the request/response
//! shape translation.

use serde_json::{json, Value};
use std::collections::HashMap;

/// Sampling params opencode already defines for this model (config/opencode
/// in the dotfiles repo): one set for "thinking on", one for "thinking
/// off" (same model and endpoint, requested with a different
/// chat_template_kwargs flag). Claude Code signals which one it wants via
/// the top-level `thinking` field on the Anthropic request.
pub fn thinking_enabled(body: &Value) -> bool {
    body.get("thinking")
        .and_then(|t| t.get("type"))
        .and_then(Value::as_str)
        == Some("enabled")
}

fn apply_sampling_params(openai_body: &mut Value, thinking: bool) {
    if thinking {
        openai_body["temperature"] = json!(0.6);
        openai_body["top_p"] = json!(0.95);
        openai_body["top_k"] = json!(20);
        openai_body["min_p"] = json!(0);
        openai_body["presence_penalty"] = json!(1.0);
    } else {
        openai_body["temperature"] = json!(0.7);
        openai_body["top_p"] = json!(0.8);
        openai_body["top_k"] = json!(20);
        openai_body["min_p"] = json!(0);
        openai_body["presence_penalty"] = json!(1.5);
        openai_body["chat_template_kwargs"] = json!({ "enable_thinking": false });
    }
}

fn system_text(body: &Value) -> Option<String> {
    match body.get("system") {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        Some(Value::Array(blocks)) => {
            let joined: String = blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n");
            if joined.is_empty() {
                None
            } else {
                Some(joined)
            }
        }
        _ => None,
    }
}

fn block_text(block: &Value) -> Option<&str> {
    if block.get("type").and_then(Value::as_str) == Some("text") {
        block.get("text").and_then(Value::as_str)
    } else {
        None
    }
}

/// A tool_result's `content` can itself be a string or an array of blocks
/// (usually text). Flatten it to a single string for the OpenAI `tool`
/// message, which only supports plain string content.
fn tool_result_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(block_text)
            .collect::<Vec<_>>()
            .join("\n"),
        other => other.to_string(),
    }
}

fn convert_message(role: &str, content: &Value, out: &mut Vec<Value>) {
    if let Some(s) = content.as_str() {
        if !s.is_empty() {
            out.push(json!({ "role": role, "content": s }));
        }
        return;
    }
    let Some(blocks) = content.as_array() else { return };

    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut tool_results: Vec<(String, String)> = Vec::new();

    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(t) = block.get("text").and_then(Value::as_str) {
                    text_parts.push(t.to_string());
                }
            }
            Some("tool_use") if role == "assistant" => {
                let id = block.get("id").and_then(Value::as_str).unwrap_or("");
                let name = block.get("name").and_then(Value::as_str).unwrap_or("");
                let input = block.get("input").cloned().unwrap_or(json!({}));
                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": serde_json::to_string(&input).unwrap_or_else(|_| "{}".to_string()),
                    }
                }));
            }
            Some("tool_result") if role == "user" => {
                let id = block
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let content = block.get("content").cloned().unwrap_or(json!(""));
                tool_results.push((id, tool_result_text(&content)));
            }
            _ => {}
        }
    }

    // Tool results become their own `tool` messages first (they answer a
    // preceding assistant tool_calls message), then any accompanying text
    // becomes a follow-up user message.
    for (id, text) in tool_results {
        out.push(json!({ "role": "tool", "tool_call_id": id, "content": text }));
    }

    let joined_text = text_parts.join("\n");
    if role == "assistant" {
        if !joined_text.is_empty() || !tool_calls.is_empty() {
            let mut msg = json!({ "role": "assistant" });
            msg["content"] = if joined_text.is_empty() {
                Value::Null
            } else {
                json!(joined_text)
            };
            if !tool_calls.is_empty() {
                msg["tool_calls"] = json!(tool_calls);
            }
            out.push(msg);
        }
    } else if !joined_text.is_empty() {
        out.push(json!({ "role": "user", "content": joined_text }));
    }
}

fn convert_tools(body: &Value) -> Option<Value> {
    let tools = body.get("tools")?.as_array()?;
    if tools.is_empty() {
        return None;
    }
    let converted: Vec<Value> = tools
        .iter()
        .map(|t| {
            json!({
                "type": "function",
                "function": {
                    "name": t.get("name").and_then(Value::as_str).unwrap_or(""),
                    "description": t.get("description").and_then(Value::as_str).unwrap_or(""),
                    "parameters": t.get("input_schema").cloned().unwrap_or(json!({"type":"object","properties":{}})),
                }
            })
        })
        .collect();
    Some(json!(converted))
}

fn convert_tool_choice(body: &Value) -> Option<Value> {
    let tc = body.get("tool_choice")?;
    match tc.get("type").and_then(Value::as_str) {
        Some("auto") => Some(json!("auto")),
        Some("none") => Some(json!("none")),
        Some("any") => Some(json!("required")),
        Some("tool") => {
            let name = tc.get("name").and_then(Value::as_str).unwrap_or("");
            Some(json!({ "type": "function", "function": { "name": name } }))
        }
        _ => None,
    }
}

/// Translate a full Anthropic Messages API request body into the
/// OpenAI-compatible chat-completions body llama-server expects.
pub fn anthropic_to_openai_request(body: &Value, model_id: &str) -> Value {
    let thinking = thinking_enabled(body);

    let mut messages: Vec<Value> = Vec::new();
    if let Some(sys) = system_text(body) {
        messages.push(json!({ "role": "system", "content": sys }));
    }
    if let Some(anthropic_messages) = body.get("messages").and_then(Value::as_array) {
        for m in anthropic_messages {
            let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
            if let Some(content) = m.get("content") {
                convert_message(role, content, &mut messages);
            }
        }
    }

    let mut openai_body = json!({
        "model": model_id,
        "messages": messages,
        "stream": body.get("stream").and_then(Value::as_bool).unwrap_or(false),
    });

    if let Some(max_tokens) = body.get("max_tokens") {
        openai_body["max_tokens"] = max_tokens.clone();
    }
    if let Some(tools) = convert_tools(body) {
        openai_body["tools"] = tools;
    }
    if let Some(tool_choice) = convert_tool_choice(body) {
        openai_body["tool_choice"] = tool_choice;
    }

    apply_sampling_params(&mut openai_body, thinking);
    openai_body
}

/// Very rough token estimate (chars/4) for the `/v1/messages/count_tokens`
/// endpoint - Claude Code only uses this for context-window bookkeeping and
/// tolerates an approximate count; the real limit enforcement happens
/// server-side against the model's actual context window.
pub fn estimate_input_tokens(body: &Value) -> u64 {
    let mut chars = 0usize;
    if let Some(sys) = system_text(body) {
        chars += sys.len();
    }
    if let Some(messages) = body.get("messages").and_then(Value::as_array) {
        chars += serde_json::to_string(messages).map(|s| s.len()).unwrap_or(0);
    }
    if let Some(tools) = body.get("tools") {
        chars += serde_json::to_string(tools).map(|s| s.len()).unwrap_or(0);
    }
    ((chars as f64) / 4.0).ceil() as u64
}

fn map_stop_reason(finish_reason: Option<&str>) -> &'static str {
    match finish_reason {
        Some("tool_calls") => "tool_use",
        Some("length") => "max_tokens",
        _ => "end_turn",
    }
}

/// Translate a complete (non-streaming) OpenAI chat-completion response
/// into an Anthropic `message` object.
pub fn openai_response_to_anthropic(resp: &Value, model_id: &str) -> Value {
    let choice = &resp["choices"][0];
    let message = &choice["message"];
    let finish_reason = choice.get("finish_reason").and_then(Value::as_str);

    let mut content: Vec<Value> = Vec::new();
    if let Some(reasoning) = message.get("reasoning_content").and_then(Value::as_str) {
        if !reasoning.is_empty() {
            content.push(json!({ "type": "thinking", "thinking": reasoning }));
        }
    }
    if let Some(text) = message.get("content").and_then(Value::as_str) {
        if !text.is_empty() {
            content.push(json!({ "type": "text", "text": text }));
        }
    }
    if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
        for tc in tool_calls {
            let id = tc.get("id").and_then(Value::as_str).unwrap_or("");
            let name = tc["function"].get("name").and_then(Value::as_str).unwrap_or("");
            let args_str = tc["function"].get("arguments").and_then(Value::as_str).unwrap_or("{}");
            let input: Value = serde_json::from_str(args_str).unwrap_or(json!({}));
            content.push(json!({ "type": "tool_use", "id": id, "name": name, "input": input }));
        }
    }

    json!({
        "id": resp.get("id").and_then(Value::as_str).unwrap_or("msg_local"),
        "type": "message",
        "role": "assistant",
        "model": model_id,
        "content": content,
        "stop_reason": map_stop_reason(finish_reason),
        "stop_sequence": Value::Null,
        "usage": {
            "input_tokens": resp["usage"].get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0),
            "output_tokens": resp["usage"].get("completion_tokens").and_then(Value::as_u64).unwrap_or(0),
        }
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Thinking,
    Text,
    ToolUse,
}

/// Incrementally converts a stream of OpenAI `chat.completion.chunk` delta
/// objects into the Anthropic Messages streaming SSE event sequence
/// (message_start / content_block_start+delta+stop / message_delta /
/// message_stop), tracking which content block is currently open so text,
/// thinking, and tool-call segments each get their own block as required
/// by the Anthropic wire format.
pub struct AnthropicStreamState {
    message_id: String,
    model: String,
    next_index: usize,
    current_block: Option<(BlockKind, usize)>,
    /// OpenAI tool_call stream index -> (anthropic block index, whether the
    /// block_start has been emitted yet - it's deferred until we see a
    /// name, since the first delta for a tool call carries id+name but the
    /// arguments may start empty).
    tool_blocks: HashMap<u64, usize>,
    output_chars: usize,
    started: bool,
}

impl AnthropicStreamState {
    pub fn new(message_id: String, model: String) -> Self {
        Self {
            message_id,
            model,
            next_index: 0,
            current_block: None,
            tool_blocks: HashMap::new(),
            output_chars: 0,
            started: false,
        }
    }

    pub fn message_start_event(&mut self) -> String {
        self.started = true;
        sse_event(
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.model,
                    "stop_reason": Value::Null,
                    "stop_sequence": Value::Null,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                }
            }),
        )
    }

    fn open_block(&mut self, kind: BlockKind, content_block: Value) -> String {
        let idx = self.next_index;
        self.next_index += 1;
        self.current_block = Some((kind, idx));
        sse_event(
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": idx,
                "content_block": content_block,
            }),
        )
    }

    fn close_current_block(&mut self, out: &mut Vec<String>) {
        if let Some((_, idx)) = self.current_block.take() {
            out.push(sse_event(
                "content_block_stop",
                &json!({ "type": "content_block_stop", "index": idx }),
            ));
        }
    }

    /// Feed one OpenAI `delta` object (`choices[0].delta`). Returns the
    /// Anthropic SSE events it produces, in order.
    pub fn feed_delta(&mut self, delta: &Value) -> Vec<String> {
        let mut out = Vec::new();

        if let Some(reasoning) = delta.get("reasoning_content").and_then(Value::as_str) {
            if !reasoning.is_empty() {
                if !matches!(self.current_block, Some((BlockKind::Thinking, _))) {
                    self.close_current_block(&mut out);
                    out.push(self.open_block(BlockKind::Thinking, json!({"type": "thinking", "thinking": ""})));
                }
                let idx = self.current_block.unwrap().1;
                self.output_chars += reasoning.len();
                out.push(sse_event(
                    "content_block_delta",
                    &json!({
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "thinking_delta", "thinking": reasoning},
                    }),
                ));
            }
        }

        if let Some(text) = delta.get("content").and_then(Value::as_str) {
            if !text.is_empty() {
                if !matches!(self.current_block, Some((BlockKind::Text, _))) {
                    self.close_current_block(&mut out);
                    out.push(self.open_block(BlockKind::Text, json!({"type": "text", "text": ""})));
                }
                let idx = self.current_block.unwrap().1;
                self.output_chars += text.len();
                out.push(sse_event(
                    "content_block_delta",
                    &json!({
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "text_delta", "text": text},
                    }),
                ));
            }
        }

        if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for tc in tool_calls {
                let openai_idx = tc.get("index").and_then(Value::as_u64).unwrap_or(0);
                let anthropic_idx = if let Some(&idx) = self.tool_blocks.get(&openai_idx) {
                    idx
                } else {
                    self.close_current_block(&mut out);
                    let id = tc.get("id").and_then(Value::as_str).unwrap_or("").to_string();
                    let name = tc["function"].get("name").and_then(Value::as_str).unwrap_or("").to_string();
                    out.push(self.open_block(
                        BlockKind::ToolUse,
                        json!({"type": "tool_use", "id": id, "name": name, "input": {}}),
                    ));
                    let idx = self.current_block.unwrap().1;
                    self.tool_blocks.insert(openai_idx, idx);
                    idx
                };
                if let Some(args) = tc["function"].get("arguments").and_then(Value::as_str) {
                    if !args.is_empty() {
                        out.push(sse_event(
                            "content_block_delta",
                            &json!({
                                "type": "content_block_delta",
                                "index": anthropic_idx,
                                "delta": {"type": "input_json_delta", "partial_json": args},
                            }),
                        ));
                    }
                }
            }
        }

        out
    }

    /// Close whatever block is open and emit `message_delta` +
    /// `message_stop`. Call exactly once, at the end of the stream.
    pub fn finish(&mut self, finish_reason: Option<&str>) -> Vec<String> {
        let mut out = Vec::new();
        self.close_current_block(&mut out);
        let output_tokens = ((self.output_chars as f64) / 4.0).ceil() as u64;
        out.push(sse_event(
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": {"stop_reason": map_stop_reason(finish_reason), "stop_sequence": Value::Null},
                "usage": {"output_tokens": output_tokens},
            }),
        ));
        out.push(sse_event("message_stop", &json!({ "type": "message_stop" })));
        out
    }
}

fn sse_event(event: &str, data: &Value) -> String {
    format!("event: {event}\ndata: {}\n\n", data)
}
