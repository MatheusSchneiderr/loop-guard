//! loop-guard: a thin reverse proxy in front of gpu-server-hard (llama-server)
//! that detects a real, reproduced failure mode - Qwen3.6-35B-A3B getting
//! stuck restating the same dead-end hypothesis in different words inside
//! its own <think> block, sometimes even while making real tool calls,
//! never converging on an answer. Fixed reasoning-token budgets can't fix
//! this (a legitimately hard task and a stuck loop both need "however long
//! it takes" - there's no N that's right for both), and neither can a
//! timeout (same flaw, different unit). See tracker.rs for the actual
//! content-based detection (bag-of-words cosine similarity between
//! reasoning "steps", segmented at this model's own restart markers).
//!
//! On a hit: cancel the in-flight request, call llama-server's own
//! /apply-template to get the exact rendered prompt, splice in everything
//! generated so far plus a forced "</think>" closing tag and a short
//! explicit nudge, and continue via the raw /completion endpoint - the
//! same "budget forcing" technique from the s1 reasoning-scaling paper and
//! vLLM's ThinkingTokenBudgetLogitsProcessor, just triggered by content
//! instead of a token count, one layer above the engine instead of inside
//! it. The continuation targets the SAME llama-server slot the original
//! request was using (via /slots + id_slot) so it resumes from the
//! existing KV cache instead of reprocessing the whole trace from
//! scratch - the first (C++) prototype of this got that wrong (a racy
//! before/after /slots diff), confirmed live to sometimes miss and fall
//! back to a ~60-90s full reprocess; this version queries /slots once,
//! right at the moment of detection, and asks directly.
//!
//! Second implementation, in Rust instead of the original C++ prototype -
//! same httplib-class reverse-proxy shape as this project's existing
//! gpu-guard, but rewritten for memory-safety guarantees (no manual
//! lifetime/thread-synchronization code to get subtly wrong) given this is
//! network-facing, AI-authored, and sits in the hot path of a daily-driver
//! coding assistant.

mod anthropic;
mod tracker;

use axum::{
    body::Body,
    extract::State,
    http::{HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use bytes::Bytes;
use chrono::Local;
use serde_json::{json, Value};
use std::env;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tracker::StepTracker;

#[derive(Clone)]
struct AppState {
    upstream_base: String,
    threshold: f64,
    min_step_words: usize,
    verbose: bool,
    client: reqwest::Client,
    request_counter: Arc<AtomicU64>,
    model_name: String,
}

fn now_stamp() -> String {
    Local::now().format("%Y-%m-%dT%H:%M:%S").to_string()
}

fn env_or(name: &str, fallback: &str) -> String {
    env::var(name).ok().filter(|v| !v.is_empty()).unwrap_or_else(|| fallback.to_string())
}
fn env_or_f64(name: &str, fallback: f64) -> f64 {
    env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(fallback)
}
fn env_or_usize(name: &str, fallback: usize) -> usize {
    env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(fallback)
}

/// Splits a raw SSE byte chunk into complete "data: ..." JSON payloads,
/// carrying any partial trailing line over in `carry`. Mirrors the same
/// helper from the original C++ prototype.
fn extract_sse_payloads(chunk: &str, carry: &mut String) -> Vec<String> {
    let mut out = Vec::new();
    carry.push_str(chunk);
    let data = std::mem::take(carry);
    let mut rest = data.as_str();
    loop {
        match rest.find('\n') {
            None => {
                *carry = rest.to_string();
                break;
            }
            Some(nl) => {
                let line = &rest[..nl];
                rest = &rest[nl + 1..];
                let line = line.strip_suffix('\r').unwrap_or(line);
                if let Some(payload) = line.strip_prefix("data: ") {
                    if payload != "[DONE]" {
                        out.push(payload.to_string());
                    }
                }
            }
        }
    }
    out
}

fn make_chat_chunk(id: &str, model: &str, created: i64, delta: Value, finish_reason: Option<&str>) -> String {
    let chunk = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }]
    });
    format!("data: {}\n\n", chunk)
}

/// Query /slots and return the id of whichever slot is currently
/// processing something, at the exact moment of the call. Assumes at most
/// one concurrent generation in flight (true for this deployment - the
/// title-generation contention that used to violate this was fixed
/// separately by routing opencode's small_model at an unreachable port).
/// Returns None if zero or more than one slot is busy (ambiguous - safer
/// to fall back to no id_slot, i.e. a full reprocess, than to guess wrong
/// and corrupt an unrelated request's cache).
async fn current_processing_slot(client: &reqwest::Client, upstream_base: &str) -> Option<i64> {
    let res = client.get(format!("{upstream_base}/slots")).send().await.ok()?;
    let slots: Value = res.json().await.ok()?;
    let arr = slots.as_array()?;
    let mut found: Option<i64> = None;
    for s in arr {
        if s.get("is_processing").and_then(Value::as_bool) == Some(true) {
            if found.is_some() {
                return None; // more than one busy - ambiguous
            }
            found = s.get("id").and_then(Value::as_i64);
        }
    }
    found
}

async fn apply_template(client: &reqwest::Client, upstream_base: &str, messages: &Value) -> Option<String> {
    let res = client
        .post(format!("{upstream_base}/apply-template"))
        .json(&json!({ "messages": messages }))
        .send()
        .await
        .ok()?;
    let parsed: Value = res.json().await.ok()?;
    parsed.get("prompt").and_then(Value::as_str).map(|s| s.to_string())
}

const LOOP_NUDGE_PREFIX: &str =
    "\n\n(loop-guard: repetition detected - you were restating an earlier \
     point instead of making progress. Stop analyzing further and commit \
     to your best answer now, using whatever you've already found.)\n";

async fn health(State(state): State<AppState>) -> Response {
    match state.client.get(format!("{}/health", state.upstream_base)).send().await {
        Ok(res) => {
            let status = res.status();
            let body = res.text().await.unwrap_or_default();
            (status, [("content-type", "application/json")], body).into_response()
        }
        Err(_) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({"error": "loop-guard: upstream unreachable"})),
        )
            .into_response(),
    }
}

async fn chat_completions(State(state): State<AppState>, body: Bytes) -> Response {
    let parsed: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "loop-guard: invalid JSON body"})),
            )
                .into_response();
        }
    };

    let wants_stream = parsed.get("stream").and_then(Value::as_bool).unwrap_or(false);

    if !wants_stream {
        // Non-streaming requests can't be intervened on mid-flight the
        // same way (no incremental deltas to inspect before the full
        // response already exists) - pass through untouched rather than
        // pretend to protect them.
        let res = state
            .client
            .post(format!("{}/v1/chat/completions", state.upstream_base))
            .json(&parsed)
            .send()
            .await;
        return match res {
            Ok(r) => {
                let status = r.status();
                let bytes = r.bytes().await.unwrap_or_default();
                (status, [("content-type", "application/json")], bytes).into_response()
            }
            Err(_) => (
                StatusCode::BAD_GATEWAY,
                Json(json!({"error": "loop-guard: upstream unreachable"})),
            )
                .into_response(),
        };
    }

    let (tx, rx) = mpsc::channel::<String>(64);
    let request_id = format!("req-{}", state.request_counter.fetch_add(1, Ordering::Relaxed));
    let messages = parsed.get("messages").cloned().unwrap_or(json!([]));

    tokio::spawn(run_streaming_request(state, parsed, messages, request_id, tx));

    let stream = ReceiverStream::new(rx).map(|s| Ok::<_, std::io::Error>(Bytes::from(s)));
    let mut res = Response::new(Body::from_stream(stream));
    res.headers_mut().insert(
        "content-type",
        HeaderValue::from_static("text/event-stream"),
    );
    res
}

async fn run_streaming_request(
    state: AppState,
    original_body: Value,
    messages: Value,
    request_id: String,
    tx: mpsc::Sender<String>,
) {
    if state.verbose {
        eprintln!("[{}] loop-guard: {} started", now_stamp(), request_id);
    }

    let mut tracker = StepTracker::new(state.threshold, state.min_step_words, request_id.clone());
    let mut carry = String::new();
    let mut full_text = String::new();
    let mut meta_id = String::new();
    let mut meta_model = String::new();
    let mut meta_created: i64 = 0;
    let mut meta_seen = false;

    let upstream_res = match state
        .client
        .post(format!("{}/v1/chat/completions", state.upstream_base))
        .json(&original_body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(_) => {
            let _ = tx
                .send("data: {\"error\":\"loop-guard: upstream unreachable\"}\n\n".to_string())
                .await;
            return;
        }
    };

    let mut byte_stream = upstream_res.bytes_stream();
    let mut triggered_reason: Option<String> = None;
    let mut saw_own_done = false;

    'outer: while let Some(chunk) = byte_stream.next().await {
        let Ok(chunk) = chunk else { break };
        let chunk_str = String::from_utf8_lossy(&chunk).to_string();
        if chunk_str.contains("data: [DONE]") {
            saw_own_done = true;
        }
        let payloads = extract_sse_payloads(&chunk_str, &mut carry);

        for p in &payloads {
            let Ok(parsed): Result<Value, _> = serde_json::from_str(p) else { continue };
            if !meta_seen {
                meta_id = parsed.get("id").and_then(Value::as_str).unwrap_or("chatcmpl-loop-guard").to_string();
                meta_model = parsed.get("model").and_then(Value::as_str).unwrap_or("loop-guard").to_string();
                meta_created = parsed.get("created").and_then(Value::as_i64).unwrap_or(0);
                meta_seen = true;
            }
            let delta = &parsed["choices"][0]["delta"];
            let piece = delta
                .get("reasoning_content")
                .and_then(Value::as_str)
                .or_else(|| delta.get("content").and_then(Value::as_str))
                .unwrap_or("");
            if !piece.is_empty() {
                full_text.push_str(piece);
                for (event, hit) in tracker.feed(piece) {
                    if state.verbose {
                        eprintln!(
                            "[{}] loop-guard: {} step {} best-match={:?} similarity={:.6}",
                            now_stamp(),
                            request_id,
                            event.step_idx,
                            event.best_match_idx,
                            event.similarity
                        );
                    }
                    if let Some(h) = hit {
                        triggered_reason = Some(h.reason);
                        break;
                    }
                }
            }
            if triggered_reason.is_some() {
                break 'outer;
            }
        }
        if triggered_reason.is_none() {
            if tx.send(chunk_str).await.is_err() {
                return; // client disconnected
            }
        }
    }

    let Some(reason) = triggered_reason else {
        // Natural completion (or upstream error) - passthrough already
        // forwarded everything (including upstream's own [DONE], if it
        // sent one) - only add our own if it didn't, to avoid a duplicate.
        if !saw_own_done {
            let _ = tx.send("data: [DONE]\n\n".to_string()).await;
        }
        return;
    };

    eprintln!("[{}] loop-guard: TRIGGERED - {}", now_stamp(), reason);

    // --- Budget-forcing intervention ---
    let slot_id = current_processing_slot(&state.client, &state.upstream_base).await;

    let Some(base_prompt) = apply_template(&state.client, &state.upstream_base, &messages).await else {
        let _ = tx.send("data: {\"error\":\"loop-guard: apply-template failed\"}\n\n".to_string()).await;
        return;
    };

    let continuation_prompt = format!("{base_prompt}{full_text}{LOOP_NUDGE_PREFIX}</think>\n\n");

    let _ = tx
        .send(make_chat_chunk(
            &meta_id,
            &meta_model,
            meta_created,
            json!({"reasoning_content": format!("{LOOP_NUDGE_PREFIX}</think>\n\n")}),
            None,
        ))
        .await;

    let mut completion_req = json!({
        "prompt": continuation_prompt,
        "stream": true,
        "n_predict": original_body.get("max_tokens").and_then(Value::as_i64).unwrap_or(4096),
    });
    if let Some(id) = slot_id {
        // Reuse the exact slot that held this trace's KV cache - without
        // this, llama-server's own slot auto-selection isn't guaranteed to
        // pick it back up (confirmed live, twice, on the original
        // prototype: it picked an idle slot instead, forcing a full
        // multi-thousand-token reprocess - 60-90s of visible silence -
        // instead of resuming from the cache already sitting right there).
        completion_req["id_slot"] = json!(id);
    }
    for key in ["temperature", "top_p", "top_k", "min_p", "presence_penalty"] {
        if let Some(v) = original_body.get(key) {
            completion_req[key] = v.clone();
        }
    }

    let completion_res = match state
        .client
        .post(format!("{}/completion", state.upstream_base))
        .json(&completion_req)
        .send()
        .await
    {
        Ok(r) => r,
        Err(_) => {
            let _ = tx.send("data: [DONE]\n\n".to_string()).await;
            return;
        }
    };

    let mut carry2 = String::new();
    let mut stream2 = completion_res.bytes_stream();
    while let Some(chunk) = stream2.next().await {
        let Ok(chunk) = chunk else { break };
        let chunk_str = String::from_utf8_lossy(&chunk).to_string();
        let payloads = extract_sse_payloads(&chunk_str, &mut carry2);
        for p in payloads {
            let Ok(parsed): Result<Value, _> = serde_json::from_str(&p) else { continue };
            if let Some(content) = parsed.get("content").and_then(Value::as_str) {
                if !content.is_empty() {
                    let out = make_chat_chunk(&meta_id, &meta_model, meta_created, json!({"content": content}), None);
                    if tx.send(out).await.is_err() {
                        return;
                    }
                }
            }
            if parsed.get("stop").and_then(Value::as_bool) == Some(true) {
                let out = make_chat_chunk(&meta_id, &meta_model, meta_created, json!({}), Some("stop"));
                let _ = tx.send(out).await;
            }
        }
    }
    let _ = tx.send("data: [DONE]\n\n".to_string()).await;
}

// --- Anthropic Messages API front (for Claude Code) ---
//
// Claude Code only speaks the Anthropic Messages API, not the
// OpenAI-compatible format the `/v1/chat/completions` route above (and
// opencode) use. These two handlers translate a Messages API request into
// the same OpenAI-shaped request the rest of this file already knows how
// to drive (including the loop-detection/budget-forcing intervention),
// then translate the response back. See anthropic.rs for the actual
// field-by-field translation.

async fn messages_count_tokens(Json(body): Json<Value>) -> Json<Value> {
    Json(json!({ "input_tokens": anthropic::estimate_input_tokens(&body) }))
}

async fn messages(State(state): State<AppState>, body: Bytes) -> Response {
    let parsed: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": {"type": "invalid_request_error", "message": "loop-guard: invalid JSON body"}})),
            )
                .into_response();
        }
    };

    let openai_body = anthropic::anthropic_to_openai_request(&parsed, &state.model_name);
    let wants_stream = openai_body.get("stream").and_then(Value::as_bool).unwrap_or(false);

    if !wants_stream {
        let res = state
            .client
            .post(format!("{}/v1/chat/completions", state.upstream_base))
            .json(&openai_body)
            .send()
            .await;
        return match res {
            Ok(r) => {
                let status = r.status();
                match r.json::<Value>().await {
                    Ok(openai_resp) if status.is_success() => {
                        let anthropic_resp = anthropic::openai_response_to_anthropic(&openai_resp, &state.model_name);
                        (StatusCode::OK, Json(anthropic_resp)).into_response()
                    }
                    _ => (
                        StatusCode::BAD_GATEWAY,
                        Json(json!({"error": {"type": "api_error", "message": "loop-guard: upstream error"}})),
                    )
                        .into_response(),
                }
            }
            Err(_) => (
                StatusCode::BAD_GATEWAY,
                Json(json!({"error": {"type": "api_error", "message": "loop-guard: upstream unreachable"}})),
            )
                .into_response(),
        };
    }

    let (tx, rx) = mpsc::channel::<String>(64);
    let request_id = format!("req-{}", state.request_counter.fetch_add(1, Ordering::Relaxed));
    let messages = openai_body.get("messages").cloned().unwrap_or(json!([]));

    tokio::spawn(run_messages_stream(state, openai_body, messages, request_id, tx));

    let stream = ReceiverStream::new(rx).map(|s| Ok::<_, std::io::Error>(Bytes::from(s)));
    let mut res = Response::new(Body::from_stream(stream));
    res.headers_mut().insert(
        "content-type",
        HeaderValue::from_static("text/event-stream"),
    );
    res
}

async fn run_messages_stream(
    state: AppState,
    original_body: Value,
    request_messages: Value,
    request_id: String,
    tx: mpsc::Sender<String>,
) {
    if state.verbose {
        eprintln!("[{}] loop-guard: {} (anthropic) started", now_stamp(), request_id);
    }

    let message_id = format!("msg-{}", request_id);
    let mut anthropic_state = anthropic::AnthropicStreamState::new(message_id, state.model_name.clone());
    let _ = tx.send(anthropic_state.message_start_event()).await;

    let mut tracker = StepTracker::new(state.threshold, state.min_step_words, request_id.clone());
    let mut carry = String::new();
    let mut full_text = String::new();

    let upstream_res = match state
        .client
        .post(format!("{}/v1/chat/completions", state.upstream_base))
        .json(&original_body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(_) => {
            for ev in anthropic_state.finish(Some("stop")) {
                let _ = tx.send(ev).await;
            }
            return;
        }
    };

    let mut byte_stream = upstream_res.bytes_stream();
    let mut triggered_reason: Option<String> = None;
    let mut finish_reason: Option<String> = None;

    // llama-server sends nothing at all while prefilling a long prompt
    // (Claude Code's system prompt + full tool schemas routinely run
    // 10-40k tokens, which can take minutes to prefill on this quantized
    // model). Claude Code's own stream watchdogs treat total silence on
    // the wire as a dead connection and abort+retry - which, since a retry
    // resends the same huge prompt to a *different*, now-busy GPU slot
    // instead of reusing the one already mid-prefill, only compounds the
    // problem (confirmed live: a retry left two duplicate prefills
    // competing for the same 4 GPU slots, each slower than either alone).
    // An SSE comment line (leading ':') is invisible to any SSE/Anthropic
    // parser but still counts as bytes-on-the-wire, keeping every one of
    // Claude Code's watchdogs (byte-level, event-level, body-idle) from
    // ever seeing true silence.
    let mut keepalive = tokio::time::interval(std::time::Duration::from_secs(10));
    keepalive.tick().await; // first tick fires immediately - consume it

    'outer: loop {
        let chunk = tokio::select! {
            biased;
            chunk = byte_stream.next() => chunk,
            _ = keepalive.tick() => {
                if tx.send(": keep-alive\n\n".to_string()).await.is_err() {
                    return; // client disconnected
                }
                continue 'outer;
            }
        };
        let Some(chunk) = chunk else { break };
        let Ok(chunk) = chunk else { break };
        let chunk_str = String::from_utf8_lossy(&chunk).to_string();
        let payloads = extract_sse_payloads(&chunk_str, &mut carry);

        for p in &payloads {
            let Ok(parsed): Result<Value, _> = serde_json::from_str(p) else { continue };
            let delta = &parsed["choices"][0]["delta"];

            let piece = delta
                .get("reasoning_content")
                .and_then(Value::as_str)
                .or_else(|| delta.get("content").and_then(Value::as_str))
                .unwrap_or("");
            if !piece.is_empty() {
                full_text.push_str(piece);
                for (event, hit) in tracker.feed(piece) {
                    if state.verbose {
                        eprintln!(
                            "[{}] loop-guard: {} step {} best-match={:?} similarity={:.6}",
                            now_stamp(),
                            request_id,
                            event.step_idx,
                            event.best_match_idx,
                            event.similarity
                        );
                    }
                    if let Some(h) = hit {
                        triggered_reason = Some(h.reason);
                        break;
                    }
                }
            }
            if triggered_reason.is_some() {
                break 'outer;
            }

            if let Some(fr) = parsed["choices"][0].get("finish_reason").and_then(Value::as_str) {
                finish_reason = Some(fr.to_string());
            }

            for ev in anthropic_state.feed_delta(delta) {
                if tx.send(ev).await.is_err() {
                    return; // client disconnected
                }
            }
        }
    }

    let Some(reason) = triggered_reason else {
        for ev in anthropic_state.finish(finish_reason.as_deref()) {
            let _ = tx.send(ev).await;
        }
        return;
    };

    eprintln!("[{}] loop-guard: TRIGGERED (anthropic) - {}", now_stamp(), reason);

    // --- Budget-forcing intervention, same technique as the OpenAI path ---
    let slot_id = current_processing_slot(&state.client, &state.upstream_base).await;

    let Some(base_prompt) = apply_template(&state.client, &state.upstream_base, &request_messages).await else {
        for ev in anthropic_state.finish(Some("stop")) {
            let _ = tx.send(ev).await;
        }
        return;
    };

    let continuation_prompt = format!("{base_prompt}{full_text}{LOOP_NUDGE_PREFIX}</think>\n\n");

    for ev in anthropic_state.feed_delta(&json!({"reasoning_content": format!("{LOOP_NUDGE_PREFIX}</think>\n\n")})) {
        let _ = tx.send(ev).await;
    }

    let mut completion_req = json!({
        "prompt": continuation_prompt,
        "stream": true,
        "n_predict": original_body.get("max_tokens").and_then(Value::as_i64).unwrap_or(4096),
    });
    if let Some(id) = slot_id {
        completion_req["id_slot"] = json!(id);
    }
    for key in ["temperature", "top_p", "top_k", "min_p", "presence_penalty"] {
        if let Some(v) = original_body.get(key) {
            completion_req[key] = v.clone();
        }
    }

    let completion_res = match state
        .client
        .post(format!("{}/completion", state.upstream_base))
        .json(&completion_req)
        .send()
        .await
    {
        Ok(r) => r,
        Err(_) => {
            for ev in anthropic_state.finish(Some("stop")) {
                let _ = tx.send(ev).await;
            }
            return;
        }
    };

    let mut carry2 = String::new();
    let mut stream2 = completion_res.bytes_stream();
    let mut stopped = false;
    while let Some(chunk) = stream2.next().await {
        let Ok(chunk) = chunk else { break };
        let chunk_str = String::from_utf8_lossy(&chunk).to_string();
        let payloads = extract_sse_payloads(&chunk_str, &mut carry2);
        for p in payloads {
            let Ok(parsed): Result<Value, _> = serde_json::from_str(&p) else { continue };
            if let Some(content) = parsed.get("content").and_then(Value::as_str) {
                if !content.is_empty() {
                    for ev in anthropic_state.feed_delta(&json!({"content": content})) {
                        if tx.send(ev).await.is_err() {
                            return;
                        }
                    }
                }
            }
            if parsed.get("stop").and_then(Value::as_bool) == Some(true) {
                stopped = true;
            }
        }
    }
    let _ = stopped; // budget-forced continuations always end in a natural stop
    for ev in anthropic_state.finish(Some("stop")) {
        let _ = tx.send(ev).await;
    }
}

#[tokio::main]
async fn main() {
    let listen_host = env_or("LOOP_GUARD_HOST", "127.0.0.1");
    let listen_port = env_or("LOOP_GUARD_PORT", "8898");
    let upstream_host = env_or("LOOP_UPSTREAM_HOST", "127.0.0.1");
    let upstream_port = env_or("LOOP_UPSTREAM_PORT", "8901");
    let threshold = env_or_f64("LOOP_GUARD_THRESHOLD", 0.35);
    let min_step_words = env_or_usize("LOOP_GUARD_MIN_STEP_WORDS", 15);
    let verbose = env_or("LOOP_GUARD_VERBOSE", "0") == "1";
    let model_name = env_or("LOOP_GUARD_MODEL_NAME", "qwen3.6-35b-a3b-gpu");

    let state = AppState {
        upstream_base: format!("http://{upstream_host}:{upstream_port}"),
        threshold,
        min_step_words,
        verbose,
        client: reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(3600))
            .build()
            .expect("failed to build HTTP client"),
        request_counter: Arc::new(AtomicU64::new(0)),
        model_name,
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(messages_count_tokens))
        .with_state(state);

    let addr = format!("{listen_host}:{listen_port}");
    println!(
        "loop-guard listening on {addr}, forwarding to {upstream_host}:{upstream_port}, similarity threshold={threshold}, verbose={}",
        if verbose { "on" } else { "off" }
    );
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("failed to bind");
    axum::serve(listener, app).await.expect("server error");
}
