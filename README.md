# loop-guard

A reverse proxy for `llama-server` (llama.cpp) that detects a real,
reproduced failure mode in reasoning models: getting stuck restating the
same dead-end hypothesis in different words inside a `<think>` block,
sometimes even while making real tool calls, never converging on an
answer.

Fixed reasoning-token budgets can't fix this — a legitimately hard task
and a stuck loop both need "however long it takes," so there's no token
count that's correct for both. Neither can a timeout — same flaw,
different unit.

## How it works

Instead of a length-based guess, `loop-guard` segments the streaming
reasoning text into "steps" at the model's own restart markers ("wait",
"actually", "hold on", "hmm" — which turn out to be a free, built-in
signal for where one line of thought ends and the next begins) and
compares each new step's content against every earlier step in the same
trace, via a hand-rolled bag-of-words cosine similarity with an
in-session IDF reweighting. No ML framework, no model weights — just
word-frequency maps.

A step that's just a restatement of an earlier one triggers immediately,
regardless of whether that happens 3 steps in or 300. A genuinely novel,
converging reasoning trace never trips it, no matter how long it
legitimately runs.

On a hit: the in-flight request is cancelled, `llama-server`'s own
`/apply-template` endpoint gives the exact rendered prompt, everything
generated so far gets spliced in with a forced `</think>` closing tag and
a short nudge naming what's being repeated, and generation continues via
the raw `/completion` endpoint — the same "budget forcing" technique from
the s1 reasoning-scaling paper and vLLM's `ThinkingTokenBudgetLogitsProcessor`,
just triggered by content instead of a token count, one layer above the
engine instead of inside it. The continuation targets the exact
`llama-server` slot the original request was using (via `/slots` +
`id_slot`) so it resumes from the existing KV cache instead of
reprocessing the whole trace from scratch.

Validated against a real, reproduced stuck-in-a-loop transcript (see
`src/tracker.rs`'s test suite) — fires early, and never false-positives
on a genuinely non-repeating trace.

## Usage

```
LOOP_GUARD_PORT=8901 \
LOOP_UPSTREAM_HOST=127.0.0.1 LOOP_UPSTREAM_PORT=8902 \
LOOP_GUARD_THRESHOLD=0.35 LOOP_GUARD_MIN_STEP_WORDS=15 \
LOOP_GUARD_VERBOSE=0 \
./loop-guard
```

Point your OpenAI-compatible client at `LOOP_GUARD_PORT` instead of
`llama-server` directly; `loop-guard` forwards everything transparently
except streaming `/v1/chat/completions` requests, which it inspects
incrementally as described above. Non-streaming requests are passed
through untouched (no incremental deltas to inspect before the response
already exists).

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `LOOP_GUARD_HOST` | `127.0.0.1` | Listen address |
| `LOOP_GUARD_PORT` | `8898` | Listen port |
| `LOOP_UPSTREAM_HOST` | `127.0.0.1` | `llama-server` host |
| `LOOP_UPSTREAM_PORT` | `8901` | `llama-server` port |
| `LOOP_GUARD_THRESHOLD` | `0.35` | Cosine similarity threshold to trigger |
| `LOOP_GUARD_MIN_STEP_WORDS` | `15` | Minimum words before a segment counts as a step |
| `LOOP_GUARD_VERBOSE` | `0` | Set to `1` to log every step's similarity score, not just triggers |

## Why Rust

A first prototype was written in C++ (same shape as a sibling reverse
proxy in the deployment this was built for). It was rewritten in Rust for
the memory-safety guarantees around the concurrent stream-bridging logic
— this sits in the hot path of a daily-driver coding assistant, and the
C++ prototype's own hand-rolled slot-cache-targeting fix had a real,
twice-confirmed-live race bug. Built without `reqwest`'s default TLS
backend, since this only ever talks to `127.0.0.1` over plain HTTP —
dropping it cut RSS from ~17MB to ~5MB.

## License

MIT.
