//! Segments streaming reasoning text into "steps" at this model's own
//! restart markers ("wait", "actually", "hold on", ...) and detects when a
//! new step is just a semantic restatement of an earlier one in the same
//! trace, via a hand-rolled bag-of-words cosine similarity with an
//! in-session IDF reweighting - no ML framework, no model weights, just
//! word-frequency maps. See main.rs's module doc for the full rationale.

use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use std::time::Instant;

fn stopwords() -> &'static HashSet<&'static str> {
    static WORDS: OnceLock<HashSet<&'static str>> = OnceLock::new();
    WORDS.get_or_init(|| {
        [
            "the", "a", "an", "is", "this", "that", "but", "and", "or", "to", "of", "in", "on",
            "at", "it", "be", "as", "with", "for", "was", "were", "are", "i", "we", "you", "not",
            "no", "so", "if", "then", "let", "me", "look", "looking", "more", "carefully",
            "just", "now", "wait", "actually", "hold", "hmm", "oh", "again", "here", "there",
            "would", "could", "should", "which", "what", "how", "do", "does", "did", "has",
            "have", "had", "will", "can", "may", "might", "s", "t", "re", "ve", "ll", "d", "m",
        ]
        .into_iter()
        .collect()
    })
}

// Restart markers this model emits mid-thought, used as step boundaries.
// Matched only at the start of a sentence/paragraph (after ". ", a
// newline, or the very start of the text) to avoid splitting on
// incidental mid-sentence uses (e.g. "wait for the response").
fn restart_marker_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)(?:^|\n|\.\s)\s*(?:But\s+)?(?:Oh\s+)?(Wait|Actually|Hold on|Hmm)\b")
            .unwrap()
    })
}

fn tokenize(text: &str) -> Vec<String> {
    let stop = stopwords();
    text.split(|c: char| !c.is_alphanumeric())
        .map(|w| w.to_lowercase())
        .filter(|w| w.len() > 1 && !stop.contains(w.as_str()))
        .collect()
}

type TermFreq = HashMap<String, f64>;

fn term_freq(tokens: &[String]) -> TermFreq {
    let mut tf = TermFreq::new();
    for t in tokens {
        *tf.entry(t.clone()).or_insert(0.0) += 1.0;
    }
    tf
}

/// Result of finalizing one step: always reported (for verbose tuning
/// logs), regardless of whether it crossed the repetition threshold.
pub struct StepEvent {
    pub step_idx: usize,
    pub best_match_idx: Option<usize>,
    pub similarity: f64,
}

/// A confirmed repetition hit.
pub struct LoopDetected {
    pub reason: String,
}

pub struct StepTracker {
    threshold: f64,
    min_step_words: usize,
    request_id: String,
    buffer: String,
    last_consumed: usize,
    steps: Vec<TermFreq>,
    step_texts: Vec<String>,
    start: Instant,
}

impl StepTracker {
    pub fn new(threshold: f64, min_step_words: usize, request_id: impl Into<String>) -> Self {
        Self {
            threshold,
            min_step_words,
            request_id: request_id.into(),
            buffer: String::new(),
            last_consumed: 0,
            steps: Vec::new(),
            step_texts: Vec::new(),
            start: Instant::now(),
        }
    }

    /// Feed a newly-arrived text delta. Returns every step finalized by
    /// this delta (usually 0 or 1, but a big delta could contain more than
    /// one marker), each paired with whether it triggered detection.
    pub fn feed(&mut self, delta: &str) -> Vec<(StepEvent, Option<LoopDetected>)> {
        self.buffer.push_str(delta);
        let mut results = Vec::new();

        loop {
            let mut split_at: Option<usize> = None;
            for m in restart_marker_regex().captures_iter(&self.buffer) {
                let g = m.get(1).unwrap();
                let pos = g.start();
                if pos > self.last_consumed {
                    let candidate = &self.buffer[self.last_consumed..pos];
                    if tokenize(candidate).len() >= self.min_step_words {
                        split_at = Some(pos);
                    }
                }
            }
            let Some(pos) = split_at else { break };
            let finished = self.buffer[self.last_consumed..pos].to_string();
            self.last_consumed = pos;
            results.push(self.finalize_step(&finished));
            // Loop again in case the same delta contains multiple markers
            // past the one we just consumed.
        }
        results
    }

    /// Call once at the natural end of generation (or right before forcing
    /// a close) to check whatever's left in the buffer as a final step.
    pub fn flush(&mut self) -> Option<(StepEvent, Option<LoopDetected>)> {
        let remaining = self.buffer[self.last_consumed..].to_string();
        self.last_consumed = self.buffer.len();
        if tokenize(&remaining).len() < self.min_step_words {
            return None;
        }
        Some(self.finalize_step(&remaining))
    }

    fn finalize_step(&mut self, text: &str) -> (StepEvent, Option<LoopDetected>) {
        let tokens = tokenize(text);
        let tf = term_freq(&tokens);

        // In-session IDF: document frequency counted across steps seen so
        // far INCLUDING this new one, recomputed fresh each time - cheap,
        // since a real reasoning trace realistically has a handful to a
        // few dozen steps by the time this matters.
        let mut all: Vec<&TermFreq> = self.steps.iter().collect();
        all.push(&tf);
        let mut df: HashMap<&str, usize> = HashMap::new();
        for step_tf in &all {
            let seen: HashSet<&str> = step_tf.keys().map(|s| s.as_str()).collect();
            for w in seen {
                *df.entry(w).or_insert(0) += 1;
            }
        }
        let n_docs = all.len() as f64;
        let idf = |w: &str| -> f64 {
            let d = *df.get(w).unwrap_or(&1) as f64;
            (1.0 + n_docs / d).ln()
        };
        let weighted = |t: &TermFreq| -> HashMap<String, f64> {
            t.iter().map(|(w, c)| (w.clone(), c * idf(w))).collect()
        };
        let cosine = |a: &HashMap<String, f64>, b: &HashMap<String, f64>| -> f64 {
            let mut dot = 0.0;
            let mut na = 0.0;
            for (w, va) in a {
                na += va * va;
                if let Some(vb) = b.get(w) {
                    dot += va * vb;
                }
            }
            let nb: f64 = b.values().map(|v| v * v).sum();
            if na <= 0.0 || nb <= 0.0 {
                0.0
            } else {
                dot / (na.sqrt() * nb.sqrt())
            }
        };

        let new_vec = weighted(&tf);
        let mut best_sim = 0.0_f64;
        let mut best_idx: Option<usize> = None;
        for (i, step_tf) in self.steps.iter().enumerate() {
            let sim = cosine(&new_vec, &weighted(step_tf));
            if sim > best_sim {
                best_sim = sim;
                best_idx = Some(i);
            }
        }

        self.steps.push(tf);
        self.step_texts.push(text.to_string());
        let this_idx = self.steps.len() - 1;

        let event = StepEvent {
            step_idx: this_idx,
            best_match_idx: best_idx,
            similarity: best_sim,
        };

        let hit = if let Some(bi) = best_idx {
            if best_sim >= self.threshold {
                let snippet: String = self.step_texts[bi].chars().take(160).collect();
                Some(LoopDetected {
                    reason: format!(
                        "request={} step {} repeats step {} (similarity {:.6}, elapsed={:.1}s): \"{}\"",
                        self.request_id,
                        this_idx,
                        bi,
                        best_sim,
                        self.start.elapsed().as_secs_f64(),
                        snippet
                    ),
                })
            } else {
                None
            }
        } else {
            None
        };

        (event, hit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The exact 13 steps transcribed from a real, reproduced stuck-in-a-
    // loop session (four screenshots, 2026-09-11) - the same transcript
    // already validated offline with scikit-learn's TfidfVectorizer (first
    // detection at step 5 there) and with the original C++ port (fired at
    // fixture step 2, due to its own finer-grained internal resplitting).
    // This is a real regression test against real model behavior, not a
    // synthetic fixture.
    const REAL_LOOP_STEPS: &[&str] = &[
        r#"Actually, wait - I just realized something. Looking at line 410:
        errors.AppendFormat("File '%s' exceeds 100MB limit.", fileName)
        This is using StringBuilder.AppendFormat with a format string that
        contains %s. But StringBuilder.AppendFormat uses {0}, {1}, etc., not
        %s. So this is a bug - the format string should use {0} instead of
        %s. But the error is reported at line 189, not 410. Let me check if
        there's some compilation behavior where the error gets reported at
        the wrong line."#,
        r#"Actually, let me look at the sprintf template more carefully. The
        template from line 189-289 contains %s placeholders. Let me count
        them: Line 84 relative %s successHtml, Line 85 relative %s
        errorHtml, Line 86 relative upload%s queryString, Line 94 relative
        value quote %s quote safePath. That's 4 %s placeholders and 4
        arguments successHtml errorHtml queryString safePath. This looks
        correct."#,
        r#"But wait! Looking at line 86 again: form method post action
        upload%s enctype multipart form data. The upload%s is meant to
        insert the query string path. But actually looking at the sprintf
        arguments on line 289 relative line 101 successHtml errorHtml
        queryString safePath."#,
        r#"I can see the template clearly now. The issue is on line 71:
        color hash 0366 semicolon d6 - this should be color hash 0366d6.
        But that's a CSS issue, not a format string issue."#,
        r#"Wait, I need to look more carefully. The error says Bad format
        specifier semicolon at line 189. Let me check if there are any
        percent characters followed by special characters in the template.
        Looking at the lines: Line 35 width 100 percent semicolon this has
        percent but it's not preceded by percent. Line 71 color hash 0366
        semicolon d6 semicolon CSS issue. Actually, I don't see any percent
        quote in the template. The percent s placeholders are on lines 84,
        85, 86, and 94 all followed by quote which is valid the percent s
        will be replaced with strings."#,
        r#"Wait! Let me look at line 86 more carefully: form method post
        action upload percent s enctype multipart form data. The upload
        percent s has percent s which is a format specifier. But this is
        inside the triple quoted string that's being passed to sprintf. The
        issue is that the percent s here is meant to be part of the HTML
        replaced with the query string path but it's being interpreted as a
        sprintf format specifier. Actually, this should work fine since
        sprintf will replace percent s with queryString."#,
        r#"Oh wait! I just realized - looking at the CSS on line 71: color
        hash 0366 semicolon d6 semicolon - there's a semicolon inside the
        color value. This is invalid CSS, but shouldn't cause a format
        string error. Let me look at the actual error message again Bad
        format specifier semicolon this means F sharp found a percent
        followed by quote somewhere. Let me search for this pattern more
        carefully."#,
        r#"I see the format strings. Let me look at the createUploadHtml
        function more carefully. The sprintf template is a triple quoted
        string from line 189 to 289. Let me check for any issues with the
        percent characters. Looking at line 274: form method post action
        upload percent s enctype multipart form data. This has percent s
        which should be a format specifier. But wait this is inside the
        HTML template string not a separate sprintf call."#,
        r#"Wait, let me count the percent s in the template: Line 272
        percent s successHtml, Line 273 percent s errorHtml, Line 274
        upload percent s queryString, Line 282 percent s safePath. That's 4
        percent s placeholders and 4 arguments. But wait, line 274 has
        upload percent s where the percent s is meant to be replaced with
        the query string path. So this should work."#,
        r#"But wait! The error says Bad format specifier semicolon which
        means there's a malformed format specifier somewhere. Let me look
        more carefully at the template. Actually, I wonder if there's a
        hidden Unicode character. Let me check the raw bytes around line
        189."#,
        r#"The color code should be 0366d6 not 0366 semicolon d6. But this
        is CSS, not a format string issue unless there's some interaction."#,
        r#"Actually wait - let me look at line 189 column 17 more carefully.
        The error says Bad format specifier semicolon this means somewhere
        in the format string there's a percent followed by quote. Let me
        search for any percent followed by special characters in the HTML
        template."#,
        r#"Oh wait! I think I see it now. Looking at line 269: a href back
        link larr Back to catalog. The larr is an HTML entity, not a format
        specifier. That's fine. Hmm, let me look for any percent in the
        template. I don't see any obvious issues. Maybe the issue is with a
        Unicode character that looks like a regular character but isn't? Or
        maybe there's an encoding issue? Let me check the file encoding."#,
    ];

    #[test]
    fn fires_on_real_reproduced_loop_early() {
        let mut tracker = StepTracker::new(0.35, 15, "test-loop");
        let mut fired_at: Option<usize> = None;
        for (i, step) in REAL_LOOP_STEPS.iter().enumerate() {
            let chunk = format!("{step} Wait, ");
            for (_, hit) in tracker.feed(&chunk) {
                if hit.is_some() && fired_at.is_none() {
                    fired_at = Some(i);
                }
            }
        }
        assert!(fired_at.is_some(), "must fire at all on a real reproduced loop");
        assert!(
            fired_at.unwrap() <= 8,
            "must fire within the first 8 of 13 steps (detects it QUICK, not just eventually), fired at {:?}",
            fired_at
        );
    }

    #[test]
    fn never_fires_on_genuinely_distinct_steps() {
        let mut tracker = StepTracker::new(0.35, 15, "test-clean");
        let distinct_steps = [
            "Wait, let me check the imports first, nothing unusual there.",
            "Actually, let me check the routing table next, that looks fine too, all endpoints registered correctly.",
            "Hold on, let me check the database connection string, that's using the right environment variable.",
            "Actually, let me check the middleware order, authentication runs before authorization as expected.",
            "Wait, let me check the serialization settings, camelCase policy is applied consistently.",
            "Hmm, let me check the logging configuration, structured JSON output is enabled correctly.",
        ];
        let mut fired = false;
        for step in distinct_steps {
            let chunk = format!("{step} Wait, ");
            for (_, hit) in tracker.feed(&chunk) {
                if hit.is_some() {
                    fired = true;
                }
            }
        }
        assert!(!fired, "must never fire on genuinely non-repeating steps");
    }
}
