//! Word Error Rate and timing-drift scoring.
//!
//! WER is always `(S + D + I) / N_ref`, so it asks "how badly did the
//! hypothesis fail to reproduce the reference?". Functions in this module
//! take `reference` as their first argument and `hypothesis` as their second
//! — keep them in that order and you don't have to think about which is
//! which.
//!
//! # Pipeline
//!
//! Inputs are pre-structured word lists (`WerWord` = text + start + end),
//! so callers feed `Transcriber::transcribe(...).words()` on the hypothesis
//! side and the parquet `words` column on the reference side — no
//! whitespace-splitting of a flattened transcript.
//!
//! `compute_wer` runs the classic word-level Levenshtein with backtracking
//! and returns the alignment alongside the S/D/I/N counts. Timing isn't
//! scored inside `compute_wer`; feed the returned alignment into
//! `timing_stats` to get per-matched-pair midpoint drift, mergeable across
//! files for corpus-level aggregation.

/// Lowercase, fold `й → и` and `ё → е` (both spellings are inconsistent
/// between Golos labels and what the model emits), and strip non-alphanumeric
/// chars from both ends. Cyrillic-aware (delegates to Unicode
/// `char::is_alphanumeric` and `str::to_lowercase`).
pub fn normalize_word(w: &str) -> String {
    w.to_lowercase()
        .replace('й', "и")
        .replace('ё', "е")
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_string()
}

/// One word with its time interval, used as input to `compute_wer`.
/// Reference words come from the parquet `words` column; hypothesis words
/// come from `Transcriber::transcribe(...).words()`.
#[derive(Debug, Clone)]
pub struct WerWord {
    /// Normalized text. Use `WerWord::new` to normalize automatically.
    pub text: String,
    pub start: f32,
    pub end: f32,
}

impl WerWord {
    pub fn new(text: &str, start: f32, end: f32) -> Self {
        Self {
            text: normalize_word(text),
            start,
            end,
        }
    }

    pub fn midpoint(&self) -> f32 {
        0.5 * (self.start + self.end)
    }
}

/// One step in a reference/hypothesis alignment.
#[derive(Debug, Clone, Copy)]
pub enum AlignOp {
    Match { ref_idx: usize, hyp_idx: usize },
    Sub { ref_idx: usize, hyp_idx: usize },
    Del { ref_idx: usize },
    Ins { hyp_idx: usize },
}

/// WER counts plus the alignment that produced them. Timing isn't scored
/// here — feed `alignment` into `timing_stats` to get drift metrics.
#[derive(Debug, Clone)]
pub struct WerResult {
    pub substitutions: usize,
    pub deletions: usize,
    pub insertions: usize,
    pub ref_len: usize,
    pub alignment: Vec<AlignOp>,
}

impl WerResult {
    pub fn wer(&self) -> f64 {
        if self.ref_len == 0 {
            if self.insertions == 0 { 0.0 } else { 1.0 }
        } else {
            (self.substitutions + self.deletions + self.insertions) as f64 / self.ref_len as f64
        }
    }
}

/// Word-level Levenshtein with backtracking. Match cost is 0 iff `.text`
/// fields are equal (caller is responsible for normalization, which
/// `WerWord::new` already does).
pub fn compute_wer(reference: &[WerWord], hypothesis: &[WerWord]) -> WerResult {
    let m = reference.len();
    let n = hypothesis.len();
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for (i, row) in dp.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, val) in dp[0].iter_mut().enumerate() {
        *val = j;
    }
    for i in 1..=m {
        for j in 1..=n {
            let cost = if reference[i - 1].text == hypothesis[j - 1].text {
                0
            } else {
                1
            };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    let mut alignment = Vec::new();
    let (mut i, mut j) = (m, n);
    let (mut subs, mut dels, mut ins) = (0usize, 0usize, 0usize);
    while i > 0 || j > 0 {
        let cur = dp[i][j];
        if i > 0 && j > 0 {
            let cost = if reference[i - 1].text == hypothesis[j - 1].text {
                0
            } else {
                1
            };
            if cur == dp[i - 1][j - 1] + cost {
                if cost == 0 {
                    alignment.push(AlignOp::Match {
                        ref_idx: i - 1,
                        hyp_idx: j - 1,
                    });
                } else {
                    alignment.push(AlignOp::Sub {
                        ref_idx: i - 1,
                        hyp_idx: j - 1,
                    });
                    subs += 1;
                }
                i -= 1;
                j -= 1;
                continue;
            }
        }
        if i > 0 && cur == dp[i - 1][j] + 1 {
            alignment.push(AlignOp::Del { ref_idx: i - 1 });
            dels += 1;
            i -= 1;
        } else {
            alignment.push(AlignOp::Ins { hyp_idx: j - 1 });
            ins += 1;
            j -= 1;
        }
    }
    alignment.reverse();
    WerResult {
        substitutions: subs,
        deletions: dels,
        insertions: ins,
        ref_len: m,
        alignment,
    }
}

/// Per-matched-pair timing deltas. Mergeable for corpus-level aggregation.
#[derive(Debug, Clone, Default)]
pub struct TimingStats {
    pub matched_pairs: usize,
    /// `|hyp.midpoint - ref.midpoint|` for each matched pair (unsorted).
    pub mid_abs_deltas: Vec<f32>,
    /// Signed `hyp.start - ref.start` for each matched pair (unsorted).
    pub start_signed_deltas: Vec<f32>,
    /// Signed `hyp.end - ref.end`.
    pub end_signed_deltas: Vec<f32>,
    /// Matched pairs whose midpoint drift exceeds the configured threshold.
    pub high_drift_pairs: usize,
}

impl TimingStats {
    pub fn merge(&mut self, other: &TimingStats) {
        self.matched_pairs += other.matched_pairs;
        self.mid_abs_deltas.extend_from_slice(&other.mid_abs_deltas);
        self.start_signed_deltas
            .extend_from_slice(&other.start_signed_deltas);
        self.end_signed_deltas
            .extend_from_slice(&other.end_signed_deltas);
        self.high_drift_pairs += other.high_drift_pairs;
    }

    pub fn mean_abs_mid(&self) -> f32 {
        if self.mid_abs_deltas.is_empty() {
            0.0
        } else {
            self.mid_abs_deltas.iter().sum::<f32>() / self.mid_abs_deltas.len() as f32
        }
    }

    fn percentile_abs_mid(&self, q: f32) -> f32 {
        if self.mid_abs_deltas.is_empty() {
            return 0.0;
        }
        let mut s = self.mid_abs_deltas.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((s.len() as f32 - 1.0) * q.clamp(0.0, 1.0)).round() as usize;
        s[idx]
    }

    pub fn median_abs_mid(&self) -> f32 {
        self.percentile_abs_mid(0.5)
    }
    pub fn p95_abs_mid(&self) -> f32 {
        self.percentile_abs_mid(0.95)
    }
}

/// For every `Match` op in `alignment`, record the timestamp delta between
/// hypothesis and reference. `drift_threshold_s` is the midpoint-drift bar
/// above which a matched pair is counted as "high-drift".
pub fn timing_stats(
    reference: &[WerWord],
    hypothesis: &[WerWord],
    alignment: &[AlignOp],
    drift_threshold_s: f32,
) -> TimingStats {
    let mut stats = TimingStats::default();
    for op in alignment {
        if let AlignOp::Match { ref_idx, hyp_idx } = *op {
            let r = &reference[ref_idx];
            let h = &hypothesis[hyp_idx];
            let dmid = (h.midpoint() - r.midpoint()).abs();
            stats.mid_abs_deltas.push(dmid);
            stats.start_signed_deltas.push(h.start - r.start);
            stats.end_signed_deltas.push(h.end - r.end);
            stats.matched_pairs += 1;
            if dmid > drift_threshold_s {
                stats.high_drift_pairs += 1;
            }
        }
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ws(words: &[&str]) -> Vec<WerWord> {
        words
            .iter()
            .enumerate()
            .map(|(i, w)| WerWord::new(w, i as f32, i as f32 + 0.5))
            .collect()
    }

    #[test]
    fn wer_exact_match() {
        let r = ws(&["a", "b", "c"]);
        let res = compute_wer(&r, &r);
        assert_eq!(res.wer(), 0.0);
        let t = timing_stats(&r, &r, &res.alignment, 1.0);
        assert_eq!(t.matched_pairs, 3);
        assert_eq!(t.high_drift_pairs, 0);
    }

    #[test]
    fn wer_total_edits_invariant_to_alignment_choice() {
        // "TWO," normalizes to "two" so it matches. From there, the algorithm
        // can either pick (del "three", ins "five") or (sub "three"->"four",
        // sub "four"->"five") — both have edit distance 2, so we assert on
        // totals and WER rather than on the specific split.
        let r = ws(&["one", "two", "three", "four"]);
        let h = ws(&["one", "TWO,", "four", "five"]);
        let res = compute_wer(&r, &h);
        assert_eq!(res.ref_len, 4);
        assert_eq!(res.substitutions + res.deletions + res.insertions, 2);
        assert!((res.wer() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn high_drift_pair_is_flagged() {
        let r = vec![WerWord::new("слово", 1.0, 1.5)];
        let h = vec![WerWord::new("Слово", 5.0, 5.5)];
        let res = compute_wer(&r, &h);
        let t_self = timing_stats(&r, &r, &res.alignment, 1.0);
        assert_eq!(t_self.high_drift_pairs, 0);
        let t_drifted = timing_stats(&r, &h, &res.alignment, 1.0);
        assert_eq!(t_drifted.matched_pairs, 1);
        assert_eq!(t_drifted.high_drift_pairs, 1);
    }

    #[test]
    fn normalize_strips_punct_and_lowers_cyrillic() {
        assert_eq!(normalize_word("Привет,"), "привет");
        assert_eq!(normalize_word("«Слово»."), "слово");
    }

    #[test]
    fn normalize_folds_short_i_to_i() {
        // Golos references write "и" where standard Russian uses "й", so the
        // bench folds the model's "й" output back to "и" before comparison.
        assert_eq!(normalize_word("Саратовский"), normalize_word("Саратовскии"));
        assert_eq!(normalize_word("медицинскиЙ"), "медицинскии");
    }

    #[test]
    fn normalize_folds_yo_to_ye() {
        // "ё" vs "е" disagreements would otherwise show up as false subs.
        assert_eq!(normalize_word("всё"), normalize_word("все"));
        assert_eq!(normalize_word("Ёлка"), "елка");
    }
}
