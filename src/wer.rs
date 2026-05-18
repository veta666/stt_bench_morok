//! Word Error Rate and timing-drift scoring.
//!
//! Normalization + alignment come from `transcription_normalization`:
//! lowercase, `ё → е`, edge-punctuation strip, hyphen equivalence
//! (`вице-президент` ≡ `вице президент` ≡ `вицепрезидент`), numeral
//! collapsing for digit literals, cardinals and ordinals
//! (`сто двадцать три` ≡ `123`).
//!
//! On top of that we only add what upstream is missing for our dataset:
//!   * `й → и` pre-fold — Golos labels conflate the two and upstream
//!     doesn't fold them.
//!   * `TimingStats` aggregator — upstream gives us per-op `OpTiming`
//!     (ref/hyp `TimeSpan`s, N-M unions already handled); we just walk
//!     `Match` ops and accumulate midpoint / start / end deltas across a
//!     file or a corpus.
//!
//! Everything else (types, ops, alignment, per-op timing) is re-exported
//! from upstream; no shadow `WerWord` / `WerResult` / `AlignOp` here, and
//! no hand-rolled token-range → input-word-range span math anymore.

use std::ops::Range;

use transcription_normalization::compare;
pub use transcription_normalization::{
    AlignmentResult, Canonical, Language, Op, OpTiming, TimeSpan, Token, Word,
};

/// Fold `й → и` (both cases). Apply this when *constructing* `Word`s
/// from the dataset / model output — upstream tokenization doesn't know
/// about the Golos-specific й/и conflation, and we don't want to re-fold
/// inside `compute_wer` on every call.
pub fn fold_short_i(s: &str) -> String {
    if s.contains(['Й', 'й']) {
        s.replace('й', "и").replace('Й', "И")
    } else {
        s.to_owned()
    }
}

/// Russian Golos alignment. Words are expected to already have `й → и`
/// applied (see [`fold_short_i`]).
pub fn compute_wer(reference: &[Word], hypothesis: &[Word]) -> AlignmentResult {
    compare(reference, hypothesis, Language::Russian)
}

/// Input-word range covered by a single upstream token (its `word_indices`
/// are guaranteed contiguous-ascending by upstream's tokenizer). Used by
/// `pretty.rs` to render the *original* input words for a token — upstream
/// owns the timing, but the surface text is ours.
pub fn token_word_range(tok: &Token) -> Range<usize> {
    let first = tok.word_indices.first().copied().unwrap_or(0);
    let last = tok.word_indices.last().copied().unwrap_or(first);
    first..last + 1
}

/// Same as [`token_word_range`] but for a slice of tokens — used for the
/// many-to-many `Op::Match { ref_range, hyp_range }`.
pub fn tokens_word_range(tokens: &[Token]) -> Range<usize> {
    let first = tokens
        .iter()
        .flat_map(|t| t.word_indices.first().copied())
        .min()
        .unwrap_or(0);
    let last = tokens
        .iter()
        .flat_map(|t| t.word_indices.last().copied())
        .max()
        .unwrap_or(first);
    first..last + 1
}

#[derive(Debug, Clone, Default)]
pub struct TimingStats {
    pub matched_pairs: usize,
    /// `|hyp_mid - ref_mid|` per `Match` op (span midpoints when the op
    /// covers multiple input words).
    pub mid_abs_deltas: Vec<f64>,
    pub start_signed_deltas: Vec<f64>,
    pub end_signed_deltas: Vec<f64>,
    /// `Match` pairs whose midpoint drift exceeds the configured threshold.
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

    pub fn mean_abs_mid(&self) -> f64 {
        if self.mid_abs_deltas.is_empty() {
            0.0
        } else {
            self.mid_abs_deltas.iter().sum::<f64>() / self.mid_abs_deltas.len() as f64
        }
    }

    fn percentile_abs_mid(&self, q: f64) -> f64 {
        if self.mid_abs_deltas.is_empty() {
            return 0.0;
        }
        let mut s = self.mid_abs_deltas.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((s.len() as f64 - 1.0) * q.clamp(0.0, 1.0)).round() as usize;
        s[idx]
    }

    pub fn median_abs_mid(&self) -> f64 {
        self.percentile_abs_mid(0.5)
    }
    pub fn p95_abs_mid(&self) -> f64 {
        self.percentile_abs_mid(0.95)
    }

    /// Returns `(median, p95)` of `|hyp_mid - ref_mid|` with a single sort —
    /// cheaper than calling `median_abs_mid` + `p95_abs_mid` back-to-back.
    pub fn percentiles_abs_mid(&self) -> (f64, f64) {
        if self.mid_abs_deltas.is_empty() {
            return (0.0, 0.0);
        }
        let mut s = self.mid_abs_deltas.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = s.len() as f64 - 1.0;
        let pick = |q: f64| s[(n * q.clamp(0.0, 1.0)).round() as usize];
        (pick(0.5), pick(0.95))
    }
}

/// Walk every `Op::Match` and record midpoint / start / end deltas from
/// upstream's `op_timing` (which already handles N-M unions).
pub fn timing_stats(result: &AlignmentResult, drift_threshold_s: f64) -> TimingStats {
    let mut stats = TimingStats::default();
    for op in &result.ops {
        if !matches!(op, Op::Match { .. }) {
            continue;
        }
        let OpTiming {
            ref_span: Some(r),
            hyp_span: Some(h),
        } = result.op_timing(op)
        else {
            continue;
        };
        let dmid = (0.5 * (h.start + h.end) - 0.5 * (r.start + r.end)).abs();
        stats.mid_abs_deltas.push(dmid);
        stats.start_signed_deltas.push(h.start - r.start);
        stats.end_signed_deltas.push(h.end - r.end);
        stats.matched_pairs += 1;
        if dmid > drift_threshold_s {
            stats.high_drift_pairs += 1;
        }
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ws(words: &[&str]) -> Vec<Word> {
        words
            .iter()
            .enumerate()
            .map(|(i, t)| Word::new(fold_short_i(t), i as f64, i as f64 + 0.5))
            .collect()
    }

    #[test]
    fn exact_match_has_no_errors_and_three_matched_pairs() {
        let r = ws(&["а", "б", "в"]);
        let res = compute_wer(&r, &r);
        assert_eq!(res.errors(), 0);
        let t = timing_stats(&res, 1.0);
        assert_eq!(t.matched_pairs, 3);
        assert_eq!(t.high_drift_pairs, 0);
    }

    #[test]
    fn short_i_fold_prevents_false_sub() {
        // Golos writes "саратовский" as "саратовскии"; model emits "й".
        let r = ws(&["саратовскии"]);
        let h = ws(&["саратовский"]);
        assert_eq!(compute_wer(&r, &h).errors(), 0);
    }

    #[test]
    fn yo_fold_prevents_false_sub() {
        let r = ws(&["ёлка"]);
        let h = ws(&["елка"]);
        assert_eq!(compute_wer(&r, &h).errors(), 0);
    }

    #[test]
    fn numeral_run_matches_digit_form() {
        let r = ws(&["сто", "двадцать", "три", "рубля"]);
        let h = vec![Word::new("123", 0.0, 2.5), Word::new("рубля", 3.0, 3.5)];
        let res = compute_wer(&r, &h);
        assert_eq!(res.errors(), 0);
        assert_eq!(res.ops.len(), 2);
        let t = timing_stats(&res, 10.0);
        assert_eq!(t.matched_pairs, 2);
        assert_eq!(t.high_drift_pairs, 0);
    }

    #[test]
    fn hyphen_split_vs_joined_is_a_match() {
        let r = ws(&["вице-президент", "сказал"]);
        let h = ws(&["вице", "президент", "сказал"]);
        assert_eq!(compute_wer(&r, &h).errors(), 0);
    }

    #[test]
    fn drift_flagged_above_threshold() {
        let r = vec![Word::new("слово", 1.0, 1.5)];
        let h = vec![Word::new("Слово", 5.0, 5.5)];
        let res = compute_wer(&r, &h);
        let t = timing_stats(&res, 1.0);
        assert_eq!(t.matched_pairs, 1);
        assert_eq!(t.high_drift_pairs, 1);
    }
}
