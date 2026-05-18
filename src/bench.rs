//! Bench drivers: pull audio from the HF dataset parquet (or a custom WAV),
//! transcribe, score against the dataset's ground-truth `words` column,
//! render.
//!
//! Three entry points share one internal scoring helper:
//!
//! * [`run_dataset`] — stream the whole parquet, accumulate corpus-level
//!   stats, emit the summary panel.
//! * [`run_idx`] — single-row mode: pull just one `idx` from the parquet
//!   and emit the pretty per-file panel.
//! * [`run_custom_wav`] — escape hatch for an arbitrary WAV file outside
//!   the dataset (no ground truth, so no WER).
//!
//! The model is loaded once by the caller and threaded in by mutable
//! reference, so a corpus run pays the HF download / weight-load cost
//! exactly once.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use morok_model::audio::Splitter;
use morok_model::gigaam::{TranscribeResult, Transcriber};
use tracing::info;

use crate::dataset::{DatasetIter, DatasetRow, find_row};
use crate::pretty::{print_single, print_summary};
use crate::wer::{AlignmentResult, TimingStats, Word, compute_wer, fold_short_i, timing_stats};
use crate::{load_wav, load_wav_from_bytes};

/// Convert the model's word stream into `Word` form. `й → и` is folded
/// here at the boundary, so the rest of the pipeline never has to.
pub fn hyp_words(result: &TranscribeResult) -> Vec<Word> {
    result
        .words()
        .filter(|w| !w.text.trim().is_empty())
        .map(|w| Word::new(fold_short_i(&w.text), w.start as f64, w.end as f64))
        .collect()
}

/// Per-file row consumed by the corpus summary.
#[derive(Debug, Clone)]
pub struct FileScore {
    pub idx: Option<i32>,
    pub path: PathBuf,
    pub duration_s: f32,
    pub transcribe_s: f32,
    pub wer: f64,
    pub subs: usize,
    pub dels: usize,
    pub ins: usize,
    /// Reference token count after upstream tokenization (denominator of `wer`).
    pub ref_len: usize,
    pub matched_pairs: usize,
    pub high_drift_pairs: usize,
    pub mean_abs_mid_s: f64,
}

impl FileScore {
    fn from_result(
        idx: Option<i32>,
        path: PathBuf,
        duration_s: f32,
        transcribe_s: f32,
        result: &AlignmentResult,
        timing: &TimingStats,
    ) -> Self {
        Self {
            idx,
            path,
            duration_s,
            transcribe_s,
            wer: result.wer(),
            subs: result.substitutions(),
            dels: result.deletions(),
            ins: result.insertions(),
            ref_len: result.ref_token_count(),
            matched_pairs: timing.matched_pairs,
            high_drift_pairs: timing.high_drift_pairs,
            mean_abs_mid_s: timing.mean_abs_mid(),
        }
    }
}

struct Scored {
    transcribe_dt: Duration,
    hyp: Vec<Word>,
    result: AlignmentResult,
    timing: TimingStats,
}

fn score_waveform<S: Splitter>(
    transcriber: &mut Transcriber<S>,
    waveform: &[f32],
    sample_rate: u32,
    reference: &[Word],
    drift_threshold_s: f64,
) -> Result<Scored, Box<dyn Error>> {
    let t = Instant::now();
    let raw = transcriber.transcribe(waveform, sample_rate)?;
    let transcribe_dt = t.elapsed();

    let hyp = hyp_words(&raw);
    let result = compute_wer(reference, &hyp);
    let timing = timing_stats(&result, drift_threshold_s);
    Ok(Scored {
        transcribe_dt,
        hyp,
        result,
        timing,
    })
}

pub fn run_dataset<S: Splitter>(
    transcriber: &mut Transcriber<S>,
    dataset: &Path,
    limit: usize,
    worst_n: usize,
    drift_threshold_s: f32,
) -> Result<(), Box<dyn Error>> {
    info!("streaming dataset {}", dataset.display());

    let drift = drift_threshold_s as f64;
    let mut total_timing = TimingStats::default();
    let mut total_dur_s = 0.0f32;
    let mut total_xt_s = 0.0f32;
    let mut total_errors = 0usize;
    let mut total_ref_len = 0usize;
    let mut per_file: Vec<FileScore> = Vec::new();

    let started = Instant::now();
    let mut processed = 0usize;
    for row in DatasetIter::open(dataset)? {
        if limit > 0 && processed >= limit {
            break;
        }
        let row = row?;
        let (waveform, sr) = load_wav_from_bytes(&row.audio_bytes)?;
        let duration_s = waveform.len() as f32 / sr as f32;
        let scored = score_waveform(transcriber, &waveform, sr, &row.words, drift)?;

        total_timing.merge(&scored.timing);
        total_dur_s += duration_s;
        total_xt_s += scored.transcribe_dt.as_secs_f32();
        total_errors +=
            scored.result.substitutions() + scored.result.deletions() + scored.result.insertions();
        total_ref_len += scored.result.ref_token_count();

        per_file.push(FileScore::from_result(
            Some(row.idx),
            PathBuf::from(format!("idx{:05}", row.idx)),
            duration_s,
            scored.transcribe_dt.as_secs_f32(),
            &scored.result,
            &scored.timing,
        ));

        processed += 1;
        if processed.is_multiple_of(10) {
            let elapsed = started.elapsed().as_secs_f32();
            let rate = processed as f32 / elapsed;
            let rolling = if total_ref_len == 0 {
                0.0
            } else {
                total_errors as f64 / total_ref_len as f64
            };
            info!(
                "  {processed} rows  rate={rate:.1}/s  elapsed={elapsed:.0}s  rolling WER={:.2}%",
                rolling * 100.0,
            );
        }
    }

    if per_file.is_empty() {
        return Err("dataset is empty".into());
    }

    print_summary(
        &per_file,
        &total_timing,
        total_dur_s,
        total_xt_s,
        worst_n,
        drift,
    );
    Ok(())
}

pub fn run_idx<S: Splitter>(
    transcriber: &mut Transcriber<S>,
    dataset: &Path,
    idx: i32,
    drift_threshold_s: f32,
) -> Result<(), Box<dyn Error>> {
    let row = find_row(dataset, idx)?
        .ok_or_else(|| -> Box<dyn Error> { format!("idx {idx} not found in dataset").into() })?;
    print_dataset_row(transcriber, &row, drift_threshold_s as f64)
}

fn print_dataset_row<S: Splitter>(
    transcriber: &mut Transcriber<S>,
    row: &DatasetRow,
    drift_threshold_s: f64,
) -> Result<(), Box<dyn Error>> {
    let (waveform, sr) = load_wav_from_bytes(&row.audio_bytes)?;
    let duration_s = waveform.len() as f32 / sr as f32;
    info!(
        "transcribing dataset idx={} ({duration_s:.1}s @ {sr} Hz)",
        row.idx
    );

    let scored = score_waveform(transcriber, &waveform, sr, &row.words, drift_threshold_s)?;
    let label = PathBuf::from(format!("dataset[idx={}]", row.idx));
    print_single(
        &label,
        Some(row.idx),
        duration_s,
        scored.transcribe_dt.as_secs_f32(),
        &row.words,
        &scored.hyp,
        true,
        &scored.result,
        &scored.timing,
        drift_threshold_s,
    );
    Ok(())
}

pub fn run_custom_wav<S: Splitter>(
    transcriber: &mut Transcriber<S>,
    path: &Path,
    drift_threshold_s: f32,
) -> Result<(), Box<dyn Error>> {
    let (waveform, sr) = load_wav(path)?;
    let duration_s = waveform.len() as f32 / sr as f32;
    info!(
        "transcribing custom WAV {} ({duration_s:.1}s @ {sr} Hz)",
        path.display()
    );

    let scored = score_waveform(transcriber, &waveform, sr, &[], drift_threshold_s as f64)?;
    print_single(
        path,
        None,
        duration_s,
        scored.transcribe_dt.as_secs_f32(),
        &[],
        &scored.hyp,
        false,
        &scored.result,
        &scored.timing,
        drift_threshold_s as f64,
    );
    Ok(())
}
