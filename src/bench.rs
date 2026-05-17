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
use crate::wer::{TimingStats, WerResult, WerWord, compute_wer, timing_stats};
use crate::{load_wav, load_wav_from_bytes};

/// Convert the model's word stream into our normalized `WerWord` form,
/// dropping any tokens that normalize to empty strings.
pub fn hyp_words(result: &TranscribeResult) -> Vec<WerWord> {
    result
        .words()
        .map(|w| WerWord::new(&w.text, w.start, w.end))
        .filter(|w| !w.text.is_empty())
        .collect()
}

/// One row of per-file results, accumulated by `run_dataset` and consumed
/// by the summary printer.
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
    pub ref_len: usize,
    pub matched_pairs: usize,
    pub high_drift_pairs: usize,
    pub mean_abs_mid_s: f32,
}

/// Output of [`score_waveform`]: timing + hypothesis + WER + drift stats.
struct Scored {
    transcribe_dt: Duration,
    hyp: Vec<WerWord>,
    wer: WerResult,
    timing: TimingStats,
}

/// Transcribe `waveform`, score against `reference` (use `&[]` for no
/// ground truth). Returns all the inputs both the per-file panel and the
/// corpus accumulator need.
fn score_waveform<S: Splitter>(
    transcriber: &mut Transcriber<S>,
    waveform: &[f32],
    sample_rate: u32,
    reference: &[WerWord],
    drift_threshold_s: f32,
) -> Result<Scored, Box<dyn Error>> {
    let t = Instant::now();
    let result = transcriber.transcribe(waveform, sample_rate)?;
    let transcribe_dt = t.elapsed();

    let hyp = hyp_words(&result);
    let wer = compute_wer(reference, &hyp);
    let timing = timing_stats(reference, &hyp, &wer.alignment, drift_threshold_s);
    Ok(Scored {
        transcribe_dt,
        hyp,
        wer,
        timing,
    })
}

/// Stream the whole dataset, transcribe each row, emit the corpus summary.
pub fn run_dataset<S: Splitter>(
    transcriber: &mut Transcriber<S>,
    dataset: &Path,
    limit: usize,
    worst_n: usize,
    drift_threshold_s: f32,
) -> Result<(), Box<dyn Error>> {
    info!("streaming dataset {}", dataset.display());

    let mut totals = WerResult {
        substitutions: 0,
        deletions: 0,
        insertions: 0,
        ref_len: 0,
        alignment: Vec::new(),
    };
    let mut total_timing = TimingStats::default();
    let mut total_dur_s = 0.0f32;
    let mut total_xt_s = 0.0f32;
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
        let scored = score_waveform(transcriber, &waveform, sr, &row.words, drift_threshold_s)?;

        totals.substitutions += scored.wer.substitutions;
        totals.deletions += scored.wer.deletions;
        totals.insertions += scored.wer.insertions;
        totals.ref_len += scored.wer.ref_len;
        total_timing.merge(&scored.timing);
        total_dur_s += duration_s;
        total_xt_s += scored.transcribe_dt.as_secs_f32();

        per_file.push(FileScore {
            idx: Some(row.idx),
            path: PathBuf::from(format!("idx{:05}", row.idx)),
            duration_s,
            transcribe_s: scored.transcribe_dt.as_secs_f32(),
            wer: scored.wer.wer(),
            subs: scored.wer.substitutions,
            dels: scored.wer.deletions,
            ins: scored.wer.insertions,
            ref_len: scored.wer.ref_len,
            matched_pairs: scored.timing.matched_pairs,
            high_drift_pairs: scored.timing.high_drift_pairs,
            mean_abs_mid_s: scored.timing.mean_abs_mid(),
        });

        processed += 1;
        if processed.is_multiple_of(10) {
            let elapsed = started.elapsed().as_secs_f32();
            let rate = processed as f32 / elapsed;
            info!(
                "  {processed} rows  rate={rate:.1}/s  elapsed={elapsed:.0}s  rolling WER={:.2}%",
                totals.wer() * 100.0,
            );
        }
    }

    if per_file.is_empty() {
        return Err("dataset is empty".into());
    }

    print_summary(
        &per_file,
        &totals,
        &total_timing,
        total_dur_s,
        total_xt_s,
        worst_n,
        drift_threshold_s,
    );
    Ok(())
}

/// Single-row mode: fetch one `idx` from the dataset and emit the per-file
/// panel.
pub fn run_idx<S: Splitter>(
    transcriber: &mut Transcriber<S>,
    dataset: &Path,
    idx: i32,
    drift_threshold_s: f32,
) -> Result<(), Box<dyn Error>> {
    let row = find_row(dataset, idx)?
        .ok_or_else(|| -> Box<dyn Error> { format!("idx {idx} not found in dataset").into() })?;
    print_dataset_row(transcriber, &row, drift_threshold_s)
}

fn print_dataset_row<S: Splitter>(
    transcriber: &mut Transcriber<S>,
    row: &DatasetRow,
    drift_threshold_s: f32,
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
        &scored.wer,
        &scored.timing,
        drift_threshold_s,
    );
    Ok(())
}

/// Escape hatch: transcribe an arbitrary WAV from disk, no ground truth.
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

    let scored = score_waveform(transcriber, &waveform, sr, &[], drift_threshold_s)?;
    print_single(
        path,
        None,
        duration_s,
        scored.transcribe_dt.as_secs_f32(),
        &[],
        &scored.hyp,
        false,
        &scored.wer,
        &scored.timing,
        drift_threshold_s,
    );
    Ok(())
}
