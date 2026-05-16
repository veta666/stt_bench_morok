//! Bench drivers: load a WAV, transcribe, score against ground truth, render.
//!
//! Two entry points: [`run_single`] for the pretty per-file panel and
//! [`run_dir`] for the corpus-level summary. The model is loaded once by the
//! caller (`main`) and threaded in by mutable reference, so a directory run
//! pays the HF download / weight-load cost exactly once.

use std::collections::HashMap;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;

use morok_model::gigaam::{TranscribeResult, Transcriber};
use morok_model::silero_vad::SileroVadSplitter;
use tracing::info;

use crate::load_wav;
use crate::pretty::{print_single, print_summary};
use crate::truth::TruthRow;
use crate::wer::{TimingStats, WerResult, WerWord, compute_wer, timing_stats};

/// Convert the model's word stream into our normalized `WerWord` form,
/// dropping any tokens that normalize to empty strings.
pub fn hyp_words(result: &TranscribeResult) -> Vec<WerWord> {
    result
        .words()
        .map(|w| WerWord::new(&w.text, w.start, w.end))
        .filter(|w| !w.text.is_empty())
        .collect()
}

/// One row of per-file results, accumulated by `run_dir` and consumed by
/// the summary printer.
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

/// Transcribe one WAV and emit the colored per-file panel.
pub fn run_single(
    transcriber: &mut Transcriber<SileroVadSplitter>,
    path: &Path,
    idx: Option<i32>,
    truth: &HashMap<i32, TruthRow>,
    drift_threshold_s: f32,
) -> Result<(), Box<dyn Error>> {
    let (waveform, sr) = load_wav(path)?;
    let duration_s = waveform.len() as f32 / sr as f32;

    info!(
        "transcribing {} ({:.1}s @ {} Hz)",
        path.display(),
        duration_s,
        sr
    );
    let t = Instant::now();
    let result = transcriber.transcribe(&waveform, sr)?;
    let dt = t.elapsed();

    let hyp = hyp_words(&result);
    let truth_row = idx.and_then(|i| truth.get(&i));
    let ref_words: Vec<WerWord> = truth_row.map(|r| r.words.clone()).unwrap_or_default();

    let wer = compute_wer(&ref_words, &hyp);
    let timing = timing_stats(&ref_words, &hyp, &wer.alignment, drift_threshold_s);

    print_single(
        path,
        idx,
        duration_s,
        dt.as_secs_f32(),
        &ref_words,
        &hyp,
        truth_row.is_some(),
        &wer,
        &timing,
        drift_threshold_s,
    );
    Ok(())
}

/// Transcribe every `.wav` in `dir`, accumulate corpus-level stats, and emit
/// the summary panel. Files are processed in sorted-name order.
pub fn run_dir(
    transcriber: &mut Transcriber<SileroVadSplitter>,
    dir: &Path,
    truth: &HashMap<i32, TruthRow>,
    limit: usize,
    worst_n: usize,
    drift_threshold_s: f32,
) -> Result<(), Box<dyn Error>> {
    let wavs = collect_wavs(dir, limit)?;
    let total = wavs.len();
    info!("processing {} files from {}", total, dir.display());

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
    let mut per_file: Vec<FileScore> = Vec::with_capacity(total);

    let started = Instant::now();
    for (i, path) in wavs.iter().enumerate() {
        let idx = crate::cli::parse_idx_from_filename(path);
        let (waveform, sr) = load_wav(path)?;
        let duration_s = waveform.len() as f32 / sr as f32;

        let t = Instant::now();
        let result = transcriber.transcribe(&waveform, sr)?;
        let dt = t.elapsed();

        let hyp = hyp_words(&result);
        let ref_words: Vec<WerWord> = idx
            .and_then(|i| truth.get(&i))
            .map(|r| r.words.clone())
            .unwrap_or_default();

        let wer = compute_wer(&ref_words, &hyp);
        let timing = timing_stats(&ref_words, &hyp, &wer.alignment, drift_threshold_s);

        totals.substitutions += wer.substitutions;
        totals.deletions += wer.deletions;
        totals.insertions += wer.insertions;
        totals.ref_len += wer.ref_len;
        total_timing.merge(&timing);
        total_dur_s += duration_s;
        total_xt_s += dt.as_secs_f32();

        per_file.push(FileScore {
            idx,
            path: path.clone(),
            duration_s,
            transcribe_s: dt.as_secs_f32(),
            wer: wer.wer(),
            subs: wer.substitutions,
            dels: wer.deletions,
            ins: wer.insertions,
            ref_len: wer.ref_len,
            matched_pairs: timing.matched_pairs,
            high_drift_pairs: timing.high_drift_pairs,
            mean_abs_mid_s: timing.mean_abs_mid(),
        });

        if (i + 1) % 10 == 0 || i + 1 == total {
            let elapsed = started.elapsed().as_secs_f32();
            let rate = (i + 1) as f32 / elapsed;
            info!(
                "  {}/{}  rate={:.1}/s  elapsed={:.0}s  rolling WER={:.2}%",
                i + 1,
                total,
                rate,
                elapsed,
                totals.wer() * 100.0,
            );
        }
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

/// Sorted list of `.wav` files directly under `dir`, optionally capped at `limit`.
fn collect_wavs(dir: &Path, limit: usize) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut wavs: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("wav"))
        .collect();
    wavs.sort();
    if limit > 0 && wavs.len() > limit {
        wavs.truncate(limit);
    }
    if wavs.is_empty() {
        return Err(format!("no .wav files found in {}", dir.display()).into());
    }
    Ok(wavs)
}
