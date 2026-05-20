//! Python-vs-ground-truth WER bench for `scripts/gigaam_rnnt.py`.
//!
//! Two modes, same scoring path (`transcription_normalization::compare`):
//!
//! * Corpus: spawn the Python script in `--parquet` mode, stream
//!   `<idx>\t<text>` lines, score against per-row ground truth.
//! * Single (`--idx N`): extract that one row's audio bytes from the parquet
//!   into a temp WAV, invoke the Python script in single-file mode, parse
//!   its stdout, score one row.
//!
//! Python doesn't emit word timestamps, so this bench is WER-only.

use std::collections::HashMap;
use std::error::Error;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use clap::Parser;

use stt_bench_svod::dataset::{DatasetIter, find_row};
use stt_bench_svod::pretty::print_single_wer_only;
use stt_bench_svod::wer::{Word, compute_wer, fold_short_i};

#[derive(Parser, Debug)]
#[command(
    about = "Score scripts/gigaam_rnnt.py output against ground truth",
    version
)]
struct Args {
    /// Dataset parquet (same one bench.rs uses).
    #[arg(long, default_value = "data/golos_long.parquet")]
    dataset: PathBuf,

    /// Corpus mode: rows to score (0 = all). Ignored when --idx is set.
    #[arg(long, default_value_t = 100, conflicts_with = "idx")]
    limit: usize,

    /// Single-row mode: score just this dataset `idx`. Mutually exclusive with --limit.
    #[arg(long)]
    idx: Option<i32>,

    /// Path to the Python script.
    #[arg(long, default_value = "scripts/gigaam_rnnt.py")]
    script: PathBuf,

    /// Corpus mode: print the N worst-WER rows after the summary.
    #[arg(long, default_value_t = 5)]
    worst: usize,
}

struct RowScore {
    idx: i32,
    ref_len: usize,
    subs: usize,
    dels: usize,
    ins: usize,
    wer: f64,
}

fn python_bin() -> String {
    std::env::var("stt_bench_svod_PYTHON").unwrap_or_else(|_| "python".into())
}

fn tokenize_hyp(text: &str) -> Vec<Word> {
    text.split_whitespace()
        .map(|t| Word::new(fold_short_i(t), 0.0, 0.0))
        .collect()
}

#[allow(clippy::type_complexity)]
fn ground_truth(
    dataset: &Path,
    limit: usize,
) -> Result<HashMap<i32, (Vec<Word>, f32)>, Box<dyn Error>> {
    let iter = DatasetIter::open(dataset)?;
    let mut out: HashMap<i32, (Vec<Word>, f32)> = HashMap::new();
    for row in iter {
        if limit > 0 && out.len() >= limit {
            break;
        }
        let row = row?;
        out.insert(row.idx, (row.words, row.duration));
    }
    Ok(out)
}

fn run_corpus(args: &Args) -> Result<(), Box<dyn Error>> {
    eprintln!(
        "loading ground truth for first {} rows from {}",
        args.limit,
        args.dataset.display()
    );
    let truth = ground_truth(&args.dataset, args.limit)?;
    eprintln!("ground truth: {} rows", truth.len());

    eprintln!(
        "spawning {} {} --parquet {} --limit {}",
        python_bin(),
        args.script.display(),
        args.dataset.display(),
        args.limit,
    );
    let subprocess_started = Instant::now();
    let mut child = Command::new(python_bin())
        .arg(&args.script)
        .arg("--parquet")
        .arg(&args.dataset)
        .arg("--limit")
        .arg(args.limit.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;

    let stdout = child.stdout.take().ok_or("child stdout unavailable")?;
    let reader = BufReader::new(stdout);

    let mut scored: Vec<RowScore> = Vec::new();
    let mut total_subs = 0usize;
    let mut total_dels = 0usize;
    let mut total_ins = 0usize;
    let mut total_ref = 0usize;
    let mut total_dur_s = 0.0f32;
    let mut total_inference_s = 0.0f32;

    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        // 3-column TSV: idx \t inference_s \t text.
        let (idx_s, rest) = line
            .split_once('\t')
            .ok_or_else(|| format!("malformed TSV at line {}: {line:?}", line_no + 1))?;
        let (inference_s_s, text) = rest
            .split_once('\t')
            .ok_or_else(|| format!("malformed TSV at line {}: {line:?}", line_no + 1))?;
        let idx: i32 = idx_s.parse()?;
        let inference_s: f32 = inference_s_s.parse()?;
        let (reference, duration) = truth.get(&idx).map(|(w, d)| (w, *d)).ok_or_else(|| {
            format!(
                "idx {idx} from python not in first {} dataset rows",
                args.limit
            )
        })?;

        let hyp = tokenize_hyp(text);
        let result = compute_wer(reference, &hyp);
        let subs = result.substitutions();
        let dels = result.deletions();
        let ins = result.insertions();
        let ref_len = result.ref_token_count();

        total_subs += subs;
        total_dels += dels;
        total_ins += ins;
        total_ref += ref_len;
        total_dur_s += duration;
        total_inference_s += inference_s;

        scored.push(RowScore {
            idx,
            ref_len,
            subs,
            dels,
            ins,
            wer: result.wer(),
        });

        if scored.len().is_multiple_of(10) {
            let corpus = if total_ref == 0 {
                0.0
            } else {
                (total_subs + total_dels + total_ins) as f64 / total_ref as f64
            };
            eprintln!(
                "  {} rows scored, rolling WER = {:.2}%",
                scored.len(),
                corpus * 100.0
            );
        }
    }

    let status = child.wait()?;
    let subprocess_s = subprocess_started.elapsed().as_secs_f32();
    if !status.success() {
        return Err(format!("python subprocess exited with {status}").into());
    }
    if scored.is_empty() {
        return Err("python emitted no rows".into());
    }

    let corpus_wer = (total_subs + total_dels + total_ins) as f64 / total_ref as f64;
    // RTF uses the sum of per-row Python-side inference timings (encoder +
    // decoder loop only, CUDA-synchronized), matching what `bench.rs::
    // score_waveform` measures in the svod bench. Subprocess wall-clock
    // is reported separately for context — its delta vs total_inference_s
    // is roughly the Python startup + model-load overhead.
    let rtf = if total_dur_s > 0.0 {
        total_inference_s as f64 / total_dur_s as f64
    } else {
        0.0
    };
    println!();
    println!("=== gigaam_rnnt.py vs ground truth ===");
    println!("rows scored : {}", scored.len());
    println!("ref tokens  : {total_ref}");
    println!("S / D / I   : {total_subs} / {total_dels} / {total_ins}");
    println!("corpus WER  : {:.2}%", corpus_wer * 100.0);
    println!(
        "audio total : {:.1}s ({:.2}h)",
        total_dur_s,
        total_dur_s / 3600.0,
    );
    println!("subprocess  : {subprocess_s:.1}s  (wall-clock, incl. startup + model load)");
    println!("inference   : {total_inference_s:.1}s  (sum of per-row Python timings)");
    println!("RTF         : {rtf:.3}x  (inference / audio, comparable to gigaam_svod_bench)");

    if args.worst > 0 {
        let mut sorted: Vec<&RowScore> = scored.iter().collect();
        sorted.sort_by(|a, b| {
            b.wer
                .partial_cmp(&a.wer)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        println!();
        println!("worst {} rows by WER:", args.worst.min(sorted.len()));
        println!(
            "  {:>6}  {:>7}  {:>4}  {:>4}  {:>4}  {:>7}",
            "idx", "ref_len", "S", "D", "I", "WER"
        );
        for r in sorted.iter().take(args.worst) {
            println!(
                "  {:>6}  {:>7}  {:>4}  {:>4}  {:>4}  {:>6.2}%",
                r.idx,
                r.ref_len,
                r.subs,
                r.dels,
                r.ins,
                r.wer * 100.0
            );
        }
    }

    Ok(())
}

fn run_single(args: &Args, idx: i32) -> Result<(), Box<dyn Error>> {
    eprintln!("looking up idx={idx} in {}", args.dataset.display());
    let row = find_row(&args.dataset, idx)?
        .ok_or_else(|| format!("idx {idx} not found in {}", args.dataset.display()))?;

    // The Python script reads its single-file argument as a path, so we
    // materialize this row's audio.bytes (already RIFF/PCM WAV per the
    // dataset's schema) to a temp file. PID-tagged so concurrent runs
    // don't clobber each other.
    let tmp_path =
        std::env::temp_dir().join(format!("gigaam_py_idx_{idx}_pid{}.wav", std::process::id()));
    std::fs::write(&tmp_path, &row.audio_bytes)?;

    eprintln!(
        "transcribing idx={idx} via {} {} {} --show-timing",
        python_bin(),
        args.script.display(),
        tmp_path.display()
    );
    let output = Command::new(python_bin())
        .arg(&args.script)
        .arg(&tmp_path)
        .arg("--show-timing")
        .stderr(Stdio::inherit())
        .output();
    let _ = std::fs::remove_file(&tmp_path);
    let output = output?;
    if !output.status.success() {
        return Err(format!("python subprocess exited with {}", output.status).into());
    }

    // With --show-timing the script prints `<inference_s>\t<text>` so we
    // can report the same inference-only RTF as gigaam_svod_bench.
    let stdout = String::from_utf8(output.stdout)?;
    let stdout = stdout.trim();
    let (inference_s_s, hyp_text) = stdout
        .split_once('\t')
        .ok_or_else(|| format!("python single-mode output missing timing column: {stdout:?}"))?;
    let inference_s: f32 = inference_s_s.parse()?;
    let hyp = tokenize_hyp(hyp_text);
    let result = compute_wer(&row.words, &hyp);

    let label = PathBuf::from(format!("gigaam_rnnt.py[idx={idx}]"));
    print_single_wer_only(
        &label,
        Some(idx),
        row.duration,
        inference_s,
        &row.words,
        &hyp,
        &result,
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    if !args.dataset.exists() {
        return Err(format!("dataset parquet not found: {}", args.dataset.display()).into());
    }
    if !args.script.exists() {
        return Err(format!("python script not found: {}", args.script.display()).into());
    }

    match args.idx {
        Some(idx) => run_single(&args, idx),
        None => run_corpus(&args),
    }
}
