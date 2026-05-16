//! `stt_bench` binary: thin entry point. All heavy lifting lives in the
//! sibling lib modules (`cli`, `truth`, `bench`, `pretty`, `wer`).

use std::error::Error;

use clap::Parser;
use morok_model::gigaam::TranscribeOpts;
use tracing::info;

use stt_bench::bench::{run_dir, run_single};
use stt_bench::build_rnnt_transcriber;
use stt_bench::cli::{Args, parse_idx_from_filename};
use stt_bench::truth::load_truth;

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    if args.audio.is_none() && args.dir.is_none() {
        return Err("specify --audio FILE or --dir DIR".into());
    }

    let truth = load_truth(&args.truth)?;
    info!(
        "loaded {} ground-truth rows from {}",
        truth.len(),
        args.truth.display(),
    );

    let mut opts = TranscribeOpts::from_env();
    opts.word_timestamps = true; // bench always needs word-level timing
    let mut transcriber = build_rnnt_transcriber(&args.repo, &args.revision, opts)?;

    if let Some(audio) = &args.audio {
        let idx = args.idx.or_else(|| parse_idx_from_filename(audio));
        run_single(&mut transcriber, audio, idx, &truth, args.drift_threshold)?;
    } else if let Some(dir) = &args.dir {
        run_dir(
            &mut transcriber,
            dir,
            &truth,
            args.limit,
            args.worst,
            args.drift_threshold,
        )?;
    }
    Ok(())
}
