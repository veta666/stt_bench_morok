//! `stt_bench_svod` binary: thin entry point. All heavy lifting lives in the
//! sibling lib modules (`cli`, `dataset`, `bench`, `pretty`, `wer`).

use std::error::Error;

use clap::Parser;
use svod_model::audio::Splitter;
use svod_model::gigaam::{TranscribeOpts, Transcriber};

use stt_bench_svod::bench::{run_custom_wav, run_dataset, run_idx};
use stt_bench_svod::cli::{Args, SplitterChoice};
use stt_bench_svod::{build_rnnt_transcriber, build_rnnt_transcriber_fixed};

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let mut opts = TranscribeOpts::from_env();
    opts.word_timestamps = true; // bench always needs word-level timing

    match args.splitter {
        SplitterChoice::Silero => {
            let mut t = build_rnnt_transcriber(&args.repo, &args.revision, opts)?;
            dispatch(&args, &mut t)
        }
        SplitterChoice::Fixed => {
            let mut t = build_rnnt_transcriber_fixed(&args.repo, &args.revision, opts)?;
            dispatch(&args, &mut t)
        }
    }
}

fn dispatch<S: Splitter>(
    args: &Args,
    transcriber: &mut Transcriber<S>,
) -> Result<(), Box<dyn Error>> {
    if let Some(audio) = &args.audio {
        run_custom_wav(transcriber, audio, args.drift_threshold)
    } else if let Some(idx) = args.idx {
        run_idx(transcriber, &args.dataset, idx, args.drift_threshold)
    } else {
        run_dataset(
            transcriber,
            &args.dataset,
            args.limit,
            args.worst,
            args.drift_threshold,
        )
    }
}
