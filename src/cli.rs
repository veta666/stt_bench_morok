//! CLI surface: the `Args` struct (clap-derived).

use clap::{Parser, ValueEnum};
use std::path::PathBuf;

/// Which audio chunker feeds the encoder.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum SplitterChoice {
    /// Silero VAD: model-driven, splits on detected speech regions.
    Silero,
    /// Encoder-stride-aligned fixed windows; no model load. Lower quality
    /// on long-form audio (word splits at chunk seams) but faster startup.
    Fixed,
}

#[derive(Parser, Debug)]
#[command(about = "STT benchmark over GigaAM RN-T", version)]
pub struct Args {
    /// Path to the dataset parquet (audio + ground-truth words). Default is
    /// the local copy of
    /// `veta666/golos_mfa_punctuation_long/data/golos_mfa_punctuation_long_00000.parquet`.
    #[arg(long, default_value = "data/golos_long.parquet")]
    pub dataset: PathBuf,

    /// Single-row mode: transcribe just this dataset `idx` and emit the
    /// pretty per-file panel. Mutually exclusive with `--audio`.
    #[arg(long, conflicts_with = "audio")]
    pub idx: Option<i32>,

    /// Custom WAV path outside the dataset (no ground truth → no WER).
    /// Mutually exclusive with `--idx`.
    #[arg(long, conflicts_with = "idx")]
    pub audio: Option<PathBuf>,

    /// HF Hub repo for GigaAM weights.
    #[arg(long, default_value = "vpermilp/GigaAM-v3")]
    pub repo: String,

    /// HF Hub revision (RN-T head).
    #[arg(long, default_value = "e2e_rnnt")]
    pub revision: String,

    /// Corpus mode: cap the number of dataset rows processed (0 = all).
    #[arg(long, default_value_t = 0)]
    pub limit: usize,

    /// Corpus mode: print the N rows with the worst WER.
    #[arg(long, default_value_t = 5)]
    pub worst: usize,

    /// Midpoint-drift threshold (seconds) for flagging a matched pair as "off".
    #[arg(long, default_value_t = 1.0)]
    pub drift_threshold: f32,

    /// Audio chunker fed into the encoder. `silero` is the default
    /// production path; `fixed` swaps in `FixedLengthSplitter` (no VAD model).
    #[arg(long, value_enum, default_value_t = SplitterChoice::Silero)]
    pub splitter: SplitterChoice,
}
