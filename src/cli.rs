//! CLI surface: the `Args` struct (clap-derived) and small filename helpers.

use clap::Parser;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "STT benchmark over GigaAM RN-T", version)]
pub struct Args {
    /// Single WAV to transcribe; produces the pretty per-file panel.
    #[arg(long, conflicts_with = "dir")]
    pub audio: Option<PathBuf>,

    /// Directory of `combinedNNNN.wav` files; produces aggregate stats.
    #[arg(long, conflicts_with = "audio")]
    pub dir: Option<PathBuf>,

    /// Combined parquet with ground truth (`idx` + `words` columns).
    #[arg(long, default_value = "data/combined.parquet")]
    pub truth: PathBuf,

    /// Override the idx parsed from the filename (single-file mode only).
    #[arg(long)]
    pub idx: Option<i32>,

    /// Default Hugging Face Hub repo for the GigaAM RN-T weights.
    #[arg(long, default_value = "vpermilp/GigaAM-v3")]
    pub repo: String,

    /// Default revision (RN-T head) inside the repo above.
    #[arg(long, default_value = "e2e_rnnt")]
    pub revision: String,

    /// In dir mode, cap the number of files processed (0 = all).
    #[arg(long, default_value_t = 0)]
    pub limit: usize,

    /// In dir mode, print the N files with the worst WER.
    #[arg(long, default_value_t = 5)]
    pub worst: usize,

    /// Midpoint-drift threshold (seconds) for flagging a matched pair as "off".
    #[arg(long, default_value_t = 1.0)]
    pub drift_threshold: f32,
}

/// Parse a trailing run of digits in the filename stem as an i32.
/// `combined0042.wav` -> `Some(42)`; non-numeric stems return `None`.
pub fn parse_idx_from_filename(p: &Path) -> Option<i32> {
    let stem = p.file_stem()?.to_str()?;
    let digits_rev: String = stem
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if digits_rev.is_empty() {
        return None;
    }
    digits_rev.chars().rev().collect::<String>().parse().ok()
}
