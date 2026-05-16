//! STT benchmark library wrapping `morok_model::gigaam` inference.
//!
//! The functions in this module mirror the upstream RN-T example at
//! <https://github.com/npatsakula/morok/blob/main/model/examples/gigaam_rnnt_infer.rs>,
//! refactored so the load/build/transcribe steps can be reused by a benchmark
//! driver instead of being glued together inside `main`.

use std::error::Error;
use std::path::Path;

use morok_model::gigaam::{GigaAm, TranscribeOpts, Transcriber};
use morok_model::silero_vad::SileroVadSplitter;

pub use morok_model::gigaam;
pub use morok_model::silero_vad;
use tracing::info;

pub mod bench;
pub mod cli;
pub mod pretty;
pub mod truth;
pub mod wer;

pub use wer::{
    AlignOp, TimingStats, WerResult, WerWord, compute_wer, normalize_word, timing_stats,
};

/// Read a WAV file from disk and return `(samples, sample_rate)`.
pub fn load_wav(path: impl AsRef<Path>) -> Result<(Vec<f32>, u32), Box<dyn Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.0))
            .collect::<Result<_, _>>()?,
    };
    Ok((samples, spec.sample_rate))
}

/// Load the GigaAM RN-T model + Silero VAD splitter from the Hugging Face Hub
/// and return a ready-to-use `Transcriber`.
///
/// Errors if the resolved revision exposes a CTC head instead of RN-T.
pub fn build_rnnt_transcriber(
    repo: &str,
    revision: &str,
    opts: TranscribeOpts,
) -> Result<Transcriber<SileroVadSplitter>, Box<dyn Error>> {
    info!("\nLoading GigaAM RNN-T from {repo} ({revision})...");
    let model = GigaAm::from_hub_with_revision(repo, revision)?;
    if model.head.as_rnnt().is_none() {
        return Err(format!(
            "{repo}@{revision} has a CTC head, not RN-T. \
             Set MOROK_RNNT_REVISION to an RN-T revision."
        )
        .into());
    }

    let splitter = SileroVadSplitter::from_hub()?;
    Ok(Transcriber::new(model, splitter, opts)?)
}
