//! STT benchmark library wrapping `svod_model::gigaam` inference.
//!
//! The functions in this module mirror the upstream RN-T example at
//! <https://github.com/npatsakula/svod/blob/main/model/examples/gigaam_rnnt_infer.rs>,
//! refactored so the load/build/transcribe steps can be reused by a benchmark
//! driver instead of being glued together inside `main`.

use std::error::Error;
use std::path::Path;

use svod_model::audio::FixedLengthSplitter;
use svod_model::gigaam::{GigaAm, TranscribeOpts, Transcriber};
use svod_model::silero_vad::SileroVadSplitter;

pub use svod_model::audio;
pub use svod_model::gigaam;
pub use svod_model::silero_vad;
use tracing::info;

pub mod bench;
pub mod cli;
pub mod dataset;
pub mod pretty;
pub mod wer;

pub use wer::{AlignmentResult, Op, TimingStats, Token, Word, compute_wer, timing_stats};

/// Decode a WAV stream from any `Read + Seek` source into `(samples, sample_rate)`.
/// Integer samples are normalized to `[-1.0, 1.0]` by dividing by `32768.0`.
fn decode_wav<R: std::io::Read>(
    mut reader: hound::WavReader<R>,
) -> Result<(Vec<f32>, u32), Box<dyn Error>> {
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

/// Read a WAV file from disk and return `(samples, sample_rate)`.
pub fn load_wav(path: impl AsRef<Path>) -> Result<(Vec<f32>, u32), Box<dyn Error>> {
    decode_wav(hound::WavReader::open(path)?)
}

/// Decode an in-memory WAV blob (e.g. the `audio.bytes` column from the
/// dataset parquet) into `(samples, sample_rate)`.
pub fn load_wav_from_bytes(bytes: &[u8]) -> Result<(Vec<f32>, u32), Box<dyn Error>> {
    decode_wav(hound::WavReader::new(std::io::Cursor::new(bytes))?)
}

fn load_rnnt_model(repo: &str, revision: &str) -> Result<GigaAm, Box<dyn Error>> {
    info!("\nLoading GigaAM RNN-T from {repo} ({revision})...");
    let model = GigaAm::from_hub_with_revision(repo, revision)?;
    if model.head.as_rnnt().is_none() {
        return Err(format!(
            "{repo}@{revision} has a CTC head, not RN-T. \
             Set svod_RNNT_REVISION to an RN-T revision."
        )
        .into());
    }
    Ok(model)
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
    let model = load_rnnt_model(repo, revision)?;
    let splitter = SileroVadSplitter::from_hub()?;
    Ok(Transcriber::new(model, splitter, opts)?)
}

/// Same as [`build_rnnt_transcriber`] but with the no-VAD
/// [`FixedLengthSplitter`] — encoder-stride-aligned fixed windows, zero
/// extra model load. Boundary context is lost at chunk seams; expect a WER
/// regression vs. Silero on long-form audio.
pub fn build_rnnt_transcriber_fixed(
    repo: &str,
    revision: &str,
    opts: TranscribeOpts,
) -> Result<Transcriber<FixedLengthSplitter>, Box<dyn Error>> {
    let model = load_rnnt_model(repo, revision)?;
    Ok(Transcriber::new(model, FixedLengthSplitter::new(), opts)?)
}
