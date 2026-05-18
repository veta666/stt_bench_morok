# stt_bench

A long-form Russian STT benchmark for the [GigaAM RN-T](https://huggingface.co/vpermilp/GigaAM-v3)
model via the [morok](https://github.com/npatsakula/morok) crate. Runs over
[`veta666/golos_mfa_punctuation_long`](https://huggingface.co/datasets/veta666/golos_mfa_punctuation_long)
(1903 clips of ~60 s each, derived from Golos by splicing 15 short clips per
combined clip with randomized inter-clip silences). Reports corpus-level WER
and per-word timing drift.

## Setup

```bash
# Audio + ground-truth in one parquet (~2.6 GB).
mkdir -p data
wget -O data/golos_long.parquet \
  "https://huggingface.co/datasets/veta666/golos_mfa_punctuation_long/resolve/main/data/golos_mfa_punctuation_long_00000.parquet?download=true"
```

The GigaAM weights are downloaded automatically from Hugging Face on first
run.

## Usage

```bash
# Corpus mode: stream the whole dataset, emit summary + worst-N panel.
cargo run --release --bin gigaam_morok_bench

# Subset for a quick sanity check.
cargo run --release --bin gigaam_morok_bench -- --limit 50

# Single dataset row, pretty per-file panel with colored word-level diff.
cargo run --release --bin gigaam_morok_bench -- --idx 42

# Custom WAV outside the dataset (no ground truth → no WER).
cargo run --release --bin gigaam_morok_bench -- --audio /tmp/test.wav

# Swap Silero VAD for the no-VAD fixed-window splitter (faster startup,
# worse WER on long-form audio).
cargo run --release --bin gigaam_morok_bench -- --splitter fixed
```

Useful flags:

| Flag | Default | Notes |
|---|---|---|
| `--dataset PATH` | `data/golos_long.parquet` | Local parquet path |
| `--idx N` | — | Run just one dataset row |
| `--audio PATH` | — | Custom WAV (mutually exclusive with `--idx`) |
| `--limit N` | `0` (all) | Cap rows in corpus mode |
| `--worst N` | `5` | Worst-WER rows to show |
| `--drift-threshold S` | `1.0` | Seconds; flag matched-word timing drift above this |
| `--splitter {silero,fixed}` | `silero` | Audio chunker |
| `--repo` / `--revision` | GigaAM-v3 / `e2e_rnnt` | HF Hub source |

## Scoring

**WER** = `(S + D + I) / N_ref` via word-level Levenshtein with backtracking
(see `src/wer.rs`). Corpus mode reports the sum-based aggregate, not a mean of
per-file rates.

**Word normalization** before comparison: lowercase, fold `й → и` and
`ё → е` (both spellings are inconsistent between Golos labels and what the
model emits), strip non-alphanumeric chars from both ends.

**Timing** is scored separately: for every matched pair the bench records
`|hyp.midpoint − ref.midpoint|`, then reports mean / median / p95 plus a
count of "high-drift" matches (above `--drift-threshold`). Per-file mode
flags each high-drift word inline in the alignment table.

## Layout

```
src/
├── lib.rs                       # model glue: load_wav, build_rnnt_transcriber{,_fixed}
├── cli.rs                       # clap Args + SplitterChoice
├── dataset.rs                   # streaming reader for the HF parquet (audio + words)
├── bench.rs                     # run_dataset / run_idx / run_custom_wav + score_waveform
├── pretty.rs                    # ANSI single-file panel + corpus summary
├── wer.rs                       # normalize_word, compute_wer, timing_stats
└── bin/
    ├── gigaam_morok_bench.rs    # CLI entry: morok-backed bench
    └── gigaam_py_bench.rs       # CLI entry: Python-script-backed bench
```
