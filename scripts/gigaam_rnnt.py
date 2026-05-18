#!/usr/bin/env python3
"""GigaAM RN-T inference.

Two modes:

* Single file (legacy):
    `python gigaam_rnnt.py path/to/audio.wav` prints the transcription to stdout.
    With `--show-timing`, prints `<inference_s>\t<text>` instead so the Rust
    bench can compute an apples-to-apples RTF (inference only, no startup).

* Batch parquet (drives the Python-vs-Rust comparison test):
    `python gigaam_rnnt.py --parquet data/golos_long.parquet --limit 100`
    emits one TSV line per row, `<idx>\t<inference_s>\t<text>`, on stdout.
    `src/bin/gigaam_py_bench.rs` consumes that stream, scores it with
    `transcription_normalization`, and aggregates the per-row inference times
    into corpus RTF.
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import gigaam
from gigaam.preprocess import SAMPLE_RATE, load_audio

DEFAULT_AUDIO = Path(__file__).resolve().parents[1] / "data" / "combined" / "combined0001.wav"
CHUNK_SECONDS = 20
# Encoder's front conv pads (200, 200) on the raw-sample axis, and PyTorch
# reflection padding requires padding < input size — so a trailing slice
# shorter than ~1 s blows up with `Padding size should be less than the
# corresponding input dimension`. We right-pad short tails with zeros to
# clear the conv kernel, then pass the *original* length so the encoder
# masks the padded silence out of its output.
MIN_TAIL_SAMPLES = SAMPLE_RATE


def transcribe_with_timing(model, audio_path: Path) -> tuple[str, float]:
    """Run GigaAM v3 RN-T on a single audio file in fixed 20 s chunks.

    No VAD: chunks are split blindly at `CHUNK_SECONDS` boundaries, so a
    word straddling a seam can be split or duplicated. This is the no-VAD
    baseline against the `morok` crate's Silero splitter — when a row's
    WER is much worse here than in `gigaam_morok_bench`, the seam is the
    usual suspect.

    Returns `(text, inference_s)` where `inference_s` covers the encoder +
    decoder loop only — `load_audio`, tensor moves, and the final string
    join are excluded so the value lines up with `bench.rs::score_waveform`'s
    `transcribe_dt` (the basis for the morok bench's RTF).
    """
    wav = load_audio(str(audio_path)).to(model._device).to(model._dtype)
    chunk = CHUNK_SECONDS * SAMPLE_RATE
    is_cuda = model._device.type == "cuda"

    parts: list[str] = []
    # CUDA kernels are launched async; flush any pending work before/after
    # so perf_counter() captures real GPU time, not just enqueue time.
    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, wav.shape[-1], chunk):
            piece = wav[start : start + chunk]
            real_len = piece.shape[-1]
            # Short tails need zero-padding so the encoder's reflection
            # conv survives (see MIN_TAIL_SAMPLES); real_len is passed
            # separately so the encoder masks the padded silence out.
            if real_len < MIN_TAIL_SAMPLES:
                piece = F.pad(piece, (0, MIN_TAIL_SAMPLES - real_len))
            piece = piece.unsqueeze(0)
            length = torch.full([1], real_len, device=model._device)
            encoded, encoded_len = model.forward(piece, length)
            # decode() returns [(text, token_ids, token_frames)] per batch
            # element; batch=1, so [0][0] is the text of the only sample.
            parts.append(model.decoding.decode(model.head, encoded, encoded_len)[0][0])
    if is_cuda:
        torch.cuda.synchronize()
    inference_s = time.perf_counter() - t0

    return " ".join(p.strip() for p in parts if p.strip()), inference_s


def transcribe(model, audio_path: Path) -> str:
    """Backwards-compatible wrapper: same as `transcribe_with_timing` but
    drops the inference time. Used by `scripts/personal_test_compare.py`
    which imports `transcribe` directly.
    """
    text, _ = transcribe_with_timing(model, audio_path)
    return text


def run_parquet(model, parquet_path: Path, limit: int) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        sys.exit(f"--parquet mode needs pyarrow ({e}); pip install pyarrow")

    pf = pq.ParquetFile(str(parquet_path))
    processed = 0
    for batch in pf.iter_batches(batch_size=16, columns=["idx", "audio"]):
        idx_col = batch.column("idx")
        audio_col = batch.column("audio")
        for i in range(batch.num_rows):
            if limit and processed >= limit:
                return
            idx = idx_col[i].as_py()
            audio_bytes = audio_col[i].as_py()["bytes"]
            with tempfile.NamedTemporaryFile(suffix=".wav") as tf:
                tf.write(audio_bytes)
                tf.flush()
                text, inference_s = transcribe_with_timing(model, Path(tf.name))
            # TSV: idx \t inference_s \t text. text is whitespace-collapsed
            # already, so no tabs/newlines slip through and the Rust side
            # parses with two split_once calls.
            print(f"{idx}\t{inference_s:.6f}\t{text}", flush=True)
            processed += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "audio",
        nargs="?",
        type=Path,
        default=None,
        help="single-file mode: path to a WAV (defaults to data/combined/combined0001.wav)",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="batch mode: read audio.bytes rows from this parquet and emit TSV (idx \\t text)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="batch mode: cap the number of rows processed (0 = all)",
    )
    parser.add_argument(
        "--show-timing",
        action="store_true",
        help="single-file mode: print `<inference_s>\\t<text>` instead of just text",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = gigaam.load_model("rnnt", device="cpu")

    if args.parquet is not None:
        run_parquet(model, args.parquet, args.limit)
        return

    audio = args.audio.expanduser().resolve() if args.audio is not None else DEFAULT_AUDIO
    text, inference_s = transcribe_with_timing(model, audio)
    if args.show_timing:
        print(f"{inference_s:.6f}\t{text}")
    else:
        print(text)


if __name__ == "__main__":
    main()
