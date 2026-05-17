//! Streaming reader for the long-form benchmark parquet.
//!
//! Source: <https://huggingface.co/datasets/veta666/golos_mfa_punctuation_long>.
//! Download once with e.g. `wget` and pass the local path through `--dataset`.
//!
//! The audio bytes column is heavy (~2.6 GB total), so we yield one row at a
//! time — never building a `HashMap` over the whole dataset.

use std::error::Error;
use std::fs::File;
use std::path::Path;

use arrow::array::{
    Array, BinaryArray, Float32Array, Int32Array, ListArray, RecordBatch, StringArray, StructArray,
};
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};

use crate::WerWord;

/// One row of the dataset, decoded into owned form so the caller is free of
/// arrow lifetimes.
#[derive(Debug, Clone)]
pub struct DatasetRow {
    pub idx: i32,
    pub duration: f32,
    /// WAV bytes (RIFF/PCM 16 kHz mono).
    pub audio_bytes: Vec<u8>,
    /// Ground-truth words on the combined timeline (start/end in seconds).
    pub words: Vec<WerWord>,
}

/// Borrow a typed child array from a parent exposing `column_by_name` (works
/// for both `RecordBatch` and `StructArray`).
macro_rules! col {
    ($parent:expr, $name:expr, $T:ty) => {{
        let arr = $parent
            .column_by_name($name)
            .ok_or_else(|| -> Box<dyn Error> { format!("missing column `{}`", $name).into() })?;
        arr.as_any()
            .downcast_ref::<$T>()
            .ok_or_else(|| -> Box<dyn Error> {
                format!("column `{}` has unexpected arrow type", $name).into()
            })?
    }};
}

/// Streaming iterator over `(idx, duration, audio_bytes, words)` from the
/// dataset parquet. Each yielded item is `Result<DatasetRow>`.
pub struct DatasetIter {
    reader: ParquetRecordBatchReader,
    current: Option<RecordBatch>,
    row_in_batch: usize,
}

impl DatasetIter {
    pub fn open(path: &Path) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
        Ok(Self {
            reader,
            current: None,
            row_in_batch: 0,
        })
    }
}

impl Iterator for DatasetIter {
    type Item = Result<DatasetRow, Box<dyn Error>>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(batch) = self.current.as_ref() {
                if self.row_in_batch < batch.num_rows() {
                    let row = self.row_in_batch;
                    self.row_in_batch += 1;
                    return Some(decode_row(batch, row));
                }
                self.current = None;
            }
            match self.reader.next() {
                None => return None,
                Some(Err(e)) => return Some(Err(Box::new(e))),
                Some(Ok(batch)) => {
                    self.current = Some(batch);
                    self.row_in_batch = 0;
                }
            }
        }
    }
}

fn decode_row(batch: &RecordBatch, row: usize) -> Result<DatasetRow, Box<dyn Error>> {
    let idx_arr = col!(batch, "idx", Int32Array);
    let dur_arr = col!(batch, "duration", Float32Array);
    let audio_arr = col!(batch, "audio", StructArray);
    let words_arr = col!(batch, "words", ListArray);

    let audio_bytes_arr = col!(audio_arr, "bytes", BinaryArray);
    let audio_bytes = audio_bytes_arr.value(row).to_vec();

    let words_for_row = words_arr.value(row);
    let struct_arr = words_for_row
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or("`words.element` is not a struct")?;
    let text_arr = col!(struct_arr, "text", StringArray);
    let start_arr = col!(struct_arr, "start", Float32Array);
    let end_arr = col!(struct_arr, "end", Float32Array);

    let words: Vec<WerWord> = (0..struct_arr.len())
        .map(|k| WerWord::new(text_arr.value(k), start_arr.value(k), end_arr.value(k)))
        .collect();

    Ok(DatasetRow {
        idx: idx_arr.value(row),
        duration: dur_arr.value(row),
        audio_bytes,
        words,
    })
}

/// Convenience: linear scan for a single `idx`. Returns `Ok(None)` if not
/// found. Cheap enough for one-off lookups; for repeated random access build
/// your own index.
pub fn find_row(path: &Path, idx: i32) -> Result<Option<DatasetRow>, Box<dyn Error>> {
    for row in DatasetIter::open(path)? {
        let row = row?;
        if row.idx == idx {
            return Ok(Some(row));
        }
    }
    Ok(None)
}
