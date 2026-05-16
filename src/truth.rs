//! Ground-truth loader for `data/combined.parquet`.
//!
//! Reads only `idx`, `duration`, and `words` — the heavy `audio.bytes` column
//! is projected away so we never materialize the 2.5 GB WAV blob.

use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::path::Path;

use arrow::array::{Array, Float32Array, Int32Array, ListArray, StringArray, StructArray};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::WerWord;

/// One row of the combined parquet, keyed by `idx` in the surrounding map.
#[derive(Debug, Clone)]
#[allow(dead_code)] // idx duplicates the map key; duration is informational
pub struct TruthRow {
    pub idx: i32,
    pub duration: f32,
    pub words: Vec<WerWord>,
}

/// Borrow a typed child array from a parent that exposes `column_by_name`
/// (works for both `RecordBatch` and `StructArray`). Returns a typed error
/// if the column is missing or has the wrong arrow type.
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

pub fn load_truth(path: &Path) -> Result<HashMap<i32, TruthRow>, Box<dyn Error>> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;

    let schema_descr = builder.parquet_schema();
    let kept: Vec<usize> = schema_descr
        .root_schema()
        .get_fields()
        .iter()
        .enumerate()
        .filter(|(_, f)| matches!(f.name(), "idx" | "duration" | "words"))
        .map(|(i, _)| i)
        .collect();
    let projection = ProjectionMask::roots(schema_descr, kept);

    let reader = builder.with_projection(projection).build()?;
    let mut map: HashMap<i32, TruthRow> = HashMap::new();

    for batch in reader {
        let batch = batch?;
        let idx_arr = col!(batch, "idx", Int32Array);
        let dur_arr = col!(batch, "duration", Float32Array);
        let words_arr = col!(batch, "words", ListArray);

        for row in 0..batch.num_rows() {
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

            let idx = idx_arr.value(row);
            map.insert(
                idx,
                TruthRow {
                    idx,
                    duration: dur_arr.value(row),
                    words,
                },
            );
        }
    }
    Ok(map)
}
