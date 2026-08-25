//! `delta_kernel::JsonHandler` over polars-io NDJSON. Map fields appear on
//! the wire as JSON objects but our internal encoding is
//! `List<Struct<{key, value}>>`; we let polars infer freely and reshape via
//! `align` on read.

use std::io::Cursor;
use std::num::NonZeroUsize;
use std::sync::Arc;

use delta_kernel::engine_data::EngineData;
use delta_kernel::expressions::PredicateRef;
use delta_kernel::schema::{DataType as KernelDataType, SchemaRef, StructField};
use delta_kernel::{
    DeltaResult, Error, FileDataReadResultIterator, FileMeta, JsonHandler, StorageHandler,
};
use polars::io::SerReader;
use polars::io::json::{JsonFormat, JsonReader};
use polars::prelude::as_struct as polars_as_struct;
use polars::prelude::{
    Column, DataFrame, DataType as PlDataType, Expr, Field as PlField, IntoLazy, LazyFrame,
    PolarsError, Schema as PlSchema, col, concat_list, lit,
};
use polars_utils::pl_str::PlSmallStr;
use url::Url;

use crate::consts::{MAP_KEY_FIELD, MAP_VALUE_FIELD};
use crate::engine::{PolarsEngineData, select_anchored};
use crate::errors::to_kernel_err;
use crate::translation::from_kernel::{empty_typed_list_expr, null_gated};
use crate::translation::schema::KernelDataTypeExt;

use super::storage::ObjectStoreStorageHandler;

pub(crate) struct PolarsJsonHandler {
    storage: Arc<ObjectStoreStorageHandler>,
}

impl PolarsJsonHandler {
    pub(crate) fn new(storage: Arc<ObjectStoreStorageHandler>) -> Self {
        Self { storage }
    }
}

impl JsonHandler for PolarsJsonHandler {
    fn parse_json(
        &self,
        json_strings: Box<dyn EngineData>,
        output_schema: SchemaRef,
    ) -> DeltaResult<Box<dyn EngineData>> {
        let row_count = json_strings.len();
        let lines = extract_json_strings(json_strings.as_ref())?;
        if lines.len() != row_count {
            return Err(Error::Generic(format!(
                "parse_json: expected {row_count} rows, got {} strings",
                lines.len()
            )));
        }
        let mut blob = String::with_capacity(lines.iter().map(|s| s.len() + 1).sum());
        for line in &lines {
            // One document may span lines (pretty-printed stats). JSON
            // forbids raw control characters inside string literals, so a
            // raw newline can only be inter-token whitespace — replacing it
            // keeps the value and keeps the NDJSON framing one-per-line.
            if line.contains(['\n', '\r']) {
                blob.push_str(&line.replace(['\n', '\r'], " "));
            } else {
                blob.push_str(line);
            }
            blob.push('\n');
        }
        let df = parse_ndjson_inferred(blob.as_bytes())?;
        let aligned = align_lazy(df, output_schema.as_ref())?
            .collect()
            .map_err(to_kernel_err)?;
        // Contract: N input rows produce N output rows.
        if aligned.height() != row_count {
            return Err(Error::Generic(format!(
                "parse_json: {row_count} input rows produced {} output rows",
                aligned.height()
            )));
        }
        Ok(Box::new(PolarsEngineData::new(aligned)))
    }

    fn read_json_files(
        &self,
        files: &[FileMeta],
        physical_schema: SchemaRef,
        _predicate: Option<PredicateRef>,
    ) -> DeltaResult<FileDataReadResultIterator> {
        // Eagerly fetch bytes (drops the borrow on `self`); parse lazily
        // inside the returned iterator so kernel can stream commit replay.
        let slices = files
            .iter()
            .map(|f| (f.location.clone(), None))
            .collect::<Vec<_>>();
        let payloads: Vec<bytes::Bytes> = self
            .storage
            .read_files(slices)?
            .collect::<DeltaResult<_>>()?;
        let schema: SchemaRef = physical_schema;

        let iter = payloads
            .into_iter()
            .map(move |bytes| -> DeltaResult<Box<dyn EngineData>> {
                let df = parse_ndjson_inferred(&bytes)?;
                let aligned = align_lazy(df, schema.as_ref())?
                    .collect()
                    .map_err(to_kernel_err)?;
                Ok(Box::new(PolarsEngineData::new(aligned)))
            });
        Ok(Box::new(iter))
    }

    fn write_json_file(
        &self,
        _path: &Url,
        _data: Box<
            dyn Iterator<Item = DeltaResult<delta_kernel::engine_data::FilteredEngineData>>
                + Send
                + '_,
        >,
        _overwrite: bool,
    ) -> DeltaResult<()> {
        Err(Error::Unsupported(
            "polars-deltalake is read-only; write support is not yet implemented".into(),
        ))
    }
}

pub(crate) fn parse_ndjson_inferred(bytes: &[u8]) -> DeltaResult<DataFrame> {
    // polars raises a type-inference error on a reader with no JSON value;
    // a zero-byte or blank commit file is an empty batch, not a failure.
    if bytes.iter().all(u8::is_ascii_whitespace) {
        return Ok(DataFrame::empty());
    }
    let mut df = JsonReader::new(Cursor::new(bytes))
        .with_json_format(JsonFormat::JsonLines)
        .infer_schema_len(NonZeroUsize::new(usize::MAX))
        .finish()
        .map_err(to_kernel_err)?;
    df.rechunk_mut();
    Ok(df)
}

/// Reshape an inferred polars DataFrame to the kernel-declared layout:
/// missing fields → null columns, inferred `Struct{...}` map fields →
/// `List<Struct<{key, value}>>`, struct children recurse. Returned lazy so
/// the executor's concatenated commit frames pin only the raw parses; the
/// align runs inside the terminal streaming collect.
pub(crate) fn align_lazy(
    df: DataFrame,
    kernel_schema: &delta_kernel::schema::StructType,
) -> DeltaResult<LazyFrame> {
    let polars_schema = df.schema().clone();
    let select_exprs: Vec<Expr> = kernel_schema
        .fields()
        .map(|field| {
            let inferred = polars_schema.get(field.name.as_str());
            align(col(field.name.as_str()), field, inferred, None, &field.name)
                .map(|e| e.alias(PlSmallStr::from_str(field.name.as_str())))
        })
        .collect::<DeltaResult<_>>()?;

    // The file may name none of the requested fields, or every one it does
    // name aligns to a literal (an empty inferred struct for a Map).
    Ok(select_anchored(df.lazy(), &select_exprs))
}

/// Top-level callers pass `col(name)`; nested walks pass the appropriate
/// `.struct_().field_by_name(...)` subexpression. `gate` is the enclosing
/// struct's presence (`None` at top level); `path` is the dotted field
/// path for error messages.
fn align(
    source: Expr,
    field: &StructField,
    inferred: Option<&PlDataType>,
    gate: Option<&Expr>,
    path: &str,
) -> DeltaResult<Expr> {
    if let KernelDataType::Variant(_) = &field.data_type {
        return Err(Error::Unsupported(format!(
            "json align: Variant column {} is not supported",
            field.name
        )));
    }
    let aligned = match (&field.data_type, inferred) {
        (_, None) => null_expr_for_kernel(&field.data_type)?,
        // JSON encodes Map as a JSON object → polars infers `Struct{k1,…}`;
        // reshape into our `List<Struct<{key,value}>>` representation.
        (KernelDataType::Map(_), Some(PlDataType::Struct(fs))) => map_from_struct_expr(source, fs)?,
        // A Struct may contain a Map at any depth, so recurse and rebuild.
        // The rebuild must keep the source's outer validity: `as_struct`
        // alone yields a valid struct of null children for a null row, and
        // plan aggregates (`max_non_null_by(protocol, ...)`) select rows by
        // exactly that struct-level nullity.
        (KernelDataType::Struct(struct_type), Some(PlDataType::Struct(fs))) => {
            let inferred_by_name: std::collections::HashMap<&str, &PlDataType> =
                fs.iter().map(|f| (f.name.as_str(), &f.dtype)).collect();
            let presence = source.clone().is_not_null();
            let children: Vec<Expr> = struct_type
                .fields()
                .map(|child| {
                    align(
                        source.clone().struct_().field_by_name(child.name.as_str()),
                        child,
                        inferred_by_name.get(child.name.as_str()).copied(),
                        Some(&presence),
                        &format!("{path}.{}", child.name),
                    )
                    .map(|e| e.alias(PlSmallStr::from_str(child.name.as_str())))
                })
                .collect::<DeltaResult<_>>()?;
            null_gated(presence, polars_as_struct(children))
        }
        // Primitives, Arrays, and any shape mismatch — let polars cast the
        // inferred column to the kernel-declared type.
        _ => source.cast(field.data_type.to_polars().map_err(to_kernel_err)?),
    };
    if field.nullable {
        Ok(aligned)
    } else {
        Ok(require_present(aligned, gate.cloned(), path.to_string()))
    }
}

/// ScanJson contract (kernel `plans/ir/nodes.rs`): a missing value for a
/// non-nullable field is an error, not a null-fill. Row-level — a single
/// JSON line can omit a field the rest of the file has — and gated to rows
/// whose enclosing struct is present.
fn require_present(value: Expr, gate: Option<Expr>, path: String) -> Expr {
    let gate = gate.unwrap_or_else(|| lit(true));
    value.map_many(
        move |cols: &mut [Column]| {
            let nulls = cols[0].is_null();
            let present = cols[1].bool()?;
            // Either input may be a length-1 broadcast literal; `&` handles
            // that, and `.any()` ignores nulls.
            let violated = (&nulls & present).any();
            if violated {
                Err(PolarsError::ComputeError(
                    format!("json align: non-nullable field {path} is null for a present row")
                        .into(),
                ))
            } else {
                Ok(cols[0].clone())
            }
        },
        &[gate],
        |_: &PlSchema, fields: &[PlField]| Ok(fields[0].clone()),
    )
}

fn map_from_struct_expr(value_expr: Expr, fields: &[polars::prelude::Field]) -> DeltaResult<Expr> {
    if fields.is_empty() {
        // Kernel's `get_map` reads null as "data missing" for non-nullable
        // Map fields, so empty (`partitionValues: {}`) must stay typed.
        return empty_typed_list_expr(polars_as_struct(vec![
            lit("").alias(PlSmallStr::from_static(MAP_KEY_FIELD)),
            lit("").alias(PlSmallStr::from_static(MAP_VALUE_FIELD)),
        ]));
    }
    let entries: Vec<Expr> = fields
        .iter()
        .map(|f| {
            polars_as_struct(vec![
                lit(f.name.as_str()).alias(PlSmallStr::from_static(MAP_KEY_FIELD)),
                value_expr
                    .clone()
                    .struct_()
                    .field_by_name(f.name.as_str())
                    .cast(PlDataType::String)
                    .alias(PlSmallStr::from_static(MAP_VALUE_FIELD)),
            ])
        })
        .collect();
    concat_list(entries).map_err(to_kernel_err)
}

fn null_expr_for_kernel(kernel_dt: &KernelDataType) -> DeltaResult<Expr> {
    let polars_dt = kernel_dt.to_polars().map_err(to_kernel_err)?;
    Ok(lit(polars::prelude::LiteralValue::untyped_null()).cast(polars_dt))
}

/// The JsonHandler contract fixes the input to "a single column batch of
/// string type" without naming the column, so read it positionally — the
/// reference engine does the same (`json_strings.column(0)`).
fn extract_json_strings(data: &dyn EngineData) -> DeltaResult<Vec<String>> {
    let df = data
        .any_ref()
        .downcast_ref::<PolarsEngineData>()
        .ok_or_else(|| {
            Error::Generic("parse_json received EngineData that is not PolarsEngineData".into())
        })?
        .dataframe();
    if df.width() != 1 {
        return Err(Error::Generic(format!(
            "parse_json: expected a single string column, got {} columns",
            df.width()
        )));
    }
    let strings = df.columns()[0]
        .as_materialized_series()
        .str()
        .map_err(to_kernel_err)?;
    Ok(strings
        .iter()
        .map(|s| match s {
            // polars drops blank NDJSON lines; `{}` keeps the row and
            // null-fills it, matching kernel's reference decoder.
            Some(v) if !v.trim().is_empty() => v.to_string(),
            _ => "{}".to_string(),
        })
        .collect())
}

#[cfg(test)]
mod align_nullability_tests {
    use delta_kernel::schema::StructType;

    use super::*;

    fn aligned(lines: &str, schema: &StructType) -> DeltaResult<DataFrame> {
        let df = parse_ndjson_inferred(lines.as_bytes())?;
        align_lazy(df, schema)?.collect().map_err(to_kernel_err)
    }

    /// The add/remove action shape: nullable outer struct, non-nullable
    /// `p` leaf, nullable `q` leaf.
    fn action_schema() -> StructType {
        StructType::try_new([StructField::nullable(
            "a",
            KernelDataType::Struct(Box::new(
                StructType::try_new([
                    StructField::not_null("p", KernelDataType::STRING),
                    StructField::nullable("q", KernelDataType::LONG),
                ])
                .unwrap(),
            )),
        )])
        .unwrap()
    }

    /// ScanJson contract: a missing value for a non-nullable field under a
    /// present parent is an error, not a null-fill.
    #[test]
    fn missing_non_nullable_leaf_under_present_parent_errors() {
        let out = aligned(
            "{\"a\":{\"p\":\"x\",\"q\":1}}\n{\"a\":{\"q\":2}}",
            &action_schema(),
        );
        let err = out.expect_err("row with present parent and missing p must error");
        assert!(err.to_string().contains("a.p"), "got: {err}");
    }

    #[test]
    fn absent_parent_rows_pass() {
        let out = aligned("{\"a\":{\"p\":\"x\"}}\n{\"z\":5}", &action_schema()).unwrap();
        assert_eq!(out.height(), 2);
        assert_eq!(out.column("a").unwrap().null_count(), 1);
    }

    #[test]
    fn missing_nullable_leaf_null_fills() {
        let out = aligned("{\"a\":{\"p\":\"x\"}}", &action_schema()).unwrap();
        assert_eq!(out.height(), 1);
    }

    /// An all-literal select collapses to one row in polars; one output row
    /// per input line is the ScanJson contract.
    #[test]
    fn rows_survive_when_no_requested_field_is_present() {
        let schema =
            StructType::try_new([StructField::nullable("a", KernelDataType::LONG)]).unwrap();
        let out = aligned("{\"z\":1}\n{\"z\":2}\n{\"z\":3}", &schema).unwrap();
        assert_eq!(out.height(), 3);
        assert_eq!(out.column("a").unwrap().null_count(), 3);
    }

    #[test]
    fn missing_top_level_non_nullable_column_errors() {
        let schema =
            StructType::try_new([StructField::not_null("p", KernelDataType::STRING)]).unwrap();
        let err = aligned("{\"z\":1}", &schema).expect_err("whole column missing must error");
        assert!(err.to_string().contains('p'), "got: {err}");
    }

    /// A null mid-level struct gates out its non-nullable leaves; a present
    /// but empty one does not.
    #[test]
    fn gate_is_transitive_through_nesting() {
        let schema = StructType::try_new([StructField::nullable(
            "o",
            KernelDataType::Struct(Box::new(
                StructType::try_new([StructField::nullable(
                    "m",
                    KernelDataType::Struct(Box::new(
                        StructType::try_new([StructField::not_null("p", KernelDataType::STRING)])
                            .unwrap(),
                    )),
                )])
                .unwrap(),
            )),
        )])
        .unwrap();

        aligned("{\"o\":{\"m\":{\"p\":\"x\"}}}\n{\"o\":{}}", &schema)
            .expect("null mid struct must not trip the leaf check");
        let err = aligned("{\"o\":{\"m\":{}}}", &schema)
            .expect_err("present empty mid struct must error on p");
        assert!(err.to_string().contains("o.m.p"), "got: {err}");
    }
}

#[cfg(test)]
mod json_string_tests {
    use delta_kernel::schema::StructType;

    use super::*;

    /// polars silently drops a blank NDJSON line, so a NULL or empty input
    /// has to become `{}` to keep one output row per input row.
    #[test]
    fn null_and_blank_become_empty_objects() {
        let df = polars::df!("json" => [Some("{\"a\":1}"), None, Some(""), Some("  ")]).unwrap();
        let data = PolarsEngineData::new(df);
        let out = extract_json_strings(&data).unwrap();
        assert_eq!(out, vec!["{\"a\":1}", "{}", "{}", "{}"]);
    }

    /// The JsonHandler contract fixes shape and type, not the column name —
    /// kernel's own helpers build the single-column batch under the name
    /// "a", and the reference engine reads column 0 positionally.
    #[test]
    fn any_single_column_name_is_accepted() {
        let df = polars::df!("a" => [Some("{\"p\":1}")]).unwrap();
        let out = extract_json_strings(&PolarsEngineData::new(df)).unwrap();
        assert_eq!(out, vec!["{\"p\":1}"]);
    }

    /// One legal JSON document may span lines (pretty-printed stats from a
    /// foreign writer); the reference engine parses it per-document, so the
    /// NDJSON re-parse must not split it.
    #[test]
    fn pretty_printed_document_parses() {
        let url = Url::parse("file:///").unwrap();
        let rt = crate::engine::rt();
        let storage =
            Arc::new(ObjectStoreStorageHandler::new(&url, std::iter::empty(), rt).unwrap());
        let handler = PolarsJsonHandler::new(storage);

        let schema = Arc::new(
            StructType::try_new([StructField::nullable(
                "a",
                KernelDataType::LONG,
            )])
            .unwrap(),
        );
        let df = polars::df!("json" => [Some("{\n  \"a\": 3\n}")]).unwrap();
        let out = handler
            .parse_json(Box::new(PolarsEngineData::new(df)), schema)
            .unwrap();
        let df = out
            .any_ref()
            .downcast_ref::<PolarsEngineData>()
            .unwrap()
            .dataframe()
            .clone();
        assert_eq!(df.height(), 1);
        assert_eq!(
            df.column("a").unwrap().i64().unwrap().get(0),
            Some(3),
            "pretty-printed document must parse as one row"
        );
    }

    /// A reader holding no JSON value is an empty batch, not a type-inference
    /// failure — a crashed writer can leave a zero-byte commit.
    #[test]
    fn blank_input_parses_to_an_empty_frame() {
        for bytes in [&b""[..], b"\n", b"   \n  "] {
            assert_eq!(parse_ndjson_inferred(bytes).unwrap().height(), 0);
        }
    }
}
