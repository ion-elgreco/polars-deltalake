//! `delta_kernel::JsonHandler` over polars-io NDJSON. Map fields appear on
//! the wire as JSON objects but our internal encoding is
//! `List<Struct<{key, value}>>`; we let polars infer freely and reshape via
//! `align` on read.

use std::io::Cursor;
use std::num::NonZeroUsize;
use std::sync::Arc;

use delta_kernel::engine_data::EngineData;
use delta_kernel::expressions::PredicateRef;
use delta_kernel::schema::{DataType as KernelDataType, MapType, SchemaRef, StructField};
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
use crate::translation::from_kernel::{as_struct_checked, empty_typed_list_expr, null_gated};
use crate::translation::schema::{KernelDataTypeExt, KernelSchemaExt};

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

/// One or more commit files parsed as a single NDJSON document, in file
/// order. `rows_per_file` recovers per-file columns after the parse.
pub(crate) struct ParsedLog {
    pub(crate) df: DataFrame,
    pub(crate) rows_per_file: Vec<usize>,
}

/// Parse commit files as one NDJSON document, in order. Each file's row
/// count is its number of non-blank lines, which polars parses one row
/// each.
pub(crate) fn parse_commit_files(payloads: &[bytes::Bytes]) -> DeltaResult<ParsedLog> {
    let mut joined = Vec::with_capacity(payloads.iter().map(|b| b.len() + 1).sum());
    let mut rows_per_file = Vec::with_capacity(payloads.len());
    for bytes in payloads {
        rows_per_file.push(
            bytes
                .split(|&b| b == b'\n')
                .filter(|line| !line.iter().all(u8::is_ascii_whitespace))
                .count(),
        );
        joined.extend_from_slice(bytes);
        joined.push(b'\n');
    }
    let df = parse_ndjson_inferred(&joined)?;
    let expected: usize = rows_per_file.iter().sum();
    if df.height() != expected {
        return Err(Error::Generic(format!(
            "commit files parsed to {} rows but hold {expected} lines",
            df.height()
        )));
    }
    Ok(ParsedLog { df, rows_per_file })
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
    // A zero-byte commit parses to a 0-row frame, where every `align` expr is
    // a broadcast literal of length 1 — `require_present` would then read a
    // missing non-nullable column as a violated row that does not exist, and
    // the select would trip a shape error. Nothing to align over.
    if df.height() == 0 {
        return Ok(kernel_schema.empty_frame().map_err(to_kernel_err)?.lazy());
    }
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
        (KernelDataType::Map(m), Some(PlDataType::Struct(fs))) => {
            map_from_struct_expr(source, fs, m)?
        }
        // A Struct may contain a Map at any depth, so recurse and rebuild.
        // The rebuild must keep the source's outer validity: `as_struct`
        // alone yields a valid struct of null children for a null row, and
        // plan aggregates (`max_non_null_by(protocol, ...)`) select rows by
        // exactly that struct-level nullity.
        (KernelDataType::Struct(struct_type), Some(PlDataType::Struct(fs))) => {
            let inferred_by_name: std::collections::HashMap<&str, &PlDataType> =
                fs.iter().map(|f| (f.name.as_str(), &f.dtype)).collect();
            let presence = source.clone().is_not_null();
            // Composed with the enclosing gate, not replacing it: a child
            // array under a NULL ancestor may still hold values, so its own
            // validity alone would re-admit a row the ancestor gated out.
            let child_gate = match gate {
                Some(outer) => outer.clone().and(presence.clone()),
                None => presence.clone(),
            };
            let children: Vec<Expr> = struct_type
                .fields()
                .map(|child| {
                    align(
                        source.clone().struct_().field_by_name(child.name.as_str()),
                        child,
                        inferred_by_name.get(child.name.as_str()).copied(),
                        Some(&child_gate),
                        &format!("{path}.{}", child.name),
                    )
                    .map(|e| e.alias(PlSmallStr::from_str(child.name.as_str())))
                })
                .collect::<DeltaResult<_>>()?;
            null_gated(presence, as_struct_checked(children, "json align")?)
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

fn map_from_struct_expr(
    value_expr: Expr,
    fields: &[polars::prelude::Field],
    map_type: &MapType,
) -> DeltaResult<Expr> {
    // The declared entry types, not String/String: `to_polars` promises
    // `List<Struct<key,value>>` in exactly these types, and a Map whose
    // value type is not String must not read back stringified.
    let key_dt = map_type.key_type().to_polars().map_err(to_kernel_err)?;
    let value_dt = map_type.value_type().to_polars().map_err(to_kernel_err)?;
    if fields.is_empty() {
        // polars infers `Struct{}` when every occurrence of the key is `{}`.
        // Kernel's `get_map` reads null as "data missing" for non-nullable
        // Map fields, so an empty map must stay typed and non-null — but a
        // row where the key is absent has to stay NULL, so this needs the
        // same outer gate as the branch below.
        let empty = empty_typed_list_expr(polars_as_struct(vec![
            lit(polars::prelude::LiteralValue::untyped_null())
                .cast(key_dt)
                .alias(PlSmallStr::from_static(MAP_KEY_FIELD)),
            lit(polars::prelude::LiteralValue::untyped_null())
                .cast(value_dt)
                .alias(PlSmallStr::from_static(MAP_VALUE_FIELD)),
        ]))?;
        return Ok(null_gated(value_expr.is_not_null(), empty));
    }
    let entries: Vec<Expr> = fields
        .iter()
        .map(|f| {
            polars_as_struct(vec![
                lit(f.name.as_str())
                    .cast(key_dt.clone())
                    .alias(PlSmallStr::from_static(MAP_KEY_FIELD)),
                value_expr
                    .clone()
                    .struct_()
                    .field_by_name(f.name.as_str())
                    .cast(value_dt.clone())
                    .alias(PlSmallStr::from_static(MAP_VALUE_FIELD)),
            ])
        })
        .collect();
    // `concat_list` over literal keys is non-null even where the source
    // object is absent, which would turn a missing map into a map of nulls
    // and slip past the non-nullable guard. Same gate as the Struct arm.
    Ok(null_gated(
        value_expr.is_not_null(),
        concat_list(entries).map_err(to_kernel_err)?,
    ))
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

    /// The presence gates in `align` (and in the parquet arm's
    /// `non_nullable_guards`) compose every ancestor's validity rather than
    /// trusting the immediate parent's alone. That is redundant only while
    /// polars keeps a null struct row's children null — `set_outer_validity`
    /// calls `propagate_nulls_mut` today. Pin it: if it ever stops holding,
    /// the composed gate is what keeps a gated-out row from being checked.
    #[test]
    fn polars_propagates_outer_struct_nulls_to_children() {
        use polars::prelude::{IntoColumn, IntoSeries, NamedFrom, Series, StructChunked, col};

        let leaf = Series::new("p".into(), [Some("x"), None::<&str>]);
        let mid = StructChunked::from_series("m".into(), 2, [leaf].iter())
            .unwrap()
            .into_series();
        let outer = StructChunked::from_series("o".into(), 2, [mid].iter())
            .unwrap()
            .with_outer_validity(Some(polars_arrow::bitmap::Bitmap::from([true, false])));
        let df = DataFrame::new(2, vec![outer.into_series().into_column()]).unwrap();

        let child = df.column("o").unwrap().struct_().unwrap();
        assert_eq!(
            child.field_by_name("m").unwrap().null_count(),
            1,
            "an outer-null struct row must carry a null child"
        );
        let gate = df
            .lazy()
            .select([col("o")
                .struct_()
                .field_by_name("m")
                .is_not_null()
                .alias("g")])
            .collect()
            .unwrap();
        assert_eq!(
            gate.column("g").unwrap().bool().unwrap().get(1),
            Some(false),
            "the parent gate must already read false under a null ancestor"
        );
    }

    /// A Map column is rebuilt from literal keys, which produces a non-null
    /// list even where the source object is absent. Without the outer gate
    /// a missing non-nullable map reads back as a map of nulls and slips
    /// past the presence check instead of aborting.
    #[test]
    fn absent_non_nullable_map_errors() {
        use delta_kernel::schema::MapType;

        let schema = StructType::try_new([StructField::not_null(
            "pv",
            KernelDataType::Map(Box::new(MapType::new(
                KernelDataType::STRING,
                KernelDataType::STRING,
                true,
            ))),
        )])
        .unwrap();
        let err = aligned("{\"pv\":{\"p\":\"x\"}}\n{\"z\":1}", &schema)
            .expect_err("a missing non-nullable map must error, not null-fill");
        assert!(err.to_string().contains("pv"), "got: {err}");
    }

    /// polars infers `Struct{}` when every occurrence of the key is `{}`,
    /// which takes the empty-map branch. That branch names no column, so
    /// without the same outer gate its non-null empty list is broadcast over
    /// the rows where the map is absent and the presence check never fires.
    #[test]
    fn absent_non_nullable_map_errors_when_every_present_map_is_empty() {
        use delta_kernel::schema::MapType;

        let schema = StructType::try_new([StructField::not_null(
            "pv",
            KernelDataType::Map(Box::new(MapType::new(
                KernelDataType::STRING,
                KernelDataType::STRING,
                true,
            ))),
        )])
        .unwrap();
        let err = aligned("{\"pv\":{}}\n{\"z\":1}", &schema)
            .expect_err("a missing non-nullable map must error, not read as empty");
        assert!(err.to_string().contains("pv"), "got: {err}");
    }

    /// The declared Map entry types must survive alignment: this arm
    /// previously seeded and cast every entry to String whatever the
    /// schema said, so a `Map<String, Long>` read back String-valued.
    #[test]
    fn map_alignment_keeps_the_declared_entry_types() {
        use delta_kernel::schema::MapType;

        let map_dt = || {
            KernelDataType::Map(Box::new(MapType::new(
                KernelDataType::STRING,
                KernelDataType::LONG,
                true,
            )))
        };
        let schema = StructType::try_new([StructField::nullable("m", map_dt())]).unwrap();
        let expected = map_dt().to_polars().unwrap();

        // Non-empty objects: the per-key rebuild casts values to the
        // declared type.
        let out = aligned("{\"m\":{\"a\":1,\"b\":2}}", &schema).unwrap();
        let col = out.column("m").unwrap().as_materialized_series().clone();
        assert_eq!(col.dtype(), &expected);
        let entries = col.list().unwrap().get_as_series(0).unwrap();
        let values = entries
            .struct_()
            .unwrap()
            .field_by_name(MAP_VALUE_FIELD)
            .unwrap();
        assert_eq!(values.i64().unwrap().get(0), Some(1));

        // All-empty objects infer `Struct{}`; the empty branch must seed
        // the declared types too, not String/String.
        let out = aligned("{\"m\":{}}", &schema).unwrap();
        assert_eq!(
            out.column("m").unwrap().as_materialized_series().dtype(),
            &expected
        );
    }

    /// A zero-byte commit parses to a 0-row frame, where every align expr is
    /// a length-1 broadcast literal — the presence check would read the
    /// missing column as a violated row that does not exist.
    #[test]
    fn zero_row_input_aligns_without_a_presence_error() {
        let schema =
            StructType::try_new([StructField::not_null("p", KernelDataType::STRING)]).unwrap();
        let out = aligned("", &schema).expect("an empty commit is an empty batch");
        assert_eq!(out.height(), 0);
        assert_eq!(out.get_column_names(), ["p"]);
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
        // A null grandparent gates the leaf too, so the leaf's check must
        // carry every ancestor's presence, not just its parent's.
        aligned("{\"o\":{\"m\":{\"p\":\"x\"}}}\n{\"z\":1}", &schema)
            .expect("null outer struct must not trip the leaf check");
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
            StructType::try_new([StructField::nullable("a", KernelDataType::LONG)]).unwrap(),
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
