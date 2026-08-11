//! `delta_kernel::JsonHandler` over polars-io NDJSON. Map fields appear on
//! the wire as JSON objects but our internal encoding is
//! `List<Struct<{key, value}>>`; we let polars infer freely and reshape via
//! `align` on read.

use std::io::Cursor;
use std::num::NonZeroUsize;
use std::sync::Arc;

use delta_kernel::engine_data::EngineData;
use delta_kernel::expressions::{ColumnName, PredicateRef};
use delta_kernel::schema::{DataType as KernelDataType, SchemaRef, StructField};
use delta_kernel::{
    DeltaResult, Error, FileDataReadResultIterator, FileMeta, JsonHandler, StorageHandler,
};
use polars::io::SerReader;
use polars::io::json::{JsonFormat, JsonReader};
use polars::prelude::as_struct as polars_as_struct;
use polars::prelude::{
    DataFrame, DataType as PlDataType, Expr, IntoLazy, col, concat_list, lit, when,
};
use polars_utils::pl_str::PlSmallStr;
use url::Url;

use crate::consts::{MAP_KEY_FIELD, MAP_VALUE_FIELD};
use crate::engine::PolarsEngineData;
use crate::errors::to_kernel_err;
use crate::translation::from_kernel::empty_typed_list_expr;
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
            blob.push_str(line);
            blob.push('\n');
        }
        let df = parse_ndjson_inferred(blob.as_bytes())?;
        let aligned = align_dataframe(df, output_schema.as_ref())?;
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
                let aligned = align_dataframe(df, schema.as_ref())?;
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
/// `List<Struct<{key, value}>>`, struct children recurse.
pub(crate) fn align_dataframe(
    df: DataFrame,
    kernel_schema: &delta_kernel::schema::StructType,
) -> DeltaResult<DataFrame> {
    let polars_schema = df.schema().clone();
    let select_exprs: Vec<Expr> = kernel_schema
        .fields()
        .map(|field| {
            let inferred = polars_schema.get(field.name.as_str());
            align(col(field.name.as_str()), field, inferred)
                .map(|e| e.alias(PlSmallStr::from_str(field.name.as_str())))
        })
        .collect::<DeltaResult<_>>()?;

    df.lazy()
        .select(select_exprs)
        .collect()
        .map_err(to_kernel_err)
}

/// Top-level callers pass `col(name)`; nested walks pass the appropriate
/// `.struct_().field_by_name(...)` subexpression.
fn align(source: Expr, field: &StructField, inferred: Option<&PlDataType>) -> DeltaResult<Expr> {
    if let KernelDataType::Variant(_) = &field.data_type {
        return Err(Error::Unsupported(format!(
            "json align: Variant column {} is not supported",
            field.name
        )));
    }
    let Some(inferred) = inferred else {
        return null_expr_for_kernel(&field.data_type);
    };
    match (&field.data_type, inferred) {
        // JSON encodes Map as a JSON object → polars infers `Struct{k1,…}`;
        // reshape into our `List<Struct<{key,value}>>` representation.
        (KernelDataType::Map(_), PlDataType::Struct(fs)) => map_from_struct_expr(source, fs),
        // A Struct may contain a Map at any depth, so recurse and rebuild.
        // The rebuild must keep the source's outer validity: `as_struct`
        // alone yields a valid struct of null children for a null row, and
        // plan aggregates (`max_non_null_by(protocol, ...)`) select rows by
        // exactly that struct-level nullity.
        (KernelDataType::Struct(struct_type), PlDataType::Struct(fs)) => {
            let inferred_by_name: std::collections::HashMap<&str, &PlDataType> =
                fs.iter().map(|f| (f.name.as_str(), &f.dtype)).collect();
            let children: Vec<Expr> = struct_type
                .fields()
                .map(|child| {
                    align(
                        source.clone().struct_().field_by_name(child.name.as_str()),
                        child,
                        inferred_by_name.get(child.name.as_str()).copied(),
                    )
                    .map(|e| e.alias(PlSmallStr::from_str(child.name.as_str())))
                })
                .collect::<DeltaResult<_>>()?;
            Ok(when(source.clone().is_not_null())
                .then(polars_as_struct(children))
                .otherwise(lit(polars::prelude::LiteralValue::untyped_null())))
        }
        // Primitives, Arrays, and any shape mismatch — let polars cast the
        // inferred column to the kernel-declared type.
        _ => Ok(source.cast(field.data_type.to_polars().map_err(to_kernel_err)?)),
    }
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

fn extract_json_strings(data: &dyn EngineData) -> DeltaResult<Vec<String>> {
    use delta_kernel::engine_data::{GetData, RowVisitor};
    use delta_kernel::schema::{DataType, PrimitiveType};

    struct Collector {
        out: Vec<String>,
    }

    impl RowVisitor for Collector {
        fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
            static NAMES: std::sync::OnceLock<Vec<ColumnName>> = std::sync::OnceLock::new();
            static TYPES: std::sync::OnceLock<Vec<DataType>> = std::sync::OnceLock::new();
            let names = NAMES.get_or_init(|| vec![ColumnName::new(["json"])]);
            let types = TYPES.get_or_init(|| vec![DataType::Primitive(PrimitiveType::String)]);
            (names.as_slice(), types.as_slice())
        }
        fn visit<'a>(
            &mut self,
            row_count: usize,
            getters: &[&'a dyn GetData<'a>],
        ) -> DeltaResult<()> {
            let getter = getters[0];
            for i in 0..row_count {
                let s: Option<&str> = getter.get_str(i, "json")?;
                self.out.push(s.unwrap_or_default().to_string());
            }
            Ok(())
        }
    }

    let mut collector = Collector {
        out: Vec::with_capacity(data.len()),
    };
    let names = [ColumnName::new(["json"])];
    data.visit_rows(&names, &mut collector)?;
    Ok(collector.out)
}
