//! Declarative metadata scan plan → resolved file list + per-file rewrites.
//!
//! `Scan::declarative_metadata_scan_plan` hands back a kernel plan whose
//! rows are the scan's live `add` actions — log replay, stats skipping, and
//! kernel-predicate partition pruning already applied. The plan runs on the
//! `PolarsPlanExecutor`; each output row becomes one [`ScanFileMeta`]: the
//! data-file path, the physical→logical select list (column-mapping renames
//! + typed partition literals), and the materialized deletion vector.

use std::collections::HashMap;
use std::str::FromStr;
use std::sync::OnceLock;

use delta_kernel::actions::deletion_vector::{DeletionVectorDescriptor, DeletionVectorStorageType};
use delta_kernel::engine_data::{GetData, RowVisitor};
use delta_kernel::expressions::ColumnName;
use delta_kernel::plans::Operation;
use delta_kernel::scan::Scan;
use delta_kernel::schema::{DataType as KernelDataType, MapType, StructField, StructType};
use delta_kernel::table_features::ColumnMappingMode;
use delta_kernel::{DeltaResult, Engine};
use polars::prelude::as_struct as polars_as_struct;
use polars::prelude::{DataFrame, Expr, LiteralValue, col, lit, when};
use polars_utils::pl_path::PlRefPath;
use polars_utils::pl_str::PlSmallStr;

use crate::consts::{MAP_KEY_FIELD, MAP_VALUE_FIELD};
use crate::engine::{PolarsEngine, PolarsEngineData, path_for_polars_io, resolve_series_path};
use crate::scan::predicate::renames_nested_fields;
use crate::translation::from_kernel::series_value_lit;
use crate::translation::schema::KernelDataTypeExt;

/// Per-file work to apply post-read: physical→logical select + DV keep-mask.
pub(crate) struct LogicalRewrite {
    /// Select list producing the logical frame from the physical read:
    /// `col(physical).alias(logical)` renames plus typed partition-value
    /// literals, in logical-schema order. `None` when the physical frame is
    /// already logical (non-partitioned, non-column-mapped).
    pub(crate) select: Option<Vec<Expr>>,
    /// Per-file DV state — sorted deleted row indices + cursor of how many
    /// rows of the file have been consumed by prior batches. `None` if the
    /// file has no DV.
    pub(crate) dv: Option<DvState>,
}

pub(crate) struct DvState {
    /// Sorted ascending row indices to drop. Consumed entries are drained
    /// off the front as batches are processed.
    pub(crate) deleted: Vec<u64>,
    /// Absolute row offset within the file already covered by past batches.
    pub(crate) cursor: u64,
}

pub(crate) struct ScanFileMeta {
    pub(crate) path: PlRefPath,
    pub(crate) rewrite: LogicalRewrite,
    pub(crate) partition_values: HashMap<String, String>,
}

/// The metadata plan drained into bulk-read inputs.
pub(crate) struct ResolvedScan {
    pub(crate) files: Vec<ScanFileMeta>,
    /// `FILE_ID_COL` value (= `PlRefPath::as_str()`) → index in `files`.
    pub(crate) path_index: HashMap<String, usize>,
}

/// One `add` row pulled out of a plan output batch via the row visitor.
struct AddRow {
    path: String,
    dv: Option<DeletionVectorDescriptor>,
    partition_values: HashMap<String, String>,
}

/// A projected logical field's physical source: the next physical column
/// (rename) or a per-file partition literal (looked up in
/// `add.partitionValues_parsed` by physical partition name).
enum FieldSource {
    Data { physical: String },
    Partition { physical: String },
}

pub(crate) fn resolve_scan(scan: &Scan, engine: &PolarsEngine) -> anyhow::Result<ResolvedScan> {
    let executor = engine
        .plan_executor()
        .expect("PolarsEngine always provides a plan executor");

    let plan = scan
        .declarative_metadata_scan_plan(engine as &dyn Engine)
        .map_err(|e| anyhow::anyhow!("declarative_metadata_scan_plan failed: {e:#}"))?;
    let Some(plan) = plan else {
        return Ok(ResolvedScan {
            files: Vec::new(),
            path_index: HashMap::new(),
        });
    };

    let batches = executor
        .execute_op(Operation::QueryPlan(plan))
        .and_then(|r| r.into_data())
        .map_err(|e| anyhow::anyhow!("metadata plan execution failed: {e:#}"))?;

    let table_root = scan.table_root().clone();
    // Protocol-aware effective mode: matches how kernel resolved the physical
    // schema, including stale `physicalName` annotations under mode `none`.
    let mode = scan.snapshot().table_configuration().column_mapping_mode();
    let sources = field_sources(scan.logical_schema(), scan.physical_schema(), mode);
    // Identity frames need no per-file select at all. Nested renames count:
    // a top-level name can survive column mapping while a child does not.
    let needs_select = sources.iter().any(|(f, s)| match s {
        FieldSource::Partition { .. } => true,
        FieldSource::Data { physical } => {
            physical != f.name.as_str() || renames_nested_fields(&f.data_type, mode)
        }
    });

    let storage = engine.storage_handler();
    let mut files: Vec<ScanFileMeta> = Vec::new();
    let mut path_index: HashMap<String, usize> = HashMap::new();

    for batch in batches {
        let batch = batch.map_err(|e| anyhow::anyhow!("metadata plan batch failed: {e:#}"))?;
        let rows = visit_add_rows(batch.as_ref())?;
        let polars_batch = batch
            .any_ref()
            .downcast_ref::<PolarsEngineData>()
            .ok_or_else(|| anyhow::anyhow!("metadata plan returned non-PolarsEngineData"))?;
        let partition_lits = partition_literals(polars_batch.dataframe(), &sources, rows.len())?;

        for (row, lits) in rows.into_iter().zip(partition_lits) {
            let abs = table_root
                .join(&row.path)
                .map_err(|e| anyhow::anyhow!("failed to resolve add path {}: {e}", row.path))?;
            let pl_path = path_for_polars_io(&abs)?;

            // `Vec<u64>` of deleted row indices is far smaller than a
            // `Vec<bool>` keep-mask for sparse deletes.
            let dv = row
                .dv
                .map(|descriptor| -> anyhow::Result<DvState> {
                    let deleted = descriptor
                        .row_indexes(storage.clone(), &table_root)
                        .map_err(|e| anyhow::anyhow!("deletion vector read failed: {e:#}"))?;
                    Ok(DvState { deleted, cursor: 0 })
                })
                .transpose()?;

            let select = needs_select.then(|| build_select(&sources, lits, mode));

            let idx = files.len();
            path_index.insert(pl_path.as_str().to_string(), idx);
            files.push(ScanFileMeta {
                path: pl_path,
                rewrite: LogicalRewrite { select, dv },
                partition_values: row.partition_values,
            });
        }
    }

    Ok(ResolvedScan { files, path_index })
}

/// Pair each projected logical field with its physical source. Partition
/// columns are exactly the logical fields whose physical name is absent
/// from the physical (file) schema.
///
/// This name-diff re-derives the partition/rename subset of kernel's
/// five-variant `FieldTransformSpec`, which (with `Scan::state_info`) is
/// private at the pinned rev. A future logical-only field that is not a
/// partition column — row tracking, CDF metadata — would be misclassified
/// as one here and fail downstream with a partition-shaped error.
fn field_sources<'a>(
    logical: &'a StructType,
    physical: &StructType,
    mode: ColumnMappingMode,
) -> Vec<(&'a StructField, FieldSource)> {
    let physical_names: std::collections::HashSet<&str> =
        physical.fields().map(|f| f.name.as_str()).collect();
    logical
        .fields()
        .map(|f| {
            let phys = f.physical_name(mode).to_string();
            let source = if physical_names.contains(phys.as_str()) {
                FieldSource::Data { physical: phys }
            } else {
                FieldSource::Partition { physical: phys }
            };
            (f, source)
        })
        .collect()
}

/// Per-row typed partition literals, one `Vec<Expr>` per add row, aligned
/// with `sources` (empty per-row vec when the table is unpartitioned). The
/// values come from `add.partitionValues_parsed`, which the metadata plan
/// populates via `MapToStruct` with the physical partition schema.
fn partition_literals(
    df: &DataFrame,
    sources: &[(&StructField, FieldSource)],
    row_count: usize,
) -> anyhow::Result<Vec<Vec<Expr>>> {
    let partition_fields: Vec<(&StructField, &str)> = sources
        .iter()
        .filter_map(|(f, s)| match s {
            FieldSource::Partition { physical } => Some((*f, physical.as_str())),
            FieldSource::Data { .. } => None,
        })
        .collect();
    if partition_fields.is_empty() {
        return Ok(vec![Vec::new(); row_count]);
    }

    // Cast per field, not per row: the per-row literals then carry the
    // output dtype already and stay plain literals.
    let series_per_field = partition_fields
        .iter()
        .map(|(f, phys)| {
            let path = ColumnName::new(["add", "partitionValues_parsed", phys]);
            let series = resolve_series_path(df, &path).map_err(|e| {
                anyhow::anyhow!(
                    "partition column '{}' missing from parsed partition values: {e:#}",
                    f.name
                )
            })?;
            Ok(series.cast(&f.data_type.to_polars()?)?)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    (0..row_count)
        .map(|row| {
            partition_fields
                .iter()
                .zip(series_per_field.iter())
                .map(|((field, _), series)| -> anyhow::Result<Expr> {
                    series_value_lit(series, row, field.name.as_str())
                        .map_err(|e| anyhow::anyhow!("partition value read: {e:#}"))
                })
                .collect()
        })
        .collect()
}

/// Logical-order select list: partition literals in place, physical→logical
/// renames for data columns.
/// Rename nested struct fields to their logical names; `None` if nothing
/// below `dtype` is renamed.
///
/// Polars names struct fields from the data, so `alias` reaches only the top
/// level. Kernel's engine has no plan node for this either — its evaluator's
/// `apply_schema` renames every level as a side effect.
fn logical_names(expr: Expr, dtype: &KernelDataType, mode: ColumnMappingMode) -> Option<Expr> {
    if !renames_nested_fields(dtype, mode) {
        return None;
    }
    let renamed = match dtype {
        KernelDataType::Struct(fields) => {
            let children: Vec<Expr> = fields
                .fields()
                .map(|f| {
                    let child = expr.clone().struct_().field_by_name(f.physical_name(mode));
                    logical_names(child.clone(), &f.data_type, mode)
                        .unwrap_or(child)
                        .alias(PlSmallStr::from_str(f.name.as_str()))
                })
                .collect();
            // `as_struct` alone makes every row valid; downstream filters
            // select on this column's own nulls.
            when(expr.clone().is_not_null())
                .then(polars_as_struct(children))
                .otherwise(lit(LiteralValue::untyped_null()))
        }
        // Elements, keys and values are anonymous; only structs inside them
        // have names.
        KernelDataType::Array(array) => {
            let element = logical_names(Expr::Element, array.element_type(), mode)?;
            expr.list().eval(element)
        }
        KernelDataType::Map(map) => {
            let entry = |name: &'static str, dtype: &KernelDataType| {
                let field = Expr::Element.struct_().field_by_name(name);
                logical_names(field.clone(), dtype, mode)
                    .unwrap_or(field)
                    .alias(PlSmallStr::from_static(name))
            };
            expr.list().eval(polars_as_struct(vec![
                entry(MAP_KEY_FIELD, map.key_type()),
                entry(MAP_VALUE_FIELD, map.value_type()),
            ]))
        }
        _ => return None,
    };
    Some(renamed)
}

fn build_select(
    sources: &[(&StructField, FieldSource)],
    lits: Vec<Expr>,
    mode: ColumnMappingMode,
) -> Vec<Expr> {
    let mut lit_iter = lits.into_iter();
    sources
        .iter()
        .map(|(field, source)| match source {
            FieldSource::Partition { .. } => lit_iter
                .next()
                .expect("one literal per partition field by construction"),
            FieldSource::Data { physical } => {
                let read = col(PlSmallStr::from_str(physical.as_str()));
                logical_names(read.clone(), &field.data_type, mode)
                    .unwrap_or(read)
                    .alias(PlSmallStr::from_str(field.name.as_str()))
            }
        })
        .collect()
}

/// Extract (path, DV descriptor, partition-values map) per add row through
/// the kernel row-visitor machinery — the same getters kernel's own log
/// replay uses.
fn visit_add_rows(batch: &dyn delta_kernel::EngineData) -> anyhow::Result<Vec<AddRow>> {
    struct Visitor {
        rows: Vec<AddRow>,
    }

    fn names_and_types() -> &'static (Vec<ColumnName>, Vec<KernelDataType>) {
        static CELL: OnceLock<(Vec<ColumnName>, Vec<KernelDataType>)> = OnceLock::new();
        CELL.get_or_init(|| {
            (
                vec![
                    ColumnName::new(["add", "path"]),
                    ColumnName::new(["add", "partitionValues"]),
                    ColumnName::new(["add", "deletionVector", "storageType"]),
                    ColumnName::new(["add", "deletionVector", "pathOrInlineDv"]),
                    ColumnName::new(["add", "deletionVector", "offset"]),
                    ColumnName::new(["add", "deletionVector", "sizeInBytes"]),
                    ColumnName::new(["add", "deletionVector", "cardinality"]),
                ],
                vec![
                    KernelDataType::STRING,
                    MapType::new(KernelDataType::STRING, KernelDataType::STRING, true).into(),
                    KernelDataType::STRING,
                    KernelDataType::STRING,
                    KernelDataType::INTEGER,
                    KernelDataType::INTEGER,
                    KernelDataType::LONG,
                ],
            )
        })
    }

    impl RowVisitor for Visitor {
        fn selected_column_names_and_types(
            &self,
        ) -> (&'static [ColumnName], &'static [KernelDataType]) {
            let (names, types) = names_and_types();
            (names.as_slice(), types.as_slice())
        }

        fn visit<'a>(
            &mut self,
            row_count: usize,
            getters: &[&'a dyn GetData<'a>],
        ) -> DeltaResult<()> {
            for i in 0..row_count {
                let path: Option<&str> = getters[0].get_str(i, "add.path")?;
                let Some(path) = path else {
                    return Err(delta_kernel::Error::Generic(
                        "metadata plan emitted a null add.path".into(),
                    ));
                };
                let partition_values = getters[1]
                    .get_map(i, "add.partitionValues")?
                    .map(|m| m.materialize())
                    .unwrap_or_default();

                let storage_type: Option<&str> = getters[2].get_str(i, "storageType")?;
                let dv = match storage_type {
                    None => None,
                    Some(st) => {
                        let path_or_inline: &str =
                            getters[3].get_str(i, "pathOrInlineDv")?.ok_or_else(|| {
                                delta_kernel::Error::Generic(
                                    "deletionVector.pathOrInlineDv is null".into(),
                                )
                            })?;
                        let offset: Option<i32> = getters[4].get_int(i, "offset")?;
                        let size_in_bytes: i32 =
                            getters[5].get_int(i, "sizeInBytes")?.ok_or_else(|| {
                                delta_kernel::Error::Generic(
                                    "deletionVector.sizeInBytes is null".into(),
                                )
                            })?;
                        let cardinality: i64 =
                            getters[6].get_long(i, "cardinality")?.ok_or_else(|| {
                                delta_kernel::Error::Generic(
                                    "deletionVector.cardinality is null".into(),
                                )
                            })?;
                        Some(DeletionVectorDescriptor::try_new(
                            DeletionVectorStorageType::from_str(st)?,
                            path_or_inline,
                            offset,
                            size_in_bytes,
                            cardinality,
                        )?)
                    }
                };

                self.rows.push(AddRow {
                    path: path.to_string(),
                    dv,
                    partition_values,
                });
            }
            Ok(())
        }
    }

    let mut visitor = Visitor {
        rows: Vec::with_capacity(batch.len()),
    };
    visitor
        .visit_rows_of(batch)
        .map_err(|e| anyhow::anyhow!("add-row visit failed: {e:#}"))?;
    Ok(visitor.rows)
}
