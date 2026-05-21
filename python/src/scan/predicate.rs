//! Predicate plumbing: extract polars `Expr` from Python, split AND chains,
//! detect column mapping, and rewrite logical → physical column names.

use std::collections::{BTreeSet, HashMap, HashSet};

use delta_kernel::schema::{MetadataValue, StructField, StructType};
use delta_kernel::table_features::ColumnMappingMode;
use delta_kernel::table_properties::TableProperties;
use polars::prelude::{Column, DataFrame, Expr, IntoLazy};
use polars_plan::dsl::Operator;
use polars_utils::pl_str::PlSmallStr;
use pyo3::prelude::*;

use crate::scan::plan::ScanFileMeta;
use crate::translation::schema::KernelDataTypeExt;

/// A single conjunct of the user predicate, with its kernel-translatability
/// cached so `build_iter` doesn't re-translate every scan.
pub(crate) struct Conjunct {
    pub(crate) expr: Expr,
    pub(crate) kernel_translatable: bool,
}

/// Per-conjunct routing. Buckets are **non-disjoint** — a conjunct can
/// appear in multiple buckets when several layers evaluate it. E.g. a
/// translatable data conjunct is in both `kernel` (file-level stats) and
/// `parquet_filter` (row-group + row).
#[derive(Default)]
pub(crate) struct ConjunctClassification {
    /// Kernel `with_predicate` — file-level stats-based skipping. Every
    /// kernel-translatable conjunct, regardless of which columns it touches.
    pub(crate) kernel: Vec<Expr>,
    /// Pushed to the parquet reader via `.filter` above `scan_parquet` —
    /// row-group skipping + row-level filter. Rewritten to physical names
    /// for column-mapped tables.
    pub(crate) parquet_filter: Vec<Expr>,
    /// Option-2 file pruning via polars eval on partition values.
    pub(crate) partition_prune: Vec<Expr>,
    /// Applied inside `LogicalScanIter` after `transform_to_logical`
    /// materializes partition columns.
    pub(crate) post_transform: Vec<Expr>,
}

/// Route each conjunct to the layer(s) that will evaluate it.
pub(crate) fn classify_conjuncts(
    conjuncts: &[Conjunct],
    column_mapped: bool,
    logical_schema: &StructType,
    physical_schema: &StructType,
) -> ConjunctClassification {
    let mut out = ConjunctClassification::default();
    for c in conjuncts {
        if c.kernel_translatable {
            out.kernel.push(c.expr.clone());
        }
        let partition_only = touches_partition_only(&c.expr, logical_schema, physical_schema);
        let for_parquet = if column_mapped {
            rewrite_predicate_to_physical(&c.expr, logical_schema, physical_schema)
        } else {
            predicate_only_touches_data_columns(&c.expr, physical_schema).then(|| c.expr.clone())
        };
        if let Some(e) = for_parquet {
            out.parquet_filter.push(e);
        } else if !c.kernel_translatable && partition_only {
            out.partition_prune.push(c.expr.clone());
        } else if !partition_only {
            // Mixed atomic (touches partition + data) — kernel may best-effort
            // file-skip, but rows in surviving files still need row-level eval
            // once partition cols are materialized.
            out.post_transform.push(c.expr.clone());
        }
        // Translatable + partition-only: kernel exact-skips, no further work.
    }
    out
}

/// Gnarly workaround:
/// JSON instead of bincode: bincode encodes enum variants positionally, and
/// our feature subset shifts `FunctionExpr` discriminants relative to the
/// Python wheel's full-feature build — variant-name keying survives that.
pub(crate) fn extract_expr_via_json(predicate: &Bound<'_, PyAny>) -> PyResult<Expr> {
    let py = predicate.py();
    let pyexpr = predicate.getattr("_pyexpr")?;
    let buf = py.import("io")?.getattr("BytesIO")?.call0()?;
    pyexpr.call_method1("serialize_json", (&buf,))?;
    let bytes: Vec<u8> = buf.call_method0("getvalue")?.extract()?;
    serde_json::from_slice::<Expr>(&bytes).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("failed to deserialize polars Expr: {e}"))
    })
}

/// Split a top-level `AND` chain so each conjunct can be routed
/// independently (kernel file-skipping vs polars-io row-group pushdown).
pub(crate) fn flatten_and_conjuncts(expr: &Expr) -> Vec<&Expr> {
    fn walk<'a>(expr: &'a Expr, acc: &mut Vec<&'a Expr>) {
        match expr {
            Expr::BinaryExpr {
                left,
                op: Operator::And,
                right,
            } => {
                walk(left, acc);
                walk(right, acc);
            }
            other => acc.push(other),
        }
    }
    let mut out = Vec::new();
    walk(expr, &mut out);
    out
}

/// Inverse of [`flatten_and_conjuncts`]. `None` for an empty input.
pub(crate) fn conjunction(mut conjuncts: Vec<Expr>) -> Option<Expr> {
    let first = conjuncts.pop()?;
    Some(conjuncts.into_iter().fold(first, |acc, e| acc.and(e)))
}

/// Kernel's `StructField::physical_name(mode)` is `pub(crate)`, so read the
/// underlying metadata key ourselves.
const PHYSICAL_NAME_KEY: &str = "delta.columnMapping.physicalName";

fn physical_name(field: &StructField) -> &str {
    match field.metadata.get(PHYSICAL_NAME_KEY) {
        Some(MetadataValue::String(s)) => s.as_str(),
        _ => field.name.as_str(),
    }
}

pub(crate) fn has_column_mapping(props: &TableProperties) -> bool {
    matches!(
        props.column_mapping_mode,
        Some(ColumnMappingMode::Id | ColumnMappingMode::Name)
    )
}

/// Used to drop predicates touching partition columns — kernel adds those
/// post-read via `Transform`, so the parquet reader can't see them.
pub(crate) fn predicate_only_touches_data_columns(
    expr: &Expr,
    physical_schema: &StructType,
) -> bool {
    let phys_names: std::collections::HashSet<&str> =
        physical_schema.fields().map(|f| f.name.as_str()).collect();
    polars_plan::utils::expr_to_leaf_column_names(expr)
        .iter()
        .all(|n| phys_names.contains(n.as_str()))
}

/// Polars-driven partition pruning for predicates kernel can't translate
/// (e.g. `partition_col.dt.year() == 2024`). Returns the indices of files
/// whose partition values satisfy `partition_conjuncts`.
pub(crate) fn file_skip_via_partition_eval(
    partition_conjuncts: &[Expr],
    files: &[ScanFileMeta],
    logical_schema: &StructType,
) -> anyhow::Result<HashSet<usize>> {
    const FILE_IDX_COL: &str = "__pldl_file_idx__";
    // BTreeSet for one-pass dedup with sorted iteration order.
    let partition_cols: BTreeSet<&str> = files
        .iter()
        .flat_map(|f| f.partition_values.keys().map(String::as_str))
        .collect();

    let mut columns: Vec<Column> = partition_cols
        .iter()
        .map(|name| -> anyhow::Result<Column> {
            let field = logical_schema
                .field(name)
                .ok_or_else(|| anyhow::anyhow!("partition column not in logical schema: {name}"))?;
            let vals: Vec<Option<&str>> = files
                .iter()
                .map(|f| f.partition_values.get(*name).map(String::as_str))
                .collect();
            Column::new(PlSmallStr::from_str(name), vals.as_slice())
                .cast(&field.data_type.to_polars()?)
                .map_err(|e| anyhow::anyhow!("cast partition col {name}: {e:#}"))
        })
        .collect::<anyhow::Result<_>>()?;
    let idx_vals: Vec<u32> = (0..files.len() as u32).collect();
    columns.push(Column::new(
        PlSmallStr::from_static(FILE_IDX_COL),
        idx_vals.as_slice(),
    ));

    let pred = conjunction(partition_conjuncts.to_vec())
        .expect("caller gates on non-empty partition_skip_conjuncts");
    let surviving = DataFrame::new(files.len(), columns)
        .map_err(|e| anyhow::anyhow!("partition DF build: {e:#}"))?
        .lazy()
        .filter(pred)
        .select([polars::prelude::col(PlSmallStr::from_static(FILE_IDX_COL))])
        .collect()
        .map_err(|e| anyhow::anyhow!("partition eval: {e:#}"))?;
    let chunked = surviving
        .column(FILE_IDX_COL)
        .and_then(|c| c.u32())
        .map_err(|e| anyhow::anyhow!("read file-idx col: {e:#}"))?;
    Ok(chunked.into_iter().flatten().map(|x| x as usize).collect())
}

/// Returns true iff `expr` references any column that is in `logical_schema`
/// but not in `physical_schema` (i.e. a partition column). Used to gate
/// untranslatable conjuncts for [`file_skip_via_partition_eval`].
pub(crate) fn touches_partition_only(
    expr: &Expr,
    logical_schema: &StructType,
    physical_schema: &StructType,
) -> bool {
    let phys_names: HashSet<&str> = physical_schema.fields().map(|f| f.name.as_str()).collect();
    let logical_names: HashSet<&str> = logical_schema.fields().map(|f| f.name.as_str()).collect();
    let referenced = polars_plan::utils::expr_to_leaf_column_names(expr);
    !referenced.is_empty()
        && referenced.iter().all(|n| {
            let s = n.as_str();
            logical_names.contains(s) && !phys_names.contains(s)
        })
}

/// Only meaningful when column mapping is active — see [`has_column_mapping`].
/// `None` if the predicate references a partition column or an unknown name.
pub(crate) fn rewrite_predicate_to_physical(
    expr: &Expr,
    logical_schema: &StructType,
    physical_schema: &StructType,
) -> Option<Expr> {
    let phys_names: std::collections::HashSet<&str> =
        physical_schema.fields().map(|f| f.name.as_str()).collect();
    let mut logical_to_phys: HashMap<String, PlSmallStr> = HashMap::new();
    for field in logical_schema.fields() {
        let phys = physical_name(field);
        if phys_names.contains(phys) {
            logical_to_phys.insert(field.name.to_string(), PlSmallStr::from_str(phys));
        }
    }
    let referenced = polars_plan::utils::expr_to_leaf_column_names(expr);
    if !referenced
        .iter()
        .all(|n| logical_to_phys.contains_key(n.as_str()))
    {
        return None;
    }
    let rewritten = expr.clone().map_expr(|node| match node {
        Expr::Column(name) => match logical_to_phys.get(name.as_str()) {
            Some(phys) => Expr::Column(phys.clone()),
            None => Expr::Column(name),
        },
        other => other,
    });
    Some(rewritten)
}
