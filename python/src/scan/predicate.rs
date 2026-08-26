//! Predicate plumbing: extract polars `Expr` from Python, split AND chains,
//! detect column mapping, and rewrite logical → physical column names.

use std::collections::{BTreeSet, HashMap, HashSet};

use delta_kernel::schema::{DataType as KernelDataType, StructField, StructType};
use delta_kernel::table_features::ColumnMappingMode;
use polars::prelude::{Column, DataFrame, Expr, IntoLazy};
use polars_plan::dsl::Operator;
use polars_utils::pl_str::PlSmallStr;
use pyo3::prelude::*;

use crate::scan::plan::ScanFileMeta;

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
    /// Exact file pruning via polars eval on partition values — the only
    /// layer that evaluates a partition-only conjunct, since kernel's
    /// stats-based skipping keeps every file it cannot decide.
    pub(crate) partition_prune: Vec<Expr>,
    /// Applied inside `LogicalScanIter` after the physical→logical select
    /// materializes partition columns.
    pub(crate) post_transform: Vec<Expr>,
}

/// Route each conjunct to the layer(s) that will evaluate it.
pub(crate) fn classify_conjuncts(
    conjuncts: &[Conjunct],
    mode: ColumnMappingMode,
    logical_schema: &StructType,
    physical_schema: &StructType,
) -> ConjunctClassification {
    let column_mapped = mode != ColumnMappingMode::None;
    let mut out = ConjunctClassification::default();
    for c in conjuncts {
        if c.kernel_translatable {
            out.kernel.push(c.expr.clone());
        }
        let partition_only = touches_partition_only(&c.expr, logical_schema, physical_schema, mode);
        let for_parquet = if column_mapped {
            rewrite_predicate_to_physical(&c.expr, logical_schema, physical_schema, mode)
        } else {
            predicate_only_touches_data_columns(&c.expr, physical_schema).then(|| c.expr.clone())
        };
        if let Some(e) = for_parquet {
            out.parquet_filter.push(e);
        } else if partition_only {
            // Every partition-only conjunct, translatable or not. Kernel's
            // file skipping is conservative — a NULL verdict (a NULL partition
            // value) and an expression its evaluator has no rule for both
            // *keep* the file — while polars deletes its own filter node once
            // we accept the predicate, so an unevaluated conjunct returns rows
            // it excludes.
            out.partition_prune.push(c.expr.clone());
        } else {
            // Mixed atomic (touches partition + data) — kernel may best-effort
            // file-skip, but rows in surviving files still need row-level eval
            // once partition cols are materialized.
            out.post_transform.push(c.expr.clone());
        }
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

/// True when column mapping renames anything below the top level of `dtype`.
pub(crate) fn renames_nested_fields(dtype: &KernelDataType, mode: ColumnMappingMode) -> bool {
    match dtype {
        KernelDataType::Struct(fields) => fields.fields().any(|f| {
            f.physical_name(mode) != f.name.as_str() || renames_nested_fields(&f.data_type, mode)
        }),
        KernelDataType::Array(array) => renames_nested_fields(array.element_type(), mode),
        KernelDataType::Map(map) => {
            renames_nested_fields(map.key_type(), mode)
                || renames_nested_fields(map.value_type(), mode)
        }
        _ => false,
    }
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
///
/// `add.partitionValues` is keyed by *physical* name, while the conjuncts
/// name logical columns, so the frame is built physical-keyed and emitted
/// under logical names.
pub(crate) fn file_skip_via_partition_eval(
    partition_conjuncts: &[Expr],
    files: &[ScanFileMeta],
    logical_schema: &StructType,
    mode: ColumnMappingMode,
) -> anyhow::Result<HashSet<usize>> {
    const FILE_IDX_COL: &str = "__pldl_file_idx__";
    let by_logical: HashMap<&str, &StructField> = logical_schema
        .fields()
        .map(|f| (f.name.as_str(), f))
        .collect();
    // Only the columns the conjuncts name: an unreferenced partition key —
    // a stale one a foreign writer left behind, or one whose type polars
    // cannot build — must not fail or slow down a skip that never reads it.
    // BTreeSet for one-pass dedup with sorted iteration order.
    let referenced: BTreeSet<&str> = partition_conjuncts
        .iter()
        .flat_map(polars_plan::utils::expr_to_leaf_column_names)
        .filter_map(|n| by_logical.get_key_value(n.as_str()).map(|(k, _)| *k))
        .collect();

    let mut columns: Vec<Column> = referenced
        .iter()
        .map(|logical| -> anyhow::Result<Column> {
            let field = by_logical[logical];
            let physical = field.physical_name(mode);
            let vals: Vec<Option<&str>> = files
                .iter()
                .map(|f| f.partition_values.get(physical).map(String::as_str))
                .collect();
            let raw = Column::new(PlSmallStr::from_str(field.name.as_str()), vals.as_slice());
            // Kernel's `parse_scalar`, the same grammar the select list uses
            // to materialize these values: a polars cast rejects spellings
            // Delta mandates (`2024-01-15 10:30:00`) and would prune away
            // every file it nulls.
            crate::translation::parse_partition_column(&raw, &field.data_type)
                .map_err(|e| anyhow::anyhow!("parse partition col {}: {e:#}", field.name))
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
    Ok(chunked.iter().flatten().map(|x| x as usize).collect())
}

/// Returns true iff `expr` references any column that is in `logical_schema`
/// but not in `physical_schema` (i.e. a partition column). Used to gate
/// untranslatable conjuncts for [`file_skip_via_partition_eval`].
pub(crate) fn touches_partition_only(
    expr: &Expr,
    logical_schema: &StructType,
    physical_schema: &StructType,
    mode: ColumnMappingMode,
) -> bool {
    let phys_names: HashSet<&str> = physical_schema.fields().map(|f| f.name.as_str()).collect();
    let referenced = polars_plan::utils::expr_to_leaf_column_names(expr);
    // Physical names: under column mapping no logical name is in the file
    // schema, so every column would look like a partition column.
    !referenced.is_empty()
        && referenced.iter().all(|n| {
            logical_schema
                .field(n.as_str())
                .is_some_and(|f| !phys_names.contains(f.physical_name(mode)))
        })
}

/// Only meaningful when column mapping is active (`mode` is `Id` or `Name`).
/// `None` if the predicate references a partition column or an unknown name.
pub(crate) fn rewrite_predicate_to_physical(
    expr: &Expr,
    logical_schema: &StructType,
    physical_schema: &StructType,
    mode: ColumnMappingMode,
) -> Option<Expr> {
    let phys_names: std::collections::HashSet<&str> =
        physical_schema.fields().map(|f| f.name.as_str()).collect();
    let mut logical_to_phys: HashMap<String, PlSmallStr> = HashMap::new();
    for field in logical_schema.fields() {
        let phys = field.physical_name(mode);
        // A flat name rewrite fixes only the root, so nested-renamed columns
        // stay out: the check below then declines the whole predicate, which
        // runs after the physical→logical select on logical names instead.
        if phys_names.contains(phys) && !renames_nested_fields(&field.data_type, mode) {
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

#[cfg(test)]
mod partition_prune_tests {
    use polars::prelude::{col, lit};
    use polars_utils::pl_path::PlRefPath;

    use super::*;
    use crate::scan::plan::LogicalRewrite;

    /// Kernel writes `add.partitionValues` keyed by *physical* name.
    fn file(region: &str) -> ScanFileMeta {
        ScanFileMeta {
            path: PlRefPath::new(format!("/t/{region}.parquet")),
            rewrite: LogicalRewrite {
                select: None,
                dv: None,
            },
            partition_values: [("col-3".to_string(), region.to_string())]
                .into_iter()
                .collect(),
        }
    }

    /// The conjunct names logical columns, so the pruning frame has to be
    /// emitted under logical names even though its keys arrive physical.
    #[test]
    fn column_mapped_keys_resolve_to_logical_names() {
        let logical =
            StructType::try_new([StructField::nullable("region", KernelDataType::STRING)
                .with_metadata([("delta.columnMapping.physicalName", "col-3")])])
            .unwrap();
        let files = [file("EU"), file("US")];

        let surviving = file_skip_via_partition_eval(
            &[col("region").eq(lit("EU"))],
            &files,
            &logical,
            ColumnMappingMode::Name,
        )
        .unwrap();
        assert_eq!(surviving, HashSet::from([0usize]));
    }
}
