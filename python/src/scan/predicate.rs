//! Predicate plumbing: extract polars `Expr` from Python, split AND chains,
//! detect column mapping, and rewrite logical → physical column names.

use std::collections::HashMap;

use delta_kernel::schema::{MetadataValue, StructField, StructType};
use delta_kernel::table_features::ColumnMappingMode;
use delta_kernel::table_properties::TableProperties;
use polars::prelude::Expr;
use polars_plan::dsl::Operator;
use polars_utils::pl_str::PlSmallStr;
use pyo3::prelude::*;

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
