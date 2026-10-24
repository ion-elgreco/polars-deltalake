//! Predicate plumbing: extract polars `Expr` from Python, split AND chains,
//! detect column mapping, and rewrite logical → physical column names.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use delta_kernel::schema::{DataType as KernelDataType, PrimitiveType, StructField, StructType};
use delta_kernel::table_features::ColumnMappingMode;
use polars::prelude::{
    AnyValue, Column, DataFrame, DataType as PlDataType, Expr, IntoLazy, LiteralValue,
    Scalar as PlScalar, Series, TimeZone, lit,
};
use polars_plan::dsl::function_expr::{BooleanFunction, FunctionExpr, StructFunction};
use polars_plan::dsl::{DataTypeExpr, Operator, StringFunction};
use polars_utils::pl_str::PlSmallStr;
use pyo3::prelude::*;

use crate::scan::plan::ScanFileMeta;
use crate::translation::schema::resolve_leaf_dtype;
use crate::translation::to_kernel::column_path;

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
    partition_cols: &[String],
) -> ConjunctClassification {
    let column_mapped = mode != ColumnMappingMode::None;
    let mut out = ConjunctClassification::default();
    for c in conjuncts {
        if c.kernel_translatable {
            out.kernel.push(c.expr.clone());
        }
        let partition_only = touches_partition_only(&c.expr, partition_cols);
        let for_parquet = if touches_nested_field(&c.expr) {
            // polars evaluates a pushed predicate on the file's own struct,
            // before `missing_struct_fields: Insert` adds the fields a file
            // predates, so it errors on files older than the field.
            None
        } else if column_mapped {
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
            // Mixed atomic (touches partition + data) or nested-field —
            // kernel may best-effort file-skip, but rows in surviving files
            // still need row-level eval once the columns are materialized.
            out.post_transform.push(c.expr.clone());
        }
    }
    out
}

fn touches_nested_field(expr: &Expr) -> bool {
    polars_plan::utils::has_expr(expr, |e| {
        matches!(
            e,
            Expr::Function {
                function: FunctionExpr::StructExpr(StructFunction::FieldByName(_)),
                ..
            }
        )
    })
}

/// Delta reads a tz-less timestamp literal as UTC against a `TIMESTAMP`
/// column (Spark's session zone); polars refuses the comparison outright.
/// Retype the literal once, here, so kernel stats, the parquet filter,
/// partition pruning and the post-scan filter all compare instants.
pub(crate) fn coerce_naive_datetime_literals(expr: Expr, schema: &StructType) -> Expr {
    let is_utc_column = |e: &Expr| {
        column_path(e).is_some_and(|path| {
            matches!(
                resolve_leaf_dtype(schema, path.iter().map(|s| s.as_str())),
                Some(KernelDataType::Primitive(PrimitiveType::Timestamp))
            )
        })
    };
    expr.map_expr(|node| match node {
        Expr::BinaryExpr { left, op, right } if is_comparison(op) => {
            let (left_utc, right_utc) = (is_utc_column(&left), is_utc_column(&right));
            let retype = |side: Arc<Expr>, other_is_utc: bool| {
                if other_is_utc {
                    Arc::new(literal_as_utc(Arc::unwrap_or_clone(side)))
                } else {
                    side
                }
            };
            Expr::BinaryExpr {
                left: retype(left, right_utc),
                op,
                right: retype(right, left_utc),
            }
        }
        Expr::Function { input, function }
            if matches!(
                function,
                FunctionExpr::Boolean(
                    BooleanFunction::IsBetween { .. } | BooleanFunction::IsIn { .. }
                )
            ) && input.first().is_some_and(is_utc_column) =>
        {
            let input = input
                .into_iter()
                .enumerate()
                .map(|(i, e)| if i == 0 { e } else { literal_as_utc(e) })
                .collect();
            Expr::Function { input, function }
        }
        other => other,
    })
}

fn is_comparison(op: Operator) -> bool {
    matches!(
        op,
        Operator::Eq
            | Operator::EqValidity
            | Operator::NotEq
            | Operator::NotEqValidity
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq
    )
}

/// A naive datetime literal in any of the spellings polars produces — a
/// scalar, a set, a `cast`, or SQL's `TIMESTAMP'…'` (`strptime`) — read as
/// UTC. Anything else is returned unchanged.
fn literal_as_utc(e: Expr) -> Expr {
    let utc = |tu| PlDataType::Datetime(tu, Some(TimeZone::UTC));
    let cast_series = |s: &Series| match s.dtype() {
        PlDataType::Datetime(tu, None) => s.cast(&utc(*tu)).ok(),
        _ => None,
    };
    match e {
        Expr::Literal(LiteralValue::Scalar(s)) => match (s.dtype(), s.value()) {
            (
                PlDataType::Datetime(tu, None),
                AnyValue::Datetime(v, _, _) | AnyValue::DatetimeOwned(v, _, _),
            ) => Expr::Literal(LiteralValue::Scalar(PlScalar::new(
                utc(*tu),
                AnyValue::DatetimeOwned(*v, *tu, Some(Arc::new(TimeZone::UTC))),
            ))),
            (PlDataType::List(_), AnyValue::List(series)) => match cast_series(series) {
                Some(cast) => lit(cast).implode(true),
                None => Expr::Literal(LiteralValue::Scalar(s)),
            },
            _ => Expr::Literal(LiteralValue::Scalar(s)),
        },
        Expr::Literal(LiteralValue::Series(series)) => match cast_series(&series) {
            Some(cast) => lit(cast),
            None => Expr::Literal(LiteralValue::Series(series)),
        },
        Expr::Cast {
            expr,
            dtype: DataTypeExpr::Literal(PlDataType::Datetime(tu, None)),
            options,
        } => Expr::Cast {
            expr,
            dtype: DataTypeExpr::Literal(utc(tu)),
            options,
        },
        Expr::Function {
            input,
            function:
                FunctionExpr::StringExpr(StringFunction::Strptime(
                    DataTypeExpr::Literal(PlDataType::Datetime(tu, None)),
                    opts,
                )),
        } => Expr::Function {
            input,
            function: FunctionExpr::StringExpr(StringFunction::Strptime(
                DataTypeExpr::Literal(utc(tu)),
                opts,
            )),
        },
        other => other,
    }
}

#[cfg(test)]
mod literal_coercion_tests {
    use polars::prelude::{NamedFrom, TimeUnit, col};

    use super::*;
    use crate::translation::to_kernel::polars_expr_to_kernel_predicate;

    fn schema() -> StructType {
        StructType::try_new([
            StructField::nullable("ts", KernelDataType::TIMESTAMP),
            StructField::nullable("ntz", KernelDataType::TIMESTAMP_NTZ),
        ])
        .unwrap()
    }

    fn naive(v: i64) -> Expr {
        Expr::Literal(LiteralValue::Scalar(PlScalar::new(
            PlDataType::Datetime(TimeUnit::Microseconds, None),
            AnyValue::DatetimeOwned(v, TimeUnit::Microseconds, None),
        )))
    }

    fn utc() -> PlDataType {
        PlDataType::Datetime(TimeUnit::Microseconds, Some(TimeZone::UTC))
    }

    fn literal_dtype(e: &Expr) -> PlDataType {
        match e {
            Expr::Literal(LiteralValue::Scalar(s)) => s.dtype().clone(),
            Expr::Literal(LiteralValue::Series(s)) => s.dtype().clone(),
            Expr::Alias(inner, _) => literal_dtype(inner),
            other => panic!("not a literal: {other:?}"),
        }
    }

    fn rhs(e: &Expr) -> &Expr {
        match e {
            Expr::BinaryExpr { right, .. } => right,
            other => panic!("not a binary expr: {other:?}"),
        }
    }

    #[test]
    fn naive_literal_against_utc_column_reads_as_utc() {
        let out = coerce_naive_datetime_literals(col("ts").lt(naive(7)), &schema());
        assert_eq!(literal_dtype(rhs(&out)), utc());
        // Kernel sees a `Timestamp` scalar, so file skipping stays on.
        assert!(polars_expr_to_kernel_predicate(&out, &schema()).is_some());
        // Either side.
        let out = coerce_naive_datetime_literals(naive(7).gt(col("ts")), &schema());
        match &out {
            Expr::BinaryExpr { left, .. } => assert_eq!(literal_dtype(left), utc()),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn ntz_column_keeps_its_naive_literal() {
        let out = coerce_naive_datetime_literals(col("ntz").lt(naive(7)), &schema());
        assert_eq!(
            literal_dtype(rhs(&out)),
            PlDataType::Datetime(TimeUnit::Microseconds, None)
        );
    }

    #[test]
    fn set_bounds_follow_the_column() {
        let out = coerce_naive_datetime_literals(
            col("ts").is_between(naive(1), naive(2), polars::prelude::ClosedInterval::Both),
            &schema(),
        );
        let Expr::Function { input, .. } = &out else {
            panic!("{out:?}");
        };
        assert_eq!(literal_dtype(&input[1]), utc());
        assert_eq!(literal_dtype(&input[2]), utc());

        let set = Series::new("".into(), [Some(1i64), Some(2)])
            .cast(&PlDataType::Datetime(TimeUnit::Microseconds, None))
            .unwrap();
        let out = coerce_naive_datetime_literals(col("ts").is_in(lit(set), false), &schema());
        let Expr::Function { input, .. } = &out else {
            panic!("{out:?}");
        };
        assert_eq!(literal_dtype(&input[1]), utc());
        assert!(polars_expr_to_kernel_predicate(&out, &schema()).is_some());
    }

    #[test]
    fn sql_timestamp_literal_strptime_targets_utc() {
        let strptime = lit("2024-03-01 00:00:00").str().strptime(
            PlDataType::Datetime(TimeUnit::Microseconds, None),
            polars::prelude::StrptimeOptions::default(),
            lit("raise"),
        );
        let out = coerce_naive_datetime_literals(col("ts").lt(strptime), &schema());
        let Expr::Function { function, .. } = rhs(&out) else {
            panic!("{out:?}");
        };
        assert!(matches!(
            function,
            FunctionExpr::StringExpr(StringFunction::Strptime(DataTypeExpr::Literal(dt), _)) if *dt == utc()
        ));
    }

    #[test]
    fn nested_field_conjuncts_stay_out_of_the_parquet_filter() {
        let schema = StructType::try_new([StructField::nullable(
            "info",
            KernelDataType::Struct(Box::new(
                StructType::try_new([StructField::nullable("email", KernelDataType::STRING)])
                    .unwrap(),
            )),
        )])
        .unwrap();
        let conjuncts = [Conjunct {
            expr: col("info").struct_().field_by_name("email").is_not_null(),
            kernel_translatable: true,
        }];
        let routing =
            classify_conjuncts(&conjuncts, ColumnMappingMode::None, &schema, &schema, &[]);
        assert!(routing.parquet_filter.is_empty());
        assert_eq!(routing.post_transform.len(), 1);
        assert_eq!(
            routing.kernel.len(),
            1,
            "file-level stats skipping still applies"
        );
    }
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

/// Leaf column names `exprs` reference that `schema` does not declare.
/// `BTreeSet` for one-pass dedup with sorted iteration order.
pub(crate) fn columns_outside_schema<'a>(
    exprs: impl IntoIterator<Item = &'a Expr>,
    schema: &StructType,
) -> BTreeSet<String> {
    exprs
        .into_iter()
        .flat_map(polars_plan::utils::expr_to_leaf_column_names)
        .filter(|n| !schema.contains(n.as_str()))
        .map(|n| n.to_string())
        .collect()
}

/// Used to drop predicates touching partition columns — kernel adds those
/// post-read via `Transform`, so the parquet reader can't see them.
pub(crate) fn predicate_only_touches_data_columns(
    expr: &Expr,
    physical_schema: &StructType,
) -> bool {
    columns_outside_schema([expr], physical_schema).is_empty()
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
            // An absent key is a NULL partition value, not a missing column:
            // kernel's `MapItem::materialize` drops null-valued entries, so a
            // partition whose value is NULL in every add action carries no key
            // at all. The caller already gated on the snapshot's partition
            // list, so `None` here means NULL.
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

/// Returns true iff every column `expr` references is a partition column.
/// Used to gate conjuncts for [`file_skip_via_partition_eval`].
///
/// `partition_cols` is the snapshot's own list (kernel's
/// `logical_partition_columns`), the same authority `resolve_scan` uses.
/// Deriving partitionhood as "logical minus physical" instead would
/// misread any other logical-only name — a projected-away data column, a
/// future rev's row-tracking field — as a partition column and route it to
/// a layer that cannot evaluate it.
pub(crate) fn touches_partition_only(expr: &Expr, partition_cols: &[String]) -> bool {
    let referenced = polars_plan::utils::expr_to_leaf_column_names(expr);
    !referenced.is_empty()
        && referenced
            .iter()
            .all(|n| partition_cols.iter().any(|p| p == n.as_str()))
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
            file: delta_kernel::FileMeta {
                location: url::Url::parse(&format!("file:///t/{region}.parquet")).unwrap(),
                last_modified: 0,
                size: 0,
            },
            path: PlRefPath::new(format!("/t/{region}.parquet")),
            rewrite: LogicalRewrite {
                select: None,
                dv: None,
            },
            partition_values: [("col-3".to_string(), region.to_string())]
                .into_iter()
                .collect(),
            num_records: None,
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

    /// Kernel's `MapItem::materialize` drops null-valued entries, so a
    /// partition that is NULL in every add action carries no key at all.
    /// That is a legal log, not a missing column: prune on NULL, don't fail.
    #[test]
    fn absent_partition_key_reads_as_null() {
        let logical =
            StructType::try_new([StructField::nullable("region", KernelDataType::STRING)]).unwrap();
        let files = [no_partition_values(), no_partition_values()];

        let surviving = file_skip_via_partition_eval(
            &[col("region").is_null()],
            &files,
            &logical,
            ColumnMappingMode::None,
        )
        .unwrap();
        assert_eq!(surviving, HashSet::from([0usize, 1usize]));

        let none_match = file_skip_via_partition_eval(
            &[col("region").eq(lit("EU"))],
            &files,
            &logical,
            ColumnMappingMode::None,
        )
        .unwrap();
        assert!(none_match.is_empty());
    }

    fn no_partition_values() -> ScanFileMeta {
        ScanFileMeta {
            partition_values: HashMap::new(),
            ..file("EU")
        }
    }

    /// A projected-away data column is logical-only too, so inferring
    /// partitionhood from "logical minus physical" would route it to the
    /// partition pruner, which cannot evaluate it.
    #[test]
    fn projected_away_data_column_is_not_partition_only() {
        let partition_cols = ["region".to_string()];
        assert!(touches_partition_only(
            &col("region").eq(lit("EU")),
            &partition_cols
        ));
        assert!(!touches_partition_only(
            &col("b").gt(lit(15)),
            &partition_cols
        ));
        // Mixed atomic: one partition column is not enough.
        assert!(!touches_partition_only(
            &col("region").eq(lit("EU")).and(col("b").gt(lit(15))),
            &partition_cols
        ));
    }
}
