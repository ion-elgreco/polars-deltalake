//! Polars `Expr` → delta-kernel `Predicate` translation, used by
//! `TableScan::configure` to push polars filters into the kernel scan
//! (which then applies them to per-file parquet stats during planning).
//!
//! Best-effort: recognises column refs, literals, comparisons, AND/OR,
//! IsNull, IsBetween, IsIn, NOT, and boolean constants. Anything else
//! returns `None` and the caller falls back to polars-side filtering.

use delta_kernel::expressions::{
    ColumnName, DecimalData, Expression, JunctionPredicateOp, Predicate, Scalar,
};
use delta_kernel::schema::{DataType as KernelDataType, DecimalType, PrimitiveType, StructType};
use polars::prelude::{
    AnyValue, DataType as PlDataType, Expr, LiteralValue, Operator, Series, TimeUnit,
};
use polars_plan::dsl::DataTypeExpr;
use polars_plan::dsl::function_expr::{BooleanFunction, FunctionExpr, StructFunction};
use polars_plan::plans::DynLiteralValue;
use polars_utils::pl_str::PlSmallStr;

use crate::translation::schema::KernelDataTypeExt;

/// Above this, refuse the IsIn → OR-chain rewrite
const MAX_IN_LIST_KERNEL_EXPANSION: usize = 512;

/// Try to translate a polars filter `Expr` into a kernel `Predicate`. Returns
/// `None` if any sub-expression is outside the supported subset.
pub(crate) fn polars_expr_to_kernel_predicate(
    expr: &Expr,
    schema: &StructType,
) -> Option<Predicate> {
    if let Some(name) = column_ref(expr, schema) {
        return Some(Predicate::from_expr(name));
    }
    match expr {
        Expr::BinaryExpr { left, op, right } => translate_binary(left, *op, right, schema),
        Expr::Function { input, function } => translate_function(input, function, schema),
        Expr::Cast {
            expr: inner, dtype, ..
        } if cast_is_droppable(inner, dtype, schema) => {
            polars_expr_to_kernel_predicate(inner, schema)
        }
        Expr::Alias(inner, _) => polars_expr_to_kernel_predicate(inner, schema),
        Expr::Literal(lit) => lit_bool(lit).map(Predicate::literal),
        _ => None,
    }
}

/// Kernel column reference for `expr`, following `struct.field(...)` chains.
///
/// `None` unless the path ends at a primitive leaf: kernel's
/// `PhysicalPredicate::try_new` resolves nothing else, and *errors the whole
/// scan* on a reference it cannot resolve.
fn column_ref(expr: &Expr, schema: &StructType) -> Option<ColumnName> {
    let path = column_path(expr)?;
    column_leaf_type(&path, schema)?;
    Some(ColumnName::new(path.iter().map(|s| s.to_string())))
}

/// Primitive leaf type `path` resolves to, or `None` if any segment is
/// missing or a non-struct stands mid-path.
fn column_leaf_type<'a>(path: &[PlSmallStr], schema: &'a StructType) -> Option<&'a PrimitiveType> {
    match crate::translation::schema::resolve_leaf_dtype(schema, path.iter().map(|s| s.as_str()))? {
        KernelDataType::Primitive(p) => Some(p),
        _ => None,
    }
}

/// Whether dropping a cast leaves a predicate kernel can safely skip files
/// with. Kernel prunes a file when the pushed predicate is false for its
/// stats, so a cast that *changes values* must not be dropped: on a
/// `Float64` column holding `1.4`, `col.cast(Int32) == 1` is true while the
/// stripped `col == 1` is false, and the file holding the matching row is
/// pruned away. Only a lossless widening over a resolvable column survives
/// the trip — a raw expression can carry an unfolded value-changing cast on
/// a literal, so a non-column operand declines.
fn cast_is_droppable(inner: &Expr, dtype: &DataTypeExpr, schema: &StructType) -> bool {
    let Some(path) = column_path(inner) else {
        return false;
    };
    let Some(from) = column_leaf_type(&path, schema) else {
        return false;
    };
    let DataTypeExpr::Literal(target) = dtype else {
        return false;
    };
    let Ok(from_pl) = KernelDataType::Primitive(from.clone()).to_polars() else {
        return false;
    };
    from_pl == *target
        || matches!(
            (&from_pl, target),
            (
                PlDataType::Int8,
                PlDataType::Int16
                    | PlDataType::Int32
                    | PlDataType::Int64
                    | PlDataType::Float32
                    | PlDataType::Float64
            ) | (
                PlDataType::Int16,
                PlDataType::Int32
                    | PlDataType::Int64
                    | PlDataType::Float32
                    | PlDataType::Float64
            ) | (
                PlDataType::Int32,
                PlDataType::Int64 | PlDataType::Float64
            ) | (PlDataType::Float32, PlDataType::Float64)
        )
}

/// Primitive leaf type a translated kernel column reference resolves to.
fn column_leaf_prim(name: &ColumnName, schema: &StructType) -> Option<PrimitiveType> {
    let path: Vec<PlSmallStr> = name.iter().map(|s| PlSmallStr::from_str(s)).collect();
    column_leaf_type(&path, schema).cloned()
}

fn scalar_numeric_prim(s: &Scalar) -> Option<PrimitiveType> {
    Some(match s {
        Scalar::Byte(_) => PrimitiveType::Byte,
        Scalar::Short(_) => PrimitiveType::Short,
        Scalar::Integer(_) => PrimitiveType::Integer,
        Scalar::Long(_) => PrimitiveType::Long,
        Scalar::Float(_) => PrimitiveType::Float,
        Scalar::Double(_) => PrimitiveType::Double,
        _ => return None,
    })
}

fn numeric_prim(t: &PrimitiveType) -> bool {
    matches!(
        t,
        PrimitiveType::Byte
            | PrimitiveType::Short
            | PrimitiveType::Integer
            | PrimitiveType::Long
            | PrimitiveType::Float
            | PrimitiveType::Double
    )
}

/// Convert a numeric scalar to `target` only when the value survives the
/// round trip exactly. Kernel compares stats strictly same-type, so a
/// literal left wider than its column never skips; an inexact conversion
/// must decline — `f32_col == 2.9f64` is false on every row, while the
/// narrowed `f32_col == 2.9f32` is satisfiable.
fn narrow_scalar_exact(s: &Scalar, target: &PrimitiveType) -> Option<Scalar> {
    let int_val = |s: &Scalar| -> Option<i64> {
        Some(match s {
            Scalar::Byte(v) => i64::from(*v),
            Scalar::Short(v) => i64::from(*v),
            Scalar::Integer(v) => i64::from(*v),
            Scalar::Long(v) => *v,
            _ => return None,
        })
    };
    Some(match target {
        PrimitiveType::Byte => Scalar::Byte(i8::try_from(int_val(s)?).ok()?),
        PrimitiveType::Short => Scalar::Short(i16::try_from(int_val(s)?).ok()?),
        PrimitiveType::Integer => Scalar::Integer(i32::try_from(int_val(s)?).ok()?),
        PrimitiveType::Long => Scalar::Long(int_val(s)?),
        PrimitiveType::Float => match s {
            Scalar::Double(v) => {
                let f = *v as f32;
                if f64::from(f) == *v {
                    Scalar::Float(f)
                } else {
                    return None;
                }
            }
            _ => {
                let i = int_val(s)?;
                let f = i as f32;
                if f as i64 == i {
                    Scalar::Float(f)
                } else {
                    return None;
                }
            }
        },
        PrimitiveType::Double => match s {
            Scalar::Float(v) => Scalar::Double(f64::from(*v)),
            _ => {
                let i = int_val(s)?;
                let f = i as f64;
                if f as i64 == i {
                    Scalar::Double(f)
                } else {
                    return None;
                }
            }
        },
        _ => return None,
    })
}

/// Align a numeric literal operand to its column's exact type. Non-numeric
/// pairings pass through untouched; an inexact numeric conversion declines
/// the conjunct.
fn align_numeric_literal(
    l: Expression,
    r: Expression,
    schema: &StructType,
) -> Option<(Expression, Expression)> {
    let align = |name: &ColumnName, s: Scalar| -> Option<Scalar> {
        let Some(target) = column_leaf_prim(name, schema).filter(numeric_prim) else {
            return Some(s);
        };
        match scalar_numeric_prim(&s) {
            Some(sp) if sp == target => Some(s),
            Some(_) => narrow_scalar_exact(&s, &target),
            None => Some(s),
        }
    };
    Some(match (l, r) {
        (Expression::Column(name), Expression::Literal(s)) => {
            let s = align(&name, s)?;
            (Expression::Column(name), Expression::Literal(s))
        }
        (Expression::Literal(s), Expression::Column(name)) => {
            let s = align(&name, s)?;
            (Expression::Literal(s), Expression::Column(name))
        }
        other => other,
    })
}

/// `SelectFields` is excluded: its selector expands to names later, so it is
/// not a path yet.
fn column_path(expr: &Expr) -> Option<Vec<PlSmallStr>> {
    match expr {
        Expr::Column(name) => Some(vec![name.clone()]),
        Expr::Alias(inner, _) => column_path(inner),
        Expr::Function {
            input,
            function: FunctionExpr::StructExpr(StructFunction::FieldByName(name)),
        } => {
            let mut path = column_path(input.first()?)?;
            path.push(name.clone());
            Some(path)
        }
        _ => None,
    }
}

fn lit_bool(lit: &LiteralValue) -> Option<bool> {
    let LiteralValue::Scalar(s) = lit else {
        return None;
    };
    match s.as_any_value() {
        AnyValue::Boolean(b) => Some(b),
        _ => None,
    }
}

/// Like [`lit_bool`] but peels Cast/Alias — the binary fold sees literals
/// under inferred casts.
fn extract_bool_literal(expr: &Expr) -> Option<bool> {
    match expr {
        Expr::Literal(lit) => lit_bool(lit),
        Expr::Cast { expr: inner, .. } | Expr::Alias(inner, _) => extract_bool_literal(inner),
        _ => None,
    }
}

/// `expr == lit(true)` → `expr`; `expr == lit(false)` → `NOT expr`;
/// `expr != lit(true)` → `NOT expr`; `expr != lit(false)` → `expr`.
fn fold_boolean_equality(
    left: &Expr,
    right: &Expr,
    is_ne: bool,
    schema: &StructType,
) -> Option<Predicate> {
    let (other, lit_val) = match (extract_bool_literal(left), extract_bool_literal(right)) {
        (Some(b), None) => (right, b),
        (None, Some(b)) => (left, b),
        // Both literals or neither — let the regular comparison path handle it.
        _ => return None,
    };
    let inner = polars_expr_to_kernel_predicate(other, schema)?;
    Some(if is_ne ^ lit_val {
        inner
    } else {
        Predicate::not(inner)
    })
}

fn translate_binary(
    left: &Expr,
    op: Operator,
    right: &Expr,
    schema: &StructType,
) -> Option<Predicate> {
    if let Some(junction_op) = match op {
        Operator::And | Operator::LogicalAnd => Some(JunctionPredicateOp::And),
        Operator::Or | Operator::LogicalOr => Some(JunctionPredicateOp::Or),
        _ => None,
    } {
        return Some(Predicate::junction(
            junction_op,
            [
                polars_expr_to_kernel_predicate(left, schema)?,
                polars_expr_to_kernel_predicate(right, schema)?,
            ],
        ));
    }

    // Optimizer doesn't fold `<pred> == lit(bool)` / `!=`, and a predicate
    // can't sit on a comparison's expression side — fold here so e.g.
    // `(col > 0) == pl.lit(True)` survives as `col > 0`.
    if matches!(op, Operator::Eq | Operator::NotEq) {
        if let Some(folded) = fold_boolean_equality(left, right, op == Operator::NotEq, schema) {
            return Some(folded);
        }
    }

    let l = polars_expr_to_kernel_expression(left, schema)?;
    let r = polars_expr_to_kernel_expression(right, schema)?;
    let (l, r) = align_numeric_literal(l, r, schema)?;
    Some(match op {
        Operator::Eq => Predicate::eq(l, r),
        Operator::NotEq => Predicate::ne(l, r),
        Operator::Lt => Predicate::lt(l, r),
        Operator::Gt => Predicate::gt(l, r),
        Operator::LtEq => Predicate::le(l, r),
        Operator::GtEq => Predicate::ge(l, r),
        // `ne_missing` / `eq_missing` — kernel's null-aware `Distinct`.
        Operator::NotEqValidity => Predicate::distinct(l, r),
        Operator::EqValidity => Predicate::not(Predicate::distinct(l, r)),
        _ => return None,
    })
}

fn translate_function(
    input: &[Expr],
    function: &FunctionExpr,
    schema: &StructType,
) -> Option<Predicate> {
    use polars::prelude::ClosedInterval;
    match function {
        FunctionExpr::Boolean(BooleanFunction::IsNull) => Some(Predicate::is_null(
            polars_expr_to_kernel_expression(input.first()?, schema)?,
        )),
        FunctionExpr::Boolean(BooleanFunction::IsNotNull) => Some(Predicate::is_not_null(
            polars_expr_to_kernel_expression(input.first()?, schema)?,
        )),
        FunctionExpr::Boolean(BooleanFunction::IsIn { .. }) => {
            let lhs = input.first()?;
            let rhs = input.get(1)?;
            let elements = extract_set_elements(rhs)?;
            if elements.len() > MAX_IN_LIST_KERNEL_EXPANSION {
                tracing::debug!(
                    target: "polars_deltalake::pushdown",
                    n = elements.len(),
                    cap = MAX_IN_LIST_KERNEL_EXPANSION,
                    "is_in list exceeds kernel expansion cap; \
                     polars-io will handle row filtering without file pruning",
                );
                return None;
            }
            let lhs_kernel = polars_expr_to_kernel_expression(lhs, schema)?;
            // Numeric elements narrow to the column's exact type; an inexact
            // element can equal no column value, so its disjunct drops.
            let elements: Vec<Scalar> = match &lhs_kernel {
                Expression::Column(name) => {
                    match column_leaf_prim(name, schema).filter(numeric_prim) {
                        Some(target) => elements
                            .into_iter()
                            .filter_map(|s| match scalar_numeric_prim(&s) {
                                Some(sp) if sp == target => Some(s),
                                Some(_) => narrow_scalar_exact(&s, &target),
                                None => Some(s),
                            })
                            .collect(),
                        None => elements,
                    }
                }
                _ => elements,
            };
            // `x IN []` — including a set with no representable element —
            // is vacuously false.
            if elements.is_empty() {
                return Some(Predicate::literal(false));
            }
            // Flatten to `lhs == v1 OR ...` rather than `BinaryPredicateOp::In`:
            // kernel's `eval_pred_in` (kernel_predicates/mod.rs) is a `None //
            //
            // TODO: revert to `BinaryPredicateOp::In` once kernel's pruning
            // evaluators implement `eval_pred_in`.
            Some(Predicate::or_from(elements.into_iter().map(|s| {
                Predicate::eq(lhs_kernel.clone(), Expression::Literal(s))
            })))
        }
        FunctionExpr::Boolean(BooleanFunction::IsBetween { closed }) => {
            // [value, low, high] → (value cmp low) AND (value cmp high) with
            // strict vs non-strict picked from the interval shape.
            let value = polars_expr_to_kernel_expression(input.first()?, schema)?;
            let low = polars_expr_to_kernel_expression(input.get(1)?, schema)?;
            let high = polars_expr_to_kernel_expression(input.get(2)?, schema)?;
            let (value, low) = align_numeric_literal(value, low, schema)?;
            let (value, high) = align_numeric_literal(value, high, schema)?;
            let (lo, hi) = match closed {
                ClosedInterval::Both => (
                    Predicate::ge(value.clone(), low),
                    Predicate::le(value, high),
                ),
                ClosedInterval::Left => (
                    Predicate::ge(value.clone(), low),
                    Predicate::lt(value, high),
                ),
                ClosedInterval::Right => (
                    Predicate::gt(value.clone(), low),
                    Predicate::le(value, high),
                ),
                ClosedInterval::None => (
                    Predicate::gt(value.clone(), low),
                    Predicate::lt(value, high),
                ),
            };
            Some(Predicate::and(lo, hi))
        }
        FunctionExpr::Boolean(BooleanFunction::Not) | FunctionExpr::Negate => Some(Predicate::not(
            polars_expr_to_kernel_predicate(input.first()?, schema)?,
        )),
        FunctionExpr::Boolean(BooleanFunction::AllHorizontal) => {
            translate_junction(input, JunctionPredicateOp::And, schema)
        }
        FunctionExpr::Boolean(BooleanFunction::AnyHorizontal) => {
            translate_junction(input, JunctionPredicateOp::Or, schema)
        }
        _ => None,
    }
}

fn translate_junction(
    args: &[Expr],
    op: JunctionPredicateOp,
    schema: &StructType,
) -> Option<Predicate> {
    let preds: Vec<_> = args
        .iter()
        .map(|a| polars_expr_to_kernel_predicate(a, schema))
        .collect::<Option<_>>()?;
    Some(Predicate::junction(op, preds))
}

/// Kernel stores timestamps as microseconds-since-epoch and the stats
/// column distinguishes `Timestamp` (tz-aware, treated as UTC) from
/// `TimestampNtz`; emitting the wrong variant trips the comparison check at
/// file-skipping time. Overflow on ms → µs returns `None` so we skip
/// pushdown rather than feed kernel a wrong stat value.
fn datetime_scalar(v: i64, tu: TimeUnit, has_tz: bool) -> Option<Scalar> {
    let micros = match tu {
        TimeUnit::Microseconds => v,
        TimeUnit::Milliseconds => v.checked_mul(1_000)?,
        TimeUnit::Nanoseconds => {
            if v % 1_000 != 0 {
                return None;
            }
            v / 1_000
        }
    };
    Some(if has_tz {
        Scalar::Timestamp(micros)
    } else {
        Scalar::TimestampNtz(micros)
    })
}

/// `None` for nulls and any variant that has no signed-Long-or-narrower
/// representation (e.g. UInt64 ≥ 2⁶³). The caller decides whether `None`
/// is an error or a skip signal.
fn any_value_to_scalar(av: &AnyValue<'_>) -> Option<Scalar> {
    Some(match av {
        AnyValue::Null => return None,
        AnyValue::Boolean(b) => Scalar::Boolean(*b),
        AnyValue::Int8(v) => Scalar::Byte(*v),
        AnyValue::Int16(v) => Scalar::Short(*v),
        AnyValue::Int32(v) => Scalar::Integer(*v),
        AnyValue::Int64(v) => Scalar::Long(*v),
        AnyValue::UInt8(v) => Scalar::Integer(i32::from(*v)),
        AnyValue::UInt16(v) => Scalar::Integer(i32::from(*v)),
        AnyValue::UInt32(v) => Scalar::Long(i64::from(*v)),
        // UInt64 > i64::MAX has no Long encoding — refuse rather than wrap.
        AnyValue::UInt64(v) => Scalar::Long(i64::try_from(*v).ok()?),
        AnyValue::Float32(v) => Scalar::Float(*v),
        AnyValue::Float64(v) => Scalar::Double(*v),
        AnyValue::String(s) => Scalar::String(s.to_string()),
        AnyValue::StringOwned(s) => Scalar::String(s.to_string()),
        AnyValue::Binary(b) => Scalar::Binary(b.to_vec()),
        AnyValue::BinaryOwned(b) => Scalar::Binary(b.clone()),
        AnyValue::Date(d) => Scalar::Date(*d),
        AnyValue::Datetime(v, tu, tz) => datetime_scalar(*v, *tu, tz.is_some())?,
        AnyValue::DatetimeOwned(v, tu, tz) => datetime_scalar(*v, *tu, tz.is_some())?,
        // Delta caps decimals at precision/scale 38, both fitting in u8.
        AnyValue::Decimal(v, p, s) => {
            let dt = DecimalType::try_new(u8::try_from(*p).ok()?, u8::try_from(*s).ok()?).ok()?;
            Scalar::Decimal(DecimalData::try_new(*v, dt).ok()?)
        }
        _ => return None,
    })
}

/// Translate a polars `Expr` into a kernel `Expression` (the operand-side,
/// not predicate-side). Only column references and concrete literals are
/// supported — anything else aborts pushdown.
fn polars_expr_to_kernel_expression(expr: &Expr, schema: &StructType) -> Option<Expression> {
    if let Some(name) = column_ref(expr, schema) {
        return Some(Expression::Column(name));
    }
    match expr {
        Expr::Alias(inner, _) => polars_expr_to_kernel_expression(inner, schema),
        Expr::Cast {
            expr: inner, dtype, ..
        } if cast_is_droppable(inner, dtype, schema) => {
            polars_expr_to_kernel_expression(inner, schema)
        }
        Expr::Literal(lit) => Some(Expression::Literal(match lit {
            LiteralValue::Scalar(s) => any_value_to_scalar(&s.as_any_value())?,
            LiteralValue::Dyn(DynLiteralValue::Str(s)) => Scalar::String(s.to_string()),
            // i128 → narrowest fitting signed kernel type.
            LiteralValue::Dyn(DynLiteralValue::Int(v)) => Scalar::Long(i64::try_from(*v).ok()?),
            LiteralValue::Dyn(DynLiteralValue::Float(v)) => Scalar::Double(*v),
            _ => return None,
        })),
        _ => None,
    }
}

/// `IsIn` set literal → kernel `Scalar`s. `None` for any null element
/// (no null-safe equality to OR with), unsupported element type, or a cast
/// around the set — its pre-cast elements are not the compared values.
/// Empty list yields `Some(vec![])`; caller short-circuits to `lit(false)`.
fn extract_set_elements(expr: &Expr) -> Option<Vec<Scalar>> {
    match expr {
        Expr::Alias(inner, _) => extract_set_elements(inner),
        Expr::Literal(LiteralValue::Series(s)) => series_to_scalars(s),
        Expr::Literal(LiteralValue::Scalar(s)) => match s.as_any_value() {
            AnyValue::List(series) => series_to_scalars(&series),
            _ => None,
        },
        _ => None,
    }
}

fn series_to_scalars(series: &Series) -> Option<Vec<Scalar>> {
    series
        .iter()
        .map(|av| match av {
            AnyValue::Null => None,
            other => any_value_to_scalar(&other),
        })
        .collect()
}

#[cfg(test)]
mod narrow_scalar_tests {
    use super::*;

    #[test]
    fn exact_double_narrows_to_float() {
        assert_eq!(
            narrow_scalar_exact(&Scalar::Double(1.5), &PrimitiveType::Float),
            Some(Scalar::Float(1.5))
        );
    }

    #[test]
    fn inexact_double_declines_float() {
        assert_eq!(
            narrow_scalar_exact(&Scalar::Double(2.9), &PrimitiveType::Float),
            None
        );
    }

    #[test]
    fn long_narrows_within_range_only() {
        assert_eq!(
            narrow_scalar_exact(&Scalar::Long(1), &PrimitiveType::Integer),
            Some(Scalar::Integer(1))
        );
        assert_eq!(
            narrow_scalar_exact(&Scalar::Long(i64::from(i32::MAX) + 1), &PrimitiveType::Integer),
            None
        );
    }

    #[test]
    fn long_to_double_requires_exactness() {
        assert_eq!(
            narrow_scalar_exact(&Scalar::Long(1 << 53), &PrimitiveType::Double),
            Some(Scalar::Double(9_007_199_254_740_992.0))
        );
        assert_eq!(
            narrow_scalar_exact(&Scalar::Long((1 << 53) + 1), &PrimitiveType::Double),
            None
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn datetime_scalar_microseconds_passthrough() {
        assert_eq!(
            datetime_scalar(1_700_000_000_000_000, TimeUnit::Microseconds, true),
            Some(Scalar::Timestamp(1_700_000_000_000_000))
        );
        assert_eq!(
            datetime_scalar(1_700_000_000_000_000, TimeUnit::Microseconds, false),
            Some(Scalar::TimestampNtz(1_700_000_000_000_000))
        );
    }

    #[test]
    fn datetime_scalar_milliseconds_scales_exactly() {
        assert_eq!(
            datetime_scalar(1_700_000_000_000, TimeUnit::Milliseconds, true),
            Some(Scalar::Timestamp(1_700_000_000_000_000))
        );
    }

    #[test]
    fn datetime_scalar_nanoseconds_divisible_succeeds() {
        // 1_700_000_000_000_000_000 ns = 1_700_000_000_000_000 µs exactly.
        assert_eq!(
            datetime_scalar(1_700_000_000_000_000_000, TimeUnit::Nanoseconds, true),
            Some(Scalar::Timestamp(1_700_000_000_000_000))
        );
    }

    #[test]
    fn datetime_scalar_nanoseconds_subus_refuses_pushdown() {
        // Bug A: a fixed rounding direction is unsound for one half of the
        // operator-direction matrix (floor breaks `<` / `<=`, ceil breaks
        // `>` / `>=`). Refuse pushdown rather than emit a possibly-wrong
        // literal; polars-io still filters row-by-row.
        assert_eq!(
            datetime_scalar(1_700_000_000_000_000_999, TimeUnit::Nanoseconds, true),
            None,
        );
        assert_eq!(datetime_scalar(-1_500, TimeUnit::Nanoseconds, false), None,);
    }

    #[test]
    fn datetime_scalar_milliseconds_overflow_refuses_pushdown() {
        // i64::MAX ms cannot be expressed as µs — `checked_mul` returns None.
        assert_eq!(
            datetime_scalar(i64::MAX, TimeUnit::Milliseconds, true),
            None
        );
    }
}
