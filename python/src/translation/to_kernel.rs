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
use delta_kernel::schema::DecimalType;
use polars::prelude::{AnyValue, Expr, LiteralValue, Operator, Series, TimeUnit};
use polars_plan::dsl::function_expr::{BooleanFunction, FunctionExpr};
use polars_plan::plans::DynLiteralValue;

/// Above this, refuse the IsIn → OR-chain rewrite
const MAX_IN_LIST_KERNEL_EXPANSION: usize = 512;

/// Try to translate a polars filter `Expr` into a kernel `Predicate`. Returns
/// `None` if any sub-expression is outside the supported subset.
pub(crate) fn polars_expr_to_kernel_predicate(expr: &Expr) -> Option<Predicate> {
    match expr {
        Expr::BinaryExpr { left, op, right } => translate_binary(left, *op, right),
        Expr::Function { input, function } => translate_function(input, function),
        // Drop the cast: kernel pruning isn't dtype-strict.
        Expr::Cast { expr: inner, .. } => polars_expr_to_kernel_predicate(inner),
        Expr::Alias(inner, _) => polars_expr_to_kernel_predicate(inner),
        Expr::Column(name) => Some(Predicate::column([name.to_string()])),
        Expr::Literal(lit) => lit_bool(lit).map(Predicate::literal),
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
fn fold_boolean_equality(left: &Expr, right: &Expr, is_ne: bool) -> Option<Predicate> {
    let (other, lit_val) = match (extract_bool_literal(left), extract_bool_literal(right)) {
        (Some(b), None) => (right, b),
        (None, Some(b)) => (left, b),
        // Both literals or neither — let the regular comparison path handle it.
        _ => return None,
    };
    let inner = polars_expr_to_kernel_predicate(other)?;
    Some(if is_ne ^ lit_val {
        inner
    } else {
        Predicate::not(inner)
    })
}

fn translate_binary(left: &Expr, op: Operator, right: &Expr) -> Option<Predicate> {
    if let Some(junction_op) = match op {
        Operator::And | Operator::LogicalAnd => Some(JunctionPredicateOp::And),
        Operator::Or | Operator::LogicalOr => Some(JunctionPredicateOp::Or),
        _ => None,
    } {
        return Some(Predicate::junction(
            junction_op,
            [
                polars_expr_to_kernel_predicate(left)?,
                polars_expr_to_kernel_predicate(right)?,
            ],
        ));
    }

    // Optimizer doesn't fold `<pred> == lit(bool)` / `!=`, and a predicate
    // can't sit on a comparison's expression side — fold here so e.g.
    // `(col > 0) == pl.lit(True)` survives as `col > 0`.
    if matches!(op, Operator::Eq | Operator::NotEq) {
        if let Some(folded) = fold_boolean_equality(left, right, op == Operator::NotEq) {
            return Some(folded);
        }
    }

    let l = polars_expr_to_kernel_expression(left)?;
    let r = polars_expr_to_kernel_expression(right)?;
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

fn translate_function(input: &[Expr], function: &FunctionExpr) -> Option<Predicate> {
    use polars::prelude::ClosedInterval;
    match function {
        FunctionExpr::Boolean(BooleanFunction::IsNull) => Some(Predicate::is_null(
            polars_expr_to_kernel_expression(input.first()?)?,
        )),
        FunctionExpr::Boolean(BooleanFunction::IsNotNull) => Some(Predicate::is_not_null(
            polars_expr_to_kernel_expression(input.first()?)?,
        )),
        FunctionExpr::Boolean(BooleanFunction::IsIn { .. }) => {
            let lhs = input.first()?;
            let rhs = input.get(1)?;
            let elements = extract_set_elements(rhs)?;
            // `x IN []` is vacuously false.
            if elements.is_empty() {
                return Some(Predicate::literal(false));
            }
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
            // Flatten to `lhs == v1 OR ...` rather than `BinaryPredicateOp::In`:
            // kernel's `eval_pred_in` (kernel_predicates/mod.rs) is a `None //
            //
            // TODO: revert to `BinaryPredicateOp::In` once kernel's pruning
            // evaluators implement `eval_pred_in`.
            let lhs_kernel = polars_expr_to_kernel_expression(lhs)?;
            Some(Predicate::or_from(elements.into_iter().map(|s| {
                Predicate::eq(lhs_kernel.clone(), Expression::Literal(s))
            })))
        }
        FunctionExpr::Boolean(BooleanFunction::IsBetween { closed }) => {
            // [value, low, high] → (value cmp low) AND (value cmp high) with
            // strict vs non-strict picked from the interval shape.
            let value = polars_expr_to_kernel_expression(input.first()?)?;
            let low = polars_expr_to_kernel_expression(input.get(1)?)?;
            let high = polars_expr_to_kernel_expression(input.get(2)?)?;
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
            polars_expr_to_kernel_predicate(input.first()?)?,
        )),
        FunctionExpr::Boolean(BooleanFunction::AllHorizontal) => {
            translate_junction(input, JunctionPredicateOp::And)
        }
        FunctionExpr::Boolean(BooleanFunction::AnyHorizontal) => {
            translate_junction(input, JunctionPredicateOp::Or)
        }
        _ => None,
    }
}

fn translate_junction(args: &[Expr], op: JunctionPredicateOp) -> Option<Predicate> {
    let preds: Vec<_> = args
        .iter()
        .map(polars_expr_to_kernel_predicate)
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
fn polars_expr_to_kernel_expression(expr: &Expr) -> Option<Expression> {
    match expr {
        Expr::Column(name) => Some(Expression::Column(ColumnName::new([name.to_string()]))),
        Expr::Alias(inner, _) => polars_expr_to_kernel_expression(inner),
        Expr::Cast { expr: inner, .. } => polars_expr_to_kernel_expression(inner),
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
/// (no null-safe equality to OR with) or unsupported element type.
/// Empty list yields `Some(vec![])`; caller short-circuits to `lit(false)`.
fn extract_set_elements(expr: &Expr) -> Option<Vec<Scalar>> {
    match expr {
        Expr::Cast { expr: inner, .. } | Expr::Alias(inner, _) => extract_set_elements(inner),
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
