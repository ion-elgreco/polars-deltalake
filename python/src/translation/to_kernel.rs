//! Polars `Expr` → delta-kernel `Predicate` translation, used by
//! `DeltaSource::try_set_predicate` to push polars filters into the kernel
//! scan (which then applies them to per-file parquet stats during planning).
//!
//! The translator is best-effort and intentionally narrow: it recognises the
//! subset of polars `Expr` shapes that have a 1:1 kernel equivalent (column
//! references, simple literals, comparison binops, AND/OR chains, IsNull,
//! IsIn). Anything else returns `None` and the caller must fall back to
//! polars-side filtering.

use delta_kernel::expressions::{
    BinaryPredicate, BinaryPredicateOp, ColumnName, Expression, JunctionPredicate,
    JunctionPredicateOp, Predicate, Scalar, UnaryPredicate, UnaryPredicateOp,
};
use polars::prelude::{AnyValue, Expr, LiteralValue, Operator};
use polars_plan::dsl::function_expr::{BooleanFunction, FunctionExpr};
use polars_plan::plans::DynLiteralValue;

/// Try to translate a polars filter `Expr` into a kernel `Predicate`. Returns
/// `None` if any sub-expression is outside the supported subset.
pub(crate) fn polars_expr_to_kernel_predicate(expr: &Expr) -> Option<Predicate> {
    match expr {
        Expr::BinaryExpr { left, op, right } => translate_binary(left, *op, right),
        Expr::Function { input, function } => translate_function(input, function),
        // Drop the cast: kernel pruning isn't dtype-strict.
        Expr::Cast { expr: inner, .. } => polars_expr_to_kernel_predicate(inner),
        Expr::Alias(inner, _) => polars_expr_to_kernel_predicate(inner),
        // Bare column reference filter — a bool column used as a predicate.
        // Emit as `BooleanExpression(Column)` rather than synthesising
        // `col == true`; both produce identical pruning in kernel's
        // null-safe expansion (`eval_pred_sql_where`), and the direct
        // variant matches the user's polars expression.
        Expr::Column(name) => Some(Predicate::BooleanExpression(Expression::Column(
            ColumnName::new([name.to_string()]),
        ))),
        _ => None,
    }
}

fn translate_binary(left: &Expr, op: Operator, right: &Expr) -> Option<Predicate> {
    // Logical AND/OR: both sides must translate.
    if let Some(junction_op) = match op {
        Operator::And | Operator::LogicalAnd => Some(JunctionPredicateOp::And),
        Operator::Or | Operator::LogicalOr => Some(JunctionPredicateOp::Or),
        _ => None,
    } {
        return Some(Predicate::Junction(JunctionPredicate {
            op: junction_op,
            preds: vec![
                polars_expr_to_kernel_predicate(left)?,
                polars_expr_to_kernel_predicate(right)?,
            ],
        }));
    }

    // Comparisons: kernel only exposes Equal / LessThan / GreaterThan as
    // primitives — `!=` and `<=` / `>=` are derived as `Not(strict opposite)`.
    let left_kernel = polars_expr_to_kernel_expression(left)?;
    let right_kernel = polars_expr_to_kernel_expression(right)?;
    let binop = |op| {
        Predicate::Binary(BinaryPredicate {
            op,
            left: Box::new(left_kernel.clone()),
            right: Box::new(right_kernel.clone()),
        })
    };
    Some(match op {
        Operator::Eq => binop(BinaryPredicateOp::Equal),
        Operator::NotEq => Predicate::Not(Box::new(binop(BinaryPredicateOp::Equal))),
        Operator::Lt => binop(BinaryPredicateOp::LessThan),
        Operator::Gt => binop(BinaryPredicateOp::GreaterThan),
        Operator::LtEq => Predicate::Not(Box::new(binop(BinaryPredicateOp::GreaterThan))),
        Operator::GtEq => Predicate::Not(Box::new(binop(BinaryPredicateOp::LessThan))),
        // Null-aware comparisons (`eq_missing` / `ne_missing` in polars):
        // kernel's `Distinct` is null-aware NotEqual, so `ne_missing` is a
        // direct emit and `eq_missing` is its negation.
        Operator::NotEqValidity => binop(BinaryPredicateOp::Distinct),
        Operator::EqValidity => Predicate::Not(Box::new(binop(BinaryPredicateOp::Distinct))),
        _ => return None,
    })
}

fn translate_function(input: &[Expr], function: &FunctionExpr) -> Option<Predicate> {
    use polars::prelude::ClosedInterval;
    match function {
        FunctionExpr::Boolean(BooleanFunction::IsNull) => {
            Some(unary_pred(UnaryPredicateOp::IsNull, input.first()?))?
        }
        FunctionExpr::Boolean(BooleanFunction::IsNotNull) => Some(Predicate::Not(Box::new(
            unary_pred(UnaryPredicateOp::IsNull, input.first()?)?,
        ))),
        FunctionExpr::Boolean(BooleanFunction::IsIn { .. }) => {
            Some(Predicate::Binary(BinaryPredicate {
                op: BinaryPredicateOp::In,
                left: Box::new(polars_expr_to_kernel_expression(input.first()?)?),
                right: Box::new(polars_expr_to_kernel_expression(input.get(1)?)?),
            }))
        }
        FunctionExpr::Boolean(BooleanFunction::IsBetween { closed }) => {
            // [value, low, high] → (value cmp low) AND (value cmp high) with
            // strict vs non-strict picked from the interval shape.
            let value = polars_expr_to_kernel_expression(input.first()?)?;
            let low = polars_expr_to_kernel_expression(input.get(1)?)?;
            let high = polars_expr_to_kernel_expression(input.get(2)?)?;
            let (lo_op, hi_op) = match closed {
                ClosedInterval::Both => (None, None),
                ClosedInterval::Left => (None, Some(BinaryPredicateOp::LessThan)),
                ClosedInterval::Right => (Some(BinaryPredicateOp::GreaterThan), None),
                ClosedInterval::None => (
                    Some(BinaryPredicateOp::GreaterThan),
                    Some(BinaryPredicateOp::LessThan),
                ),
            };
            Some(Predicate::Junction(JunctionPredicate {
                op: JunctionPredicateOp::And,
                preds: vec![
                    bounded_cmp(value.clone(), low, lo_op, BinaryPredicateOp::LessThan),
                    bounded_cmp(value, high, hi_op, BinaryPredicateOp::GreaterThan),
                ],
            }))
        }
        FunctionExpr::Boolean(BooleanFunction::Not) | FunctionExpr::Negate => Some(Predicate::Not(
            Box::new(polars_expr_to_kernel_predicate(input.first()?)?),
        )),
        FunctionExpr::Boolean(BooleanFunction::AllHorizontal) => {
            Some(junction(JunctionPredicateOp::And, input)?)
        }
        FunctionExpr::Boolean(BooleanFunction::AnyHorizontal) => {
            Some(junction(JunctionPredicateOp::Or, input)?)
        }
        _ => None,
    }
}

fn unary_pred(op: UnaryPredicateOp, arg: &Expr) -> Option<Predicate> {
    Some(Predicate::Unary(UnaryPredicate {
        op,
        expr: Box::new(polars_expr_to_kernel_expression(arg)?),
    }))
}

fn junction(op: JunctionPredicateOp, args: &[Expr]) -> Option<Predicate> {
    let preds: Option<Vec<Predicate>> = args.iter().map(polars_expr_to_kernel_predicate).collect();
    Some(Predicate::Junction(JunctionPredicate { op, preds: preds? }))
}

/// One side of an `IsBetween` decomposition: `strict_op` (provided directly
/// when `closed_op` is `Some`) or its `Not` negation (when `closed_op` is
/// `None`, meaning the inclusive side).
fn bounded_cmp(
    value: Expression,
    bound: Expression,
    closed_op: Option<BinaryPredicateOp>,
    strict_op: BinaryPredicateOp,
) -> Predicate {
    match closed_op {
        Some(op) => Predicate::Binary(BinaryPredicate {
            op,
            left: Box::new(value),
            right: Box::new(bound),
        }),
        None => Predicate::Not(Box::new(Predicate::Binary(BinaryPredicate {
            op: strict_op,
            left: Box::new(value),
            right: Box::new(bound),
        }))),
    }
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
        AnyValue::Datetime(v, _, _) | AnyValue::DatetimeOwned(v, _, _) => Scalar::Timestamp(*v),
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
