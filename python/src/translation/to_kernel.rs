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
    // `Predicate::from_expr` is a truthiness test, so only a boolean column
    // means here what it means in polars.
    if let Some(name) = boolean_column_ref(expr, schema) {
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

/// [`column_ref`] restricted to boolean leaves, for predicate position.
fn boolean_column_ref(expr: &Expr, schema: &StructType) -> Option<ColumnName> {
    let path = column_path(expr)?;
    matches!(column_leaf_type(&path, schema)?, PrimitiveType::Boolean)
        .then(|| ColumnName::new(path.iter().map(|s| s.to_string())))
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
                PlDataType::Int32 | PlDataType::Int64 | PlDataType::Float32 | PlDataType::Float64
            ) | (PlDataType::Int32, PlDataType::Int64 | PlDataType::Float64)
                | (PlDataType::Float32, PlDataType::Float64)
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

/// Outcome of retyping a numeric literal to a column's exact type.
enum Narrowing {
    /// The same value, spelled in `target`.
    Exact(Scalar),
    /// `target` has no value equal to the literal, so an equality against it
    /// is false for every row. Only sound in equality position.
    NoValueMatches,
    /// Not decidable here — the caller must decline rather than assume.
    Undecidable,
}

/// Largest magnitude below which every f64 integer is also exactly the i64 of
/// the same value; above it an integer column value can round to a different
/// f64 than itself, so comparisons stop agreeing with polars'.
const F64_EXACT_INT_LIMIT: f64 = (1u64 << 53) as f64;

/// The i64 a float names exactly, if it names one at all.
fn float_as_exact_int(v: f64) -> Option<i64> {
    (v.fract() == 0.0 && v.abs() < F64_EXACT_INT_LIMIT).then(|| v as i64)
}

/// Retype a numeric scalar to `target`. Kernel compares stats strictly
/// same-type, so a literal left wider than its column never skips; an
/// inexact conversion is not `Exact` — `f32_col == 2.9f64` is false on
/// every row, while the narrowed `f32_col == 2.9f32` is satisfiable.
fn narrow_scalar(s: &Scalar, target: &PrimitiveType) -> Narrowing {
    // A float that names no integer (fractional, NaN, infinite) equals no
    // value of an integer column; one too large to be exact is undecidable.
    let int_val = |s: &Scalar| -> Result<i64, Narrowing> {
        match s {
            Scalar::Byte(v) => Ok(i64::from(*v)),
            Scalar::Short(v) => Ok(i64::from(*v)),
            Scalar::Integer(v) => Ok(i64::from(*v)),
            Scalar::Long(v) => Ok(*v),
            Scalar::Float(v) => float_as_exact_int(f64::from(*v)).ok_or(
                if f64::from(*v).abs() < F64_EXACT_INT_LIMIT {
                    Narrowing::NoValueMatches
                } else {
                    Narrowing::Undecidable
                },
            ),
            Scalar::Double(v) => float_as_exact_int(*v).ok_or(if v.abs() < F64_EXACT_INT_LIMIT {
                Narrowing::NoValueMatches
            } else {
                Narrowing::Undecidable
            }),
            _ => Err(Narrowing::Undecidable),
        }
    };
    // Out of range means no value of `target` equals it.
    let fit = |v: Result<i64, Narrowing>, f: fn(i64) -> Option<Scalar>| match v {
        Ok(i) => match f(i) {
            Some(s) => Narrowing::Exact(s),
            None => Narrowing::NoValueMatches,
        },
        Err(n) => n,
    };
    match target {
        PrimitiveType::Byte => fit(int_val(s), |i| i8::try_from(i).ok().map(Scalar::Byte)),
        PrimitiveType::Short => fit(int_val(s), |i| i16::try_from(i).ok().map(Scalar::Short)),
        PrimitiveType::Integer => fit(int_val(s), |i| i32::try_from(i).ok().map(Scalar::Integer)),
        PrimitiveType::Long => fit(int_val(s), |i| Some(Scalar::Long(i))),
        PrimitiveType::Float => match s {
            Scalar::Double(v) => {
                let f = *v as f32;
                if f64::from(f) == *v {
                    Narrowing::Exact(Scalar::Float(f))
                } else {
                    Narrowing::NoValueMatches
                }
            }
            _ => fit(int_val(s), |i| {
                let f = i as f32;
                (f as i64 == i).then_some(Scalar::Float(f))
            }),
        },
        PrimitiveType::Double => match s {
            Scalar::Float(v) => Narrowing::Exact(Scalar::Double(f64::from(*v))),
            _ => fit(int_val(s), |i| {
                let f = i as f64;
                (f as i64 == i).then_some(Scalar::Double(f))
            }),
        },
        _ => Narrowing::Undecidable,
    }
}

/// Retype `s` to the column's primitive when both sides are numeric; a
/// non-numeric scalar passes through untouched as `Exact`. What
/// `NoValueMatches` / `Undecidable` mean is the call site's decision:
/// comparison position declines the conjunct, set position drops only
/// the element's disjunct.
fn narrow_to_column(s: Scalar, target: &PrimitiveType) -> Narrowing {
    match scalar_numeric_prim(&s) {
        Some(sp) if sp == *target => Narrowing::Exact(s),
        Some(_) => narrow_scalar(&s, target),
        None => Narrowing::Exact(s),
    }
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
        match narrow_to_column(s, &target) {
            Narrowing::Exact(s) => Some(s),
            // Comparison position can only use an exact retyping.
            Narrowing::NoValueMatches | Narrowing::Undecidable => None,
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
    if is_ne ^ lit_val {
        Some(inner)
    } else if mentions_float_column(other, schema) {
        None
    } else {
        Some(Predicate::not(inner))
    }
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
    if !nan_safe(&l, op, &r, schema) {
        return None;
    }
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

/// polars orders NaN above every number, but stats leave NaN out of
/// `maxValues` (delta-rs) or spell it unparsably (Spark), so a pushed
/// comparison that a NaN row satisfies can skip the file holding it. On a
/// float column only the shapes NaN never satisfies translate: equality, and
/// an upper bound on the column (`col < v`, `col <= v`, `v > col`, `v >= col`).
fn nan_safe(l: &Expression, op: Operator, r: &Expression, schema: &StructType) -> bool {
    if is_nan_literal(l) || is_nan_literal(r) {
        return false;
    }
    let (left_float, right_float) = (is_float_column(l, schema), is_float_column(r, schema));
    if !left_float && !right_float {
        return true;
    }
    match op {
        Operator::Eq | Operator::EqValidity => true,
        Operator::Lt | Operator::LtEq => !right_float,
        Operator::Gt | Operator::GtEq => !left_float,
        _ => false,
    }
}

fn is_float_column(e: &Expression, schema: &StructType) -> bool {
    match e {
        Expression::Column(name) => matches!(
            column_leaf_prim(name, schema),
            Some(PrimitiveType::Float | PrimitiveType::Double)
        ),
        _ => false,
    }
}

fn scalar_is_nan(s: &Scalar) -> bool {
    match s {
        Scalar::Float(v) => v.is_nan(),
        Scalar::Double(v) => v.is_nan(),
        _ => false,
    }
}

fn is_nan_literal(e: &Expression) -> bool {
    matches!(e, Expression::Literal(s) if scalar_is_nan(s))
}

/// Whether `expr` references a float column anywhere the translator walks.
/// A negation flips every comparison on it into one NaN satisfies.
fn mentions_float_column(expr: &Expr, schema: &StructType) -> bool {
    if let Some(path) = column_path(expr) {
        return matches!(
            column_leaf_type(&path, schema),
            Some(PrimitiveType::Float | PrimitiveType::Double)
        );
    }
    match expr {
        Expr::BinaryExpr { left, right, .. } => {
            mentions_float_column(left, schema) || mentions_float_column(right, schema)
        }
        Expr::Function { input, .. } => input.iter().any(|e| mentions_float_column(e, schema)),
        Expr::Cast { expr: inner, .. } | Expr::Alias(inner, _) => {
            mentions_float_column(inner, schema)
        }
        Expr::Literal(_) => false,
        // Anything else does not translate anyway.
        _ => true,
    }
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
            // Numeric elements narrow to the column's exact type. An element
            // the column can never equal drops its disjunct; one that cannot
            // be retyped at all declines the conjunct, because dropping it
            // would narrow the set kernel skips on.
            let elements: Vec<Scalar> = match &lhs_kernel {
                Expression::Column(name) => {
                    match column_leaf_prim(name, schema).filter(numeric_prim) {
                        Some(target) => {
                            let mut kept = Vec::with_capacity(elements.len());
                            for s in elements {
                                match narrow_to_column(s, &target) {
                                    Narrowing::Exact(s) => kept.push(s),
                                    Narrowing::NoValueMatches => {}
                                    Narrowing::Undecidable => return None,
                                }
                            }
                            kept
                        }
                        None => elements,
                    }
                }
                _ => elements,
            };
            // A NaN element would match NaN rows the stats cannot see.
            if elements.iter().any(scalar_is_nan) {
                return None;
            }
            // `x IN []` — including a set with no representable element —
            // is vacuously false.
            if elements.is_empty() {
                return Some(Predicate::literal(false));
            }
            // Flatten to `lhs == v1 OR ...` rather than `BinaryPredicateOp::In`:
            // kernel's pruning evaluators return `None` from `eval_pred_in`
            // (kernel_predicates/mod.rs), so `In` never skips a file.
            // TODO: revert to `BinaryPredicateOp::In` once they implement it.
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
            // NaN never lies inside an interval, so the bounds are safe on
            // a float column; only a NaN bound is not.
            if is_nan_literal(&low) || is_nan_literal(&high) {
                return None;
            }
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
        // `Negate` is arithmetic unary minus, not logical NOT, and has no
        // kernel predicate form.
        FunctionExpr::Boolean(BooleanFunction::Not) => {
            let inner = input.first()?;
            if mentions_float_column(inner, schema) {
                return None;
            }
            Some(Predicate::not(polars_expr_to_kernel_predicate(
                inner, schema,
            )?))
        }
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

    /// The comparison-position collapse the `align` closure applies.
    fn narrow_scalar_exact(s: &Scalar, target: &PrimitiveType) -> Option<Scalar> {
        match narrow_scalar(s, target) {
            Narrowing::Exact(s) => Some(s),
            Narrowing::NoValueMatches | Narrowing::Undecidable => None,
        }
    }

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
            narrow_scalar_exact(
                &Scalar::Long(i64::from(i32::MAX) + 1),
                &PrimitiveType::Integer
            ),
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

#[cfg(test)]
mod is_in_narrowing_tests {
    use super::*;
    use delta_kernel::schema::StructField;
    use polars::prelude::{NamedFrom, col, lit};

    fn long_schema() -> StructType {
        StructType::try_new([StructField::nullable("id", KernelDataType::LONG)]).unwrap()
    }

    fn translate(elements: &[f64]) -> Option<Predicate> {
        let set = Series::new(PlSmallStr::from_static("set"), elements);
        polars_expr_to_kernel_predicate(&col("id").is_in(lit(set), false), &long_schema())
    }

    /// A float-spelled element naming an integer exactly must survive
    /// narrowing. Dropping it emptied the set, and an empty set is
    /// vacuously false, so kernel pruned every file and the rows were lost.
    #[test]
    fn float_spelled_integral_elements_are_kept() {
        let pred = translate(&[1.0, 2.0]).expect("integral float elements must translate");
        assert_ne!(
            pred,
            Predicate::literal(false),
            "narrowable elements must not collapse the set to a false predicate"
        );
    }

    /// An element the column can never equal drops its own disjunct only.
    #[test]
    fn fractional_element_drops_its_disjunct() {
        let pred = translate(&[1.0, 2.5]).expect("one narrowable element must translate");
        assert_ne!(pred, Predicate::literal(false));
        assert_eq!(
            translate(&[2.5]),
            Some(Predicate::literal(false)),
            "a set no column value can equal is vacuously false"
        );
    }

    /// Beyond 2^53 a float names no i64 exactly, so the retyping is not
    /// decidable here and the whole conjunct must be declined.
    #[test]
    fn inexact_magnitude_declines_the_conjunct() {
        assert_eq!(
            translate(&[1.0e300]),
            None,
            "an undecidable element must decline, not prune"
        );
    }
}

#[cfg(test)]
mod float_nan_tests {
    use super::*;
    use delta_kernel::schema::StructField;
    use polars::prelude::{ClosedInterval, NamedFrom, col, lit};

    fn schema() -> StructType {
        StructType::try_new([
            StructField::nullable("f", KernelDataType::DOUBLE),
            StructField::nullable("n", KernelDataType::LONG),
        ])
        .unwrap()
    }

    fn translate(expr: Expr) -> Option<Predicate> {
        polars_expr_to_kernel_predicate(&expr, &schema())
    }

    #[test]
    fn lower_bounds_on_a_float_column_decline() {
        assert_eq!(translate(col("f").gt(lit(1.5))), None);
        assert_eq!(translate(col("f").gt_eq(lit(1.5))), None);
        assert_eq!(translate(lit(1.5).lt(col("f"))), None);
        assert_eq!(translate(col("f").neq(lit(1.5))), None);
        assert_eq!(translate(col("f").neq_missing(lit(1.5))), None);
    }

    #[test]
    fn upper_bounds_and_equality_on_a_float_column_translate() {
        assert!(translate(col("f").lt(lit(1.5))).is_some());
        assert!(translate(col("f").lt_eq(lit(1.5))).is_some());
        assert!(translate(lit(1.5).gt(col("f"))).is_some());
        assert!(translate(col("f").eq(lit(1.5))).is_some());
        assert!(translate(col("f").is_between(lit(1.0), lit(2.0), ClosedInterval::Both)).is_some());
    }

    #[test]
    fn integer_columns_are_untouched() {
        assert!(translate(col("n").gt(lit(1))).is_some());
        assert!(translate(col("n").neq(lit(1))).is_some());
        assert!(translate(col("n").lt(lit(1)).not()).is_some());
    }

    #[test]
    fn nan_literals_decline() {
        assert_eq!(translate(col("f").eq(lit(f64::NAN))), None);
        assert_eq!(translate(col("f").lt(lit(f64::NAN))), None);
        let set = Series::new(PlSmallStr::from_static("set"), &[1.5, f64::NAN]);
        assert_eq!(translate(col("f").is_in(lit(set), false)), None);
    }

    #[test]
    fn negated_float_comparisons_decline() {
        assert_eq!(translate(col("f").lt(lit(1.5)).not()), None);
        assert_eq!(translate(col("f").lt(lit(1.5)).eq(lit(false))), None);
        assert_eq!(translate(col("f").eq(lit(1.5)).neq(lit(true))), None);
    }
}

#[cfg(test)]
mod predicate_position_tests {
    use super::*;
    use delta_kernel::schema::StructField;
    use polars::prelude::{col, lit};

    fn long_schema() -> StructType {
        StructType::try_new([StructField::nullable("id", KernelDataType::LONG)]).unwrap()
    }

    /// `Predicate::from_expr` is a truthiness test, so a non-boolean column
    /// in predicate position would be pushed as `n IS TRUE` and skipped
    /// against stats that mean something else.
    #[test]
    fn non_boolean_column_is_not_a_predicate() {
        assert_eq!(
            polars_expr_to_kernel_predicate(&col("id"), &long_schema()),
            None,
            "a LONG column is not a truth value"
        );
        // The reported route in: `== True` folds the literal away first, so
        // the fold must not hand back a bare truthiness test on a LONG.
        let truthy = Predicate::from_expr(ColumnName::new(["id"]));
        let folded = polars_expr_to_kernel_predicate(&col("id").eq(lit(true)), &long_schema());
        assert_ne!(folded, Some(truthy.clone()));
        assert_ne!(folded, Some(Predicate::not(truthy)));
    }

    /// `Negate` is polars' arithmetic unary minus; `Not` is logical NOT.
    /// Translating the former as the latter inverts the file-skipping
    /// decision, so it must be declined instead.
    #[test]
    fn arithmetic_negate_is_not_logical_not() {
        let schema =
            StructType::try_new([StructField::nullable("flag", KernelDataType::BOOLEAN)]).unwrap();
        let negated = Expr::Function {
            input: vec![col("flag")],
            function: FunctionExpr::Negate,
        };
        assert_eq!(
            polars_expr_to_kernel_predicate(&negated, &schema),
            None,
            "Negate has no kernel predicate form"
        );
        // Logical NOT is unaffected.
        assert_eq!(
            polars_expr_to_kernel_predicate(&col("flag").not(), &schema),
            Some(Predicate::not(Predicate::from_expr(ColumnName::new([
                "flag"
            ]))))
        );
    }

    /// A boolean column still translates — the restriction is on the type,
    /// not on bare column references.
    #[test]
    fn boolean_column_is_a_predicate() {
        let schema =
            StructType::try_new([StructField::nullable("flag", KernelDataType::BOOLEAN)]).unwrap();
        assert_eq!(
            polars_expr_to_kernel_predicate(&col("flag"), &schema),
            Some(Predicate::from_expr(ColumnName::new(["flag"])))
        );
    }
}
