//! Kernel `Predicate` → polars `Expr` (boolean / comparison / junction).
//! Operands of comparison predicates are full `Expression`s, so this module
//! calls back into [`super::expr::translate_expr`] for them.

use delta_kernel::engine_data::EngineData;
use delta_kernel::expressions::{
    BinaryPredicate, BinaryPredicateOp, JunctionPredicate, JunctionPredicateOp, Predicate,
    UnaryPredicate, UnaryPredicateOp,
};
use delta_kernel::schema::StructType;
use delta_kernel::{DeltaResult, Error, PredicateEvaluator};
use polars::prelude::{Expr, IntoLazy, lit};
use polars_utils::pl_str::PlSmallStr;

use crate::consts::KERNEL_OUTPUT_COL;
use crate::engine::PolarsEngineData;
use crate::errors::to_kernel_err;

use super::downcast_engine_data;
use super::expr::translate_expr;

pub(super) struct PolarsPredicateEvaluator {
    pub(super) predicate_expr: Expr,
}

impl PredicateEvaluator for PolarsPredicateEvaluator {
    fn evaluate(&self, batch: &dyn EngineData) -> DeltaResult<Box<dyn EngineData>> {
        let df = downcast_engine_data(batch)?.dataframe().clone();
        // Per kernel contract the result is a single nullable boolean column
        // named "output".
        let result = df
            .lazy()
            .select(vec![
                self.predicate_expr
                    .clone()
                    .alias(PlSmallStr::from_static(KERNEL_OUTPUT_COL)),
            ])
            .collect()
            .map_err(to_kernel_err)?;
        Ok(Box::new(PolarsEngineData::new(result)))
    }
}

/// `input_schema` is threaded so a predicate whose inner expression is a
/// `Transform` can walk the right input fields; parquet-pushdown callers
/// without a schema context pass `None`.
pub(crate) fn translate_predicate(
    pred: &Predicate,
    input_schema: Option<&StructType>,
) -> DeltaResult<Expr> {
    match pred {
        Predicate::BooleanExpression(e) => translate_expr(e, None, input_schema),
        Predicate::Not(inner) => Ok(translate_predicate(inner, input_schema)?.not()),
        Predicate::Unary(u) => translate_unary_predicate(u, input_schema),
        Predicate::Binary(b) => translate_binary_predicate(b, input_schema),
        Predicate::Junction(j) => translate_junction_predicate(j, input_schema),
        Predicate::Opaque(_) => Err(Error::Unsupported(
            "translate_predicate: Opaque predicates are not implemented".into(),
        )),
        Predicate::Unknown(s) => Err(Error::Unsupported(format!(
            "translate_predicate: Unknown predicate {s}"
        ))),
    }
}

fn translate_unary_predicate(
    u: &UnaryPredicate,
    input_schema: Option<&StructType>,
) -> DeltaResult<Expr> {
    let inner = translate_expr(&u.expr, None, input_schema)?;
    Ok(match u.op {
        UnaryPredicateOp::IsNull => inner.is_null(),
    })
}

fn translate_binary_predicate(
    b: &BinaryPredicate,
    input_schema: Option<&StructType>,
) -> DeltaResult<Expr> {
    let lhs = translate_expr(&b.left, None, input_schema)?;
    let rhs = translate_expr(&b.right, None, input_schema)?;
    Ok(match b.op {
        BinaryPredicateOp::LessThan => lhs.lt(rhs),
        BinaryPredicateOp::GreaterThan => lhs.gt(rhs),
        BinaryPredicateOp::Equal => lhs.eq(rhs),
        // Direct null-aware inequality — matches kernel's Distinct semantics 1:1.
        BinaryPredicateOp::Distinct => lhs.neq_missing(rhs),
        BinaryPredicateOp::In => lhs.is_in(rhs, false),
    })
}

fn translate_junction_predicate(
    j: &JunctionPredicate,
    input_schema: Option<&StructType>,
) -> DeltaResult<Expr> {
    if j.preds.is_empty() {
        // Empty junction → identity element: AND() = TRUE (vacuously true),
        // OR() = FALSE (vacuously false). Matches kernel's own evaluator
        // (`finish_eval_pred_junction`).
        return Ok(lit(matches!(j.op, JunctionPredicateOp::And)));
    }
    let mut iter = j.preds.iter().map(|p| translate_predicate(p, input_schema));
    let first = iter.next().expect("non-empty")?;
    iter.try_fold(first, |acc, next| {
        let n = next?;
        Ok(match j.op {
            JunctionPredicateOp::And => acc.and(n),
            JunctionPredicateOp::Or => acc.or(n),
        })
    })
}
