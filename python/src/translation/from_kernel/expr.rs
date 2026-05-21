//! Kernel `Expression` → polars `Expr`. The dispatcher in
//! [`translate_expr`] routes each variant to the appropriate translator;
//! struct/array/scalar leaves stay here, `Transform` defers to
//! [`super::transform`], and `Predicate` defers to [`super::predicate`].

use delta_kernel::expressions::{
    BinaryExpression, BinaryExpressionOp, ColumnName, Expression, ExpressionRef, UnaryExpression,
    UnaryExpressionOp, VariadicExpression, VariadicExpressionOp,
};
use delta_kernel::schema::{DataType as KernelDataType, StructType};
use delta_kernel::{DeltaResult, Error};
use polars::prelude::as_struct as polars_as_struct;
use polars::prelude::{Expr, coalesce, col, lit, when};
use polars_utils::pl_str::PlSmallStr;

use crate::consts::{MAP_KEY_FIELD, MAP_VALUE_FIELD};
use crate::errors::to_kernel_err;
use crate::translation::schema::KernelDataTypeExt;

use super::predicate::translate_predicate;
use super::scalar::scalar_to_lit;
use super::transform::translate_transform;

/// `output_type` (when known) lets us name struct children correctly —
/// without it polars assigns generic `field_N` names and kernel visitors
/// that look up by name silently fail. `input_schema` is required by
/// `Transform` to walk input fields.
pub(super) fn translate_expr(
    expr: &Expression,
    output_type: Option<&KernelDataType>,
    input_schema: Option<&StructType>,
) -> DeltaResult<Expr> {
    match expr {
        Expression::Literal(scalar) => Ok(scalar_to_lit(scalar)),
        Expression::Column(name) => Ok(column_path_to_expr(name)),
        Expression::Predicate(pred) => translate_predicate(pred, input_schema),
        Expression::Struct(children, _) => translate_struct(children, output_type, input_schema),
        Expression::Transform(t) => {
            let target = match output_type {
                Some(KernelDataType::Struct(s)) => s,
                _ => {
                    return Err(Error::Generic(
                        "translate_expr: Transform requires a struct output_type".into(),
                    ));
                }
            };
            let schema = input_schema.ok_or_else(|| {
                Error::Generic(
                    "translate_expr: Transform requires an input_schema (nested-Transform path)"
                        .into(),
                )
            })?;
            translate_transform(t, target, schema)
        }
        Expression::Unary(u) => translate_unary_expr(u, input_schema),
        Expression::Binary(b) => translate_binary_expr(b, input_schema),
        Expression::Variadic(v) => translate_variadic_expr(v, input_schema),
        Expression::Opaque(_) => Err(Error::Unsupported(
            "translate_expr: Opaque expressions are not implemented".into(),
        )),
        Expression::Unknown(s) => Err(Error::Unsupported(format!(
            "translate_expr: Unknown expression {s}"
        ))),
        Expression::ParseJson(p) => {
            let inner = translate_expr(&p.json_expr, None, input_schema)?;
            let output_dt = KernelDataType::Struct(Box::new((*p.output_schema).clone()))
                .to_polars()
                .map_err(to_kernel_err)?;
            Ok(inner.str().json_decode(output_dt))
        }
        Expression::MapToStruct(m) => translate_map_to_struct(m, output_type, input_schema),
    }
}

pub(super) fn column_path_to_expr(name: &ColumnName) -> Expr {
    let mut iter = name.iter();
    let first = iter.next().expect("ColumnName has at least one segment");
    let mut expr = col(PlSmallStr::from_str(first));
    for segment in iter {
        expr = expr.struct_().field_by_name(segment);
    }
    expr
}

/// Maps are stored as `List<Struct<{key, value}>>`, so each output field
/// list-evals the entry whose `key` matches and surfaces its `value`.
fn translate_map_to_struct(
    m: &delta_kernel::expressions::MapToStructExpression,
    output_type: Option<&KernelDataType>,
    input_schema: Option<&StructType>,
) -> DeltaResult<Expr> {
    let target = match output_type {
        Some(KernelDataType::Struct(s)) => s,
        _ => {
            return Err(Error::Generic(
                "translate_expr: MapToStruct requires a struct output_type".into(),
            ));
        }
    };
    let map_expr = translate_expr(&m.map_expr, None, input_schema)?;
    let children: Vec<Expr> = target
        .fields()
        .map(|f| {
            let key_match = polars::prelude::Expr::Element
                .struct_()
                .field_by_name(MAP_KEY_FIELD)
                .eq(lit(f.name.as_str()));
            let value_when_match = when(key_match)
                .then(
                    polars::prelude::Expr::Element
                        .struct_()
                        .field_by_name(MAP_VALUE_FIELD),
                )
                .otherwise(lit(polars::prelude::LiteralValue::untyped_null()));
            map_expr
                .clone()
                .list()
                .eval(value_when_match)
                .list()
                .drop_nulls()
                .list()
                .first()
                .alias(PlSmallStr::from_str(f.name.as_str()))
        })
        .collect();
    Ok(polars_as_struct(children))
}

fn translate_struct(
    children: &[ExpressionRef],
    output_type: Option<&KernelDataType>,
    input_schema: Option<&StructType>,
) -> DeltaResult<Expr> {
    let names_and_types: Vec<(Option<PlSmallStr>, Option<&KernelDataType>)> = match output_type {
        Some(KernelDataType::Struct(s)) => s
            .fields()
            .map(|f| {
                (
                    Some(PlSmallStr::from_str(f.name.as_str())),
                    Some(&f.data_type),
                )
            })
            .collect(),
        _ => children.iter().map(|_| (None, None)).collect(),
    };
    if names_and_types.len() != children.len() {
        return Err(Error::Generic(format!(
            "translate_struct: struct child count {} != schema field count {}",
            children.len(),
            names_and_types.len()
        )));
    }
    let child_exprs: Vec<Expr> = children
        .iter()
        .zip(names_and_types.iter())
        .map(|(c, (name, child_type))| {
            let inner = translate_expr(c.as_ref(), *child_type, input_schema)?;
            Ok(match name {
                Some(n) => inner.alias(n.clone()),
                None => inner,
            })
        })
        .collect::<DeltaResult<_>>()?;
    Ok(polars_as_struct(child_exprs))
}

fn translate_unary_expr(
    u: &UnaryExpression,
    input_schema: Option<&StructType>,
) -> DeltaResult<Expr> {
    let inner = translate_expr(&u.expr, None, input_schema)?;
    match u.op {
        UnaryExpressionOp::ToJson => Ok(inner.struct_().json_encode()),
    }
}

fn translate_binary_expr(
    b: &BinaryExpression,
    input_schema: Option<&StructType>,
) -> DeltaResult<Expr> {
    let lhs = translate_expr(&b.left, None, input_schema)?;
    let rhs = translate_expr(&b.right, None, input_schema)?;
    Ok(match b.op {
        BinaryExpressionOp::Plus => lhs + rhs,
        BinaryExpressionOp::Minus => lhs - rhs,
        BinaryExpressionOp::Multiply => lhs * rhs,
        BinaryExpressionOp::Divide => lhs / rhs,
    })
}

fn translate_variadic_expr(
    v: &VariadicExpression,
    input_schema: Option<&StructType>,
) -> DeltaResult<Expr> {
    let children: Vec<Expr> = v
        .exprs
        .iter()
        .map(|e| translate_expr(e, None, input_schema))
        .collect::<DeltaResult<_>>()?;
    match v.op {
        VariadicExpressionOp::Coalesce => Ok(coalesce(&children)),
    }
}
