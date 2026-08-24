//! Kernel `Expression` → polars `Expr`. The dispatcher in
//! [`translate_expr`] routes each variant to the appropriate translator;
//! struct/array/scalar leaves stay here, `Transform` defers to
//! [`super::transform`], and `Predicate` defers to [`super::predicate`].

use delta_kernel::expressions::Scalar;
use delta_kernel::expressions::{
    BinaryExpression, BinaryExpressionOp, ColumnName, Expression, ExpressionRef, UnaryExpression,
    UnaryExpressionOp, VariadicExpression, VariadicExpressionOp,
};
use delta_kernel::schema::{DataType as KernelDataType, PrimitiveType, StructType};
use delta_kernel::transform_output_type;
use delta_kernel::transforms::SchemaTransform;
use delta_kernel::{DeltaResult, Error};
use polars::prelude::as_struct as polars_as_struct;
use polars::prelude::{Column, Expr, Field, LiteralValue, Schema, coalesce, col, lit, when};
use polars_utils::pl_str::PlSmallStr;
use std::borrow::Cow;

use crate::consts::{MAP_KEY_FIELD, MAP_VALUE_FIELD};
use crate::errors::to_kernel_err;
use crate::translation::schema::KernelDataTypeExt;

use super::predicate::translate_predicate;
use super::scalar::{build_series, scalar_to_lit};
use super::transform::translate_transform;

/// `output_type` (when known) lets us name struct children correctly —
/// without it polars assigns generic `field_N` names and kernel visitors
/// that look up by name silently fail. `input_schema` is required by
/// `StructPatch` to walk input fields.
pub(crate) fn translate_expr(
    expr: &Expression,
    output_type: Option<&KernelDataType>,
    input_schema: Option<&StructType>,
) -> DeltaResult<Expr> {
    match expr {
        Expression::Literal(scalar) => Ok(scalar_to_lit(scalar)),
        Expression::Column(name) => Ok(column_path_to_expr(name)),
        Expression::Predicate(pred) => translate_predicate(pred, input_schema),
        Expression::Struct(children, nullability) => {
            translate_struct(children, nullability.as_deref(), output_type, input_schema)
        }
        Expression::StructPatch(t) => {
            let target = match output_type {
                Some(KernelDataType::Struct(s)) => s,
                _ => {
                    return Err(Error::Generic(
                        "translate_expr: StructPatch requires a struct output_type".into(),
                    ));
                }
            };
            let schema = input_schema.ok_or_else(|| {
                Error::Generic(
                    "translate_expr: StructPatch requires an input_schema (nested-patch path)"
                        .into(),
                )
            })?;
            translate_transform(t, target, schema)
        }
        Expression::Cast(c) => {
            let inner = translate_expr(&c.expr, None, input_schema)?;
            let target = c.target.to_polars().map_err(to_kernel_err)?;
            Ok(inner.cast(target))
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
            // polars's `json_decode` requires the JSON value type to already
            // match the target; delta-rs writes Date/Timestamp stats as ISO
            // strings, so decode them as String first and let the struct-wide
            // cast lift each temporal field.
            let inner = translate_expr(&p.json_expr, None, input_schema)?;
            let decoded = match StringifyTemporal.transform_struct(&p.output_schema) {
                Cow::Borrowed(_) => {
                    let dt = KernelDataType::Struct(Box::new((*p.output_schema).clone()))
                        .to_polars()
                        .map_err(to_kernel_err)?;
                    inner.str().json_decode(dt)
                }
                Cow::Owned(decode_schema) => {
                    let final_dt = KernelDataType::Struct(Box::new((*p.output_schema).clone()))
                        .to_polars()
                        .map_err(to_kernel_err)?;
                    let decode_dt = KernelDataType::Struct(Box::new(decode_schema))
                        .to_polars()
                        .map_err(to_kernel_err)?;
                    inner.str().json_decode(decode_dt).cast(final_dt)
                }
            };
            Ok(decoded)
        }
        Expression::MapToStruct(m) => translate_map_to_struct(m, output_type, input_schema),
    }
}

/// Schema rewrite that turns `Date` / `Timestamp` / `TimestampNtz` primitives
/// into `String`. Used to relax the `json_decode` target so ISO-string stats
/// from delta-rs decode cleanly; a follow-up struct-wide cast lifts them.
struct StringifyTemporal;

impl<'a> SchemaTransform<'a> for StringifyTemporal {
    transform_output_type!(|'a, T| Cow<'a, T>);

    fn transform_primitive(&mut self, ptype: &'a PrimitiveType) -> Cow<'a, PrimitiveType> {
        match ptype {
            PrimitiveType::Date | PrimitiveType::Timestamp | PrimitiveType::TimestampNtz => {
                Cow::Owned(PrimitiveType::String)
            }
            _ => Cow::Borrowed(ptype),
        }
    }
}

pub(crate) fn column_path_to_expr(name: &ColumnName) -> Expr {
    let mut iter = name.iter();
    let first = iter.next().expect("ColumnName has at least one segment");
    let mut expr = col(PlSmallStr::from_str(first));
    for segment in iter {
        expr = expr.struct_().field_by_name(segment);
    }
    expr
}

/// Maps are stored as `List<Struct<{key, value}>>`, so each output field
/// list-evals the entry whose `key` matches, surfaces its `value`, and
/// parses it into the field's target type per the kernel contract: empty
/// string stays itself for string, becomes empty bytes for binary, and
/// null for every other type.
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
        .map(|f| -> DeltaResult<Expr> {
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
            let raw = map_expr
                .clone()
                .list()
                .eval(value_when_match)
                .list()
                .drop_nulls()
                .list()
                .first();
            Ok(parse_partition_string(raw, &f.data_type)?
                .alias(PlSmallStr::from_str(f.name.as_str())))
        })
        .collect::<DeltaResult<_>>()?;
    Ok(polars_as_struct(children))
}

/// Delta serialized-partition-value parse: `raw` is a nullable string expr.
///
/// Defers to the kernel's own [`PrimitiveType::parse_scalar`], which the
/// `MapToStruct` contract names as the reference, so every spelling it accepts
/// and every `ParseError` it raises match exactly. Reimplementing that grammar
/// out of polars cast and strptime rules cannot: kernel accepts spellings
/// polars rejects (unpadded dates, `%+` offsets) and its lenient counterparts
/// null unparseable values instead of failing the scan.
///
/// The empty string is the contract's one exception and never reaches
/// `parse_scalar`: it stays itself for string, becomes empty bytes for binary
/// and null for every other type.
fn parse_partition_string(raw: Expr, target: &KernelDataType) -> DeltaResult<Expr> {
    let KernelDataType::Primitive(prim) = target else {
        return Err(Error::Generic(format!(
            "MapToStruct: partition column of type {target:?} is not a primitive"
        )));
    };
    let polars_target = target.to_polars().map_err(to_kernel_err)?;
    match prim {
        // Identity under `parse_scalar`, so skip the per-row round trip that
        // partition columns of these two types would spend rebuilding.
        PrimitiveType::String => Ok(raw),
        PrimitiveType::Binary => Ok(raw.cast(polars_target)),
        _ => {
            let prim = prim.clone();
            let kernel_target = target.clone();
            let output = polars_target;
            Ok(raw.map(
                move |column| parse_partition_column(&column, &prim, &kernel_target),
                move |_: &Schema, field: &Field| {
                    Ok(Field::new(field.name().clone(), output.clone()))
                },
            ))
        }
    }
}

/// One `parse_scalar` per value, so a value no kernel-accepted spelling
/// matches fails the scan the way a broken table should.
fn parse_partition_column(
    column: &Column,
    prim: &PrimitiveType,
    target: &KernelDataType,
) -> polars::prelude::PolarsResult<Column> {
    let scalars = column
        .str()?
        .iter()
        .map(|value| match value {
            None => Ok(Scalar::Null(target.clone())),
            // The empty string has no representation in the types that
            // reach here; string and binary keep it via the arms above.
            Some("") => Ok(Scalar::Null(target.clone())),
            Some(value) => prim
                .parse_scalar(value)
                .map_err(|e| polars::prelude::PolarsError::ComputeError(e.to_string().into())),
        })
        .collect::<polars::prelude::PolarsResult<Vec<Scalar>>>()?;
    let series = build_series(
        column.name().as_str(),
        target,
        &scalars.iter().collect::<Vec<_>>(),
    )
    .map_err(|e| polars::prelude::PolarsError::ComputeError(e.to_string().into()))?;
    Ok(Column::from(series))
}

/// `CASE WHEN gate THEN value END` — NULL where `gate` is false or null
/// (polars routes null conditions to `otherwise`).
pub(crate) fn null_gated(gate: Expr, value: Expr) -> Expr {
    when(gate)
        .then(value)
        .otherwise(lit(LiteralValue::untyped_null()))
}

fn translate_struct(
    children: &[ExpressionRef],
    nullability: Option<&Expression>,
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
    let built = polars_as_struct(child_exprs);
    match nullability {
        Some(pred) => Ok(null_gated(translate_expr(pred, None, input_schema)?, built)),
        None => Ok(built),
    }
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
        VariadicExpressionOp::Array => {
            if children.is_empty() {
                return Err(Error::Unsupported(
                    "translate_expr: Array expression with no elements".into(),
                ));
            }
            polars::prelude::concat_list(children.as_slice()).map_err(to_kernel_err)
        }
    }
}

#[cfg(test)]
mod struct_nullability_tests {
    use super::*;
    use delta_kernel::expressions::Predicate;
    use delta_kernel::schema::StructField;
    use polars::prelude::{AnyValue, DataFrame, IntoLazy, df};

    fn eval(expr: &Expression, output_type: &KernelDataType, df: DataFrame) -> Column {
        let translated = translate_expr(expr, Some(output_type), None).unwrap();
        df.lazy()
            .select([translated.alias("out")])
            .collect()
            .unwrap()
            .column("out")
            .unwrap()
            .clone()
    }

    /// `Expression::Struct(children, Some(pred))` is `CASE WHEN pred THEN
    /// struct(...) END`. Mirrors the `file_action_key.deletionVector` gate in
    /// kernel's metadata scan plan.
    #[test]
    fn false_gate_nulls_struct_row() {
        let storage_type = Expression::from(ColumnName::new(["storageType"]));
        let expr = Expression::struct_with_nullability_from(
            [
                storage_type.clone(),
                Expression::from(ColumnName::new(["pathOrInlineDv"])),
            ],
            Expression::from_pred(storage_type.is_not_null()),
        );
        let output_type = KernelDataType::Struct(Box::new(
            StructType::try_new([
                StructField::nullable("storageType", KernelDataType::STRING),
                StructField::nullable("pathOrInlineDv", KernelDataType::STRING),
            ])
            .unwrap(),
        ));
        let df = df! {
            "storageType" => [Some("u"), None],
            "pathOrInlineDv" => [Some("p0"), Some("p1")],
        }
        .unwrap();

        let out = eval(&expr, &output_type, df);
        assert!(!matches!(out.get(0).unwrap(), AnyValue::Null));
        assert!(
            matches!(out.get(1).unwrap(), AnyValue::Null),
            "row with a false gate must be an outer-NULL struct, got {:?}",
            out.get(1).unwrap()
        );
    }

    /// A null gate must null the struct exactly like a false one.
    #[test]
    fn null_gate_nulls_struct_row() {
        let expr = Expression::struct_with_nullability_from(
            [Expression::from(ColumnName::new(["a"]))],
            Expression::from_pred(Predicate::from_expr(Expression::from(ColumnName::new([
                "gate",
            ])))),
        );
        let output_type = KernelDataType::Struct(Box::new(
            StructType::try_new([StructField::nullable("a", KernelDataType::LONG)]).unwrap(),
        ));
        let df = df! {
            "a" => [1i64, 2, 3],
            "gate" => [Some(true), Some(false), None],
        }
        .unwrap();

        let out = eval(&expr, &output_type, df);
        assert!(!matches!(out.get(0).unwrap(), AnyValue::Null));
        assert_eq!(
            out.null_count(),
            2,
            "false and null gates must both produce outer-NULL structs"
        );
    }
}
