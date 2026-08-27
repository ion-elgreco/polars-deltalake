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
use polars::prelude::{
    Column, DataType as PlDataType, Expr, Field, LiteralValue, NamedFrom, Schema, Series,
    StringChunked, Utf8JsonPathImpl, coalesce, col, lit, when,
};
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
            let raw = translate_expr(&p.json_expr, None, input_schema)?;
            let final_dt = KernelDataType::Struct(Box::new((*p.output_schema).clone()))
                .to_polars()
                .map_err(to_kernel_err)?;
            Ok(match StringifyTemporal.transform_struct(&p.output_schema) {
                Cow::Borrowed(_) => json_decode_lenient(raw, final_dt),
                Cow::Owned(decode_schema) => {
                    let decode_dt = KernelDataType::Struct(Box::new(decode_schema))
                        .to_polars()
                        .map_err(to_kernel_err)?;
                    json_decode_lenient(raw, decode_dt).cast(final_dt)
                }
            })
        }
        Expression::MapToStruct(m) => translate_map_to_struct(m, output_type, input_schema),
    }
}

/// The `ParseJson` contract makes unparsable input decode to NULL rather
/// than fail the query, and polars' `json_decode` does neither: it is an
/// NDJSON deserializer, so a blank value is not a row at all and the *row*
/// disappears — silently dropping the add action and every file it names —
/// while one malformed value fails the whole batch.
///
/// Blanks are nulled up front (cheap, and null decodes to a null value), and
/// a batch that still fails is retried per value so only the offenders go
/// NULL.
fn json_decode_lenient(raw: Expr, dtype: PlDataType) -> Expr {
    let output = dtype.clone();
    raw.map(
        move |column| {
            let ca = column.str()?;
            let blank = |s: &str| s.trim().is_empty();
            let blanked: Cow<StringChunked> = if ca.iter().any(|v| v.is_some_and(blank)) {
                Cow::Owned(ca.iter().map(|v| v.filter(|s| !blank(s))).collect())
            } else {
                Cow::Borrowed(ca)
            };
            // The decoder is line-oriented, so a value holding several
            // documents inflates the output; a length mismatch means some
            // value is unparsable-as-one-document and must go NULL.
            let n = blanked.len();
            let batch = match blanked.json_decode(Some(dtype.clone()), None) {
                Ok(series) if series.len() == n => Some(series),
                Ok(_) => None,
                // A whole-batch failure is indistinguishable here from one bad
                // value, but it is also how a wrong target dtype shows up —
                // and that nulls every stat, silently disabling file skipping.
                Err(e) => {
                    tracing::debug!(
                        target: "polars_deltalake::parse_json",
                        error = %e,
                        rows = n,
                        "batch json_decode failed; retrying per value",
                    );
                    None
                }
            };
            let decoded = match batch {
                Some(series) => series,
                None => {
                    let mut out = Series::new_empty(PlSmallStr::EMPTY, &dtype);
                    for value in blanked.iter() {
                        let one = Series::new(PlSmallStr::EMPTY, [value]);
                        let decoded = one
                            .str()
                            .and_then(|ca| ca.json_decode(Some(dtype.clone()), None))
                            .ok()
                            .filter(|s| s.len() == 1)
                            .unwrap_or_else(|| Series::full_null(PlSmallStr::EMPTY, 1, &dtype));
                        out.append(&decoded)?;
                    }
                    // One chunk per value otherwise; every downstream op pays.
                    out.rechunk()
                }
            };
            Ok(Column::from(decoded.with_name(column.name().clone())))
        },
        move |_: &Schema, field: &Field| Ok(Field::new(field.name().clone(), output.clone())),
    )
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
            let value = polars::prelude::Expr::Element
                .struct_()
                .field_by_name(MAP_VALUE_FIELD);
            // Rightmost matching entry with null values kept — kernel's
            // duplicate-key contract (the arrow reference takes the last).
            let raw = map_expr
                .clone()
                .list()
                .eval(value.filter(key_match))
                .list()
                .last();
            Ok(parse_partition_string(raw, &f.data_type)?
                .alias(PlSmallStr::from_str(f.name.as_str())))
        })
        .collect::<DeltaResult<_>>()?;
    // The arrow reference propagates the map's outer null buffer: a NULL
    // map row is a NULL struct row, not a valid struct of nulls.
    Ok(null_gated(
        map_expr.is_not_null(),
        as_struct_checked(children, "MapToStruct")?,
    ))
}

fn to_pl_err(e: impl std::fmt::Display) -> polars::prelude::PolarsError {
    polars::prelude::PolarsError::ComputeError(e.to_string().into())
}

/// Delta serialized-partition-value parse: `raw` is a nullable string expr.
///
/// Defers to the kernel's own [`PrimitiveType::parse_scalar`], which the
/// `MapToStruct` contract names as the reference, so every spelling it accepts
/// and every `ParseError` it raises match exactly. Reimplementing that grammar
/// out of polars cast and strptime rules cannot: kernel accepts spellings
/// polars rejects (unpadded dates, `%+` offsets) and its lenient counterparts
/// null unparsable values instead of failing the scan.
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
    // Identity under `parse_scalar`, so skip the per-row round trip a string
    // partition column would spend rebuilding itself.
    if matches!(prim, PrimitiveType::String) {
        return Ok(raw);
    }
    let output = target.to_polars().map_err(to_kernel_err)?;
    let kernel_target = target.clone();
    Ok(raw.map(
        move |column| parse_partition_column(&column, &kernel_target),
        move |_: &Schema, field: &Field| Ok(Field::new(field.name().clone(), output.clone())),
    ))
}

/// One `parse_scalar` per value, so a value no kernel-accepted spelling
/// matches fails the scan the way a broken table should. Shared with the
/// partition-pruning frame so skipping and projection read the same grammar.
pub(crate) fn parse_partition_column(
    column: &Column,
    target: &KernelDataType,
) -> polars::prelude::PolarsResult<Column> {
    let KernelDataType::Primitive(prim) = target else {
        return Err(to_pl_err(format!(
            "partition column of type {target:?} is not a primitive"
        )));
    };
    match prim {
        // Identity under `parse_scalar`; the empty string stays itself here
        // rather than becoming null.
        PrimitiveType::String => return Ok(column.clone()),
        PrimitiveType::Binary => return column.cast(&target.to_polars().map_err(to_pl_err)?),
        _ => {}
    }
    let scalars = column
        .str()?
        .iter()
        .map(|value| match value {
            None => Ok(Scalar::Null(target.clone())),
            // The empty string has no representation in the types that
            // reach here; string and binary keep it via the arms above.
            Some("") => Ok(Scalar::Null(target.clone())),
            Some(value) => prim.parse_scalar(value).map_err(to_pl_err),
        })
        .collect::<polars::prelude::PolarsResult<Vec<Scalar>>>()?;
    let series = build_series(
        column.name().as_str(),
        target,
        &scalars.iter().collect::<Vec<_>>(),
    )
    .map_err(to_pl_err)?;
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
    let built = as_struct_checked(child_exprs, "translate_struct")?;
    match nullability {
        Some(pred) => Ok(null_gated(translate_expr(pred, None, input_schema)?, built)),
        None => Ok(built),
    }
}

/// Kernel permits a zero-field struct; polars' `as_struct` `assert!`s on an
/// empty expression list, and a panic here unwinds into pyo3.
pub(crate) fn as_struct_checked(children: Vec<Expr>, context: &str) -> DeltaResult<Expr> {
    if children.is_empty() {
        return Err(Error::Unsupported(format!(
            "{context}: a struct with no fields has no polars representation"
        )));
    }
    Ok(polars_as_struct(children))
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
        // Kernel divides integers as integers and errors on a zero divisor;
        // polars `/` is always true division, so `7 / 2` would read back 3.5
        // as a Float64 where the plan declares LONG 3, and `7 / 0` as inf.
        // Kernel emits no Divide today, so refuse rather than diverge
        // silently — the day it does, this names what to implement.
        BinaryExpressionOp::Divide => {
            return Err(Error::Unsupported(
                "translate_expr: Divide has no polars form matching kernel's \
                 integer-division semantics"
                    .into(),
            ));
        }
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
            // `concat_list` splices list-typed inputs entry-wise; kernel
            // ARRAY makes each input a single element. Refuse the statically
            // detectable list inputs instead of building the wrong shape.
            if v.exprs.iter().any(|e| is_statically_list(e, input_schema)) {
                return Err(Error::Unsupported(
                    "translate_expr: ARRAY over a list-typed input is not implemented".into(),
                ));
            }
            polars::prelude::concat_list(children.as_slice()).map_err(to_kernel_err)
        }
    }
}

fn is_statically_list(e: &Expression, schema: Option<&StructType>) -> bool {
    match e {
        Expression::Literal(Scalar::Array(_)) => true,
        Expression::Literal(Scalar::Null(dt)) => matches!(dt, KernelDataType::Array(_)),
        Expression::Column(name) => resolve_column_dtype(name, schema)
            .is_some_and(|dt| matches!(dt, KernelDataType::Array(_))),
        Expression::Cast(c) => matches!(c.target, KernelDataType::Array(_)),
        // ARRAY builds a list, and COALESCE is list-typed when its arms are —
        // a catch-all here would let `ARRAY(ARRAY(1,2), ARRAY(3,4))` splice
        // into `[1,2,3,4]` instead of erroring.
        Expression::Variadic(v) => match v.op {
            VariadicExpressionOp::Array => true,
            VariadicExpressionOp::Coalesce => v.exprs.iter().any(|e| is_statically_list(e, schema)),
        },
        _ => false,
    }
}

fn resolve_column_dtype<'a>(
    name: &ColumnName,
    schema: Option<&'a StructType>,
) -> Option<&'a KernelDataType> {
    crate::translation::schema::resolve_leaf_dtype(schema?, name.iter())
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

#[cfg(test)]
mod map_to_struct_tests {
    use delta_kernel::expressions::{ColumnName, MapToStructExpression};
    use delta_kernel::schema::StructField;
    use polars::prelude::{AnyValue, IntoLazy};

    use super::*;

    fn map_to_struct_p() -> (Expression, KernelDataType) {
        let expr = Expression::MapToStruct(MapToStructExpression {
            map_expr: Box::new(Expression::from(ColumnName::new(["m"]))),
        });
        let output_type = KernelDataType::Struct(Box::new(
            StructType::try_new([StructField::nullable("p", KernelDataType::STRING)]).unwrap(),
        ));
        (expr, output_type)
    }

    /// Kernel resolves duplicate map keys to the rightmost entry including
    /// null values; leftmost-non-null diverges on both counts.
    #[test]
    fn duplicate_keys_resolve_to_rightmost_entry() {
        let lines = concat!(
            "{\"m\":[{\"key\":\"p\",\"value\":\"1\"},{\"key\":\"p\",\"value\":\"2\"}]}\n",
            "{\"m\":[{\"key\":\"p\",\"value\":\"1\"},{\"key\":\"p\",\"value\":null}]}\n",
            "{\"m\":[{\"key\":\"p\",\"value\":\"1\"}]}\n",
        );
        let df = crate::engine::parse_ndjson_inferred(lines.as_bytes()).unwrap();
        let (expr, output_type) = map_to_struct_p();
        let translated = translate_expr(&expr, Some(&output_type), None).unwrap();
        let out = df
            .lazy()
            .select([translated.struct_().field_by_name("p").alias("p")])
            .collect()
            .unwrap();
        let p = out.column("p").unwrap();
        assert_eq!(p.get(0).unwrap(), AnyValue::String("2"), "rightmost wins");
        assert!(
            matches!(p.get(1).unwrap(), AnyValue::Null),
            "a rightmost null value stays null, got {:?}",
            p.get(1).unwrap()
        );
        assert_eq!(p.get(2).unwrap(), AnyValue::String("1"));
    }

    /// A NULL map row must yield a NULL struct row, not a valid struct of
    /// null children — the arrow reference propagates the null buffer.
    #[test]
    fn null_map_row_yields_null_struct() {
        let lines = concat!(
            "{\"m\":[{\"key\":\"p\",\"value\":\"1\"}]}\n",
            "{\"m\":null}\n",
        );
        let df = crate::engine::parse_ndjson_inferred(lines.as_bytes()).unwrap();
        let (expr, output_type) = map_to_struct_p();
        let translated = translate_expr(&expr, Some(&output_type), None).unwrap();
        let out = df
            .lazy()
            .select([translated.alias("out")])
            .collect()
            .unwrap();
        let col = out.column("out").unwrap();
        assert!(!matches!(col.get(0).unwrap(), AnyValue::Null));
        assert!(
            matches!(col.get(1).unwrap(), AnyValue::Null),
            "NULL map row must be an outer-NULL struct, got {:?}",
            col.get(1).unwrap()
        );
    }
}

#[cfg(test)]
mod array_variadic_tests {
    use delta_kernel::expressions::{ColumnName, VariadicExpression, VariadicExpressionOp};
    use delta_kernel::schema::{ArrayType, StructField};
    use polars::prelude::IntoLazy;

    use super::*;

    /// `concat_list` splices list-typed inputs entry-wise; kernel ARRAY makes
    /// each input one element — statically list-typed inputs must be refused.
    #[test]
    fn list_typed_input_is_refused() {
        let schema = StructType::try_new([StructField::nullable(
            "l",
            KernelDataType::Array(Box::new(ArrayType::new(KernelDataType::LONG, true))),
        )])
        .unwrap();
        let expr = Expression::Variadic(VariadicExpression {
            op: VariadicExpressionOp::Array,
            exprs: vec![Expression::from(ColumnName::new(["l"]))],
        });
        assert!(translate_expr(&expr, None, Some(&schema)).is_err());
    }

    #[test]
    fn scalar_inputs_build_one_element_each() {
        let expr = Expression::Variadic(VariadicExpression {
            op: VariadicExpressionOp::Array,
            exprs: vec![Expression::literal(1i64), Expression::literal(2i64)],
        });
        let translated = translate_expr(&expr, None, None).unwrap();
        let df = polars::df!("x" => [0i64]).unwrap();
        let out = df
            .lazy()
            .select([translated.alias("out")])
            .collect()
            .unwrap();
        let first = out.column("out").unwrap().list().unwrap().get_as_series(0);
        assert_eq!(first.unwrap().len(), 2);
    }
}

#[cfg(test)]
mod parse_json_tests {
    use std::sync::Arc;

    use delta_kernel::schema::StructField;
    use polars::prelude::{AnyValue, IntoLazy, df};

    use super::*;

    /// Kernel contract: unparsable input, which includes the empty string,
    /// must yield NULL rather than failing the whole batch.
    #[test]
    fn empty_json_string_decodes_to_null() {
        let schema = Arc::new(
            StructType::try_new([StructField::nullable("a", KernelDataType::LONG)]).unwrap(),
        );
        let expr = Expression::parse_json(ColumnName::new(["stats"]), schema.clone());
        let output_type = KernelDataType::Struct(Box::new((*schema).clone()));
        let frame = df!("stats" => [Some("{\"a\":1}"), Some(""), None]).unwrap();

        let translated = translate_expr(&expr, Some(&output_type), None).unwrap();
        let out = frame
            .lazy()
            .select([translated.alias("out")])
            .collect()
            .unwrap();
        let col = out.column("out").unwrap();

        assert_eq!(col.len(), 3);
        assert!(!matches!(col.get(0).unwrap(), AnyValue::Null));
        assert!(
            matches!(col.get(1).unwrap(), AnyValue::Null),
            "empty string must decode to NULL, got {:?}",
            col.get(1).unwrap()
        );
        assert!(matches!(col.get(2).unwrap(), AnyValue::Null));
    }

    /// A value holding two comma-separated documents is one unparsable JSON
    /// value, but polars' NDJSON decoder happily emits two rows for it. The
    /// contract is one output row per input value, offenders going NULL.
    #[test]
    fn multi_document_value_decodes_to_null_not_extra_rows() {
        let schema = Arc::new(
            StructType::try_new([StructField::nullable("a", KernelDataType::LONG)]).unwrap(),
        );
        let expr = Expression::parse_json(ColumnName::new(["stats"]), schema.clone());
        let output_type = KernelDataType::Struct(Box::new((*schema).clone()));
        let frame = df!("stats" => [Some("{\"a\":1},{\"a\":2}"), Some("{\"a\":5}")]).unwrap();

        let translated = translate_expr(&expr, Some(&output_type), None).unwrap();
        let out = frame
            .lazy()
            .select([translated.alias("out")])
            .collect()
            .unwrap();
        let col = out.column("out").unwrap();

        assert_eq!(col.len(), 2, "one output row per input value");
        assert!(
            matches!(col.get(0).unwrap(), AnyValue::Null),
            "multi-document value must decode to NULL, got {:?}",
            col.get(0).unwrap()
        );
        assert!(!matches!(col.get(1).unwrap(), AnyValue::Null));
    }

    /// Same contract through the per-value retry arm: a malformed value
    /// forces the batch fallback, and a multi-document value inside that
    /// loop must still contribute exactly one (NULL) row.
    #[test]
    fn retry_arm_keeps_one_row_per_value() {
        let schema = Arc::new(
            StructType::try_new([StructField::nullable("a", KernelDataType::LONG)]).unwrap(),
        );
        let expr = Expression::parse_json(ColumnName::new(["stats"]), schema.clone());
        let output_type = KernelDataType::Struct(Box::new((*schema).clone()));
        let frame =
            df!("stats" => [Some("not json"), Some("{\"a\":1},{\"a\":2}"), Some("{\"a\":5}")])
                .unwrap();

        let translated = translate_expr(&expr, Some(&output_type), None).unwrap();
        let out = frame
            .lazy()
            .select([translated.alias("out")])
            .collect()
            .unwrap();
        let col = out.column("out").unwrap();

        assert_eq!(col.len(), 3, "one output row per input value");
        assert!(matches!(col.get(0).unwrap(), AnyValue::Null));
        assert!(matches!(col.get(1).unwrap(), AnyValue::Null));
        assert!(!matches!(col.get(2).unwrap(), AnyValue::Null));
    }
}

#[cfg(test)]
mod column_dtype_tests {
    use delta_kernel::schema::StructField;

    use super::*;

    /// `a.b` where `a` is not a struct has no children to resolve. The walk
    /// must not stay at the parent level and hand back the sibling `b`.
    #[test]
    fn non_struct_midpath_resolves_to_none() {
        let schema = StructType::try_new([
            StructField::nullable("a", KernelDataType::LONG),
            StructField::nullable("b", KernelDataType::STRING),
        ])
        .unwrap();
        assert_eq!(
            resolve_column_dtype(&ColumnName::new(["a"]), Some(&schema)),
            Some(&KernelDataType::LONG)
        );
        assert!(resolve_column_dtype(&ColumnName::new(["a", "b"]), Some(&schema)).is_none());
    }
}

#[cfg(test)]
mod partition_parse_agreement_tests {
    use super::*;

    /// The Expr-level parser delegates to the Column-level one so projection
    /// and partition pruning cannot disagree about the same file. Pin that
    /// on the types whose handling used to be spelled out in both.
    #[test]
    fn expr_and_column_parsers_agree() {
        use polars::prelude::{DataFrame, IntoLazy, NamedFrom, col};

        for target in [
            KernelDataType::STRING,
            KernelDataType::BINARY,
            KernelDataType::LONG,
            KernelDataType::DATE,
        ] {
            let raw: Vec<Option<&str>> = match target {
                KernelDataType::LONG => vec![Some("7"), Some(""), None],
                KernelDataType::DATE => vec![Some("2024-01-01"), Some(""), None],
                _ => vec![Some("x"), Some(""), None],
            };
            let column = Column::new("p".into(), Series::new("p".into(), raw.clone()));
            let direct = parse_partition_column(&column, &target).unwrap();

            let via_expr = DataFrame::new(3, vec![column])
                .unwrap()
                .lazy()
                .select([parse_partition_string(col("p"), &target)
                    .unwrap()
                    .alias("p")])
                .collect()
                .unwrap();
            let via_expr = via_expr.column("p").unwrap();

            assert_eq!(via_expr.dtype(), direct.dtype(), "dtype for {target:?}");
            assert!(
                via_expr.equals_missing(&direct),
                "values disagree for {target:?}"
            );
        }
    }
}

#[cfg(test)]
mod array_shape_tests {
    use delta_kernel::expressions::{VariadicExpression, VariadicExpressionOp};

    use super::*;

    /// `concat_list` splices a list-typed input entry-wise, so an ARRAY over
    /// list-typed children would build `[1,2,3,4]` where kernel means
    /// `[[1,2],[3,4]]`. Every statically list-typed shape must decline.
    #[test]
    fn array_over_list_typed_children_declines() {
        let inner = || {
            Expression::Variadic(VariadicExpression {
                op: VariadicExpressionOp::Array,
                exprs: vec![Expression::literal(1i64), Expression::literal(2i64)],
            })
        };
        for exprs in [
            vec![inner(), inner()],
            vec![
                Expression::literal(Scalar::Null(KernelDataType::Array(Box::new(
                    delta_kernel::schema::ArrayType::new(KernelDataType::LONG, true),
                )))),
                Expression::literal(1i64),
            ],
        ] {
            let nested = VariadicExpression {
                op: VariadicExpressionOp::Array,
                exprs,
            };
            assert!(
                translate_variadic_expr(&nested, None).is_err(),
                "a list-typed ARRAY input must decline, not splice",
            );
        }
    }
}

#[cfg(test)]
mod arithmetic_tests {
    use super::*;

    /// Kernel divides LONG by LONG as integers and errors on a zero divisor
    /// (`evaluate_expression`'s `div`); polars `/` is true division, so a
    /// translated `Divide` would answer 3.5 where the plan declares LONG 3.
    #[test]
    fn divide_declines_rather_than_true_divide() {
        let expr = BinaryExpression {
            op: BinaryExpressionOp::Divide,
            left: Box::new(Expression::from(ColumnName::new(["a"]))),
            right: Box::new(Expression::literal(2i64)),
        };
        assert!(
            translate_binary_expr(&expr, None).is_err(),
            "Divide has no polars form matching kernel's integer semantics",
        );
    }

    /// The other three arms agree with kernel on LONG operands and stay wired.
    #[test]
    fn plus_minus_multiply_translate() {
        for op in [
            BinaryExpressionOp::Plus,
            BinaryExpressionOp::Minus,
            BinaryExpressionOp::Multiply,
        ] {
            let expr = BinaryExpression {
                op,
                left: Box::new(Expression::from(ColumnName::new(["a"]))),
                right: Box::new(Expression::literal(2i64)),
            };
            assert!(translate_binary_expr(&expr, None).is_ok(), "{op:?}");
        }
    }
}
