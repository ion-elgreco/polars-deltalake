//! Kernel `Scalar` → polars `Series` / `Expr` / `lit`. Leaf module — no
//! calls into sibling translation modules; recursive within itself for
//! nested compound scalars (`Struct`, `Array`, `Map`).

use delta_kernel::expressions::Scalar;
use delta_kernel::schema::{DataType as KernelDataType, PrimitiveType, StructField};
use delta_kernel::{DeltaResult, Error};
use polars::prelude::as_struct as polars_as_struct;
use polars::prelude::{
    DataFrame, DataType as PlDataType, Expr, Int128Chunked, IntoColumn, IntoLazy, IntoSeries,
    NamedFrom, NewChunkedArray, Series, lit,
};
use polars_utils::pl_str::PlSmallStr;

use crate::consts::{MAP_KEY_FIELD, MAP_VALUE_FIELD};
use crate::errors::to_kernel_err;
use crate::translation::schema::KernelDataTypeExt;

/// Compound scalars (Array/Map/Struct/Decimal) round-trip through
/// `build_series` so an empty list stays `[]` instead of collapsing to null
/// — non-null typed empties are load-bearing for Delta log replay.
pub(super) fn scalar_to_lit(scalar: &Scalar) -> Expr {
    match scalar {
        Scalar::String(s) => lit(s.as_str()),
        Scalar::Long(v) => lit(*v),
        Scalar::Integer(v) => lit(*v),
        Scalar::Short(v) => lit(*v as i32),
        Scalar::Byte(v) => lit(*v as i32),
        Scalar::Float(v) => lit(*v),
        Scalar::Double(v) => lit(*v),
        Scalar::Boolean(v) => lit(*v),
        Scalar::Date(v) => lit(polars::prelude::Scalar::new_date(*v)),
        // Typed literal sidesteps a polars-0.53 cast-folding path that drops
        // the timezone when this scalar is embedded in `as_struct`.
        Scalar::Timestamp(v) => lit(polars::prelude::Scalar::new_datetime(
            *v,
            polars::prelude::TimeUnit::Microseconds,
            Some(polars::prelude::TimeZone::UTC),
        )),
        Scalar::TimestampNtz(v) => lit(polars::prelude::Scalar::new_datetime(
            *v,
            polars::prelude::TimeUnit::Microseconds,
            None,
        )),
        Scalar::Binary(b) => lit(polars::prelude::Series::new(
            PlSmallStr::from_static("__lit__"),
            vec![b.as_slice()],
        )),
        Scalar::Null(dt) => match dt.to_polars() {
            Ok(polars_dt) => lit(polars::prelude::LiteralValue::untyped_null()).cast(polars_dt),
            Err(_) => lit(polars::prelude::LiteralValue::untyped_null()),
        },
        Scalar::Array(_) | Scalar::Map(_) | Scalar::Struct(_) | Scalar::Decimal(_) => {
            build_series("__lit__", &scalar.data_type(), &[scalar])
                .map(lit)
                .unwrap_or_else(|_| lit(polars::prelude::LiteralValue::untyped_null()))
        }
    }
}

/// Mismatched scalars panic — the kernel guarantees scalar-vs-schema
/// agreement at every call site.
pub(crate) fn build_series(
    name: &str,
    dtype: &KernelDataType,
    values: &[&Scalar],
) -> DeltaResult<Series> {
    use PrimitiveType::*;

    let name_pl = PlSmallStr::from_str(name);

    match dtype {
        KernelDataType::Primitive(p) => match p {
            String => Ok(Series::new(
                name_pl,
                values
                    .iter()
                    .map(|s| match s {
                        Scalar::String(v) => Some(v.clone()),
                        Scalar::Null(_) => None,
                        other => panic_mismatch(name, "String", other),
                    })
                    .collect::<Vec<Option<std::string::String>>>(),
            )),
            Long => Ok(Series::new(
                name_pl,
                values
                    .iter()
                    .map(|s| match s {
                        Scalar::Long(v) => Some(*v),
                        Scalar::Null(_) => None,
                        other => panic_mismatch(name, "Long", other),
                    })
                    .collect::<Vec<Option<i64>>>(),
            )),
            Integer => Ok(Series::new(
                name_pl,
                values
                    .iter()
                    .map(|s| match s {
                        Scalar::Integer(v) => Some(*v),
                        Scalar::Null(_) => None,
                        other => panic_mismatch(name, "Integer", other),
                    })
                    .collect::<Vec<Option<i32>>>(),
            )),
            Short => Ok(Series::new(
                name_pl,
                values
                    .iter()
                    .map(|s| match s {
                        Scalar::Short(v) => Some(*v),
                        Scalar::Null(_) => None,
                        other => panic_mismatch(name, "Short", other),
                    })
                    .collect::<Vec<Option<i16>>>(),
            )),
            Byte => Ok(Series::new(
                name_pl,
                values
                    .iter()
                    .map(|s| match s {
                        Scalar::Byte(v) => Some(*v),
                        Scalar::Null(_) => None,
                        other => panic_mismatch(name, "Byte", other),
                    })
                    .collect::<Vec<Option<i8>>>(),
            )),
            Float => Ok(Series::new(
                name_pl,
                values
                    .iter()
                    .map(|s| match s {
                        Scalar::Float(v) => Some(*v),
                        Scalar::Null(_) => None,
                        other => panic_mismatch(name, "Float", other),
                    })
                    .collect::<Vec<Option<f32>>>(),
            )),
            Double => Ok(Series::new(
                name_pl,
                values
                    .iter()
                    .map(|s| match s {
                        Scalar::Double(v) => Some(*v),
                        Scalar::Null(_) => None,
                        other => panic_mismatch(name, "Double", other),
                    })
                    .collect::<Vec<Option<f64>>>(),
            )),
            Boolean => Ok(Series::new(
                name_pl,
                values
                    .iter()
                    .map(|s| match s {
                        Scalar::Boolean(v) => Some(*v),
                        Scalar::Null(_) => None,
                        other => panic_mismatch(name, "Boolean", other),
                    })
                    .collect::<Vec<Option<bool>>>(),
            )),
            Binary => Ok(Series::new(
                name_pl,
                values
                    .iter()
                    .map(|s| match s {
                        Scalar::Binary(v) => Some(v.clone()),
                        Scalar::Null(_) => None,
                        other => panic_mismatch(name, "Binary", other),
                    })
                    .collect::<Vec<Option<Vec<u8>>>>(),
            )),
            Date => {
                let v: Vec<Option<i32>> = values
                    .iter()
                    .map(|s| match s {
                        Scalar::Date(v) => Some(*v),
                        Scalar::Null(_) => None,
                        other => panic_mismatch(name, "Date", other),
                    })
                    .collect();
                let s = Series::new(name_pl, v);
                s.cast(&PlDataType::Date).map_err(to_kernel_err)
            }
            Timestamp | TimestampNtz => {
                let v: Vec<Option<i64>> = values
                    .iter()
                    .map(|s| match s {
                        Scalar::Timestamp(v) | Scalar::TimestampNtz(v) => Some(*v),
                        Scalar::Null(_) => None,
                        other => panic_mismatch(name, "Timestamp", other),
                    })
                    .collect();
                let s = Series::new(name_pl, v);
                let target = match p {
                    Timestamp => PlDataType::Datetime(
                        polars::prelude::TimeUnit::Microseconds,
                        Some(polars::prelude::TimeZone::UTC),
                    ),
                    _ => PlDataType::Datetime(polars::prelude::TimeUnit::Microseconds, None),
                };
                s.cast(&target).map_err(to_kernel_err)
            }
            Decimal(decimal_type) => {
                // `kernel::Decimal::bits()` is already the unscaled integer.
                // `Series::new(Vec<i128>).cast(Decimal(p,s))` would re-apply
                // the scale and multiply by 10^s, so reinterpret the raw
                // bits via `Int128Chunked::into_decimal_unchecked` instead.
                let it = values.iter().map(|s| match s {
                    Scalar::Decimal(d) => Some(d.bits()),
                    Scalar::Null(_) => None,
                    other => panic_mismatch(name, "Decimal", other),
                });
                let ca = Int128Chunked::from_iter_options(name_pl, it);
                Ok(ca
                    .into_decimal_unchecked(
                        decimal_type.precision() as usize,
                        decimal_type.scale() as usize,
                    )
                    .into_series())
            }
        },
        KernelDataType::Struct(struct_type) => {
            use polars::prelude::IntoSeries;
            let fields: Vec<&StructField> = struct_type.fields().collect();
            let row_count = values.len();

            let mut child_series: Vec<Series> = Vec::with_capacity(fields.len());
            for (field_idx, field) in fields.iter().enumerate() {
                // Per-field Null sentinel so null rows borrow `&null_sentinel`
                // instead of cloning a fresh `Scalar::Null(dt.clone())` per row.
                let null_sentinel = Scalar::Null(field.data_type.clone());
                let scalar_refs: Vec<&Scalar> = values
                    .iter()
                    .map(|row_scalar| match row_scalar {
                        Scalar::Struct(sd) => &sd.values()[field_idx],
                        Scalar::Null(_) => &null_sentinel,
                        other => panic_mismatch(name, "Struct", other),
                    })
                    .collect();
                let child = build_series(field.name.as_str(), &field.data_type, &scalar_refs)?;
                child_series.push(child);
            }
            let struct_chunked = polars::prelude::StructChunked::from_series(
                name_pl,
                row_count,
                child_series.iter(),
            )
            .map_err(to_kernel_err)?;
            Ok(struct_chunked.into_series())
        }
        KernelDataType::Array(arr) => {
            let inner_dtype = arr.element_type.to_polars().map_err(to_kernel_err)?;
            let list_dt = PlDataType::List(Box::new(inner_dtype));
            let row_exprs: Vec<Expr> = values
                .iter()
                .map(|s| -> DeltaResult<Expr> {
                    match s {
                        Scalar::Array(arr_data) => {
                            let elements: Vec<&Scalar> = arr_data.array_elements().iter().collect();
                            if elements.is_empty() {
                                empty_typed_list_expr(scalar_to_lit(&Scalar::Null(
                                    arr.element_type.clone(),
                                )))
                            } else {
                                let literals: Vec<Expr> =
                                    elements.iter().map(|e| scalar_to_lit(e)).collect();
                                polars::prelude::concat_list(literals).map_err(to_kernel_err)
                            }
                        }
                        Scalar::Null(_) => Ok(lit(polars::prelude::LiteralValue::untyped_null())
                            .cast(list_dt.clone())),
                        other => panic_mismatch(name, "Array", other),
                    }
                })
                .collect::<DeltaResult<Vec<_>>>()?;
            materialise_per_row_list_series(name, &list_dt, row_exprs)
        }
        KernelDataType::Map(map) => {
            let entries_dtype = PlDataType::Struct(vec![
                polars::prelude::Field::new(
                    PlSmallStr::from_static(MAP_KEY_FIELD),
                    map.key_type.to_polars().map_err(to_kernel_err)?,
                ),
                polars::prelude::Field::new(
                    PlSmallStr::from_static(MAP_VALUE_FIELD),
                    map.value_type.to_polars().map_err(to_kernel_err)?,
                ),
            ]);
            let list_dt = PlDataType::List(Box::new(entries_dtype));
            let row_exprs: Vec<Expr> = values
                .iter()
                .map(|s| -> DeltaResult<Expr> {
                    match s {
                        Scalar::Map(md) => {
                            let pairs = md.pairs();
                            if pairs.is_empty() {
                                empty_typed_list_expr(polars_as_struct(vec![
                                    lit("").alias(MAP_KEY_FIELD),
                                    lit("").alias(MAP_VALUE_FIELD),
                                ]))
                            } else {
                                let entries: Vec<Expr> = pairs
                                    .iter()
                                    .map(|(k, v)| {
                                        polars_as_struct(vec![
                                            scalar_to_lit(k).alias(MAP_KEY_FIELD),
                                            scalar_to_lit(v).alias(MAP_VALUE_FIELD),
                                        ])
                                    })
                                    .collect();
                                polars::prelude::concat_list(entries).map_err(to_kernel_err)
                            }
                        }
                        Scalar::Null(_) => Ok(lit(polars::prelude::LiteralValue::untyped_null())
                            .cast(list_dt.clone())),
                        other => panic_mismatch(name, "Map", other),
                    }
                })
                .collect::<DeltaResult<Vec<_>>>()?;
            materialise_per_row_list_series(name, &list_dt, row_exprs)
        }
        other => Err(Error::Unsupported(format!(
            "build_series: dtype {other:?} for column {name} is not yet implemented"
        ))),
    }
}

/// Polars idiom for a typed-empty list per row: `concat_list([seed]).list().head(0)`.
/// Plain `lit(null)` won't do — kernel's `get_list`/`get_map` distinguish
/// "no data" from "empty container".
pub(crate) fn empty_typed_list_expr(seed: Expr) -> DeltaResult<Expr> {
    Ok(polars::prelude::concat_list(vec![seed])
        .map_err(to_kernel_err)?
        .list()
        .head(lit(0i64)))
}

/// Evaluate one `Expr` per output row in a one-row placeholder DF and
/// `vstack` the results into one Series.
fn materialise_per_row_list_series(
    name: &str,
    list_dt: &PlDataType,
    row_exprs: Vec<Expr>,
) -> DeltaResult<Series> {
    let name_pl = PlSmallStr::from_str(name);
    if row_exprs.is_empty() {
        let empty = DataFrame::empty_with_schema(&polars::prelude::Schema::from_iter([(
            name_pl.clone(),
            list_dt.clone(),
        )]));
        return Ok(empty
            .column(name)
            .map_err(to_kernel_err)?
            .as_materialized_series()
            .clone());
    }
    let placeholder = DataFrame::new(
        1,
        vec![Series::new(PlSmallStr::from_static("__r__"), &[0i64]).into_column()],
    )
    .map_err(to_kernel_err)?;
    let mut frames: Vec<DataFrame> = Vec::with_capacity(row_exprs.len());
    for expr in row_exprs {
        let f = placeholder
            .clone()
            .lazy()
            .select(vec![expr.alias(name_pl.clone())])
            .collect()
            .map_err(to_kernel_err)?;
        frames.push(f);
    }
    let mut combined = frames.remove(0);
    for f in frames.iter() {
        combined = combined.vstack(f).map_err(to_kernel_err)?;
    }
    Ok(combined
        .column(name)
        .map_err(to_kernel_err)?
        .as_materialized_series()
        .clone())
}

fn panic_mismatch(name: &str, expected: &str, got: &Scalar) -> ! {
    panic!(
        "PolarsEvaluationHandler::create_many: column {name} expected {expected} scalars, got {got:?}",
    );
}
