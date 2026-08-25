//! Kernel `Scalar` → polars `Series` / `Expr` / `lit`. Leaf module — no
//! calls into sibling translation modules; recursive within itself for
//! nested compound scalars (`Struct`, `Array`, `Map`).

use delta_kernel::expressions::Scalar;
use delta_kernel::schema::{DataType as KernelDataType, PrimitiveType, StructField};
use delta_kernel::{DeltaResult, Error};
use polars::prelude::as_struct as polars_as_struct;
use polars::prelude::{
    AnyValue, DataFrame, DataType as PlDataType, Expr, Int128Chunked, IntoColumn, IntoLazy,
    IntoSeries, NamedFrom, NewChunkedArray, Scalar as PolarsScalar, Series, TimeUnit, TimeZone,
    lit,
};
use polars_utils::pl_str::PlSmallStr;

use crate::consts::{MAP_KEY_FIELD, MAP_VALUE_FIELD};
use crate::errors::to_kernel_err;
use crate::translation::schema::KernelDataTypeExt;

/// `series[row]` as a typed literal aliased to `name` — the shared
/// row-literal builder. `series` must already carry the output dtype: the
/// cast belongs on the series once, not on every row's literal, and a bare
/// literal is what the scan's fast path reads back without a polars round
/// trip.
pub(crate) fn series_value_lit(
    series: &polars::prelude::Series,
    row: usize,
    name: &str,
) -> polars::prelude::PolarsResult<Expr> {
    let value = series.get(row)?.into_static();
    let scalar = PolarsScalar::new(series.dtype().clone(), value);
    Ok(lit(scalar).alias(PlSmallStr::from_str(name)))
}

/// One aliased literal per (row, field) — the per-file constant scaffold
/// shared by partition literals and DynamicScan file constants. Each series
/// must already carry the output dtype (cast once per field, not per row).
pub(crate) fn per_row_literals(
    series_per_field: &[polars::prelude::Series],
    names: &[&str],
    row_count: usize,
) -> polars::prelude::PolarsResult<Vec<Vec<Expr>>> {
    (0..row_count)
        .map(|row| {
            series_per_field
                .iter()
                .zip(names)
                .map(|(series, name)| series_value_lit(series, row, name))
                .collect()
        })
        .collect()
}

/// Compound scalars (Array/Map/Struct/Decimal) round-trip through
/// `build_series` so an empty list stays `[]` instead of collapsing to null
/// — non-null typed empties are load-bearing for Delta log replay.
pub(crate) fn scalar_to_lit(scalar: &Scalar) -> Expr {
    match scalar {
        Scalar::String(s) => lit(s.as_str()),
        Scalar::Long(v) => lit(*v),
        Scalar::Integer(v) => lit(*v),
        // Typed, not widened: `build_series` derives its list dtype from the
        // declared element type, so an Int32 literal would disagree with a
        // `Array<Short>` / `Array<Byte>` column's null rows.
        Scalar::Short(_) | Scalar::Byte(_) => {
            lit(try_to_polars_scalar(scalar).expect("integral scalars are always representable"))
        }
        Scalar::Float(v) => lit(*v),
        Scalar::Double(v) => lit(*v),
        Scalar::Boolean(v) => lit(*v),
        Scalar::Date(v) => lit(polars::prelude::Scalar::new_date(*v)),
        // Typed literal sidesteps a polars cast-folding path that drops the
        // timezone when this scalar is embedded in `as_struct`.
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
        // Intervals only occur in kernel-side expression evaluation, never
        // in Delta data; the dtype decision (day-time µs → Duration,
        // year-month → typed Int32 month count) lives in
        // `try_to_polars_scalar`.
        Scalar::IntervalDayTime(_) | Scalar::IntervalYearMonth(_) => lit(
            try_to_polars_scalar(scalar).expect("interval scalars are always representable"),
        ),
        Scalar::Array(_) | Scalar::Map(_) | Scalar::Struct(_) | Scalar::Decimal(_) => {
            build_series("__lit__", &scalar.data_type(), &[scalar])
                .map(lit)
                .unwrap_or_else(|_| lit(polars::prelude::LiteralValue::untyped_null()))
        }
    }
}

/// Primitive kernel scalars → polars `Scalar` carrying its own dtype, so
/// `Column::new_scalar` can build a `ScalarColumn` without going through the
/// lazy planner. Returns `None` for compound / binary scalars; callers fall
/// back to the lazy `lit()` path.
pub(crate) fn try_to_polars_scalar(scalar: &Scalar) -> Option<PolarsScalar> {
    let s = match scalar {
        Scalar::String(s) => PolarsScalar::new(
            PlDataType::String,
            AnyValue::StringOwned(PlSmallStr::from_str(s.as_str())),
        ),
        Scalar::Long(v) => PolarsScalar::new(PlDataType::Int64, AnyValue::Int64(*v)),
        Scalar::Integer(v) => PolarsScalar::new(PlDataType::Int32, AnyValue::Int32(*v)),
        Scalar::Short(v) => PolarsScalar::new(PlDataType::Int16, AnyValue::Int16(*v)),
        Scalar::Byte(v) => PolarsScalar::new(PlDataType::Int8, AnyValue::Int8(*v)),
        Scalar::Float(v) => PolarsScalar::new(PlDataType::Float32, AnyValue::Float32(*v)),
        Scalar::Double(v) => PolarsScalar::new(PlDataType::Float64, AnyValue::Float64(*v)),
        Scalar::Boolean(v) => PolarsScalar::new(PlDataType::Boolean, AnyValue::Boolean(*v)),
        Scalar::Date(v) => PolarsScalar::new_date(*v),
        Scalar::Timestamp(v) => {
            PolarsScalar::new_datetime(*v, TimeUnit::Microseconds, Some(TimeZone::UTC))
        }
        Scalar::TimestampNtz(v) => PolarsScalar::new_datetime(*v, TimeUnit::Microseconds, None),
        Scalar::IntervalDayTime(v) => PolarsScalar::new(
            PlDataType::Duration(TimeUnit::Microseconds),
            AnyValue::Duration(*v, TimeUnit::Microseconds),
        ),
        Scalar::IntervalYearMonth(v) => PolarsScalar::new(PlDataType::Int32, AnyValue::Int32(*v)),
        Scalar::Decimal(d) => {
            PolarsScalar::new_decimal(d.bits(), d.precision() as usize, d.scale() as usize)
        }
        Scalar::Null(dt) => PolarsScalar::new(dt.to_polars().ok()?, AnyValue::Null),
        // Binary, Array, Map, Struct go through build_series; not worth
        // duplicating the per-row path for `Column::new_scalar`.
        _ => return None,
    };
    Some(s)
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
            IntervalDayTime => {
                let v: Vec<Option<i64>> = values
                    .iter()
                    .map(|s| match s {
                        Scalar::IntervalDayTime(v) => Some(*v),
                        Scalar::Null(_) => None,
                        other => panic_mismatch(name, "IntervalDayTime", other),
                    })
                    .collect();
                Series::new(name_pl, v)
                    .cast(&PlDataType::Duration(TimeUnit::Microseconds))
                    .map_err(to_kernel_err)
            }
            IntervalYearMonth => Ok(Series::new(
                name_pl,
                values
                    .iter()
                    .map(|s| match s {
                        Scalar::IntervalYearMonth(v) => Some(*v),
                        Scalar::Null(_) => None,
                        other => panic_mismatch(name, "IntervalYearMonth", other),
                    })
                    .collect::<Vec<Option<i32>>>(),
            )),
            Void => {
                // Void is inhabited only by NULL, and polars' Null dtype is
                // the exact match — but a mismatched scalar still panics
                // like every other arm.
                for s in values {
                    if !matches!(s, Scalar::Null(_)) {
                        panic_mismatch(name, "Void", s);
                    }
                }
                Ok(Series::new_null(name_pl, values.len()))
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

/// `build_series` panics on a scalar/field type mismatch by contract;
/// callers at untrusted boundaries run [`ensure_scalar_types`] first.
fn panic_mismatch(name: &str, expected: &str, got: &Scalar) -> ! {
    panic!("build_series: column {name} expected {expected} scalars, got {got:?}");
}

/// Boundary guard for [`build_series`]'s panic contract: every non-null
/// scalar must carry exactly the field's declared type.
pub(crate) fn ensure_scalar_types<'a>(
    scalars: impl IntoIterator<Item = &'a Scalar>,
    field: &delta_kernel::schema::StructField,
    context: &str,
) -> delta_kernel::DeltaResult<()> {
    for scalar in scalars {
        if !matches!(scalar, Scalar::Null(_)) && scalar.data_type() != field.data_type {
            return Err(delta_kernel::Error::Generic(format!(
                "{context}: scalar for {} is {}, schema declares {}",
                field.name,
                scalar.data_type(),
                field.data_type
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod interval_tests {
    use super::*;
    use delta_kernel::schema::DataType as KernelDataType;

    #[test]
    fn day_time_interval_builds_duration_series() {
        let a = Scalar::IntervalDayTime(90_000_000);
        let null = Scalar::Null(KernelDataType::Primitive(PrimitiveType::IntervalDayTime));
        let s = build_series(
            "iv",
            &KernelDataType::Primitive(PrimitiveType::IntervalDayTime),
            &[&a, &null],
        )
        .unwrap();
        assert_eq!(s.dtype(), &PlDataType::Duration(TimeUnit::Microseconds));
        let physical = s.duration().unwrap().physical();
        assert_eq!(physical.get(0), Some(90_000_000));
        assert_eq!(physical.get(1), None);
    }

    #[test]
    fn year_month_interval_builds_month_count_series() {
        let a = Scalar::IntervalYearMonth(14);
        let s = build_series(
            "iv",
            &KernelDataType::Primitive(PrimitiveType::IntervalYearMonth),
            &[&a],
        )
        .unwrap();
        assert_eq!(s.dtype(), &PlDataType::Int32);
        assert_eq!(s.i32().unwrap().get(0), Some(14));
    }

    #[test]
    fn void_builds_null_dtype_series() {
        let null = Scalar::Null(KernelDataType::Primitive(PrimitiveType::Void));
        let s = build_series(
            "v",
            &KernelDataType::Primitive(PrimitiveType::Void),
            &[&null, &null],
        )
        .unwrap();
        assert_eq!(s.dtype(), &PlDataType::Null);
        assert_eq!(s.len(), 2);
        assert_eq!(s.null_count(), 2);
    }

    #[test]
    fn interval_scalars_convert_to_typed_polars_scalars() {
        let day_time = try_to_polars_scalar(&Scalar::IntervalDayTime(5)).unwrap();
        assert_eq!(
            day_time.dtype(),
            &PlDataType::Duration(TimeUnit::Microseconds)
        );
        let year_month = try_to_polars_scalar(&Scalar::IntervalYearMonth(7)).unwrap();
        assert_eq!(year_month.dtype(), &PlDataType::Int32);
    }
}

#[cfg(test)]
mod scalar_lit_width_tests {
    use super::*;
    use polars::prelude::LiteralValue;

    /// `build_series` derives an Array's list dtype from the declared
    /// element type, so a widened literal disagrees with the null rows it
    /// has to vstack with — and silently retypes the column when there are
    /// none.
    #[test]
    fn short_and_byte_literals_keep_their_width() {
        for (scalar, expected) in [
            (Scalar::Short(1), PlDataType::Int16),
            (Scalar::Byte(1), PlDataType::Int8),
        ] {
            let expr = scalar_to_lit(&scalar);
            let Expr::Literal(LiteralValue::Scalar(s)) = &expr else {
                panic!("expected a scalar literal, got {expr:?}");
            };
            assert_eq!(s.dtype(), &expected, "for {scalar:?}");
        }
    }
}
