//! Extension traits between kernel `StructType` / `DataType` and polars /
//! polars-arrow schemas. Maps are modelled as `List<Struct<{key, value}>>`
//! in both flavours so the same shape flows through every reader.
//!
//! Bring [`KernelSchemaExt`], [`KernelDataTypeExt`], and [`ArrowSchemaExt`]
//! into scope at call sites to access the conversion methods.

use std::sync::Arc;

use delta_kernel::schema::{
    ArrayType, DataType as KernelDataType, MapType, PrimitiveType, StructField, StructType,
};
use polars::prelude::{
    ArrowDataType, DataType as PlDataType, Field as PlField, Schema as PlSchema,
};
use polars_arrow::datatypes::{ArrowSchema, Field as ArrowField};
use polars_utils::pl_str::PlSmallStr;

pub(crate) trait KernelSchemaExt {
    fn to_polars(&self) -> anyhow::Result<Arc<PlSchema>>;
}

impl KernelSchemaExt for StructType {
    fn to_polars(&self) -> anyhow::Result<Arc<PlSchema>> {
        let fields = self
            .fields()
            .map(|f| {
                f.data_type
                    .to_polars()
                    .map(|dt| (PlSmallStr::from_str(&f.name), dt))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Arc::new(PlSchema::from_iter(fields)))
    }
}

pub(crate) trait KernelDataTypeExt {
    fn to_polars(&self) -> anyhow::Result<PlDataType>;
}

impl KernelDataTypeExt for KernelDataType {
    fn to_polars(&self) -> anyhow::Result<PlDataType> {
        Ok(match self {
            KernelDataType::Primitive(p) => primitive_to_polars(p)?,
            KernelDataType::Array(arr) => {
                let ArrayType { element_type, .. } = arr.as_ref();
                PlDataType::List(Box::new(element_type.to_polars()?))
            }
            KernelDataType::Struct(s) => PlDataType::Struct(
                s.fields()
                    .map(|f| {
                        Ok::<_, anyhow::Error>(PlField::new(
                            PlSmallStr::from_str(&f.name),
                            f.data_type.to_polars()?,
                        ))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            KernelDataType::Map(map) => {
                let MapType {
                    key_type,
                    value_type,
                    ..
                } = map.as_ref();
                PlDataType::List(Box::new(PlDataType::Struct(vec![
                    PlField::new(
                        PlSmallStr::from_static(crate::consts::MAP_KEY_FIELD),
                        key_type.to_polars()?,
                    ),
                    PlField::new(
                        PlSmallStr::from_static(crate::consts::MAP_VALUE_FIELD),
                        value_type.to_polars()?,
                    ),
                ])))
            }
            KernelDataType::Variant(_) => {
                anyhow::bail!("Variant data type is not supported by polars-deltalake yet");
            }
        })
    }
}

pub(crate) trait ArrowSchemaExt {
    fn to_kernel(&self) -> anyhow::Result<StructType>;
}

impl ArrowSchemaExt for ArrowSchema {
    fn to_kernel(&self) -> anyhow::Result<StructType> {
        let fields = self
            .iter()
            .map(|(_, f)| arrow_to_kernel_field(f))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(StructType::try_new(fields)?)
    }
}

fn primitive_to_polars(p: &PrimitiveType) -> anyhow::Result<PlDataType> {
    use PrimitiveType::*;
    Ok(match p {
        String => PlDataType::String,
        Long => PlDataType::Int64,
        Integer => PlDataType::Int32,
        Short => PlDataType::Int16,
        Byte => PlDataType::Int8,
        Float => PlDataType::Float32,
        Double => PlDataType::Float64,
        Boolean => PlDataType::Boolean,
        Binary => PlDataType::Binary,
        Date => PlDataType::Date,
        // Delta stores both as µs i64; Timestamp is UTC, TimestampNtz is naive.
        Timestamp => PlDataType::Datetime(
            polars::prelude::TimeUnit::Microseconds,
            Some(polars::prelude::TimeZone::UTC),
        ),
        TimestampNtz => PlDataType::Datetime(polars::prelude::TimeUnit::Microseconds, None),
        Decimal(d) => PlDataType::Decimal(d.precision() as usize, d.scale() as usize),
    })
}

fn arrow_to_kernel_field(f: &ArrowField) -> anyhow::Result<StructField> {
    Ok(StructField::new(
        f.name.to_string(),
        arrow_to_kernel_dtype(&f.dtype)?,
        f.is_nullable,
    ))
}

fn arrow_to_kernel_dtype(dt: &ArrowDataType) -> anyhow::Result<KernelDataType> {
    Ok(match dt {
        ArrowDataType::Null => anyhow::bail!("arrow Null dtype has no kernel equivalent"),
        ArrowDataType::Boolean => KernelDataType::Primitive(PrimitiveType::Boolean),
        ArrowDataType::Int8 => KernelDataType::Primitive(PrimitiveType::Byte),
        ArrowDataType::Int16 => KernelDataType::Primitive(PrimitiveType::Short),
        ArrowDataType::Int32 => KernelDataType::Primitive(PrimitiveType::Integer),
        ArrowDataType::Int64 => KernelDataType::Primitive(PrimitiveType::Long),
        // Unsigned widen to the smallest signed kernel type that fits.
        ArrowDataType::UInt8 => KernelDataType::Primitive(PrimitiveType::Short),
        ArrowDataType::UInt16 => KernelDataType::Primitive(PrimitiveType::Integer),
        ArrowDataType::UInt32 => KernelDataType::Primitive(PrimitiveType::Long),
        ArrowDataType::UInt64 => KernelDataType::Primitive(PrimitiveType::Long),
        ArrowDataType::Float32 => KernelDataType::Primitive(PrimitiveType::Float),
        ArrowDataType::Float64 => KernelDataType::Primitive(PrimitiveType::Double),
        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View => {
            KernelDataType::Primitive(PrimitiveType::String)
        }
        ArrowDataType::Binary
        | ArrowDataType::LargeBinary
        | ArrowDataType::BinaryView
        | ArrowDataType::FixedSizeBinary(_) => KernelDataType::Primitive(PrimitiveType::Binary),
        ArrowDataType::Date32 | ArrowDataType::Date64 => {
            KernelDataType::Primitive(PrimitiveType::Date)
        }
        ArrowDataType::Timestamp(_, tz) => {
            if tz.is_some() {
                KernelDataType::Primitive(PrimitiveType::Timestamp)
            } else {
                KernelDataType::Primitive(PrimitiveType::TimestampNtz)
            }
        }
        // Delta caps decimals at precision/scale 38, both fit in u8.
        ArrowDataType::Decimal(p, s)
        | ArrowDataType::Decimal32(p, s)
        | ArrowDataType::Decimal64(p, s) => {
            let precision: u8 = u8::try_from(*p).map_err(|_| {
                anyhow::anyhow!("arrow Decimal precision {p} > 255 (Delta caps at 38)")
            })?;
            let scale: u8 = u8::try_from(*s)
                .map_err(|_| anyhow::anyhow!("arrow Decimal scale {s} > 255 (Delta caps at 38)"))?;
            KernelDataType::Primitive(PrimitiveType::Decimal(
                delta_kernel::schema::DecimalType::try_new(precision, scale)?,
            ))
        }
        ArrowDataType::List(field)
        | ArrowDataType::LargeList(field)
        | ArrowDataType::FixedSizeList(field, _) => KernelDataType::Array(Box::new(
            ArrayType::new(arrow_to_kernel_dtype(&field.dtype)?, field.is_nullable),
        )),
        ArrowDataType::Struct(fields) => {
            let kernel_fields = fields
                .iter()
                .map(arrow_to_kernel_field)
                .collect::<anyhow::Result<Vec<_>>>()?;
            KernelDataType::Struct(Box::new(StructType::try_new(kernel_fields)?))
        }
        ArrowDataType::Map(field, _) => {
            let ArrowDataType::Struct(entries) = &field.dtype else {
                anyhow::bail!(
                    "arrow Map inner field must be a Struct, got {:?}",
                    field.dtype
                );
            };
            if entries.len() != 2 {
                anyhow::bail!(
                    "arrow Map inner Struct must have 2 fields (key, value), got {}",
                    entries.len()
                );
            }
            let key_dt = arrow_to_kernel_dtype(&entries[0].dtype)?;
            let value_dt = arrow_to_kernel_dtype(&entries[1].dtype)?;
            KernelDataType::Map(Box::new(MapType::new(
                key_dt,
                value_dt,
                entries[1].is_nullable,
            )))
        }
        other => anyhow::bail!("arrow dtype {other:?} has no kernel equivalent yet"),
    })
}
