//! Conversions from delta types to polars types

use polars::prelude::{Schema as PolarsSchema, DataType as PolarsDataType, Field as PolarsField, TimeUnit};
use polars_arrow::legacy::error::{PolarsError, PolarsResult};

use itertools::Itertools;

use delta_kernel::error::Error;
use delta_kernel::schema::{ArrayType, DataType as DeltaKernelDataType, MapType, PrimitiveType, StructField, StructType};

pub(crate) const LIST_ARRAY_ROOT: &str = "element";
pub(crate) const MAP_ROOT_DEFAULT: &str = "key_value";
pub(crate) const MAP_KEY_DEFAULT: &str = "key";
pub(crate) const MAP_VALUE_DEFAULT: &str = "value";

struct DeltaStructType {
    inner: StructType,
}

struct DeltaMapType {
    inner: MapType,
}

struct DeltaStructField {
    inner: StructField,
}

impl DeltaStructField {
    pub fn new(field: StructField) -> Self {
        DeltaStructField { inner: field }
    }
}

struct DeltaArrayType {
    inner: ArrayType,
}

pub(crate) struct DeltaDataType {
    pub inner: DeltaKernelDataType,
}

impl DeltaStructType {
    pub fn new(fields: Vec<StructField>) -> Self {
        DeltaStructType {
            inner: StructType {
                type_name: "struct".into(),
                fields: fields.into_iter().map(|f| (f.name.clone(), f)).collect(),
            },
        }
    }
    pub fn fields(&self) -> impl Iterator<Item = DeltaStructField> + '_ {
        self.inner.fields.values().map(Into::into)
    }
}

impl From<&StructField> for DeltaStructField {
    fn from(value: &StructField) -> Self {
        DeltaStructField {
            inner: value.clone(),
        }
    }
}

impl From<DeltaKernelDataType> for DeltaDataType {
    fn from(value: DeltaKernelDataType) -> Self {
        DeltaDataType { inner: value }
    }
}
impl From<&DeltaKernelDataType> for DeltaDataType {
    fn from(value: &DeltaKernelDataType) -> Self {
        DeltaDataType {
            inner: value.clone(),
        }
    }
}

impl TryFrom<&DeltaStructType> for PolarsSchema {
    type Error = PolarsError;

    fn try_from(s: &DeltaStructType) -> PolarsResult<Self> {
        let fields: Vec<PolarsField> = s.fields().map(TryInto::try_into).try_collect()?;
        Ok(fields.into_iter().collect())
    }
}

impl TryFrom<DeltaStructField> for PolarsField {
    type Error = PolarsError;

    fn try_from(f: DeltaStructField) -> PolarsResult<Self> {
        let field = PolarsField::new(
            f.inner.name(),
            PolarsDataType::try_from(DeltaDataType {
                inner: f.inner.data_type().to_owned(),
            })?
        );

        Ok(field)
    }
}


impl TryFrom<PolarsField> for DeltaStructField {
    type Error = PolarsError;

    fn try_from(f: PolarsField) -> PolarsResult<Self> {
        let delta_datatype: DeltaDataType = f.data_type().try_into()?; 
        let field = DeltaStructField { inner: StructField::new(f.name, delta_datatype.inner, true)};

        Ok(field)
    }
}



impl TryFrom<&DeltaArrayType> for PolarsDataType {
    type Error = PolarsError;

    fn try_from(a: &DeltaArrayType) -> PolarsResult<Self> {
        let inner_type = DeltaDataType {inner: a.inner.element_type().clone()}.try_into()?;
        Ok(PolarsDataType::List(Box::new(inner_type)))
    }
}

impl TryFrom<DeltaArrayType> for PolarsDataType {
    type Error = PolarsError;

    fn try_from(a: DeltaArrayType) -> PolarsResult<Self> {
        Ok(PolarsDataType::try_from(&a)?)
    }
}

impl TryFrom<&DeltaMapType> for PolarsField {
    type Error = PolarsError;

    fn try_from(_: &DeltaMapType) -> PolarsResult<Self> {
        Err(PolarsError::ComputeError("Map type is not supported by polars".into()))
    }
}

impl TryFrom<DeltaMapType> for PolarsField {
    type Error = PolarsError;

    fn try_from(a: DeltaMapType) -> PolarsResult<Self> {
        Ok(PolarsField::try_from(&a)?)
    }
}

impl TryFrom<DeltaDataType> for PolarsDataType {
    type Error = PolarsError;

    fn try_from(t: DeltaDataType) -> PolarsResult<Self> {
        match t.inner {
            DeltaKernelDataType::Primitive(p) => {
                match p {
                    PrimitiveType::String => Ok(PolarsDataType::String),
                    PrimitiveType::Long => Ok(PolarsDataType::Int64), // undocumented type
                    PrimitiveType::Integer => Ok(PolarsDataType::Int32),
                    PrimitiveType::Short => Ok(PolarsDataType::Int16),
                    PrimitiveType::Byte => Ok(PolarsDataType::Int8),
                    PrimitiveType::Float => Ok(PolarsDataType::Float32),
                    PrimitiveType::Double => Ok(PolarsDataType::Float64),
                    PrimitiveType::Boolean => Ok(PolarsDataType::Boolean),
                    PrimitiveType::Binary => Ok(PolarsDataType::Binary),
                    PrimitiveType::Decimal(precision, scale) => {
                        PrimitiveType::check_decimal(precision, scale)
                            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
                        Ok(PolarsDataType::Decimal(Some(precision as usize), Some(scale as usize)))
                    }
                    PrimitiveType::Date => {
                        // A calendar date, represented as a year-month-day triple without a
                        // timezone. Stored as 4 bytes integer representing days since 1970-01-01
                        Ok(PolarsDataType::Date)
                    }
                    PrimitiveType::Timestamp => Ok(PolarsDataType::Datetime(
                        TimeUnit::Microseconds,
                        Some("UTC".into()),
                    )),
                    PrimitiveType::TimestampNtz => {
                        Ok(PolarsDataType::Datetime(TimeUnit::Microseconds, None))
                    }
                }
            }
            DeltaKernelDataType::Struct(s) => Ok(PolarsDataType::Struct(
                DeltaStructType { inner: *s }
                    .fields()
                    .map(TryInto::try_into)
                    .collect::<Result<Vec<PolarsField>, PolarsError>>()?
                    .into(),
            )),
            DeltaKernelDataType::Array(a) => Ok(
                DeltaArrayType { inner: *a }.try_into()?,
            ),
            DeltaKernelDataType::Map(_) => Err(PolarsError::ComputeError("Map type is not supported by polars".into())),
        }
    }
}

impl TryFrom<&PolarsSchema> for DeltaStructType {
    type Error = PolarsError;

    fn try_from(polars_schema: &PolarsSchema) -> PolarsResult<Self> {
        let new_fields: Vec<DeltaStructField> = polars_schema.iter_fields()
            .map(|field| TryInto::<DeltaStructField>::try_into(field))
            .collect::<Result<Vec<_>, _>>()?;
        let fields = new_fields
            .iter()
            .map(|v| v.inner.clone())
            .collect::<Vec<_>>();
        Ok(DeltaStructType {
            inner: StructType::new(fields),
        })
    }
}

impl TryFrom<&PolarsField> for DeltaStructField {
    type Error = PolarsError;

    fn try_from(polars_field: &PolarsField) -> PolarsResult<Self> {
        let delta_type: DeltaDataType = polars_field.data_type().try_into()?;
        let inner_struct = StructField::new(
            polars_field.name.clone(),
            delta_type.inner,
            true
        );
        Ok(DeltaStructField::new(inner_struct))
    }
}

impl TryFrom<&PolarsDataType> for DeltaDataType {
    type Error = PolarsError;

    fn try_from(polars_datatype: &PolarsDataType) -> PolarsResult<Self> {
        match polars_datatype {
            PolarsDataType::String => Ok(DeltaKernelDataType::Primitive(PrimitiveType::String).into()),
            PolarsDataType::Int64 => Ok(DeltaKernelDataType::Primitive(PrimitiveType::Long).into()), // undocumented type
            PolarsDataType::Int32 => Ok(DeltaKernelDataType::Primitive(PrimitiveType::Integer).into()),
            PolarsDataType::Int16 => Ok(DeltaKernelDataType::Primitive(PrimitiveType::Short).into()),
            PolarsDataType::Int8 => Ok(DeltaKernelDataType::Primitive(PrimitiveType::Byte).into()),
            PolarsDataType::UInt64 => Ok(DeltaKernelDataType::Primitive(PrimitiveType::Long).into()), // undocumented type
            PolarsDataType::UInt32 => Ok(DeltaKernelDataType::Primitive(PrimitiveType::Integer).into()),
            PolarsDataType::UInt16 => Ok(DeltaKernelDataType::Primitive(PrimitiveType::Short).into()),
            PolarsDataType::UInt8 => Ok(DeltaKernelDataType::Primitive(PrimitiveType::Byte).into()),
            PolarsDataType::Float32 => Ok(DeltaKernelDataType::Primitive(PrimitiveType::Float).into()),
            PolarsDataType::Float64 => Ok(DeltaKernelDataType::Primitive(PrimitiveType::Double).into()),
            PolarsDataType::Boolean => Ok(DeltaKernelDataType::Primitive(PrimitiveType::Boolean).into()),
            PolarsDataType::Binary => Ok(DeltaKernelDataType::Primitive(PrimitiveType::Binary).into()),
            PolarsDataType::Decimal(p, s) => {
                if s.unwrap_or(0) < 0 {
                    return Err(PolarsError::ComputeError(
                        Error::invalid_decimal("Negative scales are not supported in Delta")
                            .to_string()
                            .into(),
                    ));
                };
                DeltaKernelDataType::decimal(p.unwrap() as u8, s.unwrap() as u8)
                    .map_err(|e| PolarsError::ComputeError(e.to_string().into()))
                    .map(|v| v.into())
            }
            PolarsDataType::Date => Ok(DeltaKernelDataType::Primitive(PrimitiveType::Date).into()),
            PolarsDataType::Datetime(TimeUnit::Microseconds, None) => {
                Ok(DeltaKernelDataType::Primitive(PrimitiveType::TimestampNtz).into())
            }
            PolarsDataType::Datetime(TimeUnit::Microseconds, Some(tz))
                if tz.eq_ignore_ascii_case("utc") =>
            {
                Ok(DeltaKernelDataType::Primitive(PrimitiveType::Timestamp).into())
            }
            PolarsDataType::Struct(fields) => {
                let converted_fields: Result<Vec<DeltaStructField>, _> = fields
                    .iter()
                    .map(|field| TryInto::<DeltaStructField>::try_into(field))
                    .collect();
                Ok(DeltaKernelDataType::Struct(Box::new(StructType::new(
                    converted_fields?
                        .iter()
                        .map(|v| v.inner.clone())
                        .collect_vec(),
                )))
                .into())
            }
            PolarsDataType::List(inner_dtype) => Ok(DeltaKernelDataType::Array(Box::new(ArrayType::new(
                DeltaDataType::try_from(inner_dtype.as_ref())?.inner,
                true,
            )))
            .into()),
            s => Err(PolarsError::SchemaMismatch(
                format!("Invalid data type for Delta Lake: {:?}", s)
                    .to_string()
                    .into(),
            )),
        }
    }
}
