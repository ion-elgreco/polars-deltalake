//! Conversions from delta types to polars-arrow types

use polars_arrow::datatypes::{
    ArrowDataType, ArrowSchema, ArrowSchemaRef, Field as ArrowField, TimeUnit,
};
use polars_arrow::legacy::error::{PolarsError, PolarsResult};

use itertools::Itertools;

use delta_kernel::error::Error;
use delta_kernel::schema::{ArrayType, DataType, MapType, PrimitiveType, StructField, StructType};

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

struct DeltaDataType {
    inner: DataType,
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

impl From<DataType> for DeltaDataType {
    fn from(value: DataType) -> Self {
        DeltaDataType { inner: value }
    }
}
impl From<&DataType> for DeltaDataType {
    fn from(value: &DataType) -> Self {
        DeltaDataType {
            inner: value.clone(),
        }
    }
}

impl TryFrom<&DeltaStructType> for ArrowSchema {
    type Error = PolarsError;

    fn try_from(s: &DeltaStructType) -> PolarsResult<Self> {
        let fields: Vec<ArrowField> = s.fields().map(TryInto::try_into).try_collect()?;
        Ok(ArrowSchema::from(fields))
    }
}

impl TryFrom<DeltaStructField> for ArrowField {
    type Error = PolarsError;

    fn try_from(f: DeltaStructField) -> PolarsResult<Self> {
        let metadata = f
            .inner
            .metadata()
            .iter()
            .map(|(key, val)| Ok((key.clone(), serde_json::to_string(val)?)))
            .collect::<Result<_, serde_json::Error>>()
            .map_err(|err| PolarsError::ComputeError(err.to_string().into()))?;

        let field = ArrowField::new(
            f.inner.name(),
            ArrowDataType::try_from(DeltaDataType {
                inner: f.inner.data_type().to_owned(),
            })?,
            f.inner.is_nullable(),
        )
        .with_metadata(metadata);

        Ok(field)
    }
}

impl TryFrom<&DeltaArrayType> for ArrowField {
    type Error = PolarsError;

    fn try_from(a: &DeltaArrayType) -> PolarsResult<Self> {
        Ok(ArrowField::new(
            LIST_ARRAY_ROOT,
            ArrowDataType::try_from(DeltaDataType {
                inner: a.inner.element_type().to_owned(),
            })?,
            a.inner.contains_null(),
        ))
    }
}

impl TryFrom<DeltaArrayType> for ArrowField {
    type Error = PolarsError;

    fn try_from(a: DeltaArrayType) -> PolarsResult<Self> {
        Ok(ArrowField::try_from(&a)?)
    }
}

impl TryFrom<&DeltaMapType> for ArrowField {
    type Error = PolarsError;

    fn try_from(a: &DeltaMapType) -> PolarsResult<Self> {
        Ok(ArrowField::new(
            MAP_ROOT_DEFAULT,
            ArrowDataType::Struct(
                vec![
                    ArrowField::new(
                        MAP_KEY_DEFAULT,
                        ArrowDataType::try_from(DeltaDataType {
                            inner: a.inner.key_type().to_owned(),
                        })?,
                        false,
                    ),
                    ArrowField::new(
                        MAP_VALUE_DEFAULT,
                        ArrowDataType::try_from(DeltaDataType {
                            inner: a.inner.value_type().to_owned(),
                        })?,
                        a.inner.value_contains_null(),
                    ),
                ]
                .into(),
            ),
            false, // always non-null
        ))
    }
}

impl TryFrom<DeltaMapType> for ArrowField {
    type Error = PolarsError;

    fn try_from(a: DeltaMapType) -> PolarsResult<Self> {
        Ok(ArrowField::try_from(&a)?)
    }
}

impl TryFrom<DeltaDataType> for ArrowDataType {
    type Error = PolarsError;

    fn try_from(t: DeltaDataType) -> PolarsResult<Self> {
        match t.inner {
            DataType::Primitive(p) => {
                match p {
                    PrimitiveType::String => Ok(ArrowDataType::Utf8View),
                    PrimitiveType::Long => Ok(ArrowDataType::Int64), // undocumented type
                    PrimitiveType::Integer => Ok(ArrowDataType::Int32),
                    PrimitiveType::Short => Ok(ArrowDataType::Int16),
                    PrimitiveType::Byte => Ok(ArrowDataType::Int8),
                    PrimitiveType::Float => Ok(ArrowDataType::Float32),
                    PrimitiveType::Double => Ok(ArrowDataType::Float64),
                    PrimitiveType::Boolean => Ok(ArrowDataType::Boolean),
                    PrimitiveType::Binary => Ok(ArrowDataType::BinaryView),
                    PrimitiveType::Decimal(precision, scale) => {
                        PrimitiveType::check_decimal(precision, scale)
                            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
                        Ok(ArrowDataType::Decimal(precision as usize, scale as usize))
                    }
                    PrimitiveType::Date => {
                        // A calendar date, represented as a year-month-day triple without a
                        // timezone. Stored as 4 bytes integer representing days since 1970-01-01
                        Ok(ArrowDataType::Date32)
                    }
                    PrimitiveType::Timestamp => Ok(ArrowDataType::Timestamp(
                        TimeUnit::Microsecond,
                        Some("UTC".into()),
                    )),
                    PrimitiveType::TimestampNtz => {
                        Ok(ArrowDataType::Timestamp(TimeUnit::Microsecond, None))
                    }
                }
            }
            DataType::Struct(s) => Ok(ArrowDataType::Struct(
                DeltaStructType { inner: *s }
                    .fields()
                    .map(TryInto::try_into)
                    .collect::<Result<Vec<ArrowField>, PolarsError>>()?
                    .into(),
            )),
            DataType::Array(a) => Ok(ArrowDataType::List(Box::new(
                DeltaArrayType { inner: *a }.try_into()?,
            ))),
            DataType::Map(m) => Ok(ArrowDataType::Map(
                Box::new(DeltaMapType { inner: *m }.try_into()?),
                false,
            )),
        }
    }
}

impl TryFrom<&ArrowSchema> for DeltaStructType {
    type Error = PolarsError;

    fn try_from(arrow_schema: &ArrowSchema) -> PolarsResult<Self> {
        let new_fields: Vec<DeltaStructField> = arrow_schema
            .fields
            .iter()
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

impl TryFrom<ArrowSchemaRef> for DeltaStructType {
    type Error = PolarsError;

    fn try_from(arrow_schema: ArrowSchemaRef) -> PolarsResult<Self> {
        arrow_schema.as_ref().try_into()
    }
}

impl TryFrom<&ArrowField> for DeltaStructField {
    type Error = PolarsError;

    fn try_from(arrow_field: &ArrowField) -> PolarsResult<Self> {
        let delta_type: DeltaDataType = arrow_field.data_type().try_into()?;
        let inner_struct = StructField::new(
            arrow_field.name.clone(),
            delta_type.inner,
            arrow_field.is_nullable,
        );
        Ok(DeltaStructField::new(inner_struct))
    }
}

impl TryFrom<&ArrowDataType> for DeltaDataType {
    type Error = PolarsError;

    fn try_from(arrow_datatype: &ArrowDataType) -> PolarsResult<Self> {
        match arrow_datatype {
            ArrowDataType::Utf8 => Ok(DataType::Primitive(PrimitiveType::String).into()),
            ArrowDataType::LargeUtf8 => Ok(DataType::Primitive(PrimitiveType::String).into()),
            ArrowDataType::Utf8View => Ok(DataType::Primitive(PrimitiveType::String).into()),
            ArrowDataType::Int64 => Ok(DataType::Primitive(PrimitiveType::Long).into()), // undocumented type
            ArrowDataType::Int32 => Ok(DataType::Primitive(PrimitiveType::Integer).into()),
            ArrowDataType::Int16 => Ok(DataType::Primitive(PrimitiveType::Short).into()),
            ArrowDataType::Int8 => Ok(DataType::Primitive(PrimitiveType::Byte).into()),
            ArrowDataType::UInt64 => Ok(DataType::Primitive(PrimitiveType::Long).into()), // undocumented type
            ArrowDataType::UInt32 => Ok(DataType::Primitive(PrimitiveType::Integer).into()),
            ArrowDataType::UInt16 => Ok(DataType::Primitive(PrimitiveType::Short).into()),
            ArrowDataType::UInt8 => Ok(DataType::Primitive(PrimitiveType::Byte).into()),
            ArrowDataType::Float32 => Ok(DataType::Primitive(PrimitiveType::Float).into()),
            ArrowDataType::Float64 => Ok(DataType::Primitive(PrimitiveType::Double).into()),
            ArrowDataType::Boolean => Ok(DataType::Primitive(PrimitiveType::Boolean).into()),
            ArrowDataType::Binary => Ok(DataType::Primitive(PrimitiveType::Binary).into()),
            ArrowDataType::FixedSizeBinary(_) => {
                Ok(DataType::Primitive(PrimitiveType::Binary).into())
            }
            ArrowDataType::LargeBinary => Ok(DataType::Primitive(PrimitiveType::Binary).into()),
            ArrowDataType::BinaryView => Ok(DataType::Primitive(PrimitiveType::Binary).into()),
            ArrowDataType::Decimal(p, s) => {
                if *s < 0 {
                    return Err(PolarsError::ComputeError(
                        Error::invalid_decimal("Negative scales are not supported in Delta")
                            .to_string()
                            .into(),
                    ));
                };
                DataType::decimal(*p as u8, *s as u8)
                    .map_err(|e| PolarsError::ComputeError(e.to_string().into()))
                    .map(|v| v.into())
            }
            ArrowDataType::Date32 => Ok(DataType::Primitive(PrimitiveType::Date).into()),
            ArrowDataType::Date64 => Ok(DataType::Primitive(PrimitiveType::Date).into()),
            ArrowDataType::Timestamp(TimeUnit::Microsecond, None) => {
                Ok(DataType::Primitive(PrimitiveType::TimestampNtz).into())
            }
            ArrowDataType::Timestamp(TimeUnit::Microsecond, Some(tz))
                if tz.eq_ignore_ascii_case("utc") =>
            {
                Ok(DataType::Primitive(PrimitiveType::Timestamp).into())
            }
            ArrowDataType::Struct(fields) => {
                let converted_fields: Result<Vec<DeltaStructField>, _> = fields
                    .iter()
                    .map(|field| TryInto::<DeltaStructField>::try_into(field))
                    .collect();
                Ok(DataType::Struct(Box::new(StructType::new(
                    converted_fields?
                        .iter()
                        .map(|v| v.inner.clone())
                        .collect_vec(),
                )))
                .into())
            }
            ArrowDataType::List(field) => Ok(DataType::Array(Box::new(ArrayType::new(
                DeltaDataType::try_from(field.data_type())?.inner,
                (*field).is_nullable,
            )))
            .into()),
            ArrowDataType::LargeList(field) => Ok(DataType::Array(Box::new(ArrayType::new(
                DeltaDataType::try_from(field.data_type())?.inner,
                (*field).is_nullable,
            )))
            .into()),
            ArrowDataType::FixedSizeList(field, _) => {
                Ok(DataType::Array(Box::new(ArrayType::new(
                    DeltaDataType::try_from(field.data_type())?.inner,
                    (*field).is_nullable,
                )))
                .into())
            }
            ArrowDataType::Map(field, _) => {
                if let ArrowDataType::Struct(struct_fields) = field.data_type() {
                    let key_type = DeltaDataType::try_from(struct_fields[0].data_type())?.inner;
                    let value_type = DeltaDataType::try_from(struct_fields[1].data_type())?.inner;
                    let value_type_nullable = struct_fields[1].is_nullable;
                    Ok(DataType::Map(Box::new(MapType::new(
                        key_type,
                        value_type,
                        value_type_nullable,
                    )))
                    .into())
                } else {
                    panic!("DataType::Map should contain a struct field child");
                }
            }
            // Dictionary types are just an optimized in-memory representation of an array.
            // Schema-wise, they are the same as the value type.
            ArrowDataType::Dictionary(_, value_type, _) => {
                Ok(DeltaDataType::try_from(&**value_type)?.inner.into())
            }
            s => Err(PolarsError::SchemaMismatch(
                format!("Invalid data type for Delta Lake: {:?}", s)
                    .to_string()
                    .into(),
            )),
        }
    }
}
