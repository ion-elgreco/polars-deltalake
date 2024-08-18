use delta_kernel::engine_data::{EngineData, EngineList, EngineMap, GetData};
use delta_kernel::schema::{
    DataType as DeltaKernelDataType, PrimitiveType, Schema, SchemaRef, StructField,
};
use delta_kernel::{DataVisitor, DeltaResult, Error};

use crate::engine::polars_get_data::{
    WrappedBooleanChunked, WrappedInt32Chunked, WrappedInt64Chunked, WrappedStringChunked,
    WrappedTuple,
};
use polars::chunked_array::ChunkedArray;
use polars::datatypes::DataType;
use polars::error::polars_bail;
use polars::frame::DataFrame;
use polars::prelude::{
    GetAnyValue, LargeStringArray, PolarsDataType, StringChunked, StructChunked,
};
use polars::series::{IntoSeries, Series};
use polars_arrow::array::{Array, GenericBinaryArray, MapArray, Utf8Array};
use polars_arrow::compute::cast::{self, CastOptionsImpl};
use polars_arrow::datatypes::ArrowDataType;
use pyo3_polars::export::polars_core::utils::Container;
use std::any::Any;
use std::borrow::Borrow;

use super::polars_delta_conversion::DeltaDataType;
use super::polars_get_data::PolarsListChunked;
use tracing::{debug, warn};
/// convenient way to return an error if a condition isn't true
macro_rules! require {
    ( $cond:expr, $err:expr ) => {
        if !($cond) {
            return Err($err);
        }
    };
}

// pub struct PolarsMapArray {
//     inner: MapArray
// }

/// PolarsEngineData holds an Arrow RecordBatch, implements `EngineData` so the kernel can extract from it.
pub struct PolarsEngineData {
    data: DataFrame,
}

impl PolarsEngineData {
    /// Create a new `ArrowEngineData` from a `DataFrame`
    pub fn new(data: DataFrame) -> Self {
        PolarsEngineData { data }
    }

    /// Utility constructor to get a `Box<PolarsEngineData>` out of a `Box<dyn EngineData>`
    pub fn try_from_engine_data(engine_data: Box<dyn EngineData>) -> DeltaResult<Box<Self>> {
        engine_data
            .into_any()
            .downcast::<PolarsEngineData>()
            .map_err(|_| Error::engine_data_type("PolarsEngineData"))
    }

    /// Get a reference to the `DataFrame` this `PolarsEngineData` is wrapping
    pub fn dataframe(&self) -> &DataFrame {
        &self.data
    }
}

impl EngineData for PolarsEngineData {
    fn extract(&self, schema: SchemaRef, visitor: &mut dyn DataVisitor) -> DeltaResult<()> {
        let mut col_array = vec![];
        self.extract_columns(&mut col_array, &schema)?;
        visitor.visit(self.length(), &col_array)
    }

    fn length(&self) -> usize {
        self.data.len()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

impl From<DataFrame> for PolarsEngineData {
    fn from(value: DataFrame) -> Self {
        PolarsEngineData::new(value)
    }
}

impl From<PolarsEngineData> for DataFrame {
    fn from(value: PolarsEngineData) -> Self {
        value.data
    }
}

impl From<Box<PolarsEngineData>> for DataFrame {
    fn from(value: Box<PolarsEngineData>) -> Self {
        value.data
    }
}

/// This is a trait that allows us to query something by column name and get out an Arrow
/// `Array`. Both `RecordBatch` and `StructArray` can do this. By having our `extract_*` functions
/// just take anything that implements this trait we can use the same function to drill into
/// either. This is useful because when we're recursing into data we start with a RecordBatch, but
/// if we encounter a Struct column, it will be a `StructArray`.
trait ProvidesColumnByName {
    fn column_by_name(&self, name: &str) -> Option<&Series>;
}

impl ProvidesColumnByName for DataFrame {
    fn column_by_name(&self, name: &str) -> Option<&Series> {
        let index = self.get_column_index(name);
        index.map(|v| self.get_columns().get(v)).flatten()
    }
}

impl ProvidesColumnByName for StructChunked {
    fn column_by_name(&self, name: &str) -> Option<&Series> {
        self.fields().iter().find(|s| s.name() == name)
    }
}

// pub(crate) struct PolarsListSeries {
//     pub inner: Series
// }

impl EngineList for PolarsListChunked<'_> {
    fn len(&self, row_index: usize) -> usize {
        // Verify syntax
        self.inner.get(row_index).map(|v| v.len()).unwrap_or(0)
    }

    fn get(&self, row_index: usize, index: usize) -> String {
        let arry: Option<Box<dyn Array>> = self.inner.get(row_index);

        let nested_arry = match arry {
            Some(arr) => {
                Some(
                    cast::cast(&*arr, &ArrowDataType::LargeUtf8, CastOptionsImpl::default())
                        .expect("Couldn't cast nested array to String Array")
                        .as_any()
                        .downcast_ref::<LargeStringArray>()
                        .cloned(), // Must be some way to avoid a clone and  cast to String more efficiently
                )
            }
            _ => None,
        }
        .flatten();
        nested_arry
            .map(|arr| arr.value(index).to_string())
            .unwrap_or("None".to_string())
    }

    fn materialize(&self, row_index: usize) -> Vec<String> {
        let mut result = vec![];
        for i in 0..EngineList::len(self, row_index) {
            result.push(self.get(row_index, i));
        }
        result
    }
}

// ///  POLARS HASN'T FULLY IMPLEMENTED MAP IN POLARS_ARROW, NOR IN THE FRONT FACING API
// impl EngineMap for PolarsSeries {
//     fn get<'a>(&'a self, row_index: usize, key: &str) -> Option<&'a str> {
//         unimplemented!();
//     //     let offsets = self.inner.offsets();
//     //     let start_offset = offsets[row_index] as usize;
//     //     let count = offsets[row_index + 1] as usize - start_offset;
//     //     let keys = self.inner.keys().as_string::<i32>();
//     //     for (idx, map_key) in keys.iter().enumerate().skip(start_offset).take(count) {
//     //         if let Some(map_key) = map_key {
//     //             if key == map_key {
//     //                 // found the item
//     //                 let vals = self.inner.values().as_string::<i32>();
//     //                 return Some(vals.value(idx));
//     //             }
//     //         }
//     //     }
//     //     None
//     }

//     fn materialize(&self, row_index: usize) -> HashMap<String, String> {
//         unimplemented!();
//     //     let mut ret = HashMap::new();
//     //     let map_val = self.value(row_index);
//     //     let keys = map_val.column(0).as_string::<i32>();
//     //     let values = map_val.column(1).as_string::<i32>();
//     //     for (key, value) in keys.iter().zip(values.iter()) {
//     //         if let (Some(key), Some(value)) = (key, value) {
//     //             ret.insert(key.into(), value.into());
//     //         }
//     //     }
//     //     ret
//     // }
//     }
// }

impl PolarsEngineData {
    /// Extracts an exploded view (all leaf values), in schema order of that data contained
    /// within. `out_col_array` is filled with [`GetData`] items that can be used to get at the
    /// actual primitive types.
    ///
    /// # Arguments
    ///
    /// * `out_col_array` - the vec that leaf values will be pushed onto. it is passed as an arg to
    ///   make the recursion below easier. if we returned a [`Vec`] we would have to `extend` it each
    ///   time we encountered a struct and made the recursive call.
    /// * `schema` - the schema to extract getters for
    pub fn extract_columns<'a>(
        &'a self,
        out_col_array: &mut Vec<&dyn GetData<'a>>,
        schema: &Schema,
    ) -> DeltaResult<()> {
        debug!("Extracting column getters for {:#?}", schema);
        PolarsEngineData::extract_columns_from_array(out_col_array, schema, Some(&self.data))
    }

    fn extract_columns_from_array<'a>(
        out_col_array: &mut Vec<&dyn GetData<'a>>,
        schema: &Schema,
        array: Option<&'a dyn ProvidesColumnByName>,
    ) -> DeltaResult<()> {
        for field in schema.fields() {
            let col = array
                .and_then(|a| a.column_by_name(&field.name))
                .filter(|a| *a.dtype() != DataType::Null);
            // Note: if col is None we have either:
            //   a) encountered a column that is all nulls or,
            //   b) recursed into a optional struct that was null. In this case, array.is_none() is
            //      true and we don't need to check field nullability, because we assume all fields
            //      of a nullable struct can be null
            // So below if the field is allowed to be null, OR array.is_none() we push that,
            // otherwise we error out.
            if let Some(col) = col {
                Self::extract_column(out_col_array, field, col)?;
            } else if array.is_none() || field.is_nullable() {
                if let DeltaKernelDataType::Struct(inner_struct) = field.data_type() {
                    Self::extract_columns_from_array(out_col_array, inner_struct.as_ref(), None)?;
                } else {
                    debug!("Pushing a null field for {}", field.name);
                    out_col_array.push(&WrappedTuple { inner: () });
                }
            } else {
                return Err(Error::MissingData(format!(
                    "Found required field {}, but it's null",
                    field.name
                )));
            }
        }
        Ok(())
    }

    fn extract_column<'a>(
        out_col_array: &mut Vec<&dyn GetData<'a>>,
        field: &StructField,
        col: &'a Series,
    ) -> DeltaResult<()> {
        match (col.dtype(), &field.data_type) {
            (&DataType::Struct(ref f), DeltaKernelDataType::Struct(fields)) => {
                // both structs, so recurse into col
                let struct_array = col.struct_().unwrap();
                PolarsEngineData::extract_columns_from_array(
                    out_col_array,
                    fields,
                    Some(struct_array),
                )?;
            }
            (&DataType::Boolean, &DeltaKernelDataType::Primitive(PrimitiveType::Boolean)) => {
                debug!("Pushing boolean array for {}", field.name);
                let ca = WrappedBooleanChunked {
                    inner: col.bool().unwrap(),
                };
                out_col_array.push(&ca);
            }
            (&DataType::String, &DeltaKernelDataType::Primitive(PrimitiveType::String)) => {
                debug!("Pushing string array for {}", field.name);
                let ca = WrappedStringChunked {
                    inner: col.str().unwrap(),
                };
                out_col_array.push(&ca);
            }
            (&DataType::Int32, &DeltaKernelDataType::Primitive(PrimitiveType::Integer)) => {
                debug!("Pushing int32 array for {}", field.name);
                let ca: WrappedInt32Chunked = WrappedInt32Chunked {
                    inner: col.i32().unwrap(),
                };
                out_col_array.push(&ca);
            }
            (&DataType::Int64, &DeltaKernelDataType::Primitive(PrimitiveType::Long)) => {
                debug!("Pushing int64 array for {}", field.name);
                let ca = WrappedInt64Chunked {
                    inner: col.i64().unwrap(),
                };
                out_col_array.push(&ca);
            }
            (DataType::List(arrow_field), DeltaKernelDataType::Array(_array_type)) => {
                match arrow_field.clone().implode() {
                    DataType::String => {
                        debug!("Pushing list for {}", field.name);
                        let list = col.list().unwrap();
                        let ca = PolarsListChunked { inner: list };
                        out_col_array.push(&ca);
                    }
                    _ => {
                        return Err(Error::UnexpectedColumnType(format!(
                            "On {}: Only support lists that contain strings",
                            field.name()
                        )))
                    }
                }
            }
            (_, &DeltaKernelDataType::Map(_)) => {
                return Err(Error::unexpected_column_type(
                    "Polars doesn't support map types",
                ))
            }
            (polars_datatype, data_type) => {
                warn!(
                    "Can't extract {}. Polars Type: {polars_datatype}\n Kernel Type: {data_type}",
                    field.name
                );
                return Err(get_error_for_types(data_type, polars_datatype, &field.name));
            }
        }
        Ok(())
    }
}

fn get_error_for_types(
    delta_data_type: &DeltaKernelDataType,
    polars_data_type: &DataType,
    field_name: &str,
) -> Error {
    let expected_type: Result<DeltaDataType, _> = polars_data_type.try_into();

    match expected_type {
        Ok(expected_type) => {
            if expected_type.inner == *delta_data_type {
                Error::UnexpectedColumnType(format!(
                    "On {field_name}: Don't know how to extract something of type {polars_data_type}",
                ))
            } else {
                Error::UnexpectedColumnType(format!(
                    "Type mismatch on {field_name}: expected {polars_data_type}, got {delta_data_type}",
                ))
            }
        }
        Err(e) => Error::UnexpectedColumnType(format!(
            "On {field_name}: Unsupported data type {polars_data_type}: {e}",
        )),
    }
}
