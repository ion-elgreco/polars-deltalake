// use arrow_array::{
//     types::{GenericStringType, Int32Type, Int64Type},
//     Array, BooleanArray, GenericByteArray, GenericListArray, MapArray, OffsetSizeTrait,
//     PrimitiveArray,
// };

// use crate::{
//     engine_data::{GetData, ListItem, MapItem},
//     DeltaResult,
// };

// actual impls (todo: could macro these)

use delta_kernel::{
    engine_data::{GetData, ListItem},
    DeltaResult,
};
use polars::prelude::{BooleanChunked, Int32Chunked, Int64Chunked, ListChunked, StringChunked};

pub enum WrappedValue<'a> {
    Bool(&'a BooleanChunked),
    String(&'a StringChunked),
    Int32(&'a Int32Chunked),
    Int64(&'a Int64Chunked),
    List(&'a ListChunked),
    Null(()),
}

impl<'a> GetData<'a> for WrappedValue<'a> {
    fn get_bool(&self, row_index: usize, _field_name: &str) -> DeltaResult<Option<bool>> {
        Ok(match &self {
            WrappedValue::Bool(arr) => arr.get(row_index),
            WrappedValue::Null(_) => None,
            _ => unreachable!(),
        })
    }
    fn get_str(&self, row_index: usize, _field_name: &str) -> DeltaResult<Option<&'a str>> {
        Ok(match &self {
            WrappedValue::String(arr) => arr.get(row_index),
            WrappedValue::Null(_) => None,
            _ => unreachable!(),
        })
    }

    fn get_int(&self, row_index: usize, _field_name: &str) -> DeltaResult<Option<i32>> {
        Ok(match &self {
            WrappedValue::Int32(arr) => arr.get(row_index),
            WrappedValue::Null(_) => None,
            _ => unreachable!(),
        })
    }

    fn get_long(&self, row_index: usize, _field_name: &str) -> DeltaResult<Option<i64>> {
        Ok(match &self {
            WrappedValue::Int64(arr) => arr.get(row_index),
            WrappedValue::Null(_) => None,
            _ => unreachable!(),
        })
    }

    fn get_list(
        &'a self,
        row_index: usize,
        _field_name: &str,
    ) -> DeltaResult<Option<ListItem<'a>>> {
        Ok(match &self {
            WrappedValue::Int64(arr) => match arr.get(row_index) {
                Some(_) => Some(ListItem::new(self, row_index)),
                _ => None,
            },
            WrappedValue::Null(_) => None,
            _ => unreachable!(),
        })
    }
}
