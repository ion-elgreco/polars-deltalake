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

use delta_kernel::{engine_data::{GetData, ListItem}, DeltaResult};
use polars::{prelude::{BooleanChunked, Int32Chunked, Int64Chunked, ListChunked, StringChunked}, series::IntoSeries};

use super::polars_data::PolarsListSeries;


pub struct WrappedBooleanChunked<'a> {
    pub inner: &'a BooleanChunked
}

impl<'a> GetData<'a> for WrappedBooleanChunked<'a> {
    fn get_bool(&self, 
        row_index: usize,
         _field_name: &str) -> DeltaResult<Option<bool>> {
        Ok(self.inner.get(row_index))
    }
}


pub struct WrappedStringChunked<'a> {
    pub inner: &'a StringChunked
}

impl<'a> GetData<'a> for WrappedStringChunked<'a> {
    fn get_str(&self, 
        row_index: usize,
         _field_name: &str) -> DeltaResult<Option<&'a str>> {
        Ok(self.inner.get(row_index))
    }
}


pub struct WrappedInt32Chunked<'a> {
    pub inner: &'a Int32Chunked
}

impl<'a> GetData<'a> for WrappedInt32Chunked<'a> {
    fn get_int(&self, 
        row_index: usize,
         _field_name: &str) -> DeltaResult<Option<i32>> {
        Ok(self.inner.get(row_index))
    }
}

pub struct WrappedInt64Chunked<'a> {
    pub inner: &'a Int64Chunked
}

impl<'a> GetData<'a> for WrappedInt64Chunked<'a> {
    fn get_long(&self, 
        row_index: usize,
         _field_name: &str) -> DeltaResult<Option<i64>> {
        Ok(self.inner.get(row_index))
    }
}



pub struct PolarsListChunked<'a> {
    pub inner: &'a ListChunked
}

impl<'a> GetData<'a> for PolarsListChunked<'a>{
    fn get_list(&'a self, 
        row_index: usize,
         _field_name: &str) -> DeltaResult<Option<ListItem<'a>>> {
        match self.inner.get(row_index) {
            Some(_) => Ok(Some(ListItem::new(self, row_index))),
            _ => Ok(None)
        }
    }
}


macro_rules! impl_null_get {
    ( $(($name: ident, $typ: ty)), * ) => {
        $(
            fn $name(&'a self, _row_index: usize, _field_name: &str) -> DeltaResult<Option<$typ>> {
                Ok(None)
            }
        )*
    };
}

pub struct WrappedTuple {
    pub inner: ()
}

impl<'a> GetData<'a> for WrappedTuple {
    impl_null_get!(
        (get_bool, bool),
        (get_int, i32),
        (get_long, i64),
        (get_str, &'a str)
        // (get_list, ListItem<'a>),
        // (get_map, MapItem<'a>)
    );
}
