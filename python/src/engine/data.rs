//! `PolarsEngineData` — `delta_kernel::EngineData` over a polars `DataFrame`.
//! `visit_rows` resolves each nested column path, builds a typed
//! `PolarsGetter` per leaf, and hands the visitor `&[&dyn GetData]`.

use delta_kernel::engine_data::{
    EngineData, GetData, ListItem, MapItem, RowVisitor, StringArrayAccessor,
};
use delta_kernel::expressions::{ArrayData, ColumnName};
use delta_kernel::schema::SchemaRef;
use delta_kernel::{DeltaResult, Error};
use polars::prelude::{
    BinaryChunked, BooleanChunked, DataFrame, DataType as PlDataType, Float32Chunked,
    Float64Chunked, Int8Chunked, Int16Chunked, Int32Chunked, Int64Chunked, IntoColumn, ListChunked,
    NamedFrom, Series, StringChunked,
};
use polars_arrow::array::{Array as ArrowArray, Utf8ViewArray};

use crate::consts::{MAP_KEY_FIELD, MAP_VALUE_FIELD};
use crate::errors::to_kernel_err;

pub(crate) struct PolarsEngineData {
    df: DataFrame,
}

impl PolarsEngineData {
    pub(crate) fn new(df: DataFrame) -> Self {
        Self { df }
    }

    pub(crate) fn into_inner(self) -> DataFrame {
        self.df
    }

    pub(crate) fn dataframe(&self) -> &DataFrame {
        &self.df
    }
}

impl EngineData for PolarsEngineData {
    fn len(&self) -> usize {
        self.df.height()
    }

    fn has_field(&self, name: &ColumnName) -> bool {
        resolve_path(&self.df, name).is_ok()
    }

    fn visit_rows(
        &self,
        column_names: &[ColumnName],
        visitor: &mut dyn RowVisitor,
    ) -> DeltaResult<()> {
        // Owned Series outlive the typed getters that borrow into them.
        let owned: Vec<Series> = column_names
            .iter()
            .map(|cn| resolve_path(&self.df, cn).map_err(to_kernel_err))
            .collect::<DeltaResult<_>>()?;

        let getters: Vec<PolarsGetter<'_>> = owned
            .iter()
            .map(PolarsGetter::from_series)
            .collect::<DeltaResult<_>>()?;

        let refs: Vec<&dyn GetData<'_>> = getters.iter().map(|g| g as &dyn GetData<'_>).collect();

        visitor.visit(self.len(), &refs)
    }

    fn apply_selection_vector(
        self: Box<Self>,
        mut selection_vector: Vec<bool>,
    ) -> DeltaResult<Box<dyn EngineData>> {
        let height = self.df.height();
        if selection_vector.len() > height {
            return Err(Error::InvalidSelectionVector(format!(
                "selection vector len {} > data height {}",
                selection_vector.len(),
                height
            )));
        }
        // Per the FilteredEngineData contract: a selection vector shorter
        // than the data height means rows beyond the SV are assumed
        // selected. Pad with `true` to match height before applying.
        if selection_vector.len() < height {
            selection_vector.resize(height, true);
        }
        let mask: BooleanChunked =
            BooleanChunked::new("__pldl_dv__".into(), selection_vector.as_slice());
        let filtered = self.df.filter(&mask).map_err(to_kernel_err)?;
        Ok(Box::new(PolarsEngineData::new(filtered)))
    }

    /// Read path never reaches this — kernel only calls `append_columns`
    /// from transaction commit. Implemented anyway so the handler is
    /// complete against the trait.
    fn append_columns(
        &self,
        schema: SchemaRef,
        columns: Vec<ArrayData>,
    ) -> DeltaResult<Box<dyn EngineData>> {
        let fields: Vec<_> = schema.fields().collect();
        if fields.len() != columns.len() {
            return Err(Error::Generic(format!(
                "append_columns: schema has {} fields but got {} columns",
                fields.len(),
                columns.len()
            )));
        }

        let height = self.df.height();
        let new_cols: Vec<polars::prelude::Column> = fields
            .iter()
            .zip(columns.iter())
            .map(|(field, arr_data)| -> DeltaResult<_> {
                let scalars: Vec<&delta_kernel::expressions::Scalar> =
                    arr_data.array_elements().iter().collect();
                if scalars.len() != height {
                    return Err(Error::Generic(format!(
                        "append_columns: column {} has {} elements but data has {} rows",
                        field.name,
                        scalars.len(),
                        height
                    )));
                }
                let series = crate::translation::build_series(
                    field.name.as_str(),
                    &field.data_type,
                    &scalars,
                )?;
                Ok(series.into_column())
            })
            .collect::<DeltaResult<_>>()?;
        let new_df = self.df.hstack(&new_cols).map_err(to_kernel_err)?;
        Ok(Box::new(PolarsEngineData::new(new_df)))
    }
}

/// Resolve a kernel `ColumnName` to its leaf `Series`. The first segment
/// is a top-level column; each subsequent segment is a struct field on the
/// preceding series — kernel only emits named paths through struct nesting
/// (maps and lists are surfaced by their own getters, not path-walking).
fn resolve_path(df: &DataFrame, name: &ColumnName) -> anyhow::Result<Series> {
    let mut iter = name.iter();
    let first = iter.next().expect("ColumnName is nonempty by construction");

    let mut series: Series = df
        .column(first)
        .map_err(|e| anyhow::anyhow!("missing column {first}: {e}"))?
        .as_materialized_series()
        .clone();

    for segment in iter {
        let st = series
            .struct_()
            .map_err(|e| anyhow::anyhow!("expected struct at {segment}: {e}"))?;
        series = st
            .field_by_name(segment)
            .map_err(|e| anyhow::anyhow!("missing struct field {segment}: {e}"))?;
    }
    Ok(series)
}

enum PolarsGetter<'a> {
    Bool(&'a BooleanChunked),
    Byte(&'a Int8Chunked),
    Short(&'a Int16Chunked),
    Int(&'a Int32Chunked),
    Long(&'a Int64Chunked),
    Float(&'a Float32Chunked),
    Double(&'a Float64Chunked),
    String(&'a StringChunked),
    Binary(&'a BinaryChunked),
    Decimal(&'a polars::prelude::Int128Chunked),
    StringList(StringListGetter<'a>),
    StringMap(StringMapGetter<'a>),
}

impl<'a> PolarsGetter<'a> {
    fn from_series(series: &'a Series) -> DeltaResult<Self> {
        Ok(match series.dtype() {
            PlDataType::Boolean => Self::Bool(series.bool().map_err(to_kernel_err)?),
            PlDataType::Int8 => Self::Byte(series.i8().map_err(to_kernel_err)?),
            PlDataType::Int16 => Self::Short(series.i16().map_err(to_kernel_err)?),
            PlDataType::Int32 => Self::Int(series.i32().map_err(to_kernel_err)?),
            PlDataType::Int64 => Self::Long(series.i64().map_err(to_kernel_err)?),
            PlDataType::Float32 => Self::Float(series.f32().map_err(to_kernel_err)?),
            PlDataType::Float64 => Self::Double(series.f64().map_err(to_kernel_err)?),
            PlDataType::String => Self::String(series.str().map_err(to_kernel_err)?),
            PlDataType::Binary => Self::Binary(series.binary().map_err(to_kernel_err)?),
            // Kernel surfaces Date/Timestamp as i32 days / i64 microseconds
            // (Delta wire format); store the physical chunk so `get_date` /
            // `get_timestamp` can read it through `Self::Int` / `Self::Long`.
            PlDataType::Date => Self::Int(series.date().map_err(to_kernel_err)?.physical()),
            PlDataType::Datetime(_, _) => {
                Self::Long(series.datetime().map_err(to_kernel_err)?.physical())
            }
            PlDataType::Decimal(_, _) => {
                let chunked = series.decimal().map_err(to_kernel_err)?;
                Self::Decimal(chunked.physical())
            }
            PlDataType::List(inner) => match inner.as_ref() {
                PlDataType::String => Self::StringList(StringListGetter::new(
                    series.list().map_err(to_kernel_err)?,
                )?),
                // Polars has no native Map dtype, so we encode kernel `Map`
                // columns as `List<Struct<{key, value}>>` end-to-end (schema
                // converters, JSON read/write, expression eval). On the read
                // side we detect that shape here and route through the
                // dedicated map getter so kernel sees a `MapItem`, not a
                // list of structs.
                PlDataType::Struct(fields)
                    if fields.len() == 2
                        && fields[0].name == MAP_KEY_FIELD
                        && fields[1].name == MAP_VALUE_FIELD
                        && matches!(fields[0].dtype, PlDataType::String)
                        && matches!(fields[1].dtype, PlDataType::String) =>
                {
                    Self::StringMap(StringMapGetter::new(series.list().map_err(to_kernel_err)?)?)
                }
                other => {
                    return Err(Error::UnexpectedColumnType(format!(
                        "column {} has unsupported polars dtype List<{other:?}>; \
                         only List<String> and List<Struct<{{key, value}}>> are wired up",
                        series.name(),
                    )));
                }
            },
            // Loud failure here rather than silent null-padding downstream:
            // Struct (kernel should have descended), Array (fixed-size), etc.
            other => {
                return Err(Error::UnexpectedColumnType(format!(
                    "column {} has unsupported polars dtype {other:?}",
                    series.name(),
                )));
            }
        })
    }
}

/// Variant != method name because `get_date` reads `Self::Int` and
/// `get_timestamp` reads `Self::Long` — kernel collapses Date → Int32 and
/// Timestamp → Int64 at the GetData layer.
macro_rules! delegate_get {
    ($method:ident -> $ret:ty, $variant:ident, $label:literal) => {
        fn $method(&'a self, row: usize, field: &str) -> DeltaResult<Option<$ret>> {
            match self {
                Self::$variant(c) => Ok(c.get(row)),
                _ => Err(type_mismatch(field, $label)),
            }
        }
    };
}

impl<'a> GetData<'a> for PolarsGetter<'a> {
    delegate_get!(get_bool -> bool, Bool, "bool");
    delegate_get!(get_byte -> i8, Byte, "byte");
    delegate_get!(get_short -> i16, Short, "short");
    delegate_get!(get_int -> i32, Int, "int");
    delegate_get!(get_long -> i64, Long, "long");
    delegate_get!(get_float -> f32, Float, "float");
    delegate_get!(get_double -> f64, Double, "double");
    delegate_get!(get_date -> i32, Int, "date");
    delegate_get!(get_timestamp -> i64, Long, "timestamp");
    delegate_get!(get_decimal -> i128, Decimal, "decimal");
    delegate_get!(get_str -> &'a str, String, "string");
    delegate_get!(get_binary -> &'a [u8], Binary, "binary");
    delegate_get!(get_list -> ListItem<'a>, StringList, "list");
    delegate_get!(get_map -> MapItem<'a>, StringMap, "map");
}

/// Newtype because `StringArrayAccessor` (kernel) and `Utf8ViewArray`
/// (polars-arrow) are both foreign — orphan rules forbid a direct impl.
struct PlStringArray<'a> {
    inner: &'a Utf8ViewArray,
}

impl<'a> StringArrayAccessor for PlStringArray<'a> {
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn value(&self, index: usize) -> &str {
        self.inner.value(index)
    }
    fn is_valid(&self, index: usize) -> bool {
        self.inner.is_valid(index)
    }
}

pub(crate) struct StringListGetter<'a> {
    string_array: PlStringArray<'a>,
    offsets: &'a polars_arrow::offset::OffsetsBuffer<i64>,
    validity: Option<&'a polars_arrow::bitmap::Bitmap>,
    len: usize,
}

impl<'a> StringListGetter<'a> {
    fn new(list: &'a ListChunked) -> DeltaResult<Self> {
        let arr = downcast_single_chunk_list(list)?;
        let values = arr
            .values()
            .as_any()
            .downcast_ref::<polars_arrow::array::Utf8ViewArray>()
            .ok_or_else(|| Error::Generic("List<String> inner is not Utf8View".into()))?;
        Ok(Self {
            string_array: PlStringArray { inner: values },
            offsets: arr.offsets(),
            validity: arr.validity(),
            len: arr.len(),
        })
    }

    fn get(&'a self, row: usize) -> Option<ListItem<'a>> {
        if row >= self.len || !validity_at(self.validity, row) {
            return None;
        }
        let (start, end) = self.offsets.start_end(row);
        Some(ListItem::new(&self.string_array, start..end))
    }
}

pub(crate) struct StringMapGetter<'a> {
    keys: PlStringArray<'a>,
    values: PlStringArray<'a>,
    offsets: &'a polars_arrow::offset::OffsetsBuffer<i64>,
    validity: Option<&'a polars_arrow::bitmap::Bitmap>,
    len: usize,
}

impl<'a> StringMapGetter<'a> {
    fn new(list: &'a ListChunked) -> DeltaResult<Self> {
        let outer = downcast_single_chunk_list(list)?;
        let struct_arr = outer
            .values()
            .as_any()
            .downcast_ref::<polars_arrow::array::StructArray>()
            .ok_or_else(|| Error::Generic("Map element is not a Struct".into()))?;

        let key_arr = struct_arr
            .values()
            .first()
            .and_then(|v| {
                v.as_any()
                    .downcast_ref::<polars_arrow::array::Utf8ViewArray>()
            })
            .ok_or_else(|| Error::Generic("Map key is not Utf8View".into()))?;
        let val_arr = struct_arr
            .values()
            .get(1)
            .and_then(|v| {
                v.as_any()
                    .downcast_ref::<polars_arrow::array::Utf8ViewArray>()
            })
            .ok_or_else(|| Error::Generic("Map value is not Utf8View".into()))?;

        Ok(Self {
            keys: PlStringArray { inner: key_arr },
            values: PlStringArray { inner: val_arr },
            offsets: outer.offsets(),
            validity: outer.validity(),
            len: outer.len(),
        })
    }

    fn get(&'a self, row: usize) -> Option<MapItem<'a>> {
        if row >= self.len || !validity_at(self.validity, row) {
            return None;
        }
        let (start, end) = self.offsets.start_end(row);
        Some(MapItem::new(&self.keys, &self.values, start..end))
    }
}

fn downcast_single_chunk_list(
    list: &ListChunked,
) -> DeltaResult<&polars_arrow::array::ListArray<i64>> {
    let chunks = list.chunks();
    if chunks.len() != 1 {
        return Err(Error::Generic(format!(
            "expected single-chunk ListChunked, got {}",
            chunks.len()
        )));
    }
    chunks[0]
        .as_any()
        .downcast_ref::<polars_arrow::array::ListArray<i64>>()
        .ok_or_else(|| Error::Generic("ListChunked is not a LargeList".into()))
}

fn validity_at(bitmap: Option<&polars_arrow::bitmap::Bitmap>, row: usize) -> bool {
    bitmap.is_none_or(|b| b.get_bit(row))
}

fn type_mismatch(field: &str, want: &str) -> Error {
    Error::UnexpectedColumnType(format!(
        "{field}: requested {want} but column has a different type"
    ))
}
