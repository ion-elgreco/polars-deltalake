use std::path::PathBuf;
use std::sync::Arc;

use delta_kernel::EngineData;
use delta_kernel::JsonHandler;
use itertools::Itertools;
use polars::prelude::cloud::CloudOptions;
use polars::prelude::LazyFileListReader;
use polars::prelude::LazyJsonLineReader;
use polars::prelude::Schema as PolarsSchema;


use crate::data::polars_data::PolarsEngineData;
use crate::data::polars_delta_conversion::DeltaStructType;
pub struct PolarsJsonHandler {
    storage_options: Option<CloudOptions>,
}

impl JsonHandler for PolarsJsonHandler {
    fn parse_json(
        &self,
        json_strings: Box<dyn delta_kernel::EngineData>,
        output_schema: delta_kernel::schema::SchemaRef,
    ) -> delta_kernel::DeltaResult<Box<dyn delta_kernel::EngineData>> {
        todo!()
    }
    fn read_json_files(
        &self,
        files: &[delta_kernel::FileMeta],
        physical_schema: delta_kernel::schema::SchemaRef,
        _predicate: Option<delta_kernel::Expression>,
    ) -> delta_kernel::DeltaResult<delta_kernel::FileDataReadResultIterator> {
        if files.is_empty() {
            return Ok(Box::new(std::iter::empty()));
        }
        let schema_wrapped = DeltaStructType {
            inner: physical_schema.as_ref().clone(),
        };
        let schema = Arc::new(PolarsSchema::try_from(&schema_wrapped).unwrap());
        let mut paths = vec![];
        for file in files.iter() {
            paths.push(PathBuf::from(file.location.to_string()))
        }
        let mut json_loader = LazyJsonLineReader::new_paths(Arc::from(paths.into_boxed_slice()))
            .with_schema(Some(schema));

        if let Some(options) = self.storage_options {
            json_loader = json_loader.with
        }

            .finish()
            .map_err(|err| delta_kernel::error::Error::InternalError(err.to_string()))?
            .collect()
            .map_err(|err| delta_kernel::error::Error::InternalError(err.to_string()))?;

        let engine_data = PolarsEngineData::new(df);
        return Ok(Box::new(
            vec![Ok(Box::new(engine_data) as Box<dyn EngineData>)].into_iter(),
        ));
    }
}
