//! `Operation::QueryPlan` → polars `LazyFrame` compiler.
//!
//! `Plan::nodes` arrives topologically sorted, so each node compiles in
//! slice order into a `LazyFrame` plus the kernel schema it produces. The
//! terminal (last) node is collected with the streaming engine and handed
//! back as `PolarsEngineData` batches.

use std::num::NonZeroUsize;

use delta_kernel::engine_data::EngineData;
use delta_kernel::expressions::{Expression, Scalar};
use delta_kernel::plans::PlanResult;
use delta_kernel::plans::ir::nodes::{
    Agg, Aggregate, DynamicScan, FileType, Filter, Operator, Project, ScanFile, ScanJson,
    ScanParquet, SemiJoin, Values,
};
use delta_kernel::plans::ir::plan::{Plan, PlanNode};
use delta_kernel::schema::{
    DataType as KernelDataType, MetadataColumnSpec, SchemaRef, StructField, StructType,
};
use delta_kernel::{DeltaResult, Error};
use polars::prelude::{
    DataFrame, Expr, IntoLazy, JoinArgs, JoinType, LazyFrame, SortMultipleOptions, UnionArgs, col,
    concat, lit,
};
use polars_plan::dsl::{DslBuilder, Engine as PolarsEngineMode, ScanSources};
use polars_utils::pl_path::PlRefPath;
use polars_utils::pl_str::PlSmallStr;
use url::Url;

use crate::engine::data::resolve_path;
use crate::engine::handlers::{align_dataframe, parse_ndjson_inferred, parquet_options,
    path_for_polars_io, unified_scan_args};
use crate::engine::{COLLECT_CHUNK_ROWS, PolarsEngineData};
use crate::errors::to_kernel_err;
use crate::translation::from_kernel::{
    column_path_to_expr, projection_exprs, scalar_to_lit, translate_expr, translate_predicate,
};
use crate::translation::schema::{KernelDataTypeExt, KernelSchemaExt};

use delta_kernel::StorageHandler;

use super::PolarsPlanExecutor;

/// One evaluated plan node: the lazy pipeline plus the kernel schema its
/// rows carry (needed to translate downstream expressions).
struct NodeState {
    lf: LazyFrame,
    schema: SchemaRef,
}

/// One file to read plus the pre-built literal exprs for its
/// file-constant columns (typed + aliased).
struct FileEntry {
    location: Url,
    lits: Vec<Expr>,
}

impl PolarsPlanExecutor {
    pub(super) fn execute_query(&self, plan: Plan) -> DeltaResult<PlanResult> {
        let mut states: Vec<NodeState> = Vec::with_capacity(plan.nodes.len());
        for node in plan.nodes {
            let state = self.eval_node(node, &states)?;
            states.push(state);
        }
        let terminal = states
            .pop()
            .ok_or_else(|| Error::Generic("execute_query: plan has no nodes".into()))?;

        let _enter = self.rt.enter();
        let chunk_size = NonZeroUsize::new(COLLECT_CHUNK_ROWS);
        let batches = terminal
            .lf
            .collect_batches(PolarsEngineMode::Streaming, true, chunk_size, false)
            .map_err(to_kernel_err)?;
        let iter = batches.map(|r| -> DeltaResult<Box<dyn EngineData>> {
            let mut df = r.map_err(to_kernel_err)?;
            df.rechunk_mut();
            Ok(Box::new(PolarsEngineData::new(df)))
        });
        Ok(PlanResult::Data(Box::new(iter)))
    }

    fn eval_node(&self, node: PlanNode, states: &[NodeState]) -> DeltaResult<NodeState> {
        let PlanNode { op, inputs } = node;
        let input = |i: usize| -> DeltaResult<&NodeState> {
            inputs
                .get(i)
                .and_then(|&idx| states.get(idx))
                .ok_or_else(|| Error::Generic(format!("plan node is missing input {i}")))
        };
        match op {
            Operator::ScanJson(ScanJson {
                files,
                file_constant_columns,
                schema,
            }) => self.eval_scan(FileType::Json, files, &file_constant_columns, schema),
            Operator::ScanParquet(ScanParquet {
                files,
                file_constant_columns,
                schema,
            }) => self.eval_scan(FileType::Parquet, files, &file_constant_columns, schema),
            Operator::DynamicScan(ds) => self.eval_dynamic_scan(ds, input(0)?),
            Operator::Values(values) => eval_values(values),
            Operator::Project(project) => eval_project(project, input(0)?),
            Operator::Filter(Filter { predicate }) => {
                let inp = input(0)?;
                let pl_pred = translate_predicate(predicate.as_ref(), Some(inp.schema.as_ref()))?;
                Ok(NodeState {
                    lf: inp.lf.clone().filter(pl_pred),
                    schema: inp.schema.clone(),
                })
            }
            Operator::SemiJoin(join) => {
                let probe = input(0)?;
                let build = input(1)?;
                eval_semi_join(join, probe, build)
            }
            Operator::UnionAll(_) => {
                let frames: Vec<LazyFrame> = inputs
                    .iter()
                    .map(|&idx| {
                        states
                            .get(idx)
                            .map(|s| s.lf.clone())
                            .ok_or_else(|| Error::Generic("union input out of range".into()))
                    })
                    .collect::<DeltaResult<_>>()?;
                let schema = input(0)?.schema.clone();
                Ok(NodeState {
                    lf: concat_frames(frames, &schema)?,
                    schema,
                })
            }
            Operator::Aggregate(agg) => eval_aggregate(agg, input(0)?),
        }
    }

    /// Reads `files` as `file_type`. Constant columns are broadcast as
    /// typed literals; every frame is shaped to `schema` order before the
    /// union so vertical concat sees identical schemas.
    fn eval_scan(
        &self,
        file_type: FileType,
        files: Vec<ScanFile>,
        constant_cols: &[String],
        schema: SchemaRef,
    ) -> DeltaResult<NodeState> {
        let (read_schema, row_index) = split_scan_schema(&schema, constant_cols)?;
        let const_fields = constant_fields(&schema, constant_cols)?;

        // All-constants-equal (incl. the no-constants case) lets parquet use
        // one multi-file scan, keeping polars-io's cross-file parallelism —
        // the multi-part-checkpoint shape. Commit JSONs differ per file
        // (`version`), so they take the per-file path.
        let uniform = files
            .windows(2)
            .all(|w| w[0].file_constants == w[1].file_constants);

        let entries = files
            .into_iter()
            .map(|f| {
                if f.file_constants.len() != constant_cols.len() {
                    return Err(Error::Generic(format!(
                        "scan file has {} constants, node declares {}",
                        f.file_constants.len(),
                        constant_cols.len()
                    )));
                }
                let lits = kernel_constant_lits(&f.file_constants, &const_fields)?;
                Ok(FileEntry {
                    location: f.meta.location,
                    lits,
                })
            })
            .collect::<DeltaResult<Vec<_>>>()?;

        let lf = self.scan_entries(
            file_type,
            entries,
            uniform,
            &read_schema,
            &schema,
            row_index,
        )?;
        Ok(NodeState { lf, schema })
    }

    fn scan_entries(
        &self,
        file_type: FileType,
        entries: Vec<FileEntry>,
        uniform_constants: bool,
        read_schema: &StructType,
        output_schema: &SchemaRef,
        row_index: Option<PlSmallStr>,
    ) -> DeltaResult<LazyFrame> {
        let select = schema_order_select(output_schema);

        if entries.is_empty() {
            let empty = DataFrame::empty_with_schema(
                output_schema.to_polars().map_err(to_kernel_err)?.as_ref(),
            );
            return Ok(empty.lazy());
        }

        let frames: Vec<LazyFrame> = match file_type {
            FileType::Parquet if uniform_constants && row_index.is_none() => {
                let lits = entries[0].lits.clone();
                let paths: Vec<PlRefPath> = entries
                    .iter()
                    .map(|e| path_for_polars_io(&e.location))
                    .collect::<DeltaResult<_>>()?;
                let mut lf = self.scan_parquet_lazy(paths, read_schema, None)?;
                if !lits.is_empty() {
                    lf = lf.with_columns(lits);
                }
                vec![lf.select(select.clone())]
            }
            FileType::Parquet => entries
                .into_iter()
                .map(|e| {
                    let path = path_for_polars_io(&e.location)?;
                    let mut lf = self.scan_parquet_lazy(vec![path], read_schema, row_index.clone())?;
                    if !e.lits.is_empty() {
                        lf = lf.with_columns(e.lits);
                    }
                    Ok(lf.select(select.clone()))
                })
                .collect::<DeltaResult<_>>()?,
            FileType::Json => {
                // Whole-file fetches in one parallel `read_files` batch;
                // NDJSON parse + kernel-schema align per file.
                let slices = entries.iter().map(|e| (e.location.clone(), None)).collect();
                let payloads: Vec<bytes::Bytes> = self
                    .storage
                    .read_files(slices)?
                    .collect::<DeltaResult<_>>()?;
                payloads
                    .into_iter()
                    .zip(entries)
                    .map(|(bytes, e)| {
                        let df = parse_ndjson_inferred(&bytes)?;
                        let aligned = align_dataframe(df, read_schema)?;
                        let mut lf = aligned.lazy();
                        if let Some(name) = &row_index {
                            lf = lf.with_row_index(name.clone(), None);
                        }
                        if !e.lits.is_empty() {
                            lf = lf.with_columns(e.lits);
                        }
                        Ok(lf.select(select.clone()))
                    })
                    .collect::<DeltaResult<_>>()?
            }
        };
        concat_frames(frames, output_schema)
    }

    fn scan_parquet_lazy(
        &self,
        paths: Vec<PlRefPath>,
        read_schema: &StructType,
        row_index: Option<PlSmallStr>,
    ) -> DeltaResult<LazyFrame> {
        let options = parquet_options(read_schema)?;
        let mut args = unified_scan_args(self.cloud_opts.as_ref(), None);
        if let Some(name) = row_index {
            args.row_index = Some(polars::prelude::RowIndex { name, offset: 0 });
        }
        let lf: LazyFrame = DslBuilder::scan_parquet(ScanSources::Paths(paths.into()), options, args)
            .map_err(to_kernel_err)?
            .build()
            .into();
        Ok(lf)
    }

    /// Reads files named by `input` rows. Deletion vectors are not applied
    /// here yet — matching the kernel's own sync executor, a row with a
    /// non-null DV descriptor is an error.
    fn eval_dynamic_scan(&self, ds: DynamicScan, input: &NodeState) -> DeltaResult<NodeState> {
        let df = input
            .lf
            .clone()
            .collect_with_engine(PolarsEngineMode::Streaming)
            .map_err(to_kernel_err)?;

        let dv = resolve_path(&df, &ds.dv_column).map_err(to_kernel_err)?;
        let mut dv_present = dv.is_not_null();
        for len in 1..ds.dv_column.len() {
            let prefix = delta_kernel::expressions::ColumnName::new(ds.dv_column.iter().take(len));
            let ancestor = resolve_path(&df, &prefix).map_err(to_kernel_err)?;
            dv_present = &dv_present & &ancestor.is_not_null();
        }
        if dv_present.any() {
            return Err(Error::Unsupported(
                "DynamicScan with deletion vectors is not implemented".into(),
            ));
        }

        let path = resolve_path(&df, &ds.path_column).map_err(to_kernel_err)?;
        let path = path.str().map_err(to_kernel_err)?;
        let size = resolve_path(&df, &ds.file_size_column).map_err(to_kernel_err)?;
        let size = size.i64().map_err(to_kernel_err)?;

        let const_series: Vec<polars::prelude::Series> = ds
            .file_constant_columns
            .iter()
            .map(|name| {
                resolve_path(&df, &delta_kernel::expressions::ColumnName::new([name]))
                    .map_err(to_kernel_err)
            })
            .collect::<DeltaResult<_>>()?;
        let const_fields = constant_fields(&ds.schema, &ds.file_constant_columns)?;

        let mut entries = Vec::with_capacity(df.height());
        for row in 0..df.height() {
            let rel = path.get(row).ok_or_else(|| {
                Error::Generic("DynamicScan path must not be null".into())
            })?;
            let location = ds
                .base_url
                .join(rel)
                .map_err(|e| Error::Generic(format!("DynamicScan path join failed: {e}")))?;
            match size.get(row) {
                Some(s) if s > 0 => {}
                _ => {
                    return Err(Error::Generic(
                        "DynamicScan file size must be a positive long".into(),
                    ));
                }
            }
            let lits = const_series
                .iter()
                .zip(const_fields.iter())
                .map(|(series, field)| -> DeltaResult<Expr> {
                    let value = series.get(row).map_err(to_kernel_err)?.into_static();
                    let scalar = polars::prelude::Scalar::new(series.dtype().clone(), value);
                    let dt = field.data_type.to_polars().map_err(to_kernel_err)?;
                    Ok(lit(scalar)
                        .cast(dt)
                        .alias(PlSmallStr::from_str(field.name.as_str())))
                })
                .collect::<DeltaResult<Vec<_>>>()?;
            entries.push(FileEntry { location, lits });
        }

        let (read_schema, row_index) = split_scan_schema(&ds.schema, &ds.file_constant_columns)?;
        let schema = ds.schema.clone();
        let lf = self.scan_entries(ds.file_type, entries, false, &read_schema, &schema, row_index)?;
        Ok(NodeState { lf, schema })
    }
}

/// Splits a scan output schema into the fields read from files (everything
/// that is neither a constant nor a metadata column) and the requested
/// row-index column, erroring on unsupported metadata specs.
fn split_scan_schema(
    schema: &SchemaRef,
    constant_cols: &[String],
) -> DeltaResult<(StructType, Option<PlSmallStr>)> {
    let mut row_index = None;
    for f in schema.fields() {
        match f.get_metadata_column_spec() {
            None => {}
            Some(MetadataColumnSpec::RowIndex) => {
                row_index = Some(PlSmallStr::from_str(f.name.as_str()));
            }
            Some(other) => {
                return Err(Error::Unsupported(format!(
                    "plan scan: metadata column {other:?} is not supported"
                )));
            }
        }
    }
    let read_fields: Vec<StructField> = schema
        .fields()
        .filter(|f| {
            f.get_metadata_column_spec().is_none()
                && !constant_cols.iter().any(|c| c == f.name.as_str())
        })
        .cloned()
        .collect();
    let read_schema = StructType::try_new(read_fields)?;
    Ok((read_schema, row_index))
}

fn constant_fields<'a>(
    schema: &'a SchemaRef,
    constant_cols: &[String],
) -> DeltaResult<Vec<&'a StructField>> {
    constant_cols
        .iter()
        .map(|name| {
            schema.field(name).ok_or_else(|| {
                Error::Generic(format!(
                    "file-constant column '{name}' not found in scan schema"
                ))
            })
        })
        .collect()
}

fn kernel_constant_lits(
    constants: &[Scalar],
    fields: &[&StructField],
) -> DeltaResult<Vec<Expr>> {
    constants
        .iter()
        .zip(fields.iter())
        .map(|(scalar, field)| -> DeltaResult<Expr> {
            let dt = field.data_type.to_polars().map_err(to_kernel_err)?;
            Ok(scalar_to_lit(scalar)
                .cast(dt)
                .alias(PlSmallStr::from_str(field.name.as_str())))
        })
        .collect()
}

fn schema_order_select(schema: &SchemaRef) -> Vec<Expr> {
    schema
        .fields()
        .map(|f| col(PlSmallStr::from_str(f.name.as_str())))
        .collect()
}

fn concat_frames(frames: Vec<LazyFrame>, schema: &SchemaRef) -> DeltaResult<LazyFrame> {
    match frames.len() {
        0 => {
            let empty =
                DataFrame::empty_with_schema(schema.to_polars().map_err(to_kernel_err)?.as_ref());
            Ok(empty.lazy())
        }
        1 => Ok(frames.into_iter().next().expect("len checked")),
        _ => concat(frames, UnionArgs::default()).map_err(to_kernel_err),
    }
}

fn eval_values(values: Values) -> DeltaResult<NodeState> {
    let Values { schema, rows } = values;
    let fields: Vec<&StructField> = schema.fields().collect();
    if let Some(bad) = rows.iter().find(|r| r.len() != fields.len()) {
        return Err(Error::Generic(format!(
            "Values row has {} scalars, schema has {} fields",
            bad.len(),
            fields.len()
        )));
    }
    let df = if rows.is_empty() {
        DataFrame::empty_with_schema(schema.to_polars().map_err(to_kernel_err)?.as_ref())
    } else {
        let height = rows.len();
        let columns = fields
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let scalars: Vec<&Scalar> = rows.iter().map(|r| &r[i]).collect();
                crate::translation::build_series(f.name.as_str(), &f.data_type, &scalars)
                    .map(polars::prelude::IntoColumn::into_column)
            })
            .collect::<DeltaResult<Vec<_>>>()?;
        DataFrame::new(height, columns).map_err(to_kernel_err)?
    };
    Ok(NodeState {
        lf: df.lazy(),
        schema,
    })
}

fn eval_project(project: Project, input: &NodeState) -> DeltaResult<NodeState> {
    let Project { expr, schema } = project;
    let input_struct = input.schema.as_ref();
    let out_dt = KernelDataType::Struct(Box::new(schema.as_ref().clone()));
    let exprs: Vec<Expr> = match expr.as_ref() {
        // Struct-shaped exprs classify into per-column ops (no nested
        // struct build in the plan).
        Expression::Struct(..) | Expression::StructPatch(..) => {
            projection_exprs(input_struct, expr.as_ref(), &out_dt)?
        }
        // Whole-row expression: evaluate once, unnest per output field.
        other => {
            let struct_expr = translate_expr(other, Some(&out_dt), Some(input_struct))?;
            schema
                .fields()
                .map(|f| {
                    struct_expr
                        .clone()
                        .struct_()
                        .field_by_name(f.name.as_str())
                        .alias(PlSmallStr::from_str(f.name.as_str()))
                })
                .collect()
        }
    };
    Ok(NodeState {
        lf: input.lf.clone().select(exprs),
        schema,
    })
}

fn eval_semi_join(join: SemiJoin, probe: &NodeState, build: &NodeState) -> DeltaResult<NodeState> {
    let SemiJoin {
        inverted,
        probe_keys,
        build_keys,
    } = join;
    let left_on: Vec<Expr> = probe_keys.iter().map(column_path_to_expr).collect();
    let right_on: Vec<Expr> = build_keys.iter().map(column_path_to_expr).collect();
    let how = if inverted {
        JoinType::Anti
    } else {
        JoinType::Semi
    };
    // Default `nulls_equal: false` matches the SQL semantics the node
    // specifies: a NULL key never matches the build side.
    let lf = probe
        .lf
        .clone()
        .join(build.lf.clone(), left_on, right_on, JoinArgs::new(how));
    Ok(NodeState {
        lf,
        schema: probe.schema.clone(),
    })
}

fn eval_aggregate(agg: Aggregate, input: &NodeState) -> DeltaResult<NodeState> {
    let Aggregate {
        group_by,
        aggs,
        schema,
    } = agg;
    let out_fields: Vec<&StructField> = schema.fields().collect();
    if out_fields.len() != group_by.len() + aggs.len() {
        return Err(Error::Generic(format!(
            "Aggregate schema has {} fields, expected {} keys + {} aggs",
            out_fields.len(),
            group_by.len(),
            aggs.len()
        )));
    }

    let key_exprs: Vec<Expr> = group_by
        .iter()
        .zip(out_fields.iter())
        .map(|(k, f)| column_path_to_expr(k).alias(PlSmallStr::from_str(f.name.as_str())))
        .collect();

    let agg_exprs: Vec<Expr> = aggs
        .iter()
        .zip(out_fields[group_by.len()..].iter())
        .map(|(a, f)| {
            let e = match a {
                Agg::Min { value } => column_path_to_expr(value).min(),
                Agg::Max { value } => column_path_to_expr(value).max(),
                Agg::MinNonNullBy { value, key } => non_null_by(value, key, true),
                Agg::MaxNonNullBy { value, key } => non_null_by(value, key, false),
            };
            e.alias(PlSmallStr::from_str(f.name.as_str()))
        })
        .collect();

    let lf = if key_exprs.is_empty() {
        input.lf.clone().select(agg_exprs)
    } else {
        input.lf.clone().group_by_stable(key_exprs).agg(agg_exprs)
    };
    Ok(NodeState { lf, schema })
}

/// `value` from the row with the least (`ascending`) / greatest `key`,
/// considering only rows where both are non-null; NULL when no row
/// qualifies. See kernel `Agg::max_non_null_by` for the exact contract.
fn non_null_by(
    value: &delta_kernel::expressions::ColumnName,
    key: &delta_kernel::expressions::ColumnName,
    ascending: bool,
) -> Expr {
    let v = column_path_to_expr(value);
    let k = column_path_to_expr(key);
    let keep = v.clone().is_not_null().and(k.clone().is_not_null());
    let sorted = v
        .filter(keep.clone())
        .sort_by([k.filter(keep)], SortMultipleOptions::default());
    if ascending { sorted.first() } else { sorted.last() }
}
