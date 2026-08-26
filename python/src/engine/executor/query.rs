//! `Operation::QueryPlan` → polars `LazyFrame` compiler.
//!
//! `Plan::nodes` arrives topologically sorted, so each node compiles in
//! slice order into a `LazyFrame` plus the kernel schema it produces. The
//! terminal (last) node is collected with the streaming engine and handed
//! back as `PolarsEngineData` batches.

use delta_kernel::engine_data::EngineData;
use delta_kernel::expressions::{Expression, Scalar};
use delta_kernel::plans::PlanResult;
use delta_kernel::plans::ir::nodes::{
    Agg, Aggregate, DynamicScan, FileType, Filter, NonNullByOperands, Operator, Project, ScanFile,
    ScanJson, ScanParquet, SemiJoin, Values,
};
use delta_kernel::plans::ir::plan::{Plan, PlanNode};
use delta_kernel::schema::{DataType as KernelDataType, SchemaRef, StructField, StructType};
use delta_kernel::{DeltaResult, Error};
use polars::prelude::{
    BooleanChunked, DataType, Expr, Field as PlField, IntoLazy, JoinArgs, JoinType, LazyFrame,
    MaintainOrderJoin, NULL, PolarsError, PolarsResult, Schema as PlSchema, Series,
    SortMultipleOptions, UnionArgs, col, concat, len, lit, when,
};
use polars_utils::pl_path::PlRefPath;
use polars_utils::pl_str::PlSmallStr;
use url::Url;

use crate::engine::data::resolve_path;
use crate::engine::handlers::{
    MetadataColumns, align_lazy, dsl_parquet_scan, ensure_no_field_id_matching,
    parse_ndjson_inferred, path_for_polars_io, split_metadata_columns, unified_scan_args,
};
use crate::engine::{PolarsEngineData, select_anchored};
use crate::errors::to_kernel_err;
use crate::translation::from_kernel::{
    column_path_to_expr, per_row_literals, projection_exprs, scalar_to_lit, translate_expr,
    translate_predicate,
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
    literals: Vec<Expr>,
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

        let batches =
            crate::engine::collect_streaming_batches(terminal.lf).map_err(to_kernel_err)?;
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
                let frames: Vec<LazyFrame> = (0..inputs.len())
                    .map(|i| Ok(input(i)?.lf.clone()))
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

    /// Constant columns are broadcast as typed literals; every frame is
    /// shaped to `schema` order before the union so vertical concat sees
    /// identical schemas.
    fn eval_scan(
        &self,
        file_type: FileType,
        files: Vec<ScanFile>,
        constant_cols: &[String],
        schema: SchemaRef,
    ) -> DeltaResult<NodeState> {
        let (read_schema, meta_cols) = split_scan_schema(&schema, constant_cols)?;
        let const_cols: Vec<(&StructField, DataType)> = constant_fields(&schema, constant_cols)?
            .into_iter()
            .map(|f| Ok((f, f.data_type.to_polars().map_err(to_kernel_err)?)))
            .collect::<DeltaResult<_>>()?;

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
                let literals = kernel_constant_literals(&f.file_constants, &const_cols)?;
                Ok(FileEntry {
                    location: f.meta.location,
                    literals,
                })
            })
            .collect::<DeltaResult<Vec<_>>>()?;

        let lf = self.scan_entries(file_type, entries, &read_schema, &schema, meta_cols)?;
        Ok(NodeState { lf, schema })
    }

    fn scan_entries(
        &self,
        file_type: FileType,
        entries: Vec<FileEntry>,
        read_schema: &StructType,
        output_schema: &SchemaRef,
        meta_cols: MetadataColumns,
    ) -> DeltaResult<LazyFrame> {
        let select = crate::translation::schema::select_exprs_for_schema(output_schema);
        let row_index = meta_cols.row_index().cloned();
        let file_path = meta_cols.file_path().cloned();
        let shape = |lf: LazyFrame, literals: Vec<Expr>| {
            let lf = if literals.is_empty() {
                lf
            } else {
                lf.with_columns(literals)
            };
            lf.select(select.clone())
        };

        let Some(first) = entries.first() else {
            return concat_frames(Vec::new(), output_schema);
        };

        // Polars sizes a frame from the columns it reads, so a scan with no
        // read columns collapses to height 0 and the constants below broadcast
        // to one synthetic row per file. `ParquetHandler` takes `num_rows` from
        // the footer for this shape; `FileEntry` carries no file size to do the
        // same, so refuse rather than under-report every file's row count.
        if read_schema.fields().next().is_none() {
            return Err(Error::Unsupported(
                "plan scan: a node whose output is entirely file constants is not supported".into(),
            ));
        }

        // Equal per-file literals (checkpoint parts, V2 sidecars) collapse into
        // one multi-file scan, keeping polars-io's cross-file parallelism.
        let uniform_constants = entries[1..].iter().all(|e| e.literals == first.literals);

        let frames: Vec<LazyFrame> = match file_type {
            FileType::Parquet if uniform_constants && meta_cols.is_empty() => {
                let literals = first.literals.clone();
                let paths: Vec<PlRefPath> = entries
                    .iter()
                    .map(|e| path_for_polars_io(&e.location))
                    .collect::<DeltaResult<_>>()?;
                let lf = self.scan_parquet_lazy(paths, read_schema, None)?;
                vec![shape(lf, literals)]
            }
            FileType::Parquet => entries
                .into_iter()
                .map(|e| {
                    let path = path_for_polars_io(&e.location)?;
                    let mut lf =
                        self.scan_parquet_lazy(vec![path], read_schema, row_index.clone())?;
                    if let Some(name) = &file_path {
                        lf = lf.with_columns([lit(e.location.as_str()).alias(name.clone())]);
                    }
                    Ok(shape(lf, e.literals))
                })
                .collect::<DeltaResult<_>>()?,
            FileType::Json => {
                // Whole-file fetches in one parallel `read_files` batch;
                // NDJSON parse per file. The align select stays lazy so only
                // the raw parses are pinned until the terminal collect.
                // TODO: fetch+parse are still eager per file. Deferring them
                // into the pipeline (restoring kernel's P&M early-out) needs
                // AnonymousScan under the streaming engine; polars-stream
                // `todo!()`s on `FileScanIR::Anonymous` through 0.55.2.
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
                        let mut lf = align_lazy(df, read_schema)?;
                        if let Some(name) = &row_index {
                            lf = row_index_as_long(lf.with_row_index(name.clone(), None), name);
                        }
                        if let Some(name) = &file_path {
                            lf = lf.with_columns([lit(e.location.as_str()).alias(name.clone())]);
                        }
                        Ok(shape(lf, e.literals))
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
        // Plan contract: a field carrying `parquet.field.id` matches by ID,
        // which polars cannot express — refuse rather than null-fill.
        ensure_no_field_id_matching(read_schema)?;
        let args = unified_scan_args(self.cloud_opts.as_ref(), None);
        let mut lf = dsl_parquet_scan(paths, read_schema, args, row_index.as_ref())?;
        let guards = non_nullable_guards(read_schema);
        if !guards.is_empty() {
            lf = lf.with_columns(guards);
        }
        Ok(lf)
    }

    /// Reads files named by `input` rows. Deletion vectors are not applied
    /// here yet — matching the kernel's own sync executor, a row with a
    /// non-null DV descriptor is an error.
    fn eval_dynamic_scan(&self, ds: DynamicScan, input: &NodeState) -> DeltaResult<NodeState> {
        let df =
            crate::engine::collect_streaming_single(input.lf.clone()).map_err(to_kernel_err)?;

        // A non-null descriptor means a deletion vector whatever an ancestor's
        // validity says, so ANDing ancestors in could only hide one.
        let dv = resolve_path(&df, &ds.dv_column).map_err(to_kernel_err)?;
        if dv.null_count() != dv.len() {
            return Err(Error::Unsupported(
                "DynamicScan with deletion vectors is not implemented".into(),
            ));
        }

        let path = resolve_path(&df, &ds.path_column).map_err(to_kernel_err)?;
        let path = path.str().map_err(to_kernel_err)?;
        let size = resolve_path(&df, &ds.file_size_column).map_err(to_kernel_err)?;
        let size = size.i64().map_err(to_kernel_err)?;

        let const_fields = constant_fields(&ds.schema, &ds.file_constant_columns)?;
        // Cast per column, not per row, so the row loop only reads values.
        let const_series: Vec<polars::prelude::Series> = ds
            .file_constant_columns
            .iter()
            .zip(const_fields.iter())
            .map(|(name, field)| {
                let series = resolve_path(&df, &delta_kernel::expressions::ColumnName::new([name]))
                    .map_err(to_kernel_err)?;
                let dt = field.data_type.to_polars().map_err(to_kernel_err)?;
                series.cast(&dt).map_err(to_kernel_err)
            })
            .collect::<DeltaResult<_>>()?;

        let names: Vec<&str> = const_fields.iter().map(|f| f.name.as_str()).collect();
        let per_row =
            per_row_literals(&const_series, &names, df.height()).map_err(to_kernel_err)?;
        let mut entries = Vec::with_capacity(df.height());
        for (row, literals) in per_row.into_iter().enumerate() {
            let rel = path
                .get(row)
                .ok_or_else(|| Error::Generic("DynamicScan path must not be null".into()))?;
            let location = ds
                .base_url
                .join(rel)
                .map_err(|e| Error::Generic(format!("DynamicScan path join failed: {e}")))?;
            // Kernel's contract requires the column to be a non-null LONG and
            // nothing more; a zero-byte commit reads as an empty batch on the
            // ScanJson path, so it must not abort here either.
            match size.get(row) {
                Some(s) if s >= 0 => {}
                _ => {
                    return Err(Error::Generic(
                        "DynamicScan file size must be a non-negative long".into(),
                    ));
                }
            }
            entries.push(FileEntry { location, literals });
        }

        let (read_schema, meta_cols) = split_scan_schema(&ds.schema, &ds.file_constant_columns)?;
        let schema = ds.schema.clone();
        let lf = self.scan_entries(ds.file_type, entries, &read_schema, &schema, meta_cols)?;
        Ok(NodeState { lf, schema })
    }
}

/// Kernel's plan contract types metadata columns LONG; polars' row index
/// is IDX_DTYPE (u32). The parquet arms cast inside `dsl_parquet_scan`;
/// this casts the JSON arm's `with_row_index` column.
fn row_index_as_long(lf: LazyFrame, name: &PlSmallStr) -> LazyFrame {
    lf.with_columns([col(name.clone()).cast(DataType::Int64)])
}

/// ScanParquet contract (kernel `plans/ir/nodes.rs`): a missing value for a
/// non-nullable field is an error, not a null-fill — but polars' unified
/// scan `Insert` policies null-fill silently. Each read column carrying a
/// non-nullable constraint gets a guard riding the column itself (a bare
/// validation expr would be projection-pruned): null under a present parent
/// errors, mirroring the JSON arm's `align`. A present-but-null value in a
/// corrupt file trips the same check.
fn non_nullable_guards(read_schema: &StructType) -> Vec<Expr> {
    fn has_constraint(f: &StructField) -> bool {
        !f.nullable
            || matches!(&f.data_type, KernelDataType::Struct(inner) if inner.fields().any(has_constraint))
    }
    fn check(
        s: &Series,
        field: &StructField,
        parent_present: Option<&BooleanChunked>,
        path: &str,
    ) -> PolarsResult<()> {
        // `null_count` is O(1); the bitmaps below are not.
        if !field.nullable && s.null_count() > 0 {
            let nulls = s.is_null();
            let violated = match parent_present {
                Some(present) => (&nulls & present).any(),
                None => nulls.any(),
            };
            if violated {
                return Err(PolarsError::ComputeError(
                    format!("scan: non-nullable field {path} is null for a present row").into(),
                ));
            }
        }
        if let KernelDataType::Struct(inner) = &field.data_type {
            // Composed with the ancestor's presence, not replacing it: a
            // child array under a NULL ancestor may still hold values, so
            // its own validity alone would re-admit a gated-out row.
            let present = match parent_present {
                Some(parent) => &s.is_not_null() & parent,
                None => s.is_not_null(),
            };
            let sc = s.struct_()?;
            // Only the constrained children need visiting, and only they
            // need their dotted path built.
            for child in inner.fields().filter(|c| has_constraint(c)) {
                let child_s = sc.field_by_name(child.name.as_str())?;
                check(
                    &child_s,
                    child,
                    Some(&present),
                    &format!("{path}.{}", child.name),
                )?;
            }
        }
        Ok(())
    }
    read_schema
        .fields()
        .filter(|f| has_constraint(f))
        .map(|f| {
            let field = f.clone();
            col(f.name.as_str()).map(
                move |column| {
                    check(
                        column.as_materialized_series(),
                        &field,
                        None,
                        field.name.as_str(),
                    )?;
                    Ok(column)
                },
                |_: &PlSchema, field: &PlField| Ok(field.clone()),
            )
        })
        .collect()
}

/// Splits a scan output schema into the fields read from files (everything
/// that is neither a constant nor a metadata column) and the requested
/// synthetic metadata columns, via the same classifier the classic
/// `ParquetHandler` uses — the two read paths must not drift.
fn split_scan_schema(
    schema: &SchemaRef,
    constant_cols: &[String],
) -> DeltaResult<(StructType, MetadataColumns)> {
    let (read_schema, meta) = split_metadata_columns(schema, "plan scan")?;
    let read_fields: Vec<StructField> = read_schema
        .fields()
        .filter(|f| !constant_cols.iter().any(|c| c == f.name.as_str()))
        .cloned()
        .collect();
    Ok((StructType::try_new(read_fields)?, meta))
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

fn kernel_constant_literals(
    constants: &[Scalar],
    cols: &[(&StructField, DataType)],
) -> DeltaResult<Vec<Expr>> {
    constants
        .iter()
        .zip(cols)
        .map(|(scalar, (field, dt))| -> DeltaResult<Expr> {
            // Same untrusted proto boundary as `Values`: a mistyped constant
            // would otherwise surface as an opaque cast failure mid-collect,
            // or cast cleanly and broadcast a silently wrong NULL.
            crate::translation::ensure_scalar_types(
                std::iter::once(scalar),
                field,
                "scan file constant",
            )?;
            Ok(scalar_to_lit(scalar)
                .cast(dt.clone())
                .alias(PlSmallStr::from_str(field.name.as_str())))
        })
        .collect()
}

fn concat_frames(frames: Vec<LazyFrame>, schema: &SchemaRef) -> DeltaResult<LazyFrame> {
    match frames.len() {
        0 => Ok(schema.empty_frame().map_err(to_kernel_err)?.lazy()),
        1 => Ok(frames.into_iter().next().expect("len checked")),
        _ => concat(frames, UnionArgs::default()).map_err(to_kernel_err),
    }
}

fn eval_values(values: Values) -> DeltaResult<NodeState> {
    let Values { schema, rows } = values;
    let row_slices: Vec<&[Scalar]> = rows.iter().map(Vec::as_slice).collect();
    // Foreign plans arrive via the proto round-trip; the shared builder
    // guards the scalar/schema agreement `build_series` panics on.
    let df = crate::translation::scalar_rows_to_frame(&schema, &row_slices, "Values")?;
    Ok(NodeState {
        lf: df.lazy(),
        schema,
    })
}

fn eval_project(project: Project, input: &NodeState) -> DeltaResult<NodeState> {
    let Project { expr, schema } = project;
    let input_struct = input.schema.as_ref();
    let out_dt = KernelDataType::Struct(Box::new(schema.as_ref().clone()));
    let lf = match expr.as_ref() {
        // Struct-shaped exprs classify into per-column ops (no nested
        // struct build in the plan).
        Expression::Struct(..) | Expression::StructPatch(..) => {
            let exprs = projection_exprs(input_struct, expr.as_ref(), &out_dt)?;
            select_anchored(input.lf.clone(), &exprs)
        }
        // Whole-row expression: evaluate once under a temp name, then
        // unnest per output field — a per-field clone would re-run an
        // opaque UDF (ParseJson, MapToStruct) once per field.
        other => {
            const ROW_EXPR: &str = "__pldl_row_expr__";
            let struct_expr = translate_expr(other, Some(&out_dt), Some(input_struct))?
                .alias(PlSmallStr::from_static(ROW_EXPR));
            let unnest: Vec<Expr> = schema
                .fields()
                .map(|f| {
                    col(PlSmallStr::from_static(ROW_EXPR))
                        .struct_()
                        .field_by_name(f.name.as_str())
                        .alias(PlSmallStr::from_str(f.name.as_str()))
                })
                .collect();
            select_anchored(input.lf.clone(), std::slice::from_ref(&struct_expr)).select(unnest)
        }
    };
    Ok(NodeState { lf, schema })
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
    // Kernel's reference executor row-encodes keys, so NULL keys compare
    // equal — `nulls_equal` matches that, not SQL join semantics.
    let mut args = JoinArgs::new(how);
    args.nulls_equal = true;
    // The plan's checkpoint anti-join feeds scan file order; the default
    // `None` would randomize it per run.
    args.maintain_order = MaintainOrderJoin::Left;
    let lf = probe
        .lf
        .clone()
        .join(build.lf.clone(), left_on, right_on, args);
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
                Agg::Min(value) => column_path_to_expr(value).min(),
                Agg::Max(value) => column_path_to_expr(value).max(),
                // Kernel wants NULL for a group with no non-NULL value; polars
                // `sum` returns 0 there.
                Agg::Sum(value) => {
                    let v = column_path_to_expr(value);
                    when(v.clone().is_not_null().any(true))
                        .then(v.sum())
                        .otherwise(lit(NULL))
                        .cast(DataType::Int64)
                }
                // polars `count` excludes NULLs, `len` includes them; both
                // yield IdxSize, and kernel types these columns LONG.
                Agg::Count(value) => column_path_to_expr(value).count().cast(DataType::Int64),
                Agg::CountStar => len().cast(DataType::Int64),
                Agg::MinNonNullBy(ops) => non_null_by(ops, true),
                Agg::MaxNonNullBy(ops) => non_null_by(ops, false),
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
/// considering only rows where `null_sentinel` and `key` are both non-null;
/// NULL when no row qualifies. A winning `value` may itself be NULL and is
/// retained: kernel's scan plan sentinels on the file-action key so a winning
/// `remove` yields a NULL `add`, which is how a tombstone drops the file.
/// See kernel `Agg::max_non_null_by` for the exact contract.
fn non_null_by(ops: &NonNullByOperands, ascending: bool) -> Expr {
    let NonNullByOperands {
        value,
        null_sentinel,
        key,
    } = ops;
    let v = column_path_to_expr(value);
    let k = column_path_to_expr(key);
    let keep = column_path_to_expr(null_sentinel)
        .is_not_null()
        .and(k.clone().is_not_null());
    // Stable: duplicate keys otherwise pick a different row per run, and the
    // metadata plan resolves the winning protocol/metaData through this.
    let sorted = v.filter(keep.clone()).sort_by(
        [k.filter(keep)],
        SortMultipleOptions::default().with_maintain_order(true),
    );
    if ascending {
        sorted.first()
    } else {
        sorted.last()
    }
}

#[cfg(test)]
mod scan_entries_tests {
    use delta_kernel::schema::MetadataColumnSpec;

    use std::sync::Arc;

    use delta_kernel::schema::DataType;
    use polars::prelude::{DataType as PlDataType, ParquetWriter, lit};

    use super::*;
    use crate::engine::handlers::ObjectStoreStorageHandler;

    fn executor() -> PolarsPlanExecutor {
        let url = Url::parse("file:///").unwrap();
        let rt = crate::engine::rt();
        let storage =
            Arc::new(ObjectStoreStorageHandler::new(&url, std::iter::empty(), rt).unwrap());
        PolarsPlanExecutor::new(storage, None)
    }

    fn long_field(name: &str) -> StructField {
        StructField::nullable(name, DataType::LONG)
    }

    fn plan_scan_count(literals_per_file: [Vec<Expr>; 2]) -> usize {
        let read_schema = StructType::try_new([long_field("id")]).unwrap();
        let output_schema =
            Arc::new(StructType::try_new([long_field("id"), long_field("v")]).unwrap());
        let entries = literals_per_file
            .into_iter()
            .enumerate()
            .map(|(i, literals)| FileEntry {
                location: Url::parse(&format!("file:///t/{i}.parquet")).unwrap(),
                literals,
            })
            .collect();
        let lf = executor()
            .scan_entries(
                FileType::Parquet,
                entries,
                &read_schema,
                &output_schema,
                MetadataColumns::default(),
            )
            .unwrap();
        format!("{:?}", lf.logical_plan).matches("Scan {").count()
    }

    /// The DynamicScan sidecar shape: identical constants across files.
    #[test]
    fn equal_literals_collapse_to_one_multifile_scan() {
        let literals = || vec![lit(7i64).alias("v")];
        assert_eq!(plan_scan_count([literals(), literals()]), 1);
    }

    #[test]
    fn differing_literals_scan_per_file() {
        let literals = |n| vec![lit(n).alias("v")];
        assert_eq!(plan_scan_count([literals(1i64), literals(2i64)]), 2);
    }

    /// A read schema with no fields would report one synthetic row per file
    /// instead of the file's row count.
    #[test]
    fn a_scan_with_no_read_columns_is_refused() {
        let read_schema = StructType::try_new(Vec::<StructField>::new()).unwrap();
        let output_schema = Arc::new(StructType::try_new([long_field("v")]).unwrap());
        let entries = vec![FileEntry {
            location: Url::parse("file:///t/0.parquet").unwrap(),
            literals: vec![lit(7i64).alias("v")],
        }];
        let err = executor()
            .scan_entries(
                FileType::Parquet,
                entries,
                &read_schema,
                &output_schema,
                MetadataColumns::default(),
            )
            .err()
            .expect("a constants-only scan must not report one row per file");
        assert!(err.to_string().contains("file constants"), "got: {err}");
    }

    /// The polars behaviour that refusal guards against.
    #[test]
    fn a_frame_with_no_columns_loses_its_height() {
        let df = polars::df!("x" => [1i64, 2, 3]).unwrap();
        let empty = df
            .lazy()
            .drop(polars::prelude::cols(["x"]))
            .collect()
            .unwrap();
        assert_eq!(empty.height(), 0, "height comes from the columns read");
        let broadcast = empty
            .lazy()
            .with_columns([lit(7i64).alias("v")])
            .select([col("v")])
            .collect()
            .unwrap();
        assert_eq!(broadcast.height(), 1, "constants broadcast to one row");
    }

    /// Kernel's reference executor row-encodes join keys, so NULL keys
    /// compare equal; the polars join must set `nulls_equal` to match.
    #[test]
    fn semi_join_matches_null_keys() {
        use delta_kernel::expressions::ColumnName;
        use delta_kernel::plans::ir::nodes::SemiJoin;
        use polars::prelude::IntoLazy;

        let schema = Arc::new(StructType::try_new([long_field("k")]).unwrap());
        let state = |vals: &[Option<i64>]| NodeState {
            lf: polars::df!("k" => vals).unwrap().lazy(),
            schema: schema.clone(),
        };
        let node = |inverted| SemiJoin {
            inverted,
            probe_keys: vec![ColumnName::new(["k"])],
            build_keys: vec![ColumnName::new(["k"])],
        };
        let probe = state(&[Some(1), None]);
        let build = state(&[None]);

        let semi = eval_semi_join(node(false), &probe, &build)
            .unwrap()
            .lf
            .collect()
            .unwrap();
        assert_eq!(semi.height(), 1, "NULL probe key must match NULL build key");
        assert_eq!(semi.column("k").unwrap().null_count(), 1);

        let anti = eval_semi_join(node(true), &probe, &build)
            .unwrap()
            .lf
            .collect()
            .unwrap();
        assert_eq!(anti.height(), 1, "anti join must drop the matched NULL row");
        assert_eq!(anti.column("k").unwrap().null_count(), 0);
    }

    /// Foreign plans reach `eval_values` through the proto round-trip, so a
    /// mismatched scalar must surface as an Error, not a `build_series`
    /// panic unwinding into PyO3.
    #[test]
    fn values_scalar_type_mismatch_errors() {
        use delta_kernel::plans::ir::nodes::Values;

        let values = Values {
            schema: Arc::new(
                StructType::try_new([StructField::nullable("x", DataType::LONG)]).unwrap(),
            ),
            rows: vec![vec![Scalar::String("oops".into())]],
        };
        assert!(eval_values(values).is_err());
    }

    /// ScanParquet contract: `parquet.field.id` fields match by ID, which
    /// polars cannot express — the plan build must refuse.
    #[test]
    fn field_id_read_schema_is_rejected() {
        use delta_kernel::schema::MetadataValue;

        let read_schema = StructType::try_new([StructField::nullable("id", DataType::LONG)
            .with_metadata([("parquet.field.id", MetadataValue::Number(3))])])
        .unwrap();
        let output_schema = Arc::new(StructType::try_new([long_field("id")]).unwrap());
        let entries = vec![FileEntry {
            location: Url::parse("file:///t/0.parquet").unwrap(),
            literals: vec![],
        }];
        let err = match executor().scan_entries(
            FileType::Parquet,
            entries,
            &read_schema,
            &output_schema,
            MetadataColumns::default(),
        ) {
            Err(e) => e,
            Ok(_) => panic!("field-id read schema must be refused"),
        };
        assert!(err.to_string().contains("parquet.field.id"), "got: {err}");
    }

    fn ridx_meta() -> MetadataColumns {
        let schema = StructType::try_new([StructField::create_metadata_column(
            "ridx",
            MetadataColumnSpec::RowIndex,
        )])
        .unwrap();
        split_metadata_columns(&schema, "test").unwrap().1
    }

    /// The classic handler synthesizes FilePath; the plan path refused the
    /// identical schema. One classifier serves both, and each entry carries
    /// its own URL.
    #[test]
    fn plan_scan_synthesizes_file_path_per_file() {
        let dir = tempfile::tempdir().unwrap();
        for (name, v) in [("a.parquet", 1i64), ("b.parquet", 2)] {
            let mut df = polars::df!("id" => [v]).unwrap();
            ParquetWriter::new(std::fs::File::create(dir.path().join(name)).unwrap())
                .finish(&mut df)
                .unwrap();
        }
        let schema = Arc::new(
            StructType::try_new([
                long_field("id"),
                StructField::create_metadata_column("_file", MetadataColumnSpec::FilePath),
            ])
            .unwrap(),
        );
        let (read_schema, meta) =
            split_scan_schema(&schema, &[]).expect("the plan path supports FilePath");
        let entries = ["a.parquet", "b.parquet"]
            .map(|n| FileEntry {
                location: Url::from_file_path(dir.path().join(n)).unwrap(),
                literals: vec![],
            })
            .into_iter()
            .collect();
        let df = executor()
            .scan_entries(FileType::Parquet, entries, &read_schema, &schema, meta)
            .unwrap()
            .collect()
            .unwrap();
        assert_eq!(
            df.column("id")
                .unwrap()
                .i64()
                .unwrap()
                .iter()
                .flatten()
                .collect::<Vec<_>>(),
            [1, 2]
        );
        let files: Vec<String> = df
            .column("_file")
            .unwrap()
            .str()
            .unwrap()
            .iter()
            .flatten()
            .map(str::to_string)
            .collect();
        assert!(files[0].ends_with("a.parquet"), "got: {files:?}");
        assert!(files[1].ends_with("b.parquet"), "got: {files:?}");
    }

    /// Kernel's plan contract types metadata columns LONG; polars' native
    /// row index is IDX_DTYPE (u32). Both scan arms and the empty branch
    /// must agree on Int64.
    #[test]
    fn row_index_is_long() {
        let dir = tempfile::tempdir().unwrap();
        let pq = dir.path().join("f.parquet");
        let mut df = polars::df!("id" => [1i64, 2, 3]).unwrap();
        ParquetWriter::new(std::fs::File::create(&pq).unwrap())
            .finish(&mut df)
            .unwrap();
        let json = dir.path().join("f.json");
        std::fs::write(&json, "{\"id\": 1}\n{\"id\": 2}\n{\"id\": 3}\n").unwrap();

        let read_schema = StructType::try_new([long_field("id")]).unwrap();
        let output_schema =
            Arc::new(StructType::try_new([long_field("id"), long_field("ridx")]).unwrap());
        let executor = executor();

        for (file_type, path) in [(FileType::Parquet, &pq), (FileType::Json, &json)] {
            let entries = vec![FileEntry {
                location: Url::from_file_path(path).unwrap(),
                literals: vec![],
            }];
            let df = executor
                .scan_entries(
                    file_type,
                    entries,
                    &read_schema,
                    &output_schema,
                    ridx_meta(),
                )
                .unwrap()
                .collect()
                .unwrap();
            let ridx = df.column("ridx").unwrap();
            assert_eq!(ridx.dtype(), &PlDataType::Int64);
            assert_eq!(
                ridx.i64().unwrap().iter().flatten().collect::<Vec<_>>(),
                [0, 1, 2]
            );
        }

        let empty = executor
            .scan_entries(
                FileType::Parquet,
                vec![],
                &read_schema,
                &output_schema,
                ridx_meta(),
            )
            .unwrap()
            .collect()
            .unwrap();
        assert_eq!(empty.column("ridx").unwrap().dtype(), &PlDataType::Int64);
    }

    /// ScanParquet contract: a non-nullable field with no match in the file
    /// is an error, not a null-fill (the JSON arm enforces this via align).
    #[test]
    fn missing_non_nullable_parquet_column_errors() {
        let dir = tempfile::tempdir().unwrap();
        let pq = dir.path().join("f.parquet");
        let mut df = polars::df!("id" => [1i64, 2, 3]).unwrap();
        ParquetWriter::new(std::fs::File::create(&pq).unwrap())
            .finish(&mut df)
            .unwrap();

        let read_schema = StructType::try_new([
            StructField::not_null("id", DataType::LONG),
            StructField::not_null("req", DataType::LONG),
        ])
        .unwrap();
        let output_schema = Arc::new(read_schema.clone());
        let entries = vec![FileEntry {
            location: Url::from_file_path(&pq).unwrap(),
            literals: vec![],
        }];
        let out = executor()
            .scan_entries(
                FileType::Parquet,
                entries,
                &read_schema,
                &output_schema,
                MetadataColumns::default(),
            )
            .unwrap()
            .collect();
        let err = out.expect_err("missing non-nullable column must error");
        assert!(err.to_string().contains("req"), "got: {err}");
    }

    /// The checkpoint shape: nullable action struct, non-nullable leaf. A
    /// file whose struct lacks the leaf null-fills it for present parents.
    /// The guard walks only constrained children and short-circuits on
    /// `null_count`. Pin what that must not change: an unconstrained leaf
    /// missing from the file still null-fills instead of erroring.
    #[test]
    fn missing_nullable_struct_leaf_null_fills() {
        use polars::prelude::{IntoLazy, as_struct, col};

        let dir = tempfile::tempdir().unwrap();
        let pq = dir.path().join("n.parquet");
        let mut df = polars::df!("other" => [1i64, 2])
            .unwrap()
            .lazy()
            .select([as_struct(vec![col("other")]).alias("s")])
            .collect()
            .unwrap();
        ParquetWriter::new(std::fs::File::create(&pq).unwrap())
            .finish(&mut df)
            .unwrap();

        let read_schema = StructType::try_new([StructField::nullable(
            "s",
            DataType::Struct(Box::new(
                StructType::try_new([StructField::nullable("q", DataType::LONG)]).unwrap(),
            )),
        )])
        .unwrap();
        let output_schema = Arc::new(read_schema.clone());
        let entries = vec![FileEntry {
            location: Url::from_file_path(&pq).unwrap(),
            literals: vec![],
        }];
        let out = executor()
            .scan_entries(
                FileType::Parquet,
                entries,
                &read_schema,
                &output_schema,
                MetadataColumns::default(),
            )
            .unwrap()
            .collect()
            .expect("a nullable leaf must null-fill, not error");
        assert_eq!(out.height(), 2);
    }

    #[test]
    fn missing_non_nullable_struct_leaf_errors() {
        use polars::prelude::{IntoLazy, as_struct, col};

        let dir = tempfile::tempdir().unwrap();
        let pq = dir.path().join("f.parquet");
        let mut df = polars::df!("other" => [1i64, 2])
            .unwrap()
            .lazy()
            .select([as_struct(vec![col("other")]).alias("s")])
            .collect()
            .unwrap();
        ParquetWriter::new(std::fs::File::create(&pq).unwrap())
            .finish(&mut df)
            .unwrap();

        let read_schema = StructType::try_new([StructField::nullable(
            "s",
            DataType::Struct(Box::new(
                StructType::try_new([StructField::not_null("path", DataType::STRING)]).unwrap(),
            )),
        )])
        .unwrap();
        let output_schema = Arc::new(read_schema.clone());
        let entries = vec![FileEntry {
            location: Url::from_file_path(&pq).unwrap(),
            literals: vec![],
        }];
        let out = executor()
            .scan_entries(
                FileType::Parquet,
                entries,
                &read_schema,
                &output_schema,
                MetadataColumns::default(),
            )
            .unwrap()
            .collect();
        let err = out.expect_err("null leaf under a present parent must error");
        assert!(err.to_string().contains("s.path"), "got: {err}");
    }

    /// A projection whose exprs reference no input column (all literals)
    /// must still emit one row per input row; polars sizes a bare select
    /// from its expressions.
    #[test]
    fn all_literal_project_keeps_input_height() {
        let input = NodeState {
            lf: polars::df!("k" => [1i64, 2, 3]).unwrap().lazy(),
            schema: Arc::new(StructType::try_new([long_field("k")]).unwrap()),
        };
        let project = Project {
            expr: Arc::new(Expression::struct_from([Expression::literal(7i64)])),
            schema: Arc::new(StructType::try_new([long_field("v")]).unwrap()),
        };
        let df = eval_project(project, &input).unwrap().lf.collect().unwrap();
        assert_eq!(df.height(), 3, "literal-only projection must keep height");
        assert_eq!(df.get_column_names(), ["v"]);
    }
}

#[cfg(test)]
mod dynamic_scan_size_tests {
    use std::sync::Arc;

    use delta_kernel::expressions::ColumnName;
    use delta_kernel::schema::DataType;
    use polars::prelude::IntoLazy;

    use super::*;
    use crate::engine::handlers::ObjectStoreStorageHandler;

    /// Kernel's contract asks only for a non-null LONG. A zero-byte commit
    /// is an empty batch on the ScanJson path, so the size column must not
    /// abort the scan here either.
    #[test]
    fn zero_byte_file_is_accepted() {
        let url = Url::parse("file:///t/").unwrap();
        let rt = crate::engine::rt();
        let storage =
            Arc::new(ObjectStoreStorageHandler::new(&url, std::iter::empty(), rt).unwrap());
        let executor = PolarsPlanExecutor::new(storage, None);

        let schema: SchemaRef =
            Arc::new(StructType::try_new([StructField::nullable("id", DataType::LONG)]).unwrap());
        let df = polars::df!(
            "path" => ["empty.parquet"],
            "size" => [0i64],
            "modtime" => [0i64],
            "dv" => [None::<&str>],
        )
        .unwrap();
        let input = NodeState {
            lf: df.lazy(),
            schema: schema.clone(),
        };
        let ds = DynamicScan {
            schema,
            file_type: FileType::Parquet,
            base_url: url,
            file_constant_columns: vec![],
            path_column: ColumnName::new(["path"]),
            file_size_column: ColumnName::new(["size"]),
            last_modified_column: ColumnName::new(["modtime"]),
            dv_column: ColumnName::new(["dv"]),
        };
        executor
            .eval_dynamic_scan(ds, &input)
            .expect("a zero-byte file must not abort the scan");
    }
}

#[cfg(test)]
mod scalar_type_guard_tests {
    use delta_kernel::schema::DataType as KernelType;

    use super::*;

    /// File constants arrive over the same proto round trip as `Values`, so
    /// a mistyped one must be named here rather than surfacing as an opaque
    /// cast failure mid-collect — or casting cleanly to a wrong NULL.
    #[test]
    fn mistyped_file_constant_is_rejected() {
        let field = StructField::nullable("d", KernelType::DATE);
        let cols = [(&field, DataType::Date)];
        let err = kernel_constant_literals(&[Scalar::String("2024-01-01".into())], &cols)
            .expect_err("a String constant for a DATE column must error");
        assert!(err.to_string().contains("scan file constant"), "got: {err}");
    }

    /// A NULL constant is typed by its own field, so it stays accepted.
    #[test]
    fn null_and_matching_constants_pass() {
        let field = StructField::nullable("d", KernelType::DATE);
        let cols = [(&field, DataType::Date)];
        kernel_constant_literals(&[Scalar::Null(KernelType::DATE)], &cols).unwrap();
        kernel_constant_literals(&[Scalar::Date(19_000)], &cols).unwrap();
    }
}

#[cfg(test)]
mod aggregate_tests {
    use std::sync::Arc;

    use delta_kernel::expressions::ColumnName;
    use delta_kernel::schema::DataType as KernelType;
    use polars::prelude::{AnyValue, DataFrame, IntoLazy};

    use super::*;

    fn state(df: polars::prelude::DataFrame, fields: &[(&str, KernelType)]) -> NodeState {
        let schema: SchemaRef = Arc::new(
            StructType::try_new(
                fields
                    .iter()
                    .map(|(n, t)| StructField::nullable(*n, t.clone())),
            )
            .unwrap(),
        );
        NodeState {
            lf: df.lazy(),
            schema,
        }
    }

    fn run(
        input: &NodeState,
        group_by: &[&str],
        aggs: Vec<Agg>,
        out: &[(&str, KernelType)],
    ) -> DataFrame {
        let schema: SchemaRef = Arc::new(
            StructType::try_new(
                out.iter()
                    .map(|(n, t)| StructField::nullable(*n, t.clone())),
            )
            .unwrap(),
        );
        let agg = Aggregate {
            group_by: group_by.iter().map(|k| ColumnName::new([*k])).collect(),
            aggs,
            schema,
        };
        eval_aggregate(agg, input).unwrap().lf.collect().unwrap()
    }

    /// Kernel's scan plan sentinels `max_non_null_by` on the file-action key,
    /// so the newest action for a path wins even when it is a `remove` whose
    /// `add` is NULL. Retaining that NULL is what drops a tombstoned file; the
    /// pre-0.27 two-arg form skipped NULL values and resurrected the old add.
    #[test]
    fn winning_remove_retains_its_null_value() {
        let df = polars::df!(
            "key" => ["f1", "f1"],
            "add" => [Some(10i64), None],
            "version" => [1i64, 2],
        )
        .unwrap();
        let input = state(
            df,
            &[
                ("key", KernelType::STRING),
                ("add", KernelType::LONG),
                ("version", KernelType::LONG),
            ],
        );
        let out = run(
            &input,
            &["key"],
            vec![Agg::max_non_null_by(
                ColumnName::new(["add"]),
                ColumnName::new(["key"]),
                ColumnName::new(["version"]),
            )],
            &[("key", KernelType::STRING), ("add", KernelType::LONG)],
        );
        assert_eq!(out.column("add").unwrap().get(0).unwrap(), AnyValue::Null);
    }

    /// A NULL sentinel disqualifies its row even when that row holds the
    /// greatest key.
    #[test]
    fn null_sentinel_row_loses_despite_greatest_key() {
        let df = polars::df!(
            "val" => [Some(10i64), Some(99i64)],
            "sentinel" => [Some("present"), None],
            "version" => [1i64, 5],
        )
        .unwrap();
        let input = state(
            df,
            &[
                ("val", KernelType::LONG),
                ("sentinel", KernelType::STRING),
                ("version", KernelType::LONG),
            ],
        );
        let out = run(
            &input,
            &[],
            vec![Agg::max_non_null_by(
                ColumnName::new(["val"]),
                ColumnName::new(["sentinel"]),
                ColumnName::new(["version"]),
            )],
            &[("val", KernelType::LONG)],
        );
        assert_eq!(
            out.column("val").unwrap().get(0).unwrap(),
            AnyValue::Int64(10)
        );
    }

    /// polars `sum` returns 0 for a group with no non-NULL value; kernel's
    /// contract is NULL there.
    #[test]
    fn sum_is_null_when_no_non_null_value() {
        let df = polars::df!("v" => [None::<i64>, None]).unwrap();
        let input = state(df, &[("v", KernelType::LONG)]);
        let out = run(
            &input,
            &[],
            vec![Agg::sum(ColumnName::new(["v"]))],
            &[("v", KernelType::LONG)],
        );
        assert_eq!(out.column("v").unwrap().get(0).unwrap(), AnyValue::Null);
        assert_eq!(out.column("v").unwrap().dtype(), &DataType::Int64);
    }

    #[test]
    fn sum_adds_non_null_values() {
        let df = polars::df!("v" => [Some(3i64), None, Some(5), Some(1)]).unwrap();
        let input = state(df, &[("v", KernelType::LONG)]);
        let out = run(
            &input,
            &[],
            vec![Agg::sum(ColumnName::new(["v"]))],
            &[("v", KernelType::LONG)],
        );
        assert_eq!(out.column("v").unwrap().get(0).unwrap(), AnyValue::Int64(9));
    }

    /// The `when/then/otherwise` that gives `Sum` its NULL-on-empty semantics
    /// must still reduce to one scalar per group, not a list column.
    #[test]
    fn grouped_sum_and_count_stay_scalar_per_group() {
        let df = polars::df!(
            "g" => ["a", "a", "b"],
            "v" => [Some(3i64), Some(5), None],
        )
        .unwrap();
        let input = state(df, &[("g", KernelType::STRING), ("v", KernelType::LONG)]);
        let out = run(
            &input,
            &["g"],
            vec![
                Agg::sum(ColumnName::new(["v"])),
                Agg::count(ColumnName::new(["v"])),
                Agg::count_star(),
            ],
            &[
                ("g", KernelType::STRING),
                ("total", KernelType::LONG),
                ("n", KernelType::LONG),
                ("rows", KernelType::LONG),
            ],
        );
        assert_eq!(out.column("total").unwrap().dtype(), &DataType::Int64);
        let total = out.column("total").unwrap();
        let n = out.column("n").unwrap();
        let rows = out.column("rows").unwrap();
        // group "a" first: group_by_stable preserves first-seen order.
        assert_eq!(total.get(0).unwrap(), AnyValue::Int64(8));
        assert_eq!(n.get(0).unwrap(), AnyValue::Int64(2));
        assert_eq!(rows.get(0).unwrap(), AnyValue::Int64(2));
        // group "b" is all-NULL: sum is NULL, count 0, count_star 1.
        assert_eq!(total.get(1).unwrap(), AnyValue::Null);
        assert_eq!(n.get(1).unwrap(), AnyValue::Int64(0));
        assert_eq!(rows.get(1).unwrap(), AnyValue::Int64(1));
    }

    /// `count` skips NULLs, `count_star` counts rows, and kernel types both
    /// columns LONG.
    #[test]
    fn count_skips_nulls_and_count_star_does_not() {
        let df = polars::df!("v" => [Some(3i64), None, Some(5), Some(1)]).unwrap();
        let input = state(df, &[("v", KernelType::LONG)]);
        let out = run(
            &input,
            &[],
            vec![Agg::count(ColumnName::new(["v"])), Agg::count_star()],
            &[("n", KernelType::LONG), ("total", KernelType::LONG)],
        );
        assert_eq!(out.column("n").unwrap().get(0).unwrap(), AnyValue::Int64(3));
        assert_eq!(
            out.column("total").unwrap().get(0).unwrap(),
            AnyValue::Int64(4)
        );
        assert_eq!(out.column("n").unwrap().dtype(), &DataType::Int64);
    }
}
