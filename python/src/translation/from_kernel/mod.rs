//! Kernel `Expression` / `Predicate` → polars `Expr`, translated once at
//! build time and replayed per batch via `LazyFrame::select`.
//!
//! Submodule layout:
//! - [`expr`] — value-expression translation (the main dispatcher).
//! - [`transform`] — sparse schema-rewrite translation.
//! - [`predicate`] — boolean / comparison / junction translation.
//! - [`scalar`] — kernel `Scalar` → polars `Series` / `Expr` / `lit`.

use std::sync::Arc;

use delta_kernel::engine_data::EngineData;
use delta_kernel::expressions::{Expression, ExpressionRef, ExpressionStructPatch, Scalar};
use delta_kernel::schema::{DataType as KernelDataType, SchemaRef, StructField, StructType};
use delta_kernel::{
    DeltaResult, Error, EvaluationHandler, ExpressionEvaluator, PredicateEvaluator,
};
use polars::prelude::{
    Column, DataFrame, Expr, IntoColumn, IntoLazy, Scalar as PolarsScalar, Series, col, lit,
};
use polars_utils::pl_str::PlSmallStr;

use crate::consts::KERNEL_OUTPUT_COL;
use crate::engine::PolarsEngineData;
use crate::errors::to_kernel_err;
use crate::translation::schema::KernelSchemaExt;

mod expr;
mod predicate;
mod scalar;
mod transform;

pub(crate) use expr::{column_path_to_expr, null_gated, parse_partition_column, translate_expr};
pub(crate) use predicate::translate_predicate;
pub(crate) use scalar::{
    build_series, empty_typed_list_expr, ensure_scalar_types, per_row_literals, scalar_to_lit,
};

use predicate::PolarsPredicateEvaluator;
use scalar::try_to_polars_scalar;
use transform::{TransformSlot, translate_transform, walk_transform_slots};

/// Select-list for evaluating `expression` (with struct `output_type`) over a
/// frame shaped like `input_schema` — one aliased polars `Expr` per output
/// field. Shared by the `Project` plan node and the expression evaluator.
pub(crate) fn projection_exprs(
    input_schema: &StructType,
    expression: &Expression,
    output_type: &KernelDataType,
) -> DeltaResult<Vec<Expr>> {
    let ops = build_column_ops(input_schema, expression, output_type)?;
    Ok(ops.iter().map(op_to_expr).collect())
}

pub(crate) struct PolarsEvaluationHandler;

impl PolarsEvaluationHandler {
    pub(crate) fn new() -> Self {
        Self
    }
}

impl EvaluationHandler for PolarsEvaluationHandler {
    fn new_expression_evaluator(
        &self,
        input_schema: SchemaRef,
        expression: ExpressionRef,
        output_type: KernelDataType,
    ) -> DeltaResult<Arc<dyn ExpressionEvaluator>> {
        let ops = build_column_ops(input_schema.as_ref(), expression.as_ref(), &output_type)?;
        Ok(Arc::new(PolarsExpressionEvaluator::new(
            ops,
            input_schema.num_fields(),
        )))
    }

    fn new_predicate_evaluator(
        &self,
        input_schema: SchemaRef,
        predicate: delta_kernel::expressions::PredicateRef,
    ) -> DeltaResult<Arc<dyn PredicateEvaluator>> {
        let predicate_expr = translate_predicate(&predicate, Some(input_schema.as_ref()))?;
        Ok(Arc::new(PolarsPredicateEvaluator { predicate_expr }))
    }

    /// Read path never reaches this — kernel only calls `null_row` from
    /// checkpoint emit (`create_checkpoint_metadata_batch` builds an
    /// all-null row, then transforms one field to set `checkpointMetadata`).
    /// Implemented anyway so the handler is complete against the trait.
    fn null_row(&self, output_schema: SchemaRef) -> DeltaResult<Box<dyn EngineData>> {
        let schema = output_schema.to_polars().map_err(to_kernel_err)?;
        let columns = schema
            .iter()
            .map(|(name, dtype)| Series::new_null(name.clone(), 1).cast(dtype))
            .collect::<Result<Vec<_>, _>>()
            .map_err(to_kernel_err)?;
        let df = DataFrame::new(
            1,
            columns.into_iter().map(IntoColumn::into_column).collect(),
        )
        .map_err(to_kernel_err)?;
        Ok(Box::new(PolarsEngineData::new(df)))
    }

    /// Read path never reaches this — kernel only calls `create_many` from
    /// transaction commit (action-row batches handed to `write_json_file`)
    /// and checkpoint emit (`create_sidecar_action_batch`). Implemented
    /// anyway so the handler is complete against the trait.
    fn create_many(
        &self,
        schema: SchemaRef,
        rows: &[&[Scalar]],
    ) -> DeltaResult<Box<dyn EngineData>> {
        let row_count = rows.len();

        if row_count == 0 {
            let df =
                DataFrame::empty_with_schema(schema.to_polars().map_err(to_kernel_err)?.as_ref());
            return Ok(Box::new(PolarsEngineData::new(df)));
        }

        let fields: Vec<_> = schema.fields().collect();

        if rows.iter().any(|row| row.len() != fields.len()) {
            return Err(Error::Generic(format!(
                "create_many: expected {} scalars per row, got mismatched row widths",
                fields.len()
            )));
        }

        let columns = fields
            .iter()
            .enumerate()
            .map(|(col_idx, field)| {
                let column_scalars: Vec<&Scalar> = rows.iter().map(|row| &row[col_idx]).collect();
                // Untrusted boundary — `build_series` panics on a mismatch.
                ensure_scalar_types(column_scalars.iter().copied(), field, "create_many")?;
                build_series(&field.name, &field.data_type, &column_scalars)
            })
            .collect::<DeltaResult<Vec<_>>>()?;

        let df = DataFrame::new(
            row_count,
            columns.into_iter().map(IntoColumn::into_column).collect(),
        )
        .map_err(to_kernel_err)?;

        Ok(Box::new(PolarsEngineData::new(df)))
    }
}

/// One output column's evaluation plan. Real-world Transforms emit only
/// `Literal` (partition injection) and `Passthrough` (identity / column-map
/// rename) ops, both of which bypass the lazy planner. `Computed` is the
/// escape hatch for arbitrary expressions and forces the mixed lazy path.
enum ColumnOp {
    Literal {
        name: PlSmallStr,
        scalar: PolarsScalar,
    },
    /// `input_idx` indexes into the input DataFrame's column vector (kernel
    /// guarantees its schema matches the evaluator's `input_schema`).
    /// `input_name` is kept for the lazy fallback `col(name)` lookup.
    Passthrough {
        input_idx: usize,
        input_name: PlSmallStr,
        output: PlSmallStr,
    },
    Computed {
        /// Pre-aliased to the output column name.
        expr: Expr,
    },
}

struct PolarsExpressionEvaluator {
    ops: Vec<ColumnOp>,
    /// Both derived from `ops` at construction: recomputing them per batch
    /// would re-clone every expr tree on the log-replay hot path.
    lazy_exprs: Vec<Expr>,
    all_simple: bool,
    /// `Passthrough` addresses input columns by position, so the fast path
    /// only holds for a batch as wide as the schema it was built from.
    input_width: usize,
}

impl PolarsExpressionEvaluator {
    fn new(ops: Vec<ColumnOp>, input_width: usize) -> Self {
        let lazy_exprs = ops.iter().map(op_to_expr).collect();
        let all_simple = !ops.iter().any(|op| matches!(op, ColumnOp::Computed { .. }));
        Self {
            ops,
            lazy_exprs,
            all_simple,
            input_width,
        }
    }
}

impl ExpressionEvaluator for PolarsExpressionEvaluator {
    fn evaluate(&self, batch: &dyn EngineData) -> DeltaResult<Box<dyn EngineData>> {
        let df = downcast_engine_data(batch)?.dataframe();
        let height = df.height();

        if self.all_simple && df.width() == self.input_width {
            let input_cols = df.columns();
            let columns: Vec<Column> = self
                .ops
                .iter()
                .map(|op| match op {
                    ColumnOp::Literal { name, scalar } => {
                        Column::new_scalar(name.clone(), scalar.clone(), height)
                    }
                    ColumnOp::Passthrough {
                        input_idx, output, ..
                    } => input_cols[*input_idx].clone().with_name(output.clone()),
                    ColumnOp::Computed { .. } => unreachable!("all_simple guards this"),
                })
                .collect();
            // SAFETY: output names come from the kernel output schema (unique)
            // and every Column has length `height` by construction.
            let result = unsafe { DataFrame::new_unchecked(height, columns) };
            return Ok(Box::new(PolarsEngineData::new(result)));
        }

        // Resolves passthroughs by name, so a batch that does not match the
        // declared input schema errors here instead of silently picking the
        // column that happens to sit at that position.
        let result = crate::engine::select_anchored(df.clone().lazy(), &self.lazy_exprs)
            .collect()
            .map_err(to_kernel_err)?;
        Ok(Box::new(PolarsEngineData::new(result)))
    }
}

fn op_to_expr(op: &ColumnOp) -> Expr {
    match op {
        ColumnOp::Literal { name, scalar } => lit(scalar.clone()).alias(name.clone()),
        ColumnOp::Passthrough {
            input_name, output, ..
        } => col(input_name.clone()).alias(output.clone()),
        ColumnOp::Computed { expr } => expr.clone(),
    }
}

fn build_column_ops(
    input_schema: &StructType,
    expression: &Expression,
    output_type: &KernelDataType,
) -> DeltaResult<Vec<ColumnOp>> {
    match (output_type, expression) {
        (KernelDataType::Struct(output_struct), Expression::StructPatch(t)) => {
            build_transform_ops(t, output_struct, input_schema)
        }
        (KernelDataType::Struct(output_struct), Expression::Struct(children, nullability)) => {
            // A top-level gate would null whole rows — no DataFrame representation.
            if nullability.is_some() {
                return Err(Error::Unsupported(
                    "PolarsExpressionEvaluator: nullability predicate on a top-level output struct"
                        .into(),
                ));
            }
            let n_out = output_struct.num_fields();
            if children.len() != n_out {
                return Err(Error::Generic(format!(
                    "PolarsExpressionEvaluator: output struct has {n_out} fields but expression has {} children",
                    children.len()
                )));
            }
            children
                .iter()
                .zip(output_struct.fields())
                .map(|(child, field)| {
                    classify_single(
                        child.as_ref(),
                        &field.data_type,
                        input_schema,
                        PlSmallStr::from_str(field.name.as_str()),
                    )
                })
                .collect()
        }
        _ => Ok(vec![classify_single(
            expression,
            output_type,
            input_schema,
            PlSmallStr::from_static(KERNEL_OUTPUT_COL),
        )?]),
    }
}

/// Nested `input_path` patches (rare) fall back to the lazy path wholesale
/// — `ColumnOp::Passthrough` can't address columns inside a struct projection.
fn build_transform_ops(
    t: &ExpressionStructPatch,
    output_struct: &StructType,
    input_schema: &StructType,
) -> DeltaResult<Vec<ColumnOp>> {
    if t.input_path.is_some() {
        let struct_expr = translate_transform(t, output_struct, input_schema)?;
        return Ok(output_struct
            .fields()
            .map(|f| {
                let alias = PlSmallStr::from_str(f.name.as_str());
                ColumnOp::Computed {
                    expr: struct_expr
                        .clone()
                        .struct_()
                        .field_by_name(f.name.as_str())
                        .alias(alias),
                }
            })
            .collect());
    }

    let input_fields: Vec<&StructField> = input_schema.fields().collect();
    walk_transform_slots(t, output_struct, &input_fields)?
        .into_iter()
        .map(|slot| match slot {
            TransformSlot::Passthrough { input_idx, output } => Ok(ColumnOp::Passthrough {
                input_idx,
                input_name: PlSmallStr::from_str(input_fields[input_idx].name.as_str()),
                output: PlSmallStr::from_str(output.name.as_str()),
            }),
            TransformSlot::Translated { expr, output } => classify_single(
                expr,
                &output.data_type,
                input_schema,
                PlSmallStr::from_str(output.name.as_str()),
            ),
        })
        .collect()
}

fn classify_single(
    expression: &Expression,
    output_type: &KernelDataType,
    input_schema: &StructType,
    output_name: PlSmallStr,
) -> DeltaResult<ColumnOp> {
    match expression {
        Expression::Literal(scalar) => {
            if let Some(polars_scalar) = try_to_polars_scalar(scalar) {
                return Ok(ColumnOp::Literal {
                    name: output_name,
                    scalar: polars_scalar,
                });
            }
        }
        Expression::Column(path) if path.len() == 1 => {
            if let Some((input_idx, field)) = input_schema
                .fields()
                .enumerate()
                .find(|(_, f)| f.name.as_str() == path[0])
            {
                return Ok(ColumnOp::Passthrough {
                    input_idx,
                    input_name: PlSmallStr::from_str(field.name.as_str()),
                    output: output_name,
                });
            }
        }
        _ => {}
    }
    let translated = translate_expr(expression, Some(output_type), Some(input_schema))?;
    Ok(ColumnOp::Computed {
        expr: translated.alias(output_name),
    })
}

pub(super) fn downcast_engine_data(batch: &dyn EngineData) -> DeltaResult<&PolarsEngineData> {
    batch
        .any_ref()
        .downcast_ref::<PolarsEngineData>()
        .ok_or_else(|| {
            Error::Generic(
                "PolarsEvaluationHandler received EngineData that is not PolarsEngineData".into(),
            )
        })
}

#[cfg(test)]
mod evaluator_height_tests {
    use delta_kernel::schema::{StructField, StructType};

    use super::*;

    /// Foreign plans arrive via the proto round-trip, so scalar/schema
    /// agreement is not guaranteed; a mismatch must be an error, not a
    /// `build_series` panic unwinding through PyO3.
    #[test]
    fn create_many_rejects_scalar_type_mismatch() {
        let handler = PolarsEvaluationHandler::new();
        let schema = Arc::new(
            StructType::try_new([StructField::nullable("a", KernelDataType::LONG)]).unwrap(),
        );
        let rows: &[&[Scalar]] = &[&[Scalar::String("x".to_string())]];
        let result = handler.create_many(schema, rows);
        assert!(result.is_err(), "type mismatch must error, not panic");
    }

    /// `Passthrough` addresses input columns by position, so a batch that
    /// is narrower than the declared input schema must fall back to the
    /// name-resolving path and error — not index past the column vector.
    #[test]
    fn narrow_batch_errors_instead_of_indexing_past_the_end() {
        let handler = PolarsEvaluationHandler::new();
        let input_schema = Arc::new(
            StructType::try_new([
                StructField::nullable("a", KernelDataType::LONG),
                StructField::nullable("b", KernelDataType::LONG),
            ])
            .unwrap(),
        );
        let evaluator = handler
            .new_expression_evaluator(
                input_schema,
                Arc::new(Expression::column(["b"])),
                KernelDataType::LONG,
            )
            .unwrap();
        let df = polars::df!("a" => [1i64]).unwrap();
        let out = evaluator.evaluate(&PolarsEngineData::new(df));
        assert!(
            out.is_err(),
            "a batch narrower than the declared schema must error"
        );
    }

    /// Kernel contract: one value per input row. A `Computed` op that
    /// references no column (a struct/array/binary literal falls through
    /// `classify_single` to `Computed`) must not collapse the lazy path.
    #[test]
    fn column_free_computed_op_keeps_batch_height() {
        let evaluator = PolarsExpressionEvaluator::new(
            vec![ColumnOp::Computed {
                expr: lit(7i64).alias("v"),
            }],
            1,
        );
        let df = polars::df!("x" => [1i64, 2, 3]).unwrap();
        let out = evaluator.evaluate(&PolarsEngineData::new(df)).unwrap();
        assert_eq!(out.len(), 3, "one output row per input row");
    }
}
