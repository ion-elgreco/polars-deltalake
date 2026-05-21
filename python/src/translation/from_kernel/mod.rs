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
use delta_kernel::expressions::{Expression, ExpressionRef, Scalar};
use delta_kernel::schema::{DataType as KernelDataType, SchemaRef, StructType};
use delta_kernel::{
    DeltaResult, Error, EvaluationHandler, ExpressionEvaluator, PredicateEvaluator,
};
use polars::prelude::{DataFrame, Expr, IntoColumn, IntoLazy, Series};
use polars_utils::pl_str::PlSmallStr;

use crate::consts::KERNEL_OUTPUT_COL;
use crate::engine::PolarsEngineData;
use crate::errors::to_kernel_err;
use crate::translation::schema::KernelSchemaExt;

mod expr;
mod predicate;
mod scalar;
mod transform;

pub(crate) use predicate::translate_predicate;
pub(crate) use scalar::{build_series, empty_typed_list_expr};

use expr::translate_expr;
use predicate::PolarsPredicateEvaluator;
use transform::translate_transform;

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
        let select_exprs =
            build_select_exprs(input_schema.as_ref(), expression.as_ref(), &output_type)?;
        Ok(Arc::new(PolarsExpressionEvaluator { select_exprs }))
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

struct PolarsExpressionEvaluator {
    select_exprs: Vec<Expr>,
}

impl ExpressionEvaluator for PolarsExpressionEvaluator {
    fn evaluate(&self, batch: &dyn EngineData) -> DeltaResult<Box<dyn EngineData>> {
        let df = downcast_engine_data(batch)?.dataframe().clone();
        let result = df
            .lazy()
            .select(self.select_exprs.clone())
            .collect()
            .map_err(to_kernel_err)?;
        Ok(Box::new(PolarsEngineData::new(result)))
    }
}

/// Struct-typed output is unnested so each declared field becomes its own
/// DataFrame column; non-struct output produces one column named "output".
fn build_select_exprs(
    input_schema: &StructType,
    expression: &Expression,
    output_type: &KernelDataType,
) -> DeltaResult<Vec<Expr>> {
    match (output_type, expression) {
        (KernelDataType::Struct(output_struct), Expression::Transform(t)) => {
            let struct_expr = translate_transform(t, output_struct, input_schema)?;
            Ok(output_struct
                .fields()
                .map(|f| {
                    struct_expr
                        .clone()
                        .struct_()
                        .field_by_name(f.name.as_str())
                        .alias(PlSmallStr::from_str(f.name.as_str()))
                })
                .collect())
        }
        (KernelDataType::Struct(struct_type), Expression::Struct(children, _)) => {
            let fields: Vec<_> = struct_type.fields().collect();
            if children.len() != fields.len() {
                return Err(Error::Generic(format!(
                    "PolarsExpressionEvaluator: output struct has {} fields but expression has {} children",
                    fields.len(),
                    children.len()
                )));
            }
            children
                .iter()
                .zip(fields.iter())
                .map(|(child, field)| {
                    translate_expr(child.as_ref(), Some(&field.data_type), Some(input_schema))
                        .map(|e| e.alias(PlSmallStr::from_str(field.name.as_str())))
                })
                .collect()
        }
        _ => {
            let translated = translate_expr(expression, Some(output_type), Some(input_schema))?;
            Ok(vec![
                translated.alias(PlSmallStr::from_static(KERNEL_OUTPUT_COL)),
            ])
        }
    }
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
