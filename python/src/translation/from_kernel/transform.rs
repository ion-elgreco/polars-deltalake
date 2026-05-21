//! Kernel `Transform` → polars `as_struct(...)`. A `Transform` is a sparse
//! schema rewrite: prepend new fields, then walk the input fields applying
//! per-field replace/insert directives. Output ordering must match the
//! declared output struct position-by-position — see [`translate_transform`].

use delta_kernel::expressions::{ColumnName, Expression, ExpressionRef, FieldTransform, Transform};
use delta_kernel::schema::{DataType as KernelDataType, StructField, StructType};
use delta_kernel::{DeltaResult, Error};
use polars::prelude::as_struct as polars_as_struct;
use polars::prelude::{Expr, col};
use polars_utils::pl_str::PlSmallStr;

use super::expr::{column_path_to_expr, translate_expr};

/// Prepend computed fields, then walk input fields applying per-field
/// replace/insert directives from `field_transforms`. Output ordering must
/// match `output_struct` position-by-position — kernel consumes the output
/// schema in lockstep with prepends, pass-throughs, and inserts.
pub(super) fn translate_transform(
    t: &Transform,
    output_struct: &StructType,
    input_schema: &StructType,
) -> DeltaResult<Expr> {
    let (root_expr, input_fields): (Option<Expr>, Vec<&StructField>) = match &t.input_path {
        Some(path) => (
            Some(column_path_to_expr(path)),
            descend_struct_path(input_schema, path)?.fields().collect(),
        ),
        None => (None, input_schema.fields().collect()),
    };
    let lookup_input = |field_name: &str| match &root_expr {
        Some(root) => root.clone().struct_().field_by_name(field_name),
        None => col(PlSmallStr::from_str(field_name)),
    };

    let mut entries: Vec<Expr> = Vec::with_capacity(output_struct.fields().count());
    let mut output_iter = output_struct.fields();

    // Prepended fields fill leading output slots before any input field.
    for prep in &t.prepended_fields {
        push_translated(prep.as_ref(), input_schema, &mut output_iter, &mut entries)?;
    }

    // Walk the input schema; each input field maps to 0..N output slots
    // depending on its FieldTransform (or pass-through if absent).
    for input_field in &input_fields {
        let input_name = input_field.name.as_str();
        match classify_input_op(t.field_transforms.get(input_name)) {
            InputFieldOp::Keep => {
                push_to_next(lookup_input(input_name), &mut output_iter, &mut entries)?;
            }
            InputFieldOp::KeepThenInsert(exprs) => {
                push_to_next(lookup_input(input_name), &mut output_iter, &mut entries)?;
                for expr in exprs {
                    push_translated(expr.as_ref(), input_schema, &mut output_iter, &mut entries)?;
                }
            }
            InputFieldOp::Drop => {}
            InputFieldOp::ReplaceWith(exprs) => {
                for expr in exprs {
                    push_translated(expr.as_ref(), input_schema, &mut output_iter, &mut entries)?;
                }
            }
        }
    }

    if output_iter.next().is_some() {
        return Err(Error::Generic(
            "Transform: too many fields in output schema (input + transforms didn't fill all slots)"
                .into(),
        ));
    }

    Ok(polars_as_struct(entries))
}

/// Classification of a single input field's `FieldTransform` for the
/// translate_transform walk. Drop is implicit in the kernel encoding
/// (`is_replace=true` with empty `exprs`); naming it explicitly here
/// keeps the input-field loop a direct match on intent.
enum InputFieldOp<'a> {
    /// No `FieldTransform` — input column flows through to the next output slot.
    Keep,
    /// `is_replace=false` — pass-through, then `exprs.len()` inserts follow.
    KeepThenInsert(&'a [ExpressionRef]),
    /// `is_replace=true, exprs=[]` — input field consumes zero output slots.
    Drop,
    /// `is_replace=true, exprs.len() >= 1` — input position expands to N slots.
    ReplaceWith(&'a [ExpressionRef]),
}

fn classify_input_op(ft: Option<&FieldTransform>) -> InputFieldOp<'_> {
    match ft {
        None => InputFieldOp::Keep,
        Some(ft) if !ft.is_replace => InputFieldOp::KeepThenInsert(&ft.exprs),
        Some(ft) if ft.exprs.is_empty() => InputFieldOp::Drop,
        Some(ft) => InputFieldOp::ReplaceWith(&ft.exprs),
    }
}

fn push_to_next<'a>(
    expr: Expr,
    output_iter: &mut impl Iterator<Item = &'a StructField>,
    entries: &mut Vec<Expr>,
) -> DeltaResult<()> {
    let field = output_iter
        .next()
        .ok_or_else(|| Error::Generic("Transform: ran out of output schema fields".into()))?;
    entries.push(expr.alias(PlSmallStr::from_str(field.name.as_str())));
    Ok(())
}

fn push_translated<'a>(
    expr: &Expression,
    input_schema: &StructType,
    output_iter: &mut impl Iterator<Item = &'a StructField>,
    entries: &mut Vec<Expr>,
) -> DeltaResult<()> {
    let field = output_iter
        .next()
        .ok_or_else(|| Error::Generic("Transform: ran out of output schema fields".into()))?;
    let inner = translate_expr(expr, Some(&field.data_type), Some(input_schema))?;
    entries.push(inner.alias(PlSmallStr::from_str(field.name.as_str())));
    Ok(())
}

fn descend_struct_path<'a>(root: &'a StructType, path: &ColumnName) -> DeltaResult<&'a StructType> {
    let mut current = root;
    for segment in path.iter() {
        let field = current.field(segment).ok_or_else(|| {
            Error::Generic(format!(
                "Transform: input_path segment '{segment}' not found in input schema"
            ))
        })?;
        match &field.data_type {
            KernelDataType::Struct(inner) => current = inner,
            other => {
                return Err(Error::Generic(format!(
                    "Transform: input_path segment '{segment}' is {other:?}, not a struct"
                )));
            }
        }
    }
    Ok(current)
}
