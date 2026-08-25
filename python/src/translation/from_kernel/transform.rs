//! Kernel `ExpressionStructPatch` → polars `as_struct(...)`. A struct patch
//! is a sparse schema rewrite: prepend new fields, walk the input fields
//! applying per-field replace/insert directives, then append trailing
//! fields. Output ordering must match the declared output struct
//! position-by-position — see [`translate_transform`].

use delta_kernel::expressions::{ColumnName, Expression, ExpressionRef, ExpressionStructPatch};
use delta_kernel::schema::{DataType as KernelDataType, StructField, StructType};
use delta_kernel::{DeltaResult, Error};
use polars::prelude::as_struct as polars_as_struct;
use polars::prelude::{Expr, col};
use polars_utils::pl_str::PlSmallStr;

use super::expr::{column_path_to_expr, null_gated, translate_expr};

/// Per-output-slot intent produced by [`walk_transform_slots`]. Shared by
/// the lazy [`translate_transform`] walker and the eager `build_transform_ops`
/// planner so the Keep / KeepThenInsert / Drop / ReplaceWith dispatch lives
/// in one place.
pub(super) enum TransformSlot<'a> {
    /// Carry the input field at `input_fields[input_idx]` into the output
    /// slot. Consumers resolve the input however they need: top-level
    /// `col(name)`, nested `root.struct.field_by_name(name)`, or a direct
    /// DataFrame column index.
    Passthrough {
        input_idx: usize,
        output: &'a StructField,
    },
    /// Emit `expr` as the next output slot.
    Translated {
        expr: &'a Expression,
        output: &'a StructField,
    },
}

/// Walk a struct patch position-by-position over `input_fields` and emit one
/// [`TransformSlot`] per output slot.
pub(super) fn walk_transform_slots<'a>(
    t: &'a ExpressionStructPatch,
    output_struct: &'a StructType,
    input_fields: &[&'a StructField],
) -> DeltaResult<Vec<TransformSlot<'a>>> {
    // A non-optional patch naming a field the input doesn't have is an error
    // per the kernel contract; optional ones are silently skipped.
    let mut missing: Vec<&str> = t
        .field_patches
        .iter()
        .filter(|(name, patch)| {
            !patch.optional && !input_fields.iter().any(|f| f.name.as_str() == *name)
        })
        .map(|(name, _)| name.as_str())
        .collect();
    if !missing.is_empty() {
        // `field_patches` is a HashMap, so sort for a reproducible message.
        missing.sort_unstable();
        return Err(Error::Generic(format!(
            "StructPatch: patched field(s) {missing:?} not found in input schema"
        )));
    }

    let mut slots: Vec<TransformSlot<'a>> = Vec::with_capacity(output_struct.num_fields());
    let mut output_iter = output_struct.fields();

    for prep in &t.prepended_fields {
        slots.push(TransformSlot::Translated {
            expr: prep.as_ref(),
            output: next_output(&mut output_iter)?,
        });
    }

    for (input_idx, input_field) in input_fields.iter().enumerate() {
        // No entry keeps the field; an entry keeps it only when `keep_input`,
        // and its insertions land after the field's output position. Keeping
        // nothing and inserting nothing drops the field.
        let (passes_through, inserts): (bool, &[ExpressionRef]) =
            match t.field_patches.get(input_field.name.as_str()) {
                None => (true, &[]),
                Some(p) => (p.keep_input, &p.insertions),
            };
        if passes_through {
            slots.push(TransformSlot::Passthrough {
                input_idx,
                output: next_output(&mut output_iter)?,
            });
        }
        for expr in inserts {
            slots.push(TransformSlot::Translated {
                expr: expr.as_ref(),
                output: next_output(&mut output_iter)?,
            });
        }
    }

    for app in &t.appended_fields {
        slots.push(TransformSlot::Translated {
            expr: app.as_ref(),
            output: next_output(&mut output_iter)?,
        });
    }

    if output_iter.next().is_some() {
        return Err(Error::Generic(
            "Transform: too many fields in output schema (input + transforms didn't fill all slots)"
                .into(),
        ));
    }
    Ok(slots)
}

fn next_output<'a>(
    iter: &mut impl Iterator<Item = &'a StructField>,
) -> DeltaResult<&'a StructField> {
    iter.next()
        .ok_or_else(|| Error::Generic("Transform: ran out of output schema fields".into()))
}

/// Prepend computed fields, then walk input fields applying per-field
/// replace/insert directives from `field_patches`, then append
/// `appended_fields`. Output ordering must
/// match `output_struct` position-by-position — kernel consumes the output
/// schema in lockstep with prepends, pass-throughs, and inserts.
pub(super) fn translate_transform(
    t: &ExpressionStructPatch,
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

    let entries = walk_transform_slots(t, output_struct, &input_fields)?
        .into_iter()
        .map(|slot| -> DeltaResult<Expr> {
            let (raw, output) = match slot {
                TransformSlot::Passthrough { input_idx, output } => {
                    (lookup_input(input_fields[input_idx].name.as_str()), output)
                }
                TransformSlot::Translated { expr, output } => (
                    translate_expr(expr, Some(&output.data_type), Some(input_schema))?,
                    output,
                ),
            };
            Ok(raw.alias(PlSmallStr::from_str(output.name.as_str())))
        })
        .collect::<DeltaResult<Vec<_>>>()?;
    let rebuilt = polars_as_struct(entries);
    // A nested patch rewrites a struct-typed column; `as_struct` alone would
    // make every row valid, and plan filters (`add IS NOT NULL`) select rows
    // by exactly that outer validity.
    Ok(match &root_expr {
        Some(root) => null_gated(root.clone().is_not_null(), rebuilt),
        None => rebuilt,
    })
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

#[cfg(test)]
mod missing_patch_tests {
    use delta_kernel::expressions::{ExpressionFieldPatch, ExpressionStructPatch};

    use super::*;

    /// `field_patches` is a HashMap, so naming only the first field found
    /// makes the error depend on iteration order.
    #[test]
    fn missing_fields_are_all_named_in_sorted_order() {
        let drop = || ExpressionFieldPatch {
            keep_input: false,
            insertions: vec![],
            optional: false,
        };
        let patch = ExpressionStructPatch {
            input_path: None,
            field_patches: [("zz".to_string(), drop()), ("aa".to_string(), drop())]
                .into_iter()
                .collect(),
            prepended_fields: vec![],
            appended_fields: vec![],
        };
        let output =
            StructType::try_new([StructField::nullable("keep", KernelDataType::LONG)]).unwrap();
        let kept = StructField::nullable("keep", KernelDataType::LONG);

        let err = match walk_transform_slots(&patch, &output, &[&kept]) {
            Err(e) => e,
            Ok(_) => panic!("non-optional patches naming absent fields must error"),
        };
        let msg = err.to_string();
        assert!(msg.contains("\"aa\""), "got: {msg}");
        assert!(msg.contains("\"zz\""), "got: {msg}");
        assert!(
            msg.find("aa") < msg.find("zz"),
            "names must be sorted, got: {msg}"
        );
    }
}
