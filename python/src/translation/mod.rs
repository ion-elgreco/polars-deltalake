//! Translation between kernel and polars representations. Mirror pair:
//! `from_kernel` walks kernel ASTs to produce polars `Expr`/`Series`;
//! `to_kernel` walks polars `Expr` to produce kernel `Predicate`. The
//! shared `schema` module converts kernel `StructType`/`DataType` to and
//! from polars / polars-arrow schemas.

pub(crate) mod from_kernel;
pub(crate) mod schema;
pub(crate) mod to_kernel;

pub(crate) use from_kernel::{PolarsEvaluationHandler, build_series};
