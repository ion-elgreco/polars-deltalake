# Contributing

Thanks for considering a contribution. This project is a Polars I/O plugin written in Rust on top of [`delta-kernel-rs`](https://github.com/delta-io/delta-kernel-rs); the Python wheel is built with [`maturin`](https://www.maturin.rs/) and orchestrated via [`uv`](https://docs.astral.sh/uv/) and [`just`](https://just.systems/).

## Prerequisites

- Rust toolchain (`rustup` — version is read from the workspace).
- Python ≥ 3.10.
- [`just`](https://just.systems/) (`brew install just` / `cargo install just`).
- [`uv`](https://docs.astral.sh/uv/) (`brew install uv` / `pipx install uv`).
- Docker (only required for the cloud integration tests; otherwise they're skipped).

## Build

```bash
just develop
```

Syncs `python/.venv` with dev + test extras, then builds the `cdylib` and installs it as an editable wheel. Use `PROFILE=release just develop` for an optimised build. Always go through `just develop` — running `uv sync` directly skips the maturin step and leaves the venv with a stale `_internal.so`.

After build, drop into the env with:

```bash
source python/.venv/bin/activate
```

## Test

```bash
just test              # full suite (local + integration; integration auto-skips if Docker is missing)
just test-local        # filters out anything marked `integration` or `spark`
just test-integration  # only the Docker-backed S3 / Azure / GCS suites
just test-spark        # only the delta-spark fixtures; needs JDK 17 or 21
just test-rust         # cargo unit tests on the cdylib crate
```

`just test-spark` is not part of CI — delta-spark is the only writer that
produces column-mapped and complex-CDF tables, so those fixtures are built
locally.

Always go through `just test-rust` for the Rust side. `pyo3/extension-module`
sits behind a default cargo feature, so a plain `cargo test` builds a test
binary that never links libpython and fails on undefined pyo3 symbols; the
recipe drops the feature and points `PYO3_PYTHON` at the venv interpreter.

The integration suites spin up containers via [`testcontainers`](https://github.com/testcontainers/testcontainers-python):

- **S3** → `minio/minio`
- **Azure** → `mcr.microsoft.com/azure-storage/azurite`
- **GCS** → `tustvold/fake-gcs-server` (patched fork with XML-API support; the same image `delta-rs` uses)

Each test module is tagged `pytest.mark.integration` and gated on `docker_available()`, so a missing Docker daemon is a clean skip, not a failure.

## Lint, format, type-check

```bash
just pre-commit        # ruff fix + ruff format + pyright
just pre-commit-check  # the read-only CI variant, plus a typo scan
just check             # `cargo check` on the cdylib
```

CI (`.github/workflows/python_build.yaml`) runs `just pre-commit-check` and all three test recipes on every PR against `main`.

## Code layout

```
python/
├── polars_deltalake/        # Python entry points: scan_delta / read_delta
├── src/
│   ├── lib.rs               # pyo3 module wiring
│   ├── scan/                # TableState / TableScan pyclasses, the io-plugin source
│   ├── engine/              # custom delta_kernel::Engine (no default-engine)
│   │   ├── executor/        # PlanExecutor: kernel query plans → LazyFrame
│   │   └── handlers/        # ParquetHandler / JsonHandler / StorageHandler
│   │                        # built on polars-io + object_store
│   └── translation/         # polars-Expr ↔ kernel-Predicate + schema bridge
└── tests/                   # pytest suite; `_*.py` are shared helpers,
                             # `*_spark.py` are the delta-spark fixtures
```

The Rust side never depends on `default-engine` — every handler is implemented in-tree against polars-io so there's no second parquet/JSON stack in the wheel. Predicate pushdown is opportunistic: if a polars expression doesn't map to a kernel `Predicate` variant, the scan still produces correct output, just without kernel-side data skipping.

## Pull requests

- Keep PRs focused. One change per PR is much easier to review than several bundled together.
- Run `just pre-commit` and `just test` before pushing — both gate CI.
- Tests are expected for any user-visible behaviour change. Predicate-pushdown tweaks belong in `tests/test_kernel_translation.py` (which conjunct reaches kernel) and `tests/test_scan.py` (which rows come back).
- Commit messages should explain the *why* — the diff already shows the *what*.

## Reporting issues

Bug reports are most useful when they include:

- The smallest snippet that reproduces the problem (table layout, write that produced it, the `scan_delta` / `read_delta` call).
- The version of `polars`, `polars-deltalake`, and the writer (`deltalake`, Spark, Databricks, …) that produced the table.
- The Delta protocol version (`DESCRIBE DETAIL` in Spark or the `protocol` action in `_delta_log/00000000000000000000.json`).
