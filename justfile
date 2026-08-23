# polars-deltalake — convenience recipes for build / develop / test.

profile := env_var_or_default("PROFILE", "dev")

# Default: list available recipes.
default:
    @just --list

# Sync venv with all deps (without building rivers native extension)
venv:
    cd python && uv sync --no-install-project --all-groups

# Use `PROFILE=release just develop` for an optimized build.
develop: venv
    cd python && VIRTUAL_ENV='{{ justfile_directory() }}/python/.venv' uvx --from 'maturin[zig]' maturin develop --profile {{ profile }}

# Run the pytest suite. The `test_cloud_*` modules are tagged
# `integration`; they boot emulator containers via `testcontainers` and
# auto-skip when no Docker daemon is reachable. Filter explicitly with

# `just test-local` / `just test-integration` if you'd rather pick a side.
test:
    cd python && uv run --no-sync pytest tests/

# Non-Docker, non-Spark tests only.
test-local:
    cd python && uv run --no-sync pytest tests/ -m "not integration and not spark"

# Just the Docker-backed cloud integration suites (S3 / Azure / GCS).
test-integration:
    cd python && uv run --no-sync pytest tests/ -m integration

# PySpark-authored fixtures (column-mapped tables, complex CDF shapes). Spark 4
# needs JDK 17 or 21; macOS `java` usually points at a newer one.
test-spark:
    cd python && JAVA_HOME="${JAVA_HOME:-$(/usr/libexec/java_home -v 21 2>/dev/null || /usr/libexec/java_home -v 17 2>/dev/null)}" uv run --no-sync pytest tests/ -m spark

# Format + lint + type-check (writes fixes).
pre-commit:
    cd python && uv run --no-sync ruff check --fix .
    cd python && uv run --no-sync ruff format .
    cd python && uv run --no-sync pyright .

# CI variant: same checks but read-only (no autofix), plus typo scan.
pre-commit-check:
    cd python && uv run --no-sync ruff check .
    cd python && uv run --no-sync ruff format --check --diff .
    cd python && uv run --no-sync typos .
    cd python && uv run --no-sync pyright .

# cargo check on the cdylib crate.
check:
    cd python && cargo check

# cargo unit tests on the cdylib crate. `--no-default-features` drops
# pyo3/extension-module so the test binary links libpython — pointed at the
# venv interpreter, since the system python3 can predate abi3-py310.
test-rust: venv
    cd python && PYO3_PYTHON='{{ justfile_directory() }}/python/.venv/bin/python3' cargo test --lib --no-default-features

# Wipe build artifacts.
clean:
    rm -rf python/target python/.venv
    find python -type d -name __pycache__ -exec rm -rf {} +
    find python -name '*.so' -delete
