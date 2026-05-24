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
    cd python && VIRTUAL_ENV={{ justfile_directory() }}/python/.venv uvx --from 'maturin[zig]' maturin develop --profile {{ profile }}

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

# PySpark-authored fixtures (column-mapped CDF etc). Requires JDK 17 or 21.
test-spark:
    cd python && uv run --no-sync pytest tests/ -m spark

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

# cargo unit tests on the cdylib crate.
test-rust:
    cd python && cargo test --lib

# Wipe build artifacts.
clean:
    rm -rf python/target python/.venv
    find python -type d -name __pycache__ -exec rm -rf {} +
    find python -name '*.so' -delete
