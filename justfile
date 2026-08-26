# polars-deltalake — convenience recipes for build / develop / test.

profile := env_var_or_default("PROFILE", "dev")

# Default: list available recipes.
default:
    @just --list

# Sync venv with all deps (without building the native extension).
# `--inexact`: an exact sync treats the maturin-installed editable wheel as
# extraneous and removes it, breaking `import polars_deltalake`.
venv:
    cd python && uv sync --no-install-project --all-groups --inexact

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
# venv `just develop` built, since the system python3 can predate abi3-py310.
# `uv run` resolves the interpreter so the recipe works on Windows too, where
# the venv puts it in `Scripts/python.exe`. Deliberately not `: venv`:
# re-syncing between `just develop` and `just test-local` only re-locks
# uv.lock for no gain here.
test-rust:
    #!/usr/bin/env bash
    set -euo pipefail
    cd python
    py="$(uv run --no-sync python -c 'import sys; print(sys.executable)')" || {
        echo "just test-rust: could not resolve the venv interpreter — run 'just develop' first" >&2
        exit 1
    }
    # Windows `sys.executable` is backslash-separated, so match on a
    # forward-slash copy rather than the raw path.
    case "${py//\\//}" in
        */.venv/*) ;;
        *)
            echo "just test-rust: '$py' is not the project venv — run 'just develop' first" >&2
            exit 1
            ;;
    esac
    # Same profile as `just develop`, so CI reuses that build's dependency
    # artifacts instead of recompiling polars and delta-kernel from scratch.
    PYO3_PYTHON="$py" cargo test --lib --no-default-features --profile '{{ profile }}'

# Wipe build artifacts.
clean:
    rm -rf python/target python/.venv
    find python -type d -name __pycache__ -exec rm -rf {} +
    find python -name '*.so' -delete
