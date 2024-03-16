
SHELL=/bin/bash

venv:  ## Set up virtual environment
	python3 -m venv .venv
	.venv/bin/pip install uv
	.venv/bin/uv pip install -r requirements.txt

install: venv ## Install polars-deltalake in develop mode
	unset CONDA_PREFIX && \
	source .venv/bin/activate && maturin develop -m Cargo.toml

install-release: venv ## Install polars-deltalake in release mode
	unset CONDA_PREFIX && \
	source .venv/bin/activate && maturin develop --release -m Cargo.toml

clean: ## Clean venv and clean cargo
	-@rm -r .venv
	-@cargo clean

.PHONY: help
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort