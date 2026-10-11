.DEFAULT_GOAL := help

.PHONY: help check-uv install lock upgrade clean format lint typecheck test validate run

help: ## Show the available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --- Guards ------------------------------------------------------------------

check-uv: ## Fail with a clear message if uv is not installed
	@command -v uv >/dev/null 2>&1 || { \
		echo "error: uv is not installed or not on your PATH."; \
		echo "       Install it from https://docs.astral.sh/uv/getting-started/installation/"; \
		exit 1; \
	}

# --- Environment setup -------------------------------------------------------

install: check-uv ## Create the virtualenv and install all dependencies (including dev)
	uv sync --all-groups

lock: check-uv ## Refresh uv.lock from pyproject.toml without installing
	uv lock

upgrade: check-uv ## Upgrade all dependencies to their latest allowed versions
	uv lock --upgrade
	uv sync --all-groups

clean: ## Remove the virtualenv and cached build artifacts
	rm -rf .venv .ruff_cache .ty_cache .pytest_cache
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +

# --- Code quality ------------------------------------------------------------

format: check-uv ## Format code with ruff
	uv run ruff format .

lint: check-uv ## Lint code with ruff
	uv run ruff check --fix .

typecheck: check-uv ## Type-check code with ty
	uv run ty check

test: check-uv ## Run the test suite with pytest
	uv run pytest

validate: format lint typecheck test ## Run format, lint, typecheck, and tests

# --- Run ---------------------------------------------------------------------

run: check-uv ## Run microgpt
	uv run python -m microgpt.microgpt
