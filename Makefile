.PHONY: help setup test lint unit run clean

help:
	@echo "setup  - install dependencies into the project venv"
	@echo "test   - lint + full unit test suite"
	@echo "lint   - ruff check + ruff format --check"
	@echo "unit   - pytest"
	@echo "run    - run the tool, e.g. make run ARGS=\"compact --help\""
	@echo "clean  - remove caches and build artefacts"

setup:
	uv sync --all-groups

test: lint unit

lint:
	uv run ruff check .
	uv run ruff format --check .

unit:
	uv run pytest

run:
	uv run python -m idc $(ARGS)

clean:
	rm -rf .pytest_cache .ruff_cache dist build
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
