UV ?= uv
HOST ?= 127.0.0.1
PORT ?= 8741

.PHONY: install test lint typecheck check run dev extension-test extension-lint bench

# Local only. CI does not run this.
# make bench BENCH_ARGS="--backend none --limit 50 --seed 0"
BENCH_ARGS ?=

# Optional model backends from pyproject.toml. uv sync removes packages the
# lockfile does not list, so a plain `uv pip install mlx-lm` would not survive
# the next make. Apple silicon gets mlx by default. EXTRAS="mlx nli" adds both;
# EXTRAS= installs neither.
ifeq ($(shell uname -s)-$(shell uname -m),Darwin-arm64)
EXTRAS ?= mlx
else
EXTRAS ?=
endif

install:
	$(UV) sync --all-groups $(foreach extra,$(EXTRAS),--extra $(extra))

extension-lint:
	node --check extension/src/*.js

extension-test:
	node --test extension/test/*.test.js

test: install extension-test
	$(UV) run pytest

lint: install extension-lint
	$(UV) run ruff check
	$(UV) run ruff format --check

typecheck: install
	$(UV) run mypy

check: lint typecheck test

run: install
	VERIFAI_HOST=$(HOST) VERIFAI_PORT=$(PORT) $(UV) run verifai

dev: install
	VERIFAI_HOST=$(HOST) VERIFAI_PORT=$(PORT) $(UV) run uvicorn verifai.main:app --host $(HOST) --port $(PORT) --reload

bench: install
	$(UV) run python -m verifai.bench $(BENCH_ARGS)
