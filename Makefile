UV ?= uv
HOST ?= 127.0.0.1
PORT ?= 8741

.PHONY: install test lint typecheck check run dev extension-test extension-lint

install:
	$(UV) sync --all-groups

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
