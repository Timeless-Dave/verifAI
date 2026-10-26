UV ?= uv
HOST ?= 127.0.0.1
PORT ?= 8741

.PHONY: install test run dev

install:
	$(UV) sync --all-groups

test: install
	$(UV) run pytest

run: install
	VERIFAI_HOST=$(HOST) VERIFAI_PORT=$(PORT) $(UV) run verifai

dev: install
	VERIFAI_HOST=$(HOST) VERIFAI_PORT=$(PORT) $(UV) run uvicorn verifai.main:app --host $(HOST) --port $(PORT) --reload
