UV_ENV := PYTHONPATH=src UV_NO_EDITABLE=1
PYTHON := $(UV_ENV) uv run python -m controlsimulator

.PHONY: setup lint format test generate-smoke-data generate-full-data train-smoke train-full evaluate-smoke evaluate-full benchmark-smoke benchmark-full demo overnight

setup:
	UV_NO_EDITABLE=1 uv sync --group dev

lint:
	$(UV_ENV) uv run ruff check .

format:
	$(UV_ENV) uv run ruff format .

test:
	$(UV_ENV) uv run pytest

generate-smoke-data:
	$(PYTHON) generate-dataset --config configs/datasets/smoke_v4.yaml

generate-full-data:
	$(PYTHON) generate-dataset --config configs/datasets/full_v4.yaml

train-smoke:
	$(PYTHON) train --config configs/training/smoke_v4.yaml

train-full:
	$(PYTHON) train --config configs/training/training_v4.yaml

evaluate-smoke:
	$(PYTHON) evaluate --config configs/evaluation/smoke_v4.yaml

evaluate-full:
	$(PYTHON) evaluate --config configs/evaluation/evaluation_v4.yaml

benchmark-smoke:
	$(PYTHON) benchmark --config configs/evaluation/smoke_v4.yaml

benchmark-full:
	$(PYTHON) benchmark --config configs/evaluation/evaluation_v4.yaml

demo:
	$(PYTHON) demo --config configs/evaluation/evaluation_v4.yaml

overnight:
	$(PYTHON) overnight
