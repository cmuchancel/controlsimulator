PYTHON := uv run

.PHONY: setup lint format test generate-smoke-data generate-full-data train-smoke train-full evaluate-smoke evaluate-full benchmark-smoke benchmark-full demo overnight

setup:
	uv sync --group dev

lint:
	uv run ruff check .

format:
	uv run ruff format .

test:
	uv run pytest

generate-smoke-data:
	$(PYTHON) controlsimulator generate-dataset --config configs/datasets/smoke.yaml

generate-full-data:
	$(PYTHON) controlsimulator generate-dataset --config configs/datasets/full.yaml

train-smoke:
	$(PYTHON) controlsimulator train --config configs/training/smoke.yaml

train-full:
	$(PYTHON) controlsimulator train --config configs/training/full.yaml

evaluate-smoke:
	$(PYTHON) controlsimulator evaluate --config configs/evaluation/smoke.yaml

evaluate-full:
	$(PYTHON) controlsimulator evaluate --config configs/evaluation/full.yaml

benchmark-smoke:
	$(PYTHON) controlsimulator benchmark --config configs/evaluation/smoke.yaml

benchmark-full:
	$(PYTHON) controlsimulator benchmark --config configs/evaluation/full.yaml

demo:
	$(PYTHON) controlsimulator demo --config configs/evaluation/full.yaml

overnight:
	$(PYTHON) controlsimulator overnight

