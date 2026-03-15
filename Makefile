UV_ENV := PYTHONPATH=src UV_NO_EDITABLE=1
PYTHON := $(UV_ENV) uv run python -m controlsimulator

.PHONY: setup lint format test generate-smoke-data generate-full-data generate-campaign-smoke generate-campaign-data train-smoke train-full train-campaign-smoke train-campaign evaluate-smoke evaluate-full evaluate-campaign-smoke evaluate-campaign benchmark-smoke benchmark-full benchmark-campaign-smoke benchmark-campaign demo overnight overnight-campaign campaign-smoke

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

generate-campaign-smoke:
	$(PYTHON) generate-dataset --config configs/datasets/campaign_smoke.yaml

generate-campaign-data:
	$(PYTHON) generate-dataset --config configs/datasets/third_order_campaign.yaml

train-smoke:
	$(PYTHON) train --config configs/training/smoke_v4.yaml

train-full:
	$(PYTHON) train --config configs/training/training_v4.yaml

train-campaign-smoke:
	$(PYTHON) train --config configs/training/campaign_smoke.yaml

train-campaign:
	$(PYTHON) train --config configs/training/third_order_campaign.yaml

evaluate-smoke:
	$(PYTHON) evaluate --config configs/evaluation/smoke_v4.yaml

evaluate-full:
	$(PYTHON) evaluate --config configs/evaluation/evaluation_v4.yaml

evaluate-campaign-smoke:
	$(PYTHON) evaluate --config configs/evaluation/campaign_smoke.yaml

evaluate-campaign:
	$(PYTHON) evaluate --config configs/evaluation/third_order_campaign.yaml

benchmark-smoke:
	$(PYTHON) benchmark --config configs/evaluation/smoke_v4.yaml

benchmark-full:
	$(PYTHON) benchmark --config configs/evaluation/evaluation_v4.yaml

benchmark-campaign-smoke:
	$(PYTHON) benchmark --config configs/evaluation/campaign_smoke.yaml

benchmark-campaign:
	$(PYTHON) benchmark --config configs/evaluation/third_order_campaign.yaml

demo:
	$(PYTHON) demo --config configs/evaluation/evaluation_v4.yaml

overnight:
	$(PYTHON) overnight

campaign-smoke:
	$(PYTHON) campaign --dataset-config configs/datasets/campaign_smoke.yaml --training-config configs/training/campaign_smoke.yaml --evaluation-config configs/evaluation/campaign_smoke.yaml

overnight-campaign:
	$(PYTHON) overnight-campaign
