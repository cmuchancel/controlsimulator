# Experiment Log

## Commands Run

Environment and checks:

- `uv sync --group dev`
- `uv run ruff check .`
- `uv run pytest -q`

Smoke workflow:

- `uv run controlsimulator generate-dataset --config configs/datasets/smoke.yaml`
- `uv run controlsimulator train --config configs/training/smoke.yaml`
- `uv run controlsimulator evaluate --config configs/evaluation/smoke.yaml`
- `uv run controlsimulator benchmark --config configs/evaluation/smoke.yaml`

Full workflow:

- `uv run controlsimulator generate-dataset --config configs/datasets/full.yaml`
- `uv run controlsimulator train --config configs/training/full.yaml`
- `uv run controlsimulator evaluate --config configs/evaluation/full.yaml`
- `uv run controlsimulator benchmark --config configs/evaluation/full.yaml`

## Datasets Created

### `artifacts/datasets/smoke_v2`

- total samples: 2,160
- plants: 180
- stable fraction: 84.26%
- split counts:
  - train: 1,116
  - val: 240
  - test: 240
  - ood_test: 564

### `artifacts/datasets/full_v2`

- total samples: 43,200
- plants: 1,800
- stable fraction: 81.77%
- split counts:
  - train: 22,944
  - val: 4,920
  - test: 4,944
  - ood_test: 10,392

## Checkpoints Produced

Smoke:

- `artifacts/runs/smoke_mlp_v2/classifier.pt`
- `artifacts/runs/smoke_mlp_v2/regressor.pt`
- `artifacts/runs/smoke_mlp_v2/train_summary.json`

Full:

- `artifacts/runs/full_mlp_v2/classifier.pt`
- `artifacts/runs/full_mlp_v2/regressor.pt`
- `artifacts/runs/full_mlp_v2/train_summary.json`

## Committed Report Artifacts

Smoke:

- `reports/evaluations/smoke_mlp_v2/evaluation_summary.json`
- `reports/evaluations/smoke_mlp_v2/benchmark_summary.json`
- `reports/evaluations/smoke_mlp_v2/plots/*`

Full:

- `reports/evaluations/full_mlp_v2/evaluation_summary.json`
- `reports/evaluations/full_mlp_v2/benchmark_summary.json`
- `reports/evaluations/full_mlp_v2/plots/*`

## Elapsed Runtimes

Exact:

- `uv run pytest -q`: 11.13 s
- full evaluation command (`/usr/bin/time -p`): 9.39 s
- full benchmark command (`/usr/bin/time -p`): 35.75 s
- full classifier internal train time: 29.10 s
- full regressor internal train time: 28.81 s
- smoke classifier internal train time: 7.44 s
- smoke regressor internal train time: 3.71 s

Approximate from command progress/output:

- smoke dataset generation: about 4 s
- full dataset generation: about 1 m 54 s

Notes:

- Training times above are taken from the saved run summaries and measure the model-training portions directly.
- Dataset generation wall times were not separately wrapped with `/usr/bin/time`; the values above come from observed command progress and should be treated as approximate.
