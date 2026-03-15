# Experiment Log

## Commands Run

Environment and checks:

- `UV_NO_EDITABLE=1 uv sync --group dev`
- `PYTHONPATH=src UV_NO_EDITABLE=1 uv run ruff check src tests`
- `PYTHONPATH=src UV_NO_EDITABLE=1 uv run pytest -q`

Smoke v4 workflow:

- `/usr/bin/time -l env PYTHONPATH=src UV_NO_EDITABLE=1 uv run python -m controlsimulator generate-dataset --config configs/datasets/smoke_v4.yaml`
- `/usr/bin/time -l env PYTHONPATH=src UV_NO_EDITABLE=1 uv run python -m controlsimulator train --config configs/training/smoke_v4.yaml`
- `/usr/bin/time -l env PYTHONPATH=src UV_NO_EDITABLE=1 uv run python -m controlsimulator evaluate --config configs/evaluation/smoke_v4.yaml`
- `env PYTHONPATH=src UV_NO_EDITABLE=1 uv run python -m controlsimulator benchmark --config configs/evaluation/smoke_v4.yaml`

Full v4 workflow:

- `/usr/bin/time -l env PYTHONPATH=src UV_NO_EDITABLE=1 uv run python -m controlsimulator generate-dataset --config configs/datasets/full_v4.yaml`
- `/usr/bin/time -l env PYTHONPATH=src UV_NO_EDITABLE=1 uv run python -m controlsimulator train --config configs/training/training_v4.yaml`
- `/usr/bin/time -l env PYTHONPATH=src UV_NO_EDITABLE=1 uv run python -m controlsimulator evaluate --config configs/evaluation/evaluation_v4.yaml`
- `/usr/bin/time -l env PYTHONPATH=src UV_NO_EDITABLE=1 uv run python -m controlsimulator benchmark --config configs/evaluation/evaluation_v4.yaml`

## Datasets Created

### `artifacts/datasets/smoke_v4`

- total samples: 17,280
- plants: 960
- controllers per plant: 18
- stable fraction: 63.59%
- failure counts:
  - `unstable_closed_loop`: 4,995
- dataset size: 25,614,055 bytes

### `artifacts/datasets/full_v4`

- total samples: 3,456,000
- plants: 96,000
- controllers per plant: 36
- stable fraction: 69.17%
- split counts:
  - train: 2,257,920
  - val: 483,840
  - test: 483,876
  - ood_test: 230,364
- failure counts:
  - `unstable_closed_loop`: 1,065,466
- dataset size: 2,736,977,390 bytes

## Checkpoints Produced

Smoke v4:

- `artifacts/runs/smoke_training_v4/classifier.pt`
- `artifacts/runs/smoke_training_v4/regressor.pt`
- `artifacts/runs/smoke_training_v4/train_summary.json`

Full v4:

- `artifacts/runs/training_v4/classifier.pt`
- `artifacts/runs/training_v4/regressor.pt`
- `artifacts/runs/training_v4/train_summary.json`

## Report Artifacts Produced

Smoke v4:

- `reports/evaluations/smoke_evaluation_v4/evaluation_summary.json`
- `reports/evaluations/smoke_evaluation_v4/benchmark_summary.json`
- `reports/evaluations/smoke_evaluation_v4/plots/*`

Full v4:

- `reports/evaluations/evaluation_v4/evaluation_summary.json`
- `reports/evaluations/evaluation_v4/benchmark_summary.json`
- `reports/evaluations/evaluation_v4/family_metrics.csv`
- `reports/evaluations/evaluation_v4/sample_errors.csv`
- `reports/evaluations/evaluation_v4/plots/*`

Mirrored artifact plots:

- `artifacts/plots/full_v4/*`
- `artifacts/plots/evaluation_v4/*`

## Elapsed Runtimes

Exact wall times:

- `PYTHONPATH=src UV_NO_EDITABLE=1 uv run pytest -q`: 24.74 s
- `full_v4` dataset generation: 1,579.71 s
- `training_v4` train command: 1,528.54 s
- `evaluation_v4` evaluation command: 163.71 s
- `evaluation_v4` benchmark command: 41.24 s

Internal train times from saved summaries:

- `training_v4` classifier: 285.90 s
- `training_v4` regressor: 1,204.00 s
- `smoke_training_v4` classifier: 5.17 s
- `smoke_training_v4` regressor: 6.90 s

## Notes

- `full_v4` is 6.75x larger than `full_v3`.
- The v4 dataset remained chunk-native; consolidated trajectory files were intentionally disabled for the full run.
- The classifier remained better than majority accuracy but weaker than majority F1 on OOD.
- The regressor still beat the mean and 1-NN baselines by a wide margin on both ID and OOD trajectories.
- Single-example benchmark speed was worse than simulation on `mps`, while batch benchmark speed remained strongly in the surrogate’s favor.
