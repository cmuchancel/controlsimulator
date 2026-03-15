# Experiment Log

## Commands Run

Environment and checks:

- `uv sync --group dev`
- `uv run ruff check src/controlsimulator tests`
- `uv run pytest -q`

Smoke workflow:

- `uv run controlsimulator generate-dataset --config configs/datasets/smoke.yaml`
- `uv run controlsimulator train --config configs/training/smoke.yaml`
- `uv run controlsimulator evaluate --config configs/evaluation/smoke.yaml`
- `uv run controlsimulator benchmark --config configs/evaluation/smoke.yaml`

Scaled workflow:

- `/usr/bin/time -p uv run controlsimulator generate-dataset --config configs/datasets/full.yaml`
- `/usr/bin/time -p uv run controlsimulator train --config configs/training/full.yaml`
- `/usr/bin/time -p uv run controlsimulator evaluate --config configs/evaluation/full.yaml`
- `/usr/bin/time -p uv run controlsimulator benchmark --config configs/evaluation/full.yaml`

## Datasets Created

### `artifacts/datasets/smoke_v3`

- total samples: 5,760
- plants: 360
- controllers per plant: 16
- stable fraction: 68.35%
- failure counts:
  - `unstable_closed_loop`: 1,714
  - `control_effort_limit_exceeded`: 109

### `artifacts/datasets/full_v3`

- total samples: 512,000
- plants: 16,000
- controllers per plant: 32
- stable fraction: 67.99%
- split counts:
  - train: 313,728
  - val: 67,264
  - test: 67,264
  - ood_test: 63,744
- failure counts:
  - `unstable_closed_loop`: 157,159
  - `control_effort_limit_exceeded`: 6,721
- dataset size: 536,224,027 bytes

## Checkpoints Produced

Smoke:

- `artifacts/runs/smoke_mlp_v3/classifier.pt`
- `artifacts/runs/smoke_mlp_v3/regressor.pt`
- `artifacts/runs/smoke_mlp_v3/train_summary.json`

Scaled:

- `artifacts/runs/full_mlp_v3/classifier.pt`
- `artifacts/runs/full_mlp_v3/regressor.pt`
- `artifacts/runs/full_mlp_v3/train_summary.json`

## Report Artifacts Produced

Smoke:

- `reports/evaluations/smoke_mlp_v3/evaluation_summary.json`
- `reports/evaluations/smoke_mlp_v3/benchmark_summary.json`
- `reports/evaluations/smoke_mlp_v3/plots/*`

Scaled:

- `reports/evaluations/full_mlp_v3/evaluation_summary.json`
- `reports/evaluations/full_mlp_v3/benchmark_summary.json`
- `reports/evaluations/full_mlp_v3/family_metrics.csv`
- `reports/evaluations/full_mlp_v3/plots/*`

Additional artifact plots:

- `artifacts/plots/full_v3/*`
- `artifacts/plots/full_mlp_v3/*`

## Elapsed Runtimes

Exact wall times:

- `uv run pytest -q`: 13.79 s
- scaled dataset generation: 192.49 s
- scaled train command: 273.13 s
- scaled evaluation command: 16.12 s
- scaled benchmark command: 35.65 s

Internal train times from saved summaries:

- scaled classifier: 130.68 s
- scaled regressor: 133.85 s
- smoke classifier: 3.44 s
- smoke regressor: 2.69 s

## Notes

- The scaled dataset is about 11.85x larger than the earlier 43.2k run.
- Multiprocessing generation remained deterministic in tests when comparing `num_workers=1` vs `num_workers=2`.
- The scaled run is intentionally harder than the earlier baseline, so some accuracy metrics dropped while several transient-response metrics improved.
