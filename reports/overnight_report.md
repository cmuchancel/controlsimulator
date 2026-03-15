# Overnight Report

## What I Built

I built an end-to-end Python project for surrogate modeling of PID-controlled continuous-time systems:

- sampled stable SISO plant families from poles and optional zeros
- simulated unity-feedback unit-step closed loops with a finite-`tau_d` PID controller
- generated chunked datasets with metadata and plant-identity splits
- trained a stability classifier on all samples
- trained a trajectory regressor on stable samples only
- evaluated held-out plants, a held-out family, and runtime speedups

## Key Decisions

- Stable-only regression: unstable responses were kept for classification but excluded from the trajectory target to avoid contaminating the regression problem with diverging signals.
- Plant-centric splits: every controller sample from a given plant stays in exactly one split.
- OOD family split: `lightly_damped_second_order` is fully held out from training.
- Baselines kept simple: majority classifier and mean trajectory. They are weak, but they make the learned models easier to interpret.
- Gain ranges widened after an initial smoke run because the first ranges produced only about 3% unstable closed loops and made the classifier task too easy.

## Actual Experiment Settings

Full run config:

- dataset: `configs/datasets/full.yaml`
- training: `configs/training/full.yaml`
- evaluation: `configs/evaluation/full.yaml`
- plants: 1,800
- controllers per plant: 24
- total samples: 43,200
- horizon: 8.0 s
- time steps: 200
- derivative filter: `tau_d = 0.05`
- gain multipliers:
  - `Kp`: `[0.02, 50.0]`
  - `Ki`: `[0.01, 80.0]`
  - `Kd`: `[0.001, 25.0]`

Observed dataset composition:

- stable fraction overall: 81.77%
- stable fraction by split:
  - train: 81.13%
  - val: 80.55%
  - test: 79.94%
  - ood_test: 84.62%

Training summary:

- classifier best validation F1: 0.9896 at epoch 53
- classifier internal train time: 29.10 s
- regressor best validation RMSE: 0.0667
- regressor internal train time: 28.81 s

## Actual Results

### Stability Classifier

| Split | Accuracy | Precision | Recall | F1 | Majority Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Test | 0.9816 | 0.9992 | 0.9777 | 0.9884 | 0.7994 |
| OOD | 0.9607 | 0.9645 | 0.9900 | 0.9771 | 0.8462 |

### Trajectory Regressor

| Split | Stable Samples | RMSE | MAE | Mean Baseline RMSE | Mean Baseline MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Test | 3,952 | 0.0659 | 0.0275 | 0.2971 | 0.2346 |
| OOD | 8,794 | 0.1218 | 0.0637 | 0.3267 | 0.2516 |

### Derived Metric Error

| Split | Overshoot MAE | Rise-Time MAE | Settling-Time MAE | Steady-State Error MAE |
| --- | ---: | ---: | ---: | ---: |
| Test | 4.22 pct-pts | 0.104 s | 1.096 s | 0.0162 |
| OOD | 9.97 pct-pts | 0.173 s | 1.592 s | 0.0281 |

Coverage:

- rise time defined for 71.2% of stable test predictions and 70.3% of stable OOD predictions
- settling time defined for 43.3% of stable test predictions and 25.5% of stable OOD predictions

### Runtime

| Benchmark | Simulator | Surrogate | Speedup |
| --- | ---: | ---: | ---: |
| Single sample | 1.382 ms | 0.112 ms | 12.36x |
| Batch of 256 | 0.599 s | 0.00298 s | 201.24x |

## What Worked

- The data generator produced a useful stability mix after retuning the gain ranges.
- The classifier substantially beat the majority baseline on both ID and OOD data.
- The regressor beat the mean trajectory baseline by a large margin on both ID and OOD splits.
- Runtime speedups are large enough to justify the surrogate in screening-style workflows.
- The failure plots are informative rather than pathological; the hardest errors are mostly oscillatory stable systems, not numerical garbage.

## What Failed Or Remains Weak

- Settling-time prediction is still the weakest metric, especially on OOD lightly damped plants.
- The mean-trajectory baseline is too weak to say much about architectural quality; it only provides minimal context.
- The fixed 8 s horizon forces some hard oscillatory responses into ambiguous “not settled” territory.
- The project does not yet model delays, saturation, noise, or nonlinear plants.

## Recommended Next Steps

- Improve the plant encoding with root and frequency-domain summaries.
- Add targeted sampling near the stability boundary.
- Try a stronger but still simple regressor architecture focused on oscillatory tails.
- Extend the horizon or learn in normalized time coordinates.
- Add richer control-effort analysis and optional actuation constraints.
