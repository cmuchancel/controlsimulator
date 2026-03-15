# Overnight Report

## What I Built

I extended the existing repository rather than rewriting it. The v4 campaign focused on scale, dataset diversity, and honest evaluation:

- expanded from 512k samples to 3.456M samples
- expanded from 8 plant families to 15 plant families
- widened the plant time-scale range and added dedicated slow/fast families
- added oscillation-targeted PID gain sampling
- extended the simulation horizon from 8 s / 200 steps to 12 s / 300 steps
- refactored dataset storage, training, and evaluation to stream chunk-wise instead of assuming a single in-memory array
- added a capped 1-NN trajectory baseline
- added diagnostics against oscillation frequency, damping ratio, and plant order

## Core Decisions

- Kept plant-level splits intact and left `lightly_damped_second_order` fully held out as OOD.
- Kept the model class simple. The point of the campaign was better data and stronger evaluation, not architecture search.
- Removed the v3 control-effort exclusion from stable trajectory usability. For v4, the generator rejects only mathematically unstable, non-finite, or amplitude-exploding trajectories.
- Kept the full dataset chunk-native. At 3.456M samples and 300 time steps, forcing a single consolidated trajectory file was unnecessary and memory-hostile.
- Capped the 1-NN baseline reservoir at 30k stable training trajectories. That is enough to make the baseline informative without dominating evaluation runtime.

## Dataset Expansion Results

### Previous Full Baseline

- dataset name: `full_v3`
- plants: 16,000
- controllers per plant: 32
- total samples: 512,000
- plant families: 8

### New Scaled Dataset

- dataset name: `full_v4`
- plants: 96,000
- controllers per plant: 36
- total samples: 3,456,000
- plant families: 15
- scale factor vs `full_v3`: 6.75x

Families used:

- `first_order`
- `second_order`
- `underdamped_second_order`
- `overdamped_second_order`
- `lightly_damped_second_order`
- `highly_resonant_second_order`
- `third_order_real_poles`
- `third_order_mixed_real_complex`
- `weakly_resonant_third_order`
- `fourth_order_real`
- `fourth_order_mixed_complex`
- `two_mode_resonant`
- `near_integrator`
- `slow_dynamics_family`
- `fast_dynamics_family`

Observed dataset composition:

- stable fraction overall: 69.17%
- stable fraction by split:
  - train: 68.96%
  - val: 69.03%
  - test: 68.82%
  - ood_test: 72.25%

Failure counts:

- `unstable_closed_loop`: 1,065,466

Selected family-wise stable fractions:

| Family | Stable Fraction [%] |
| --- | ---: |
| `first_order` | 100.00 |
| `near_integrator` | 90.97 |
| `overdamped_second_order` | 79.18 |
| `highly_resonant_second_order` | 76.09 |
| `lightly_damped_second_order` | 72.25 |
| `fourth_order_mixed_complex` | 57.95 |
| `weakly_resonant_third_order` | 57.38 |
| `two_mode_resonant` | 35.55 |

Gain sampling mode counts:

- `wide_random`: 1,440,000
- `boundary_search`: 1,059,011
- `boundary_fallback_random`: 415,752
- `oscillatory_target`: 353,442
- `oscillatory_unstable_target`: 187,795

Generation runtime and size:

- dataset generation wall time: 1,579.71 s, about 26.3 min
- dataset size on disk: 2,736,977,390 bytes, about 2.55 GiB

## Training Results

Full training config:

- dataset: `configs/datasets/full_v4.yaml`
- training: `configs/training/training_v4.yaml`
- device: `mps`
- batch size: 2,048
- hidden sizes:
  - classifier: `[256, 256, 256]`
  - regressor: `[512, 512, 512]`

Training summary:

- classifier best validation F1: 0.8439 at epoch 22
- classifier internal train time: 285.90 s
- regressor best validation RMSE: 0.1725 at epoch 24
- regressor internal train time: 1,204.00 s
- full train command wall time: 1,528.54 s, about 25.5 min

## Updated Evaluation Results

### Stability Classifier

| Split | Accuracy | Precision | Recall | F1 | Majority Accuracy | Majority F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Test | 0.7980 | 0.9654 | 0.7328 | 0.8332 | 0.6882 | 0.8153 |
| OOD | 0.7625 | 0.9997 | 0.6715 | 0.8034 | 0.7225 | 0.8389 |

### Trajectory Regressor

| Split | Stable Samples | RMSE | MAE | Mean Baseline RMSE | 1-NN Baseline RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Test | 333,011 | 0.1728 | 0.0912 | 0.4311 | 0.3719 |
| OOD | 166,445 | 0.1679 | 0.0927 | 0.3603 | 0.3364 |

### Derived Metric Error

| Split | Overshoot MAE | Rise-Time MAE | Settling-Time MAE | Steady-State Error MAE |
| --- | ---: | ---: | ---: | ---: |
| Test | 10.25 pct-pts | 0.400 s | 4.339 s | 0.0782 |
| OOD | 13.43 pct-pts | 0.219 s | 4.832 s | 0.0794 |

Coverage:

- rise time defined for 66.77% of stable test predictions and 86.09% of stable OOD predictions
- settling time defined for 24.76% of stable test predictions and 37.64% of stable OOD predictions

Runtime:

| Benchmark | Simulator | Surrogate | Speedup |
| --- | ---: | ---: | ---: |
| Single sample | 1.465 ms | 2.758 ms | 0.53x |
| Batch of 512 | 0.888 s | 0.00556 s | 159.70x |

Auxiliary wall times:

- evaluation command wall time: 163.71 s, about 2.7 min
- benchmark command wall time: 41.24 s

## Scaling Effects On Accuracy

Compared with `full_v3`:

| Metric | `full_v3` | `full_v4` | Direction |
| --- | ---: | ---: | --- |
| Test classifier accuracy | 0.9127 | 0.7980 | worse |
| Test classifier F1 | 0.9324 | 0.8332 | worse |
| Test trajectory RMSE | 0.0902 | 0.1728 | worse |
| Test trajectory MAE | 0.0399 | 0.0912 | worse |
| Test overshoot MAE | 3.59 | 10.25 | worse |
| Test rise-time MAE | 0.073 | 0.400 | worse |
| Test settling-time MAE | 0.976 | 4.339 | worse |
| Test SSE MAE | 0.0454 | 0.0782 | worse |

Interpretation:

- `full_v4` is not a “better score” run than `full_v3`.
- The benchmark became much harder because the plant set is broader, the horizon is longer, the output is 300 points instead of 200, and the new families include genuinely multimodal fourth-order and resonant systems.
- The regressor still beats both the mean baseline and the capped 1-NN baseline by a wide margin.
- The classifier still beats majority accuracy, but the gap is much smaller than in earlier runs.

## Scaling Effects On OOD Performance

Compared with the earlier `full_v3` OOD results:

| Metric | `full_v3` | `full_v4` | Direction |
| --- | ---: | ---: | --- |
| OOD classifier accuracy | 0.8745 | 0.7625 | worse |
| OOD classifier F1 | 0.9028 | 0.8034 | worse |
| OOD trajectory RMSE | 0.1344 | 0.1679 | worse |
| OOD trajectory MAE | 0.0687 | 0.0927 | worse |
| OOD overshoot MAE | 6.21 | 13.43 | worse |
| OOD rise-time MAE | 0.099 | 0.219 | worse |
| OOD settling-time MAE | 1.425 | 4.832 | worse |
| OOD SSE MAE | 0.0638 | 0.0794 | worse |

Interpretation:

- The OOD family remains hard.
- The classifier is now conservative enough that it loses to the majority baseline on OOD F1 despite higher accuracy.
- The regressor still remains materially better than both baselines on OOD trajectories, but v4 makes it clear that generalization to lightly damped plants is still a real research problem.

## What Worked

- The generator scaled cleanly to 3.456M samples with deterministic chunking and multiprocessing.
- Plant-level splitting and the OOD family holdout remained intact at scale.
- The chunk-streamed train/eval refactor worked: the repo no longer needs a monolithic in-memory dataset for the full run.
- The regressor still beat both trivial baselines on both ID and OOD splits.
- Batch surrogate speedup remained strong.
- The new diagnostics surfaced real failure structure instead of generic aggregate loss:
  - error rises strongly with plant order
  - `fourth_order_mixed_complex` and `two_mode_resonant` are the hardest regression families
  - `first_order` and `near_integrator` are comparatively easy

## What Failed Or Remains Weak

- Classifier quality is now only modestly above majority accuracy and below majority F1 on OOD.
- All primary regression metrics got worse relative to `full_v3`.
- Fourth-order and multimode resonant plants remain difficult.
- Single-sample surrogate inference on `mps` is slower than direct simulation because device launch overhead dominates at batch size 1.
- Oscillation frequency alone is not the main explanatory variable for regression error; plant order and family structure matter more in this run.

## Recommended Next Steps

- Improve classifier recall near the stability boundary, possibly with focal loss or margin-aware sampling.
- Add richer plant encodings beyond padded coefficients.
- Try a stronger regressor specifically for fourth-order and multimode families.
- Separate device-level benchmark modes so CPU and MPS latency stories are both explicit.
- Add delayed or saturated plants only after the current LTI benchmark is strong enough to support them.
