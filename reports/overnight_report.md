# Overnight Report

## What I Built

I extended the existing repository rather than rewriting it. The main changes were:

- scaled dataset generation from 43.2k samples to 512k samples
- added more plant families while keeping pole-based stable sampling
- added deterministic multiprocessing for chunk generation
- added targeted near-boundary PID sampling
- switched dataset storage to Parquet plus compressed NumPy
- added dataset-health diagnostics and plots
- retrained both models and re-ran evaluation and benchmarking

## Core Decisions

- Kept plant-level splits intact and left `lightly_damped_second_order` as the OOD family.
- Kept the model architecture simple. The point of this campaign was scaling and data quality, not architectural novelty.
- Treated stable-only trajectory regression as the default. Unstable or rejected cases remain in the classifier dataset only.
- Added config fingerprinting so stale chunks from a different config cannot silently contaminate a resumed dataset.
- Allowed control-effort-limit rejections to be counted rather than abort the run, while still aborting on non-finite or amplitude-exploding trajectories.

## Dataset Expansion Results

### Previous Full Baseline

- dataset name: `full_v2`
- plants: 1,800
- controllers per plant: 24
- total samples: 43,200

### New Scaled Dataset

- dataset name: `full_v3`
- plants: 16,000
- controllers per plant: 32
- total samples: 512,000
- scale factor vs previous full run: 11.85x

Families used:

- `first_order`
- `second_order`
- `underdamped_second_order`
- `overdamped_second_order`
- `third_order_real_poles`
- `third_order_mixed_real_complex`
- `lightly_damped_second_order`
- `weakly_resonant_third_order`

Observed dataset composition:

- stable fraction overall: 67.99%
- stable fraction by split:
  - train: 67.94%
  - val: 68.24%
  - test: 68.40%
  - ood_test: 67.55%

Failure counts:

- `unstable_closed_loop`: 157,159
- `control_effort_limit_exceeded`: 6,721

Family-wise stable fraction:

| Family | Stable Fraction [%] |
| --- | ---: |
| `first_order` | 99.93 |
| `overdamped_second_order` | 73.14 |
| `second_order` | 70.76 |
| `underdamped_second_order` | 69.44 |
| `lightly_damped_second_order` | 67.55 |
| `third_order_real_poles` | 56.51 |
| `third_order_mixed_real_complex` | 53.50 |
| `weakly_resonant_third_order` | 52.28 |

Gain sampling mode counts:

- `wide_random`: 256,000
- `boundary_search`: 202,059
- `targeted_fallback_random`: 53,941

Generation runtime and size:

- dataset generation wall time: 192.49 s
- dataset size on disk: 536,224,027 bytes, about 511 MiB

## Training Results

Full training config:

- dataset: `configs/datasets/full.yaml`
- training: `configs/training/full.yaml`
- batch size: 2,048
- hidden sizes:
  - classifier: `[192, 192]`
  - regressor: `[384, 384, 384]`

Training summary:

- classifier best validation F1: 0.9400 at epoch 37
- classifier internal train time: 130.68 s
- regressor best validation RMSE: 0.0942
- regressor internal train time: 133.85 s
- full train command wall time: 273.13 s

## Updated Evaluation Results

### Stability Classifier

| Split | Accuracy | Precision | Recall | F1 | Majority Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Test | 0.9127 | 0.9908 | 0.8805 | 0.9324 | 0.6840 |
| OOD | 0.8745 | 0.9470 | 0.8625 | 0.9028 | 0.6755 |

### Trajectory Regressor

| Split | Stable Samples | RMSE | MAE | Mean Baseline RMSE | Mean Baseline MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Test | 46,011 | 0.0902 | 0.0399 | 0.3618 | 0.2786 |
| OOD | 43,057 | 0.1344 | 0.0687 | 0.3448 | 0.2647 |

### Derived Metric Error

| Split | Overshoot MAE | Rise-Time MAE | Settling-Time MAE | Steady-State Error MAE |
| --- | ---: | ---: | ---: | ---: |
| Test | 3.59 pct-pts | 0.073 s | 0.976 s | 0.0454 |
| OOD | 6.21 pct-pts | 0.099 s | 1.425 s | 0.0638 |

Coverage:

- rise time defined for 79.35% of stable test predictions and 81.23% of stable OOD predictions
- settling time defined for 29.91% of stable test predictions and 21.36% of stable OOD predictions

Runtime:

| Benchmark | Simulator | Surrogate | Speedup |
| --- | ---: | ---: | ---: |
| Single sample | 1.145 ms | 0.111 ms | 10.35x |
| Batch of 512 | 0.689 s | 0.00212 s | 324.50x |

Auxiliary wall times:

- evaluation command wall time: 16.12 s
- benchmark command wall time: 35.65 s

## Scaling Effects On Accuracy

Compared with the earlier `full_v2` run:

| Metric | `full_v2` | `full_v3` | Direction |
| --- | ---: | ---: | --- |
| Test classifier accuracy | 0.9816 | 0.9127 | worse |
| Test classifier F1 | 0.9884 | 0.9324 | worse |
| Test trajectory RMSE | 0.0659 | 0.0902 | worse |
| Test trajectory MAE | 0.0275 | 0.0399 | worse |
| Test overshoot MAE | 4.22 | 3.59 | better |
| Test rise-time MAE | 0.104 | 0.073 | better |
| Test settling-time MAE | 1.096 | 0.976 | better |
| Test SSE MAE | 0.0162 | 0.0454 | worse |

Interpretation:

- The scaled dataset is much broader and materially harder.
- Stability classification degraded because the previous dataset was substantially easier.
- Full-trajectory error also rose, but it still remains far better than the mean-trajectory baseline.
- Several shape-sensitive derived metrics improved, which suggests the larger dataset helped the regressor learn transient behavior better even while final-value bias became worse.

## Scaling Effects On OOD Performance

Compared with the earlier `full_v2` OOD results:

| Metric | `full_v2` | `full_v3` | Direction |
| --- | ---: | ---: | --- |
| OOD classifier accuracy | 0.9607 | 0.8745 | worse |
| OOD classifier F1 | 0.9771 | 0.9028 | worse |
| OOD trajectory RMSE | 0.1218 | 0.1344 | worse |
| OOD trajectory MAE | 0.0637 | 0.0687 | worse |
| OOD overshoot MAE | 9.97 | 6.21 | better |
| OOD rise-time MAE | 0.173 | 0.099 | better |
| OOD settling-time MAE | 1.592 | 1.425 | better |
| OOD SSE MAE | 0.0281 | 0.0638 | worse |

Interpretation:

- The OOD family remains genuinely hard.
- Scaling the dataset did not magically solve OOD generalization.
- It did improve several transient metrics, especially overshoot and rise time.
- The classifier remains clearly better than the majority baseline, but the gap is much smaller than on the earlier easier dataset.

## What Worked

- The large dataset generation pipeline remained deterministic and reproducible under multiprocessing.
- Plant-level splitting and the OOD family holdout were preserved correctly at scale.
- The new plant families made the benchmark materially richer.
- The regressor still beat the mean baseline by a large margin on both ID and OOD splits.
- Batch surrogate speedup became very strong at the larger benchmark size.
- The new plots are useful: error distributions, family-level bars, and PID stability slices all surfaced real behavior rather than noise.

## What Failed Or Remains Weak

- Classifier accuracy dropped sharply relative to the easier 43k dataset.
- Steady-state error became worse on both ID and OOD splits.
- First-order plants remain almost entirely stable with positive PID gains, so boundary-focused sampling is less informative there.
- Settling-time coverage is still limited because many oscillatory cases do not settle within the fixed 8 s horizon.
- The project still does not cover delays, saturation, or nonlinear plants.

## Recommended Next Steps

- Add richer plant encodings, especially pole-zero and frequency-response summaries.
- Target the classifier more aggressively near ambiguous stability boundaries.
- Improve the regressor for final-value bias and oscillatory tails.
- Separate mathematical stability from “usable under control-effort limit” more explicitly.
- Add at least one stronger baseline architecture after this scaled MLP reference point.
