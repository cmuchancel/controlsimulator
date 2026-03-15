# Evaluation Summary

## Classifier

### test
- accuracy: 0.7980
- precision: 0.9654
- recall: 0.7328
- f1: 0.8332
- majority_accuracy: 0.6882
- majority_f1: 0.8153

### ood_test
- accuracy: 0.7625
- precision: 0.9997
- recall: 0.6715
- f1: 0.8034
- majority_accuracy: 0.7225
- majority_f1: 0.8389

## Regressor

### test
- n_stable_samples: 333011
- trajectory_rmse: 0.1728
- trajectory_mae: 0.0912
- mean_baseline_trajectory_rmse: 0.4311
- mean_baseline_trajectory_mae: 0.3575
- knn_baseline_trajectory_rmse: 0.3719
- knn_baseline_trajectory_mae: 0.2364
- overshoot_pct_mae: 10.2473
- mean_baseline_overshoot_pct_mae: 26.0037
- knn_baseline_overshoot_pct_mae: 14.5919
- overshoot_pct_defined_fraction: 1.0000
- rise_time_mae: 0.3997
- mean_baseline_rise_time_mae: nan
- knn_baseline_rise_time_mae: 0.9659
- rise_time_defined_fraction: 0.6677
- settling_time_mae: 4.3394
- mean_baseline_settling_time_mae: nan
- knn_baseline_settling_time_mae: 2.8712
- settling_time_defined_fraction: 0.2476
- steady_state_error_mae: 0.0782
- mean_baseline_steady_state_error_mae: 0.2594
- knn_baseline_steady_state_error_mae: 0.1671
- steady_state_error_defined_fraction: 1.0000

### ood_test
- n_stable_samples: 166445
- trajectory_rmse: 0.1679
- trajectory_mae: 0.0927
- mean_baseline_trajectory_rmse: 0.3603
- mean_baseline_trajectory_mae: 0.2948
- knn_baseline_trajectory_rmse: 0.3364
- knn_baseline_trajectory_mae: 0.2210
- overshoot_pct_mae: 13.4270
- mean_baseline_overshoot_pct_mae: 26.0561
- knn_baseline_overshoot_pct_mae: 16.9167
- overshoot_pct_defined_fraction: 1.0000
- rise_time_mae: 0.2194
- mean_baseline_rise_time_mae: nan
- knn_baseline_rise_time_mae: 0.6713
- rise_time_defined_fraction: 0.8609
- settling_time_mae: 4.8316
- mean_baseline_settling_time_mae: nan
- knn_baseline_settling_time_mae: 3.4532
- settling_time_defined_fraction: 0.3764
- steady_state_error_mae: 0.0794
- mean_baseline_steady_state_error_mae: 0.1835
- knn_baseline_steady_state_error_mae: 0.1273
- steady_state_error_defined_fraction: 1.0000

- family_metrics_path: family_metrics.csv
- sample_errors_path: sample_errors.csv
- artifacts_plot_dir: /Users/chancelavoie/Desktop/nn_pid_simulator/artifacts/plots/evaluation_v4