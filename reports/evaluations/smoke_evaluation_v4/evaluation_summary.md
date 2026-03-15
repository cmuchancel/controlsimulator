# Evaluation Summary

## Classifier

### test
- accuracy: 0.7086
- precision: 0.8499
- recall: 0.6686
- f1: 0.7484
- majority_accuracy: 0.6481
- majority_f1: 0.7865

### ood_test
- accuracy: 0.7422
- precision: 0.8306
- recall: 0.7560
- f1: 0.7916
- majority_accuracy: 0.6476
- majority_f1: 0.7861

## Regressor

### test
- n_stable_samples: 1575
- trajectory_rmse: 0.2815
- trajectory_mae: 0.1845
- mean_baseline_trajectory_rmse: 0.4281
- mean_baseline_trajectory_mae: 0.3517
- knn_baseline_trajectory_rmse: 0.4061
- knn_baseline_trajectory_mae: 0.2654
- overshoot_pct_mae: 24.4408
- mean_baseline_overshoot_pct_mae: 29.4829
- knn_baseline_overshoot_pct_mae: 17.7507
- overshoot_pct_defined_fraction: 1.0000
- rise_time_mae: 1.1450
- mean_baseline_rise_time_mae: nan
- knn_baseline_rise_time_mae: 1.0684
- rise_time_defined_fraction: 0.6781
- settling_time_mae: 7.9596
- mean_baseline_settling_time_mae: nan
- knn_baseline_settling_time_mae: 3.4324
- settling_time_defined_fraction: 0.1581
- steady_state_error_mae: 0.1443
- mean_baseline_steady_state_error_mae: 0.2495
- knn_baseline_steady_state_error_mae: 0.1869
- steady_state_error_defined_fraction: 1.0000

### ood_test
- n_stable_samples: 746
- trajectory_rmse: 0.2410
- trajectory_mae: 0.1626
- mean_baseline_trajectory_rmse: 0.3496
- mean_baseline_trajectory_mae: 0.2834
- knn_baseline_trajectory_rmse: 0.3538
- knn_baseline_trajectory_mae: 0.2400
- overshoot_pct_mae: 22.8978
- mean_baseline_overshoot_pct_mae: 25.7315
- knn_baseline_overshoot_pct_mae: 16.2794
- overshoot_pct_defined_fraction: 1.0000
- rise_time_mae: 1.5926
- mean_baseline_rise_time_mae: nan
- knn_baseline_rise_time_mae: 0.9532
- rise_time_defined_fraction: 0.7319
- settling_time_mae: 6.4579
- mean_baseline_settling_time_mae: nan
- knn_baseline_settling_time_mae: 3.5485
- settling_time_defined_fraction: 0.2265
- steady_state_error_mae: 0.1057
- mean_baseline_steady_state_error_mae: 0.1743
- knn_baseline_steady_state_error_mae: 0.1488
- steady_state_error_defined_fraction: 1.0000

- family_metrics_path: family_metrics.csv
- sample_errors_path: sample_errors.csv
- artifacts_plot_dir: /Users/chancelavoie/Desktop/nn_pid_simulator/artifacts/plots/smoke_evaluation_v4