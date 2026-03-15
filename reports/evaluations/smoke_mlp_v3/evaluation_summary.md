# Evaluation Summary

## Classifier

### test
- accuracy: 0.7731
- precision: 0.9550
- recall: 0.7065
- f1: 0.8121
- majority_accuracy: 0.6943
- majority_f1: 0.8196

### ood_test
- accuracy: 0.7951
- precision: 0.8689
- recall: 0.8196
- f1: 0.8435
- majority_accuracy: 0.6736
- majority_f1: 0.8050

## Regressor

### test
- n_stable_samples: 511
- trajectory_rmse: 0.2372
- trajectory_mae: 0.1376
- baseline_trajectory_rmse: 0.3686
- baseline_trajectory_mae: 0.2846
- overshoot_pct_mae: 21.1047
- baseline_overshoot_pct_mae: 30.8352
- overshoot_pct_defined_fraction: 1.0000
- rise_time_mae: 0.2245
- baseline_rise_time_mae: nan
- rise_time_defined_fraction: 0.7573
- settling_time_mae: 4.5298
- baseline_settling_time_mae: nan
- settling_time_defined_fraction: 0.2935
- steady_state_error_mae: 0.1039
- baseline_steady_state_error_mae: 0.1976
- steady_state_error_defined_fraction: 1.0000

### ood_test
- n_stable_samples: 582
- trajectory_rmse: 0.1993
- trajectory_mae: 0.1244
- baseline_trajectory_rmse: 0.3362
- baseline_trajectory_mae: 0.2584
- overshoot_pct_mae: 19.0602
- baseline_overshoot_pct_mae: 22.4586
- overshoot_pct_defined_fraction: 1.0000
- rise_time_mae: 0.4026
- baseline_rise_time_mae: nan
- rise_time_defined_fraction: 0.6890
- settling_time_mae: 3.5723
- baseline_settling_time_mae: nan
- settling_time_defined_fraction: 0.2440
- steady_state_error_mae: 0.0842
- baseline_steady_state_error_mae: 0.1865
- steady_state_error_defined_fraction: 1.0000

- family_metrics_path: family_metrics.csv
- artifacts_plot_dir: /Users/chancelavoie/Desktop/nn_pid_simulator/artifacts/plots/smoke_mlp_v3