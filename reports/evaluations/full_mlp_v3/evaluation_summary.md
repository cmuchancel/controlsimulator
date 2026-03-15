# Evaluation Summary

## Classifier

### test
- accuracy: 0.9127
- precision: 0.9908
- recall: 0.8805
- f1: 0.9324
- majority_accuracy: 0.6840
- majority_f1: 0.8124

### ood_test
- accuracy: 0.8745
- precision: 0.9470
- recall: 0.8625
- f1: 0.9028
- majority_accuracy: 0.6755
- majority_f1: 0.8063

## Regressor

### test
- n_stable_samples: 46011
- trajectory_rmse: 0.0902
- trajectory_mae: 0.0399
- baseline_trajectory_rmse: 0.3618
- baseline_trajectory_mae: 0.2786
- overshoot_pct_mae: 3.5949
- baseline_overshoot_pct_mae: 30.2224
- overshoot_pct_defined_fraction: 1.0000
- rise_time_mae: 0.0730
- baseline_rise_time_mae: nan
- rise_time_defined_fraction: 0.7935
- settling_time_mae: 0.9758
- baseline_settling_time_mae: nan
- settling_time_defined_fraction: 0.2991
- steady_state_error_mae: 0.0454
- baseline_steady_state_error_mae: 0.1914
- steady_state_error_defined_fraction: 1.0000

### ood_test
- n_stable_samples: 43057
- trajectory_rmse: 0.1344
- trajectory_mae: 0.0687
- baseline_trajectory_rmse: 0.3448
- baseline_trajectory_mae: 0.2647
- overshoot_pct_mae: 6.2143
- baseline_overshoot_pct_mae: 26.8077
- overshoot_pct_defined_fraction: 1.0000
- rise_time_mae: 0.0992
- baseline_rise_time_mae: nan
- rise_time_defined_fraction: 0.8123
- settling_time_mae: 1.4251
- baseline_settling_time_mae: nan
- settling_time_defined_fraction: 0.2136
- steady_state_error_mae: 0.0638
- baseline_steady_state_error_mae: 0.1911
- steady_state_error_defined_fraction: 1.0000

- family_metrics_path: family_metrics.csv
- artifacts_plot_dir: /Users/chancelavoie/Desktop/nn_pid_simulator/artifacts/plots/full_mlp_v3