# third_order_pid_surrogate_bo

Objective: `J = settling_time + overshoot_pct/100 + steady_state_error`

```text
method                method_label           plants  mean_cost  median_cost  stability_rate  mean_runtime  mean_simulations
bayesian_optimization Bayesian Optimization 100     14.7483    11.9052      0.8200          0.2984        24.0000          
   surrogate_gradient    Surrogate Gradient 100     20.1031    18.7957      0.7200          0.0974         1.0000          
         surrogate_bo        Surrogate + BO 100     15.8317    18.1678      0.8200          0.2408        10.0000          
```