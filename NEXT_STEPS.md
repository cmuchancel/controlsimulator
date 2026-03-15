# Next Steps

1. Improve the stability classifier near the boundary.
   The v4 classifier is only modestly above majority accuracy and loses to majority F1 on the OOD family. Boundary-aware loss weighting or focal loss is the clearest next move.

2. Add richer plant encodings.
   Padded coefficients are no longer enough once fourth-order and multimode plants dominate the hard cases. Pole-zero summaries, frequency-response features, and explicit damping/bandwidth descriptors should help.

3. Strengthen the regressor for fourth-order and multimode families.
   `fourth_order_mixed_complex` and `two_mode_resonant` are now the main regression bottlenecks. A residual MLP or basis-decoder head is the next reasonable comparison.

4. Make benchmarking device-aware.
   On `mps`, single-sample inference is slower than simulation while batch inference is much faster. Future reports should separate CPU and accelerator latency so the deployment story is clearer.

5. Extend simulator scope only after the current benchmark is solved better.
   Delays, saturation, and noise are still worthwhile, but the current 15-family continuous-time LTI benchmark is already hard enough to justify another modeling round first.
