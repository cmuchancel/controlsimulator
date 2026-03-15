# controlsimulator

Research-grade starter repo for learning surrogate models of PID-controlled continuous-time systems.

The repo builds synthetic stable SISO plants, simulates unit-step closed-loop responses under PID control, trains:

- a stability classifier on all generated samples
- a trajectory regressor on stable closed-loop samples only

Artifacts, reports, and final measured results are populated by the smoke and overnight workflows described in the README sections that follow. This file will be finalized after the first end-to-end runs.
