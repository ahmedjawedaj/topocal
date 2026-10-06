# Research Plan

## Working title

**When Can We Trust a Neural PDE Solver? Spectral-Conditioned Risk Calibration Under Topology Shift**

## Problem statement

Neural operators can be much faster than numerical PDE solvers, but deployment geometries may differ structurally from training data. Topology shift can alter solution behavior and error in ways not captured by standard in-distribution metrics. A useful scientific surrogate should therefore expose not only a prediction but also whether its reliability estimate transfers to the current physical regime.

## Primary hypothesis

Physically meaningful spectral/topological descriptors improve prediction of **calibration transfer and unacceptable solution error** under structured topology shift beyond generic model uncertainty alone.

This is a falsifiable hypothesis, not a project requirement.

## Research questions

### RQ1
Does uncertainty/risk calibration degrade systematically as topology moves away from the calibration distribution?

### RQ2
Which signals predict unacceptable neural-operator error under topology shift?

- topology descriptors
- Rayleigh quotient
- low spectrum
- PDE residual
- model uncertainty
- combinations

### RQ3
Can support-aware spectral conditioning maintain lower selective risk than generic calibration/shift baselines?

### RQ4
At a fixed accepted-risk target, how many numerical-solver calls can be avoided?

### RQ5
Does the reliability mechanism transfer across operator architectures and a second PDE family?

## Primary outcome

The key outcome is **selective risk vs neural coverage**, not average RMSE alone.

Define unacceptable prediction:

`failure_i = 1[relative_error_i > epsilon]`

At a target risk alpha, report:

- accepted neural coverage
- empirical accepted-set failure rate
- confidence interval
- solver fallback rate
- end-to-end cost/runtime

## Baselines

At minimum:

1. always neural
2. always numerical
3. random fallback at matched coverage
4. topology-count threshold
5. Rayleigh quotient threshold
6. PDE-residual threshold
7. model uncertainty
8. generic OOD distance
9. generic shift-aware calibration where assumptions are satisfied
10. proposed method, only if Phase-3 evidence justifies it

## Datasets

### Primary
TopoBox-3D / controlled topology-OOD benchmark.

### Secondary
Preferred: perforated-domain elasticity generated with a reproducible FEM pipeline.

Alternative: Darcy/flow domain with obstacle/connectivity shift.

## Reproducibility

- at least 3 seeds for learned components
- frozen split manifests
- no threshold selection on final OOD test
- report all attempted primary baselines
- preserve negative results
- version dataset/code/checkpoints

## Paper claim hierarchy

Strongest allowed claim depends on evidence:

1. **observational:** feature correlates with failure
2. **predictive:** feature predicts failure on held-out regimes
3. **calibrated:** empirical risk matches declared calibration protocol
4. **risk-controlled:** finite-sample/statistical statement under explicit assumptions
5. **formal certificate:** only if theorem/proof actually supports it

Never jump levels in wording.
