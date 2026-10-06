# ADR 0005: Selective metrics use realizable thresholds

## Status
Accepted.

## Decision
Risk-coverage curves group samples with identical risk scores. Every reported operating point must correspond to a deterministic policy of the form:

`accept if risk <= threshold`.

The evaluator must not split tied scores sample-by-sample unless randomized tie-breaking is explicitly part of the policy and reported as such.

## Rationale
KNN and quantized risk estimators naturally create ties. Splitting a tie can report coverage/risk points that no deterministic deployment threshold can actually realize, overstating attainable selective performance.
