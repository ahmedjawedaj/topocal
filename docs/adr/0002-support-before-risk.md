# ADR 0002: Evaluate Calibration Support Before Risk

Status: Accepted

## Context

A numerically precise risk estimate can be misleading when a deployment query lies outside the distribution where the estimator was calibrated.

## Decision

Routing first evaluates calibration support. Unsupported inputs defer to the trusted solver without interpreting the ordinary risk score as valid.

## Consequences

This creates two distinct quantities:

- support of the reliability model
- predicted failure risk conditional on using the calibrated regime

This distinction is central to the research hypothesis.
