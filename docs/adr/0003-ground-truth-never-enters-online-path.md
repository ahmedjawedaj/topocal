# ADR 0003: Ground Truth Never Enters the Online Path

Status: Accepted

## Context

Research code can accidentally leak numerical-solver error into the feature path, producing unrealistic failure detection.

## Decision

The `Surrogate` produces only deployable features. True numerical-solver error is consumed by offline calibration/evaluation code and never by `TopoCalRouter`.

## Consequences

Tests and adapters must preserve this separation. Any future feature requiring exact ground truth is automatically offline-only.
