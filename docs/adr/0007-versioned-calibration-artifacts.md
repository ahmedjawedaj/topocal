# ADR 0007: Calibration artifacts are versioned and pickle-free

## Status
Accepted.

## Decision
Fitted support and empirical-risk baselines serialize as versioned NPZ artifacts with explicit feature schemas and hyperparameters. Experiment manifests record hashes for externally supplied checkpoints/artifacts.

## Rationale
Reliability semantics depend on fitted calibration state and feature ordering. Unversioned or opaque pickle files make results difficult to audit, unsafe to load, and easy to mismatch with a later feature pipeline.
