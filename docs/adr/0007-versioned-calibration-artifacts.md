# ADR 0007: Calibration artifacts are versioned and pickle-free

## Status
Accepted.

## Decision
Fitted support and empirical-risk baselines serialize as versioned NPZ artifacts with explicit feature schemas and hyperparameters. Experiment manifests record hashes for externally supplied checkpoints/artifacts.

## Rationale
Reliability semantics depend on fitted calibration state and feature ordering. Unversioned or opaque pickle files make results difficult to audit, unsafe to load, and easy to mismatch with a later feature pipeline.

## Fitted-state validation

Loading an artifact validates its content, not only its hash and version, because a matching hash proves integrity and says nothing about validity.

- `regularization` and `temperature` must be finite and strictly positive, both in the constructor and in artifact metadata.
- Feature names must be a non-empty list of unique, non-blank strings whose length equals the mean and precision dimensions.
- The mean and precision must be finite. The precision must be symmetric and positive semidefinite within `1e-8` relative to its largest entry (symmetry) or largest eigenvalue (curvature), using NumPy only. Accepted matrices are stored as their exact symmetric, PSD form. A singular PSD precision from pseudoinverse truncation is valid and preserved.
- An all-zero precision is rejected. It is formally PSD but gives every query distance zero and so maximum support.
- Any invalid content raises `ValueError`. Runtime exceptions raised later by a corrupted in-memory model still defer through the router, which falls back with `SupportStatus.UNAVAILABLE`.

The rule closes a fail-open case where a negative precision clamped every distance to zero, so an outlier received support 1.0.

These constraints newly reject artifacts that previously loaded. The artifact version is unchanged because the format is the same and only invalid content is refused. The private fitted fields are `init=False`, so state can no longer be injected through the constructor.
