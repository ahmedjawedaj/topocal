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

### Numerical policy for extreme values

- Symmetry and eigenvalue checks run on the precision divided by its largest absolute entry, so finite entries near the float64 maximum cannot overflow `P - P.T`, `P + P.T` or the eigensolver. The stored matrix is `0.5 * P + 0.5 * P.T`, which cannot overflow, or the rescaled projection when tiny negative eigenvalues were removed. The eigendecomposition and every computed matrix must be finite, otherwise loading raises `ValueError`.
- A huge but valid finite precision is kept. Queries then overflow to a distance of `+inf`, which maps to support 0.0. Overflow can lower support and never raise it.
- A negative distance is tolerated only when both the distance and its magnitude are finite and the distance is within `1e-8` of the magnitude. Negative infinity, NaN and unmeasurable magnitudes raise `ValueError`, so a corrupted state falls back through the router instead of being clamped to maximum support.
- Fitting rejects calibration features whose covariance overflows float64.

### Mutable hyperparameters

`regularization` and `temperature` are public fields, so they are revalidated whenever they are used: `fit` checks both, `support_score` checks `temperature`, and `save` checks both before touching the destination. Only real finite positive numbers pass. Strings, bytes and booleans are rejected rather than coerced. Valid reassignment, for example a new temperature, remains supported and is what `save` writes.
