# Architecture

## Design goal

TopoCal is a **post-hoc reliability layer**, not a neural-operator architecture. It must be possible to swap the surrogate, uncertainty estimator, support model, calibration method, and numerical solver independently.

## Components

### Problem adapter

Converts benchmark/application data into an opaque `Problem`. Core TopoCal does not own mesh semantics.

### Surrogate adapter

Runs a neural operator and returns:

- predicted solution
- named reliability features available at prediction time

No ground-truth solver error is allowed here.

### Feature pipeline

Research adapters may compute:

- Betti numbers/connectivity descriptors
- Rayleigh quotient
- Laplacian/Hodge spectral summaries
- PDE residuals
- boundary-condition residuals
- model uncertainty
- geometry statistics

Feature schema is explicit and ordered.

### Support model

Answers:

> Is this query represented by the calibration distribution in the chosen reliability-feature space?

The initial Gaussian/Mahalanobis model is only a baseline.

### Risk model

Estimates the probability of exceeding a declared application error tolerance. The initial KNN empirical estimator is only a transparent baseline.

### Router

Policy order is intentional:

1. reject unsupported queries first
2. estimate calibrated failure risk only within the supported regime
3. accept or fallback according to a declared risk threshold

### Fallback solver

An adapter around FEniCSx or another trusted solver. It is not imported by core code.

## Online invariants

The online path must never require:

- ground truth
- future solver error
- final test labels
- statistics fitted using the final test set

## Failure behavior

When reliability machinery fails closed, the preferred behavior is **fallback**, not acceptance.

Examples:

- malformed feature schema → error, not silent coercion
- unsupported topology/spectrum → fallback
- non-finite reliability feature → error/fallback in adapter layer
- risk model unavailable → fallback in application integration

## Future architecture

The eventual library may add `assess()` separately from `solve()` so researchers can analyze risk without invoking a fallback solver. Do not add this until real benchmark requirements are known.


## Feature extraction boundary

Production adapters should separate neural inference from reliability feature extraction. `PredictionBundle.features` remains optional for backward compatibility, while `FeatureExtractor` and `CompositeFeatureExtractor` support independent topology, spectral, physics, and model-uncertainty sources.

## Fail-closed routing

Reliability infrastructure is part of the safety boundary. Non-finite, out-of-range, unavailable, or unsupported reliability outputs route to the trusted fallback. Unsupported regimes return `risk=None` with an explicit status rather than a synthetic 100% risk value.

## Artifact boundary

Fitted support/risk models use versioned pickle-free artifacts with explicit feature schemas. Experiment manifests record exact config, Git state, dependency/hardware metadata, random seeds, artifact hashes, feature schema, and solver metadata.

## Dataset adapter boundary

`TopoBoxDataset` is the problem adapter for TopoBox-3D. It turns a manifest ID into geometry arrays and Hodge-heat inputs, and into an opaque `Problem` whose payload never carries the target. h5py stays behind it as an optional dependency (`topocal[data]`). The observed shard layout is recorded with every run, see ADR 0008.
