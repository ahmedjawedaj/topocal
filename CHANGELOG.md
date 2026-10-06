# Changelog

## Unreleased

### Fixed

- Ruff and strict mypy pass again, and CI no longer overrides the 90% coverage gate with 85%
- the final OOD test guard now normalizes partition input, so the string `"ood_test"` is locked exactly like `ExperimentPartition.OOD_TEST`
- `allow_final_test` unlocks the final test only when it is the boolean `True`
- invalid partition values raise a clear `ValueError` at `SplitManifest.ids_for` and `TopoBoxDataset`

### Changed

- `SplitManifest.ids_for` and `TopoBoxDataset` accept an `ExperimentPartition` or its exact string value, and `ExperimentPartition.parse` is the single normalization point
- `TopoBoxManifest.geometry_ids` and `split_fingerprint` take explicit keyword-only filters (`protocol`, `split`, `is_ood`)

## 0.1.0-dev3 - 2026-10-06

### Added

- `TopoBoxDataset`: lazy, read-only, partition-bound loading of TopoBox-3D geometry and Hodge-heat samples through `TopoBoxManifest` and `SplitManifest`
- layout validation for geometry and Hodge-heat arrays, with errors that list the keys actually present
- opt-in ground truth: `wT` and `relative_final_mass_norm` load only with `include_target=True`, and `to_problem()` always strips the target
- `inspect_shard` and `topocal inspect-shard` to record the observed HDF5 layout of a shard
- `TopoBoxDataset.provenance()` with SHA-256 of every shard read, observed layouts, and comparison with a published `SHA256SUMS.txt`
- `ExperimentManifest.dataset_provenance`
- synthetic HDF5 release fixture for tests (no real data needed in CI)
- ADR 0008
- `data` extra (`h5py`), also included in `research` and `dev`

### Changed

- split manifests must carry `TopoBoxManifest.sha256` as `dataset_fingerprint` to be used with the loader

## 0.1.0-dev2 - 2026-10-06

### Added

- fail-closed routing for invalid/unavailable support and risk outputs
- explicit `RiskStatus` and `SupportStatus` semantics
- optional independent/composite reliability feature extraction
- threshold-realizable risk-coverage metrics that group tied scores
- calibration-derived support-threshold helper
- train/calibration/OOD-dev/OOD-test split manifest with final-test lock
- richer experiment provenance: exact config, Git dirty state, dependencies, hardware, seeds, artifact hashes, feature schema, solver metadata
- versioned pickle-free support/risk model serialization
- auditable calibration bundle with model hashes, feature schema, thresholds and calibration fingerprints
- ADRs covering fail-closed reliability, tie handling, final-test locking and artifact persistence
- expanded failure-path tests; CI coverage gate raised to 90%

### Changed

- unsupported queries now report `risk=None` instead of a synthetic risk of `1.0`
- build backend moved to setuptools for a simpler editable-install path in restricted environments

## 0.1.0-dev - 2026-10-05

### Added

- project architecture and research plans
- framework-agnostic surrogate/fallback/risk/support protocols
- Rayleigh quotient primitive
- calibration-support baseline
- empirical failure-risk baseline
- selective routing policy
- risk-coverage evaluation primitives
- synthetic end-to-end demo
- CI and open-source contribution scaffolding
