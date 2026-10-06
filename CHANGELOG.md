# Changelog

## Unreleased

### Fixed

- Ruff and strict mypy pass again, and CI no longer overrides the 90% coverage gate with 85%
- the final OOD test guard now normalizes partition input, so the string `"ood_test"` is locked exactly like `ExperimentPartition.OOD_TEST`
- `allow_final_test` unlocks the final test only when it is the boolean `True`
- invalid partition values raise a clear `ValueError` at `SplitManifest.ids_for` and `TopoBoxDataset`
- support artifacts are validated on load and save: finite positive `regularization` and `temperature`, valid feature names and dimensions, finite state, and a symmetric positive semidefinite precision (NumPy-only, scale-aware `1e-8` tolerance)
- a negative precision no longer clamps distance to zero and grants an outlier support 1.0, and a corrupted in-memory model now raises so the router falls back
- `load_calibration_bundle` rejects an invalid support artifact even when its hash matches the manifest
- support validation is overflow-safe: a negative precision with a query that overflows the distance now raises and falls back instead of scoring 1.0, and a huge finite precision such as `1e308` stays finite instead of loading as infinity
- `temperature` and `regularization` are revalidated at `fit`, `support_score` and `save`, so mutating them to `inf`, `nan`, zero, a negative, a string or a boolean can no longer grant outliers maximum support or write an artifact the loader rejects. Strings and booleans are no longer coerced in the constructor either
- fitting raises `ValueError` when calibration features overflow float64 in the covariance
- `TopoBoxManifest.from_csv` loads the real upstream header (no `seed` column). `TopoBoxGeometryRecord.seed` is `int | None`, `None` when the column is absent or the cell is blank, and a malformed supplied seed raises `ValueError`. Seeds are never defaulted to 0
- short manifest rows raise `ValueError` instead of `AttributeError`
- the source distribution now includes `tests/conftest.py`, `tests/topobox_fixture.py`, the whole test tree and the public docs via `MANIFEST.in`, and excludes data, runs, caches, checkpoints and local-only guides
- license metadata uses an SPDX expression with `license-files`, so the build requires setuptools 77.0.3 or newer
- `scripts/check_distribution.py` inspects the sdist and wheel, and the new `package` CI job installs the extracted sdist with its dev extra, runs its tests from outside the checkout, and smoke-tests the wheel in a NumPy-only environment

### Changed

- `SplitManifest.ids_for` and `TopoBoxDataset` accept an `ExperimentPartition` or its exact string value, and `ExperimentPartition.parse` is the single normalization point
- breaking for invalid files only: support artifacts that are materially asymmetric, indefinite, all-zero or carry non-finite hyperparameters now raise `ValueError` on load. The artifact version is unchanged
- `GaussianSupportModel` private fitted fields are `init=False`
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
