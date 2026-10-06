# Experiment Protocol

This document prevents accidental leakage and moving-goalpost evaluation.

## Data partitions

Use four logical partitions:

1. `train` — neural operator fitting
2. `calibration` — risk/support calibration and policy thresholds
3. `ood_dev` — method development on held-out topology regimes
4. `ood_test` — final frozen topology regimes, used once for primary reporting

The code mirrors these names with `ExperimentPartition`. `OOD_TEST` is locked by default and requires an explicit `allow_final_test=True` access decision.

Where dataset size requires reuse, document the compromise explicitly.

## Split identity

A split manifest must contain:

- dataset version/hash
- sample IDs
- topology labels
- random seed
- split-generation code version
- dataset fingerprint
- disjointness validation across all four logical partitions

## Error definition

Before primary experiments, freeze:

- norm
- normalization convention
- unacceptable-error tolerance `epsilon`
- target accepted risk `alpha`

Sensitivity analyses may vary them later but must not replace the primary endpoint.

## Evaluation metrics

### Prediction

- relative L2 error
- task-appropriate physics metric

### Failure detection

- AUROC
- AUPRC

### Selective reliability

- risk-coverage curve
- area under risk-coverage curve
- accepted risk at fixed coverages
- maximum neural coverage at target risk

### Operational

- fallback rate
- neural runtime
- feature-extraction runtime
- numerical-solver runtime
- total wall-clock cost

### Group robustness

Break down by:

- topology class
- topology distance
- geometry complexity
- spectral-distance bins

## Confidence intervals

Use bootstrap intervals for aggregate empirical metrics when independence assumptions are reasonable. If multiple PDE instances share a geometry, bootstrap at the **geometry level**, not individual-instance level, to avoid pseudo-replication.

## Leakage rules

Forbidden:

- computing normalization statistics from final test
- selecting epsilon/alpha using final test
- tuning support threshold on final test
- choosing feature set after inspecting final test performance

## Kill tests

Run before method complexity increases:

1. Rayleigh threshold
2. PDE residual threshold
3. generic feature-distance baseline
4. ordinary model uncertainty
5. calibration degradation plot under increasing topology shift

If a simple baseline solves the declared problem, simplify the paper.


## Reliability output semantics

Outside calibration support, calibrated failure risk is **unknown**, not `1.0`. Reports must distinguish:

- calibrated risk
- unsupported query
- unavailable risk estimate
- invalid reliability output

Invalid/non-finite reliability values fail closed to the numerical solver.

## Selective-metric tie handling

Risk-coverage points must be realizable by a deterministic threshold. Equal risk scores are accepted as a group; do not create intermediate points by arbitrary sample ordering.


## TopoBox data access

Experiment code reads TopoBox-3D only through `TopoBoxDataset`, bound to one `ExperimentPartition`.

- `SplitManifest.dataset_fingerprint` is the SHA-256 of the TopoBox `manifest.csv` (`TopoBoxManifest.sha256`). The loader rejects any other value.
- The independent resampling unit is the geometry ID. A partition never mixes PDE instances of one geometry with another partition.
- Logical partitions are independent of the release's physical `train`, `validation`, `test_iid`, `test_ood` directories. Which shard holds a geometry is resolved through the manifest record, never through the partition.
- Ground-truth fields (`wT`, `relative_final_mass_norm`) require `include_target=True`. Use it for training and scoring only.
- Each run records `TopoBoxDataset.provenance()` in `ExperimentManifest.dataset_provenance`: manifest hash, split fingerprint, ID hash, SHA-256 of every shard read, and the observed layout of each.
- Where a published `SHA256SUMS.txt` is available, pass it to `provenance(published_sums=...)` to compare each shard hash with the release.
