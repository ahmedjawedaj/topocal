# ADR 0006: Final OOD test is locked by default

## Status
Accepted.

## Decision
Research split manifests use four logical partitions:

1. `train`
2. `calibration`
3. `ood_dev`
4. `ood_test`

Programmatic access to `ood_test` is denied unless the caller explicitly sets `allow_final_test=True`. Split manifests reject duplicate IDs within or across partitions and should use geometry IDs as the independent unit for TopoBox experiments.

## Input contract
`ExperimentPartition` is a `StrEnum`, so the string `"ood_test"` equals the enum member. Partition input is therefore normalized with `ExperimentPartition.parse` before any guard runs, and the guard tests the normalized member. Public entry points (`SplitManifest.ids_for` and `TopoBoxDataset`) accept an enum member or its exact string value and reject everything else with `ValueError`. Matching is case-sensitive and does no trimming.

The final test unlocks only when `allow_final_test` is the boolean `True`. Truthy look-alikes such as `"false"` or `1` stay locked. `TopoBoxDataset` runs the guard during construction, before any HDF5 file is opened.

## Rationale
The final topology-OOD set must not become an implicit validation set through repeated agent or researcher inspection. The guard makes final-test access a deliberate, reviewable action rather than the default.
