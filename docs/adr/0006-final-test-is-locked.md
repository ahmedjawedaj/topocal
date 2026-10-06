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

## Rationale
The final topology-OOD set must not become an implicit validation set through repeated agent or researcher inspection. The guard makes final-test access a deliberate, reviewable action rather than the default.
