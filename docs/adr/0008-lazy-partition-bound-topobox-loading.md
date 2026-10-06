# ADR 0008: TopoBox loading is lazy, partition-bound, and schema-validated

## Status
Accepted.

## Decision
`TopoBoxDataset` is the only public way to read TopoBox-3D samples. It is built from a `TopoBoxManifest`, a `SplitManifest`, and one `ExperimentPartition`.

1. Construction opens no files and writes nothing. Roots are checked on first read.
2. IDs come only from `SplitManifest.ids_for(partition, allow_final_test=...)`, so the locked `ood_test` partition cannot be reached without the explicit flag.
3. Every ID resolves through the manifest. IDs that are unknown or ambiguous across protocols are rejected at construction.
4. `SplitManifest.dataset_fingerprint` must equal `TopoBoxManifest.sha256`. A split can only be applied to the manifest it was frozen against.
5. A dataset serves only the IDs of its own partition. Any other ID raises `PermissionError`.
6. Shards are opened read-only, one sample at a time, and closed at once. `h5py` is imported only when a shard is opened.
7. Final-time solutions (`wT`, `relative_final_mass_norm`) are ground truth. They are returned only with `include_target=True`, and `TopoBoxPDEInstance.to_problem()` always strips them.
8. Loaded arrays are validated against the documented layout (dtype family, shape, finiteness, index bounds, incidence consistency, Hodge size against the geometry). Failures list the keys that were actually present.
9. `provenance()` hashes only the shards that were read and records their observed layout. `inspect_shard` and `topocal inspect-shard` expose the same layout record.

## Rationale
The experiment protocol forbids ad-hoc filtering and implicit use of the final test set. Binding data access to a split partition makes the lock hold at the point where bytes are read, not only where IDs are listed. Keeping ground truth opt-in supports the rule that solver error never enters the online path. The documented layout has not been confirmed against real shards, so validation and layout recording are what make a wrong assumption visible instead of silent.

## Consequences
Hashing is limited to touched shards, so a run never pays for the full 27 GB release. Reopening a shard per sample is simple and safe for multiprocessing but not the fastest option. A handle cache can be added behind the same API if profiling shows it matters.
