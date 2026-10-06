# Literature Map

This is a working implementation-oriented map, not a complete related-work section.

## Topology shift benchmark

**Beyond Arbitrary Geometry: Topology Generalization in Neural PDE Operators** (2026)

- establishes topology as a controlled OOD axis
- introduces TopoBox-3D
- reports strong error association with spectral quantities, including the initial Rayleigh quotient
- implication: “topology causes failure” and “Rayleigh predicts failure” are not TopoCal novelty claims

Paper: https://arxiv.org/abs/2609.05860
Dataset: https://huggingface.co/datasets/cppyyy/TopoBox-3D

## Topology-preserving operator learning

**Topology-Preserving Neural Operator Learning via Hodge Decomposition / Hodge Spectral Duality** (ICML 2026)

- explicitly encodes Hodge/topological structure into an operator architecture
- public MIT-licensed code exists
- implication: TopoCal should be post-hoc and architecture-agnostic rather than proposing another topology-aware operator by default

Paper: https://arxiv.org/abs/2605.13834
Code: https://github.com/ContinuumCoder/Hodge-Spectral-Duality

## General neural operator ecosystem

Official PyTorch ecosystem implementation:
https://github.com/NeuralOperator/neuraloperator

Use adapters rather than vendoring model code.

## Numerical solver

FEniCSx is the intended first fallback integration. Current 2026 stable line is 0.11.

Docs: https://docs.fenicsproject.org/

## Research collision to monitor

Continuously re-check:

- neural/numerical multi-fidelity routing
- conformal neural operators
- weighted conformal prediction under shift
- a-posteriori PDE error estimators
- topology/spectral OOD detection
- selective prediction/risk control

Before submission, rerun a dedicated nearest-prior search and record it in the paper appendix/research log.

## TopoBox-3D release layout (as documented, not yet confirmed locally)

Source: the dataset's own `TopoBox-3D/DATASET.md` on Hugging Face. The loader encodes this layout and fails loudly on any difference.

- `TopoBox-3D/packed/protocol_{A..D}/{train,validation,test_iid,test_ood}/shard_NNNN.h5` holds `/shared/*` and `/samples/<geometry_id>/*` (points, tetra, edges, faces, incidence COO, SDFs, harmonic bases, regular grid features, metadata)
- `TopoBox-3D-HodgeHeat/protocol_*/<split>/shard_NNNN.h5` holds `/samples/<geometry_id>/k{0,1,2}/{w0,wT,mass,harmonic_basis,...}` with four initial-condition configs (non_harmonic, weak_harmonic, balanced, strong_harmonic)
- 5,280 geometries and 63,360 instances, which is 12 instances per geometry (3 degrees times 4 configs)
- The paper does not state a precomputed Rayleigh quotient in the release, so it must be computed from `w0`, the stiffness operator, and `mass`

Open points to confirm on real files:

- the exact key names and dtypes in both shard types
- whether `manifest.csv` uses `beta1`/`beta2`/`geometry_family`/`seed` (what `TopoBoxManifest` requires) or `beta_0`/`beta_1`/`beta_2`/`sample_count_k*` (what `DATASET.md` lists)
- the path format inside `SHA256SUMS.txt`
