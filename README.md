# TopoCal

**Spectral-conditioned risk calibration and selective fallback for neural PDE operators under topology shift.**

TopoCal is a research-first open-source framework for answering a practical question:

> When is a neural PDE surrogate safe enough to use, and when should the system defer to a numerical solver?

The project focuses on **structured topology shift**, where a deployment geometry contains tunnels, cavities, connectivity patterns, or spectral regimes not represented in training/calibration data.

## Status

**Pre-alpha research implementation.** The current release establishes the software contracts, calibration primitives, support detector, routing policy, reproducibility plan, and a working synthetic vertical slice. It does **not** claim production safety guarantees yet.

## Research hypothesis

Prediction quality, uncertainty calibration, and calibration transfer are different problems. Under topology shift, a neural operator may produce a plausible field while its ordinary confidence estimate is itself unreliable.

TopoCal therefore separates:

1. **surrogate prediction** — `u_hat = model(x)`
2. **support estimation** — is this query similar to the calibrated physical/spectral regime?
3. **risk calibration** — estimated probability that error exceeds an application tolerance
4. **selective routing** — accept the neural result or defer to a trusted solver

## Architecture

```text
PDE problem
   │
   ├──► neural operator ───────────────► predicted field
   │                                        │
   ├──► topology/spectral features          │
   ├──► physics residuals ◄─────────────────┘
   └──► model uncertainty
                    │
                    ▼
        calibration-support detector
                    │
            ┌───────┴────────┐
            │                │
       supported        unsupported
            │                │
            ▼                ▼
      calibrated risk   numerical solver
            │
       ┌────┴────┐
       │         │
     accept    fallback
```

## Quick start

Use Python 3.11 or 3.12. Core runtime requires only NumPy. Clone the source before installing:

```bash
git clone https://github.com/ahmedjawedaj/topocal.git
cd topocal
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest

topocal demo
```

The demo creates a synthetic calibration distribution, fits a support model and empirical failure-risk calibrator, then routes new samples through either a mock neural surrogate or deterministic fallback.

## Contributing and current hardening work

Forks and contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md) and
[open issues](https://github.com/ahmedjawedaj/topocal/issues). Use
[Discussions](https://github.com/ahmedjawedaj/topocal/discussions) for questions and design proposals.

Open issues track known correctness, dataset-manifest, CI, and source-distribution
defects. The current baseline
passes 121 tests with 95% coverage on Python 3.11/3.12, while lint and type checks fail.
Do not use this development snapshot for a production reliability decision.

## Core API

```python
from topocal import TopoCalRouter

# Runnable synthetic example; replace both adapters for real PDE use.
import numpy as np
from topocal import Problem
from topocal.demo import build_demo_router

router = build_demo_router()
problem = Problem("example", np.array([0.1, -0.8, -1.0, -0.7]))

result = router.solve(problem)

print(result.decision)       # accept_neural | fallback_solver
print(result.risk)           # float when calibrated; None when unsupported/unavailable
print(result.risk_status)    # calibrated | unsupported | unavailable | invalid
print(result.support_score)  # support heuristic or None if unavailable/invalid
print(result.support_status) # supported | outside_support | unavailable | invalid
print(result.reason)
```

The neural operator and fallback solver are injected through protocols, so TopoCal does not require a specific PDE architecture or FEM package. Reliability features may be returned by legacy surrogate adapters or extracted independently through `FeatureExtractor` / `CompositeFeatureExtractor`, keeping topology, spectral, physics, and model-specific signals separate from inference.

## Reliability semantics

TopoCal is fail-closed by design. Invalid, non-finite, unavailable, or out-of-support reliability outputs trigger the numerical fallback. An unsupported query reports `risk=None`; it is **not** assigned a fictitious 100% failure probability.

Risk-coverage metrics group tied risk scores, so every reported operating point is realizable by a deterministic `risk <= threshold` deployment policy.

## Reproducibility artifacts

Phase-0 support/risk models serialize as versioned, pickle-free NPZ artifacts. A calibration bundle records:

- feature schema
- risk/support policy thresholds
- calibration dataset and split fingerprints
- SHA-256 hashes of fitted model artifacts

Experiment manifests record exact config, Git commit/dirty state, dependency and hardware metadata, random seeds, artifact hashes, feature schema, and solver metadata. Final `ood_test` split access is locked by default through `SplitManifest`.

## TopoBox-3D data loading

**Experimental adapter:** the loader is tested only on a synthetic release, and real
shards remain unverified until `topocal inspect-shard` has been run on them. Data is hosted by
[the upstream TopoBox-3D project](https://huggingface.co/datasets/cppyyy/TopoBox-3D);
keep downloads outside Git. `runs/splits.json` must be created under the
[four-way experiment protocol](docs/EXPERIMENT_PROTOCOL.md); it is not a bundled file.

```python
from topocal.datasets import TopoBoxDataset, TopoBoxManifest
from topocal.experiments import ExperimentPartition, SplitManifest

manifest = TopoBoxManifest.from_csv("data/TopoBox-3D/manifest.csv")
split = SplitManifest.from_json("runs/splits.json")  # dataset_fingerprint == manifest.sha256
train = TopoBoxDataset(
    manifest=manifest,
    split_manifest=split,
    partition=ExperimentPartition.TRAIN,
    geometry_root="data/TopoBox-3D/packed",
    solution_root="data/TopoBox-3D-HodgeHeat",
)
key = train.instance_keys()[0]
instance = train.load_instance(key)                       # inputs only
scored = train.load_instance(key, include_target=True)    # ground truth, for training and scoring
problem = scored.to_problem()                             # target is always stripped
provenance = train.provenance()                           # shard hashes and observed layout
```

Loading is lazy and read-only, needs `pip install 'topocal[data]'`, and the locked `ood_test` partition still requires `allow_final_test=True`. See [`docs/adr/0008-lazy-partition-bound-topobox-loading.md`](docs/adr/0008-lazy-partition-bound-topobox-loading.md).

## Planned adapters

- official `neuraloperator` models
- TopoBox-3D benchmark protocol (the data loader is in place, see below)
- Hodge/topological and spectral descriptors
- FEniCSx fallback solver
- second PDE family: perforated-domain elasticity or Darcy flow

See [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) and the open research issues.

## Scientific guardrails

TopoCal will not market a generic uncertainty score as a mathematical guarantee. Any coverage/risk claim must state:

- calibration distribution
- test distribution
- topology split
- error tolerance
- confidence/risk target
- support assumptions
- number of calibration/test samples
- whether the guarantee is empirical, conformal, asymptotic, or formal

## Repository navigation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — software architecture
- [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) — paper and experiment plan
- [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) — reproducible evaluation protocol
- [`docs/adr/`](docs/adr/) — architecture decisions

## License

Apache-2.0. See [`LICENSE`](LICENSE).
