# Development

## Minimal local path

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
make check
python -m topocal.cli demo
```

## Heavy research environment

Install core/dev first, then add research packages as needed. FEniCSx should be installed using its official environment instructions rather than forced into the core wheel.

## Branching

Recommended:

- `main` stays reproducible
- short-lived feature branches
- experiment-only changes still require configs and artifact outputs

## Versioning

Use semantic versioning for software interfaces. Research artifact versions should additionally record dataset/split/model checkpoint identifiers.

## TopoBox-3D data

The 27 GB release stays outside the repository. Install the reader with `pip install -e '.[data]'` (already part of `dev`).

Before trusting the loader on real data, record the layout of one shard of each kind and check that `missing_expected` and `unexpected` are empty:

```bash
topocal inspect-shard path/to/TopoBox-3D/packed/protocol_B/train/shard_0000.h5 --kind geometry
topocal inspect-shard path/to/TopoBox-3D-HodgeHeat/protocol_B/train/shard_0000.h5 --kind hodge_heat
```

Tests use a synthetic release written at test time by `tests/topobox_fixture.py`.
