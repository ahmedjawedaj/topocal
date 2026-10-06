# Contributing to TopoCal

TopoCal welcomes software improvements, reproducible research, documentation, and
negative results. It is pre-alpha: interfaces may change, and the synthetic demo does
not establish real PDE reliability or a formal risk guarantee.

## Start here

- Read [README](README.md) and [architecture](docs/ARCHITECTURE.md).
- Browse [open issues](https://github.com/ahmedjawedaj/topocal/issues).
  `good first issue` marks bounded starter work; `help wanted` marks work seeking contributors.
- For method changes, also read the [research plan](docs/RESEARCH_PLAN.md),
  [experiment protocol](docs/EXPERIMENT_PROTOCOL.md), and [literature map](docs/LITERATURE_MAP.md).
- Discuss substantial API or research changes in an issue before building them.

## Set up a fork

Fork [ahmedjawedaj/topocal](https://github.com/ahmedjawedaj/topocal), then:

```bash
git clone https://github.com/YOUR_USERNAME/topocal.git
cd topocal
git remote add upstream https://github.com/ahmedjawedaj/topocal.git
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
git switch -c fix/short-description
```

Python 3.11 and 3.12 are the CI targets. Windows users can activate the environment
with `.venv\Scripts\Activate.ps1`. Heavy research packages and real data are optional;
the ordinary tests generate small synthetic fixtures.

## Before submitting

```bash
ruff check .
mypy src/topocal
pytest --cov=topocal --cov-report=term-missing --cov-fail-under=90
topocal demo
```

Do not weaken checks to make a PR pass. Add meaningful regression tests for behavior
changes, including failure paths. Public APIs need types, docstrings, and a runnable
example. Follow the existing style; keep unrelated formatting out of behavioral changes.

## Open a pull request

1. Commit focused changes with an imperative summary, such as `Fix string partition guard`.
2. Push your branch and open a PR against `main`; use a draft while work is incomplete.
3. Explain the problem, resulting behavior, and checks performed. Link the issue with
   `Fixes #NUMBER` only when all its acceptance criteria are complete.
4. Update docs and ADRs when changing a contract. Disclose private data dependencies,
   upstream code attribution, and any changes to experiment assumptions.
5. Address review comments. Merge requires CI and maintainer review; squash merge is preferred.

To refresh a fork before opening a PR:

```bash
git fetch upstream
git rebase upstream/main
```

Do not force-push the shared `main` branch. If rebasing your own PR branch, use
`git push --force-with-lease` and coordinate with anyone sharing that branch.

## Research contributions

Ground truth stays offline. Calibration, threshold selection, and model development
must follow the four-way split protocol; the final OOD test stays locked. Compare
against simple baselines and retain negative comparisons. A report must identify the
dataset/split, checkpoint, error tolerance, risk target, calibration size, coverage,
accepted failure rate, fallback rate, runtime, seeds, and applicable confidence intervals.
Resample related PDE instances at the geometry level.

Do not commit datasets, checkpoints, credentials, or generated run directories.
Link reproducible external artifacts and record hashes and provenance instead.

## Community and attribution

Participation follows the [Code of Conduct](CODE_OF_CONDUCT.md). Vulnerabilities use
the private route in [SECURITY.md](SECURITY.md). Contributions are made under this
repository's Apache-2.0 license; retain applicable upstream notices. Cite substantive
research sources and acknowledge collaborators in the PR/report. No CLA or signed-off
commit requirement is currently configured.

There is no promised review turnaround; keep reproduction instructions self-contained.
