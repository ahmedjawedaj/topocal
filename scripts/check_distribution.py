"""Verify built TopoCal distributions and optionally extract the sdist safely.

Build success and ``twine check`` both passed on an sdist that omitted test support files, so
this script inspects the archives directly. It uses only the standard library.

Usage:
    python scripts/check_distribution.py DIST_DIR
    python scripts/check_distribution.py DIST_DIR --extract EMPTY_DIR   # prints extracted root
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SDIST_REQUIRED = (
    "LICENSE",
    "README.md",
    "CHANGELOG.md",
    "pyproject.toml",
    "MANIFEST.in",
    "tests/conftest.py",
    "tests/topobox_fixture.py",
    "docs/DEVELOPMENT.md",
    "docs/EXPERIMENT_PROTOCOL.md",
    "docs/adr/0006-final-test-is-locked.md",
    "src/topocal/__init__.py",
    "src/topocal/datasets/topobox_loader.py",
)

# Local-only guides, data, run outputs, caches and checkpoints must never ship.
FORBIDDEN_PREFIXES = (
    "agents/",
    "plans/",
    "data/",
    "runs/",
    "checkpoints/",
    "wandb/",
    ".github/",
    "docs/CLAUDE_IMPLEMENTATION_GUIDE.md",
    "docs/OPEN_SOURCE_READINESS.md",
    "docs/GITHUB_SETUP.md",
    "docs/ENGINEERING_STANDARDS.md",
    "docs/MAINTAINING.md",
    "AGENTS.md",
    "CLAUDE.md",
    "HANDOFF.md",
)
FORBIDDEN_PARTS = ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache")
FORBIDDEN_SUFFIXES = (".h5", ".hdf5", ".npz", ".pt", ".pth", ".ckpt", ".pyc", ".coverage")


def find_distributions(dist: Path) -> tuple[Path, Path]:
    sdists = sorted(dist.glob("*.tar.gz"))
    wheels = sorted(dist.glob("*.whl"))
    if len(sdists) != 1 or len(wheels) != 1:
        raise SystemExit(f"expected exactly one sdist and one wheel in {dist}")
    return sdists[0], wheels[0]


def sdist_members(path: Path) -> dict[str, str]:
    """Map archive-relative member name to its root-stripped path."""

    with tarfile.open(path, "r:gz") as archive:
        names = [member.name for member in archive.getmembers() if member.isfile()]
    roots = {name.split("/", 1)[0] for name in names}
    if len(roots) != 1:
        raise SystemExit(f"sdist must have a single root directory, found {sorted(roots)}")
    return {name.split("/", 1)[1]: name for name in names if "/" in name}


def check_sdist(path: Path, repo: Path) -> list[str]:
    problems: list[str] = []
    members = sdist_members(path)
    for required in SDIST_REQUIRED:
        if required not in members:
            problems.append(f"sdist is missing {required}")
    # Every test module and helper in the repository must travel with the sdist, because the
    # tests import each other and `tests/conftest.py`.
    for test_file in sorted((repo / "tests").rglob("*.py")):
        relative = test_file.relative_to(repo).as_posix()
        if "__pycache__" in relative:
            continue
        if relative not in members:
            problems.append(f"sdist is missing {relative}")
    for name in members:
        if name.startswith(FORBIDDEN_PREFIXES):
            problems.append(f"sdist contains local-only or data path {name}")
        if any(part in name.split("/") for part in FORBIDDEN_PARTS):
            problems.append(f"sdist contains cache path {name}")
        if name.endswith(FORBIDDEN_SUFFIXES):
            problems.append(f"sdist contains artifact file {name}")
    return problems


def _metadata(text: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for line in text.split("\n\n", 1)[0].splitlines():
        if ": " in line and not line.startswith(" "):
            key, value = line.split(": ", 1)
            result.setdefault(key, []).append(value)
    return result


def check_license_metadata(text: str, label: str) -> list[str]:
    problems: list[str] = []
    fields = _metadata(text)
    if fields.get("License-Expression") != ["Apache-2.0"]:
        problems.append(f"{label} METADATA lacks License-Expression: Apache-2.0")
    if "LICENSE" not in fields.get("License-File", []):
        problems.append(f"{label} METADATA does not list License-File: LICENSE")
    if any(item.startswith("License ::") for item in fields.get("Classifier", [])):
        problems.append(f"{label} METADATA still uses a deprecated License classifier")
    return problems


def check_wheel(path: Path) -> list[str]:
    problems: list[str] = []
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        metadata_names = [n for n in names if n.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            return [f"wheel must have one METADATA file, found {metadata_names}"]
        problems += check_license_metadata(
            archive.read(metadata_names[0]).decode("utf-8"), "wheel"
        )
        if not any(n.endswith(".dist-info/licenses/LICENSE") for n in names):
            problems.append("wheel does not contain dist-info/licenses/LICENSE")
        if not any(n == "topocal/__init__.py" for n in names):
            problems.append("wheel does not contain topocal/__init__.py")
        for name in names:
            if name.startswith(("tests/", "docs/", "data/", "runs/")):
                problems.append(f"wheel contains non-package path {name}")
            if name.endswith(FORBIDDEN_SUFFIXES):
                problems.append(f"wheel contains artifact file {name}")
    return problems


def check_sdist_metadata(path: Path) -> list[str]:
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if member.name.count("/") == 1 and member.name.endswith("/PKG-INFO"):
                handle = archive.extractfile(member)
                assert handle is not None
                return check_license_metadata(handle.read().decode("utf-8"), "sdist")
    return ["sdist has no PKG-INFO"]


def extract_sdist(path: Path, destination: Path) -> Path:
    """Extract with the stdlib ``data`` filter, which rejects traversal and unsafe links."""

    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise SystemExit(f"{destination} must be empty")
    with tarfile.open(path, "r:gz") as archive:
        archive.extractall(destination, filter="data")
    (root,) = [item for item in destination.iterdir() if item.is_dir()]
    return root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dist", type=Path, help="directory holding one sdist and one wheel")
    parser.add_argument("--repo", type=Path, default=REPO, help="repository root to compare")
    parser.add_argument("--extract", type=Path, help="empty directory to extract the sdist into")
    args = parser.parse_args(argv)

    sdist, wheel = find_distributions(args.dist)
    problems = check_sdist(sdist, args.repo) + check_sdist_metadata(sdist) + check_wheel(wheel)
    if problems:
        for problem in problems:
            print(f"FAIL {problem}", file=sys.stderr)
        return 1
    if args.extract is not None:
        print(extract_sdist(sdist, args.extract))
    else:
        print(f"OK {sdist.name} {wheel.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
