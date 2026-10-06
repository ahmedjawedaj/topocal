"""Reproducibility metadata for experiments."""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    experiment: str
    created_at: str
    python: str
    platform: str
    git_commit: str | None
    git_dirty: bool | None
    dataset_fingerprint: str
    split_fingerprint: str
    config_sha256: str
    config: dict[str, Any]
    dependencies: dict[str, str]
    hardware: dict[str, str]
    random_seeds: dict[str, int] = field(default_factory=dict)
    artifacts_sha256: dict[str, str] = field(default_factory=dict)
    feature_schema: tuple[str, ...] = ()
    solver: dict[str, str] = field(default_factory=dict)
    dataset_provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        experiment: str,
        dataset_fingerprint: str,
        split_fingerprint: str,
        config: dict[str, Any],
        repo_root: str | Path | None = None,
        random_seeds: dict[str, int] | None = None,
        artifact_paths: dict[str, str | Path] | None = None,
        feature_schema: tuple[str, ...] = (),
        solver: dict[str, str] | None = None,
        dataset_provenance: dict[str, Any] | None = None,
    ) -> ExperimentManifest:
        config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        artifacts = {
            name: file_sha256(path) for name, path in (artifact_paths or {}).items()
        }
        return cls(
            experiment=experiment,
            created_at=datetime.now(UTC).isoformat(),
            python=sys.version.split()[0],
            platform=platform.platform(),
            git_commit=_git_commit(repo_root),
            git_dirty=_git_dirty(repo_root),
            dataset_fingerprint=dataset_fingerprint,
            split_fingerprint=split_fingerprint,
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            config=dict(config),
            dependencies=_package_versions(
                (
                    "topocal",
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "torch",
                    "neuraloperator",
                    "fenics-dolfinx",
                )
            ),
            hardware=_hardware_metadata(),
            random_seeds=dict(random_seeds or {}),
            artifacts_sha256=artifacts,
            feature_schema=tuple(feature_schema),
            solver=dict(solver or {}),
            dataset_provenance=dict(dataset_provenance or {}),
        )

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def write_bundle(self, directory: str | Path) -> None:
        """Write manifest plus the exact experiment config as separate auditable files."""

        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        self.write_json(destination / "manifest.json")
        (destination / "config.json").write_text(
            json.dumps(self.config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo_root: str | Path | None) -> str | None:
    result = _run_git(["rev-parse", "HEAD"], repo_root)
    return result.strip() if result else None


def _git_dirty(repo_root: str | Path | None) -> bool | None:
    result = _run_git(["status", "--porcelain"], repo_root, preserve_empty=True)
    if result is None:
        return None
    return bool(result.strip())


def _run_git(
    args: list[str],
    repo_root: str | Path | None,
    *,
    preserve_empty: bool = False,
) -> str | None:
    cwd = Path(repo_root) if repo_root is not None else None
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    if preserve_empty:
        return completed.stdout
    return completed.stdout or None


def _package_versions(packages: tuple[str, ...]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _hardware_metadata() -> dict[str, str]:
    result = {
        "machine": platform.machine() or "unknown",
        "processor": platform.processor() or "unknown",
    }
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError):
        return result
    version = getattr(torch, "__version__", None)
    if version is not None:
        result["torch"] = str(version)
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and bool(cuda.is_available()):
        result["cuda_available"] = "true"
        result["gpu"] = str(cuda.get_device_name(0))
        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        if cuda_version:
            result["cuda"] = str(cuda_version)
    else:
        result["cuda_available"] = "false"
    return result
