"""Versioned persistence for Phase-0 calibration artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from topocal.calibration.empirical import EmpiricalRiskCalibrator
from topocal.experiments.manifest import file_sha256
from topocal.features.support import GaussianSupportModel

_BUNDLE_VERSION = 1


@dataclass(frozen=True, slots=True)
class CalibrationBundleManifest:
    artifact_version: int
    feature_schema: tuple[str, ...]
    max_failure_risk: float
    min_support_score: float
    calibration_dataset_fingerprint: str
    calibration_split_fingerprint: str
    support_model_sha256: str
    risk_model_sha256: str

    def __post_init__(self) -> None:
        if self.artifact_version != _BUNDLE_VERSION:
            raise ValueError("unsupported calibration bundle version")
        if not self.feature_schema or len(set(self.feature_schema)) != len(self.feature_schema):
            raise ValueError("feature_schema must be non-empty and unique")
        if not 0.0 <= self.max_failure_risk <= 1.0:
            raise ValueError("max_failure_risk must be in [0, 1]")
        if not 0.0 <= self.min_support_score <= 1.0:
            raise ValueError("min_support_score must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class CalibrationBundle:
    support_model: GaussianSupportModel
    risk_model: EmpiricalRiskCalibrator
    manifest: CalibrationBundleManifest


def save_calibration_bundle(
    directory: str | Path,
    *,
    support_model: GaussianSupportModel,
    risk_model: EmpiricalRiskCalibrator,
    feature_schema: tuple[str, ...],
    max_failure_risk: float,
    min_support_score: float,
    calibration_dataset_fingerprint: str,
    calibration_split_fingerprint: str,
) -> CalibrationBundleManifest:
    """Write auditable calibration state without pickle."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    support_path = destination / "support_model.npz"
    risk_path = destination / "risk_model.npz"
    support_model.save(support_path)
    risk_model.save(risk_path)

    manifest = CalibrationBundleManifest(
        artifact_version=_BUNDLE_VERSION,
        feature_schema=feature_schema,
        max_failure_risk=max_failure_risk,
        min_support_score=min_support_score,
        calibration_dataset_fingerprint=calibration_dataset_fingerprint,
        calibration_split_fingerprint=calibration_split_fingerprint,
        support_model_sha256=file_sha256(support_path),
        risk_model_sha256=file_sha256(risk_path),
    )
    (destination / "feature_schema.json").write_text(
        json.dumps({"names": list(feature_schema)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "thresholds.json").write_text(
        json.dumps(
            {
                "max_failure_risk": max_failure_risk,
                "min_support_score": min_support_score,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (destination / "calibration_manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_calibration_bundle(directory: str | Path) -> CalibrationBundle:
    """Load a bundle and verify model-file hashes before use."""

    source = Path(directory)
    data = json.loads((source / "calibration_manifest.json").read_text(encoding="utf-8"))
    manifest = CalibrationBundleManifest(
        artifact_version=int(data["artifact_version"]),
        feature_schema=tuple(str(item) for item in data["feature_schema"]),
        max_failure_risk=float(data["max_failure_risk"]),
        min_support_score=float(data["min_support_score"]),
        calibration_dataset_fingerprint=str(data["calibration_dataset_fingerprint"]),
        calibration_split_fingerprint=str(data["calibration_split_fingerprint"]),
        support_model_sha256=str(data["support_model_sha256"]),
        risk_model_sha256=str(data["risk_model_sha256"]),
    )
    support_path = source / "support_model.npz"
    risk_path = source / "risk_model.npz"
    if file_sha256(support_path) != manifest.support_model_sha256:
        raise ValueError("support model hash mismatch")
    if file_sha256(risk_path) != manifest.risk_model_sha256:
        raise ValueError("risk model hash mismatch")
    return CalibrationBundle(
        support_model=GaussianSupportModel.load(support_path),
        risk_model=EmpiricalRiskCalibrator.load(risk_path),
        manifest=manifest,
    )
