from pathlib import Path

import numpy as np
import pytest

from topocal.calibration.artifacts import load_calibration_bundle, save_calibration_bundle
from topocal.calibration.empirical import EmpiricalRiskCalibrator
from topocal.features.support import GaussianSupportModel
from topocal.types import FeatureVector

NAMES = ("rayleigh", "residual")


def fitted_models() -> tuple[GaussianSupportModel, EmpiricalRiskCalibrator]:
    vectors = [
        FeatureVector(np.array([float(i), float(i % 2)]), NAMES) for i in range(8)
    ]
    support = GaussianSupportModel().fit(vectors)
    risk = EmpiricalRiskCalibrator(neighbors=3).fit(vectors, [0, 0, 0, 0, 1, 1, 1, 1])
    return support, risk


def test_calibration_bundle_roundtrip(tmp_path: Path) -> None:
    support, risk = fitted_models()
    manifest = save_calibration_bundle(
        tmp_path,
        support_model=support,
        risk_model=risk,
        feature_schema=NAMES,
        max_failure_risk=0.05,
        min_support_score=0.1,
        calibration_dataset_fingerprint="dataset",
        calibration_split_fingerprint="split",
    )
    loaded = load_calibration_bundle(tmp_path)
    assert loaded.manifest == manifest
    query = FeatureVector(np.array([2.0, 0.0]), NAMES)
    assert loaded.support_model.support_score(query) == pytest.approx(support.support_score(query))
    assert loaded.risk_model.predict_failure_probability(query) == pytest.approx(
        risk.predict_failure_probability(query)
    )
    assert (tmp_path / "feature_schema.json").exists()
    assert (tmp_path / "thresholds.json").exists()


def test_calibration_bundle_detects_tampering(tmp_path: Path) -> None:
    support, risk = fitted_models()
    save_calibration_bundle(
        tmp_path,
        support_model=support,
        risk_model=risk,
        feature_schema=NAMES,
        max_failure_risk=0.05,
        min_support_score=0.1,
        calibration_dataset_fingerprint="dataset",
        calibration_split_fingerprint="split",
    )
    with (tmp_path / "support_model.npz").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_calibration_bundle(tmp_path)
