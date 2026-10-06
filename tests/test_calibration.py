from pathlib import Path

import numpy as np
import pytest

from topocal.calibration.empirical import EmpiricalRiskCalibrator
from topocal.types import FeatureVector

NAMES = ("risk_signal",)


def fv(value: float) -> FeatureVector:
    return FeatureVector(np.array([value]), NAMES)


def fitted_model() -> EmpiricalRiskCalibrator:
    features = [fv(float(v)) for v in range(20)]
    failures = np.array([0] * 10 + [1] * 10)
    return EmpiricalRiskCalibrator(neighbors=5).fit(features, failures)


def test_empirical_risk_tracks_local_failures() -> None:
    model = fitted_model()
    assert model.predict_failure_probability(fv(2.0)) < model.predict_failure_probability(fv(17.0))


def test_risk_artifact_roundtrip(tmp_path: Path) -> None:
    model = fitted_model()
    path = tmp_path / "risk.npz"
    model.save(path)
    restored = EmpiricalRiskCalibrator.load(path)
    assert restored.predict_failure_probability(fv(17.0)) == pytest.approx(
        model.predict_failure_probability(fv(17.0))
    )


def test_empirical_risk_rejects_bad_configuration() -> None:
    with pytest.raises(ValueError, match="neighbors"):
        EmpiricalRiskCalibrator(neighbors=0)
    with pytest.raises(ValueError, match="smoothing"):
        EmpiricalRiskCalibrator(smoothing=-1.0)


def test_empirical_risk_rejects_schema_mismatch() -> None:
    model = fitted_model()
    with pytest.raises(ValueError, match="schema"):
        model.predict_failure_probability(FeatureVector(np.array([1.0]), ("other",)))


def test_empirical_risk_fit_validation() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        EmpiricalRiskCalibrator().fit([], [])
    with pytest.raises(ValueError, match="one value"):
        EmpiricalRiskCalibrator().fit([fv(1.0)], [0, 1])
    with pytest.raises(ValueError, match="binary"):
        EmpiricalRiskCalibrator().fit([fv(1.0)], [2])
    with pytest.raises(ValueError, match="same schema"):
        EmpiricalRiskCalibrator().fit(
            [fv(1.0), FeatureVector(np.array([2.0]), ("other",))],
            [0, 1],
        )


def test_empirical_risk_requires_fitted_model() -> None:
    with pytest.raises(RuntimeError, match="not been fitted"):
        EmpiricalRiskCalibrator().predict_failure_probability(fv(1.0))
