from pathlib import Path

import numpy as np
import pytest

from topocal.features.support import GaussianSupportModel, support_threshold_from_calibration
from topocal.types import FeatureVector

NAMES = ("a", "b")


def fv(a: float, b: float) -> FeatureVector:
    return FeatureVector(np.array([a, b]), NAMES)


def fitted_model() -> GaussianSupportModel:
    return GaussianSupportModel(temperature=2.0).fit(
        [fv(-1, 0), fv(0, -1), fv(1, 0), fv(0, 1), fv(0, 0)]
    )


def test_support_is_higher_near_calibration_cloud() -> None:
    model = fitted_model()
    assert model.support_score(fv(0.1, 0.1)) > model.support_score(fv(10, 10))


def test_support_artifact_roundtrip(tmp_path: Path) -> None:
    model = fitted_model()
    path = tmp_path / "support.npz"
    model.save(path)
    restored = GaussianSupportModel.load(path)
    assert restored.support_score(fv(0.5, 0.5)) == pytest.approx(model.support_score(fv(0.5, 0.5)))


def test_support_threshold_comes_from_calibration_scores() -> None:
    threshold = support_threshold_from_calibration([0.1, 0.2, 0.4, 0.8], lower_quantile=0.25)
    assert threshold == pytest.approx(0.175)


@pytest.mark.parametrize("scores", [[], [np.nan], [-0.1], [1.1]])
def test_support_threshold_rejects_invalid_scores(scores: list[float]) -> None:
    with pytest.raises(ValueError):
        support_threshold_from_calibration(scores)


def test_support_requires_fitted_model() -> None:
    with pytest.raises(RuntimeError, match="not been fitted"):
        GaussianSupportModel().support_score(fv(0, 0))


def test_support_configuration_and_fit_validation() -> None:
    with pytest.raises(ValueError, match="regularization"):
        GaussianSupportModel(regularization=0)
    with pytest.raises(ValueError, match="temperature"):
        GaussianSupportModel(temperature=0)
    with pytest.raises(ValueError, match="at least two"):
        GaussianSupportModel().fit([fv(0, 0)])
    with pytest.raises(ValueError, match="same names"):
        GaussianSupportModel().fit(
            [fv(0, 0), FeatureVector(np.array([1.0, 1.0]), ("x", "y"))]
        )


def test_support_rejects_schema_mismatch() -> None:
    model = fitted_model()
    with pytest.raises(ValueError, match="schema"):
        model.support_score(FeatureVector(np.array([0.0, 0.0]), ("x", "y")))


def test_support_threshold_rejects_invalid_quantile() -> None:
    with pytest.raises(ValueError, match="lower_quantile"):
        support_threshold_from_calibration([0.2, 0.4], lower_quantile=1.2)
