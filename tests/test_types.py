import numpy as np
import pytest

from topocal.types import FeatureVector, PredictionBundle


def test_feature_vector_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        FeatureVector(np.array([1.0, 2.0]), ("x", "x"))


def test_feature_vector_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        FeatureVector(np.array([np.nan]), ("x",))


def test_prediction_bundle_rejects_nonfinite_solution() -> None:
    with pytest.raises(ValueError, match="finite"):
        PredictionBundle(np.array([np.inf]))
