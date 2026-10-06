from dataclasses import dataclass

import numpy as np
import pytest

from topocal.features.pipeline import CompositeFeatureExtractor
from topocal.types import FeatureVector, PredictionBundle, Problem


@dataclass
class Extractor:
    name: str
    value: float

    def extract(self, problem: Problem, prediction: PredictionBundle) -> FeatureVector:
        return FeatureVector(np.array([self.value]), (self.name,))


def test_composite_feature_extractor_concatenates_named_sources() -> None:
    extractor = CompositeFeatureExtractor((Extractor("spectral", 1.0), Extractor("physics", 2.0)))
    result = extractor.extract(Problem("x", None), PredictionBundle(np.array([0.0])))
    assert result.names == ("spectral", "physics")
    assert result.values.tolist() == [1.0, 2.0]


def test_composite_feature_extractor_rejects_duplicate_names() -> None:
    extractor = CompositeFeatureExtractor((Extractor("x", 1.0), Extractor("x", 2.0)))
    with pytest.raises(ValueError, match="unique"):
        extractor.extract(Problem("x", None), PredictionBundle(np.array([0.0])))


def test_composite_feature_extractor_requires_sources() -> None:
    with pytest.raises(ValueError, match="at least one"):
        CompositeFeatureExtractor(())
