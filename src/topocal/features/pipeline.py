"""Composable reliability-feature extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from topocal.types import FeatureExtractor, FeatureVector, PredictionBundle, Problem


@dataclass(frozen=True, slots=True)
class CompositeFeatureExtractor:
    """Concatenate independent topology/spectral/physics/model feature extractors."""

    extractors: tuple[FeatureExtractor, ...]

    def __post_init__(self) -> None:
        if not self.extractors:
            raise ValueError("at least one feature extractor is required")

    def extract(self, problem: Problem, prediction: PredictionBundle) -> FeatureVector:
        vectors = [extractor.extract(problem, prediction) for extractor in self.extractors]
        names = tuple(name for vector in vectors for name in vector.names)
        if len(set(names)) != len(names):
            raise ValueError("composite feature names must be unique")
        values = np.concatenate([vector.values for vector in vectors])
        return FeatureVector(values=values, names=names)
