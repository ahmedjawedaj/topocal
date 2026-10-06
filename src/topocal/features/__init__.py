"""Reliability feature extraction and support baselines."""

from topocal.features.pipeline import CompositeFeatureExtractor
from topocal.features.spectral import rayleigh_quotient
from topocal.features.support import GaussianSupportModel, support_threshold_from_calibration

__all__ = [
    "CompositeFeatureExtractor",
    "GaussianSupportModel",
    "rayleigh_quotient",
    "support_threshold_from_calibration",
]
