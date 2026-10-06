"""Calibration-support estimators and threshold utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from topocal.types import FeatureVector

Array = npt.NDArray[np.float64]
_SUPPORT_ARTIFACT_VERSION = 1


@dataclass(slots=True)
class GaussianSupportModel:
    """Transparent regularized Mahalanobis support baseline.

    The score is exp(-0.5 * d^2 / temperature), where d is a regularized Mahalanobis
    distance. The score is a ranking/support heuristic, not a probability or formal coverage
    guarantee. Thresholds should be selected on a calibration partition, never on final test.
    """

    regularization: float = 1e-6
    temperature: float = 1.0
    _mean: Array | None = None
    _precision: Array | None = None
    _names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.regularization <= 0:
            raise ValueError("regularization must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

    def fit(self, features: list[FeatureVector]) -> GaussianSupportModel:
        if len(features) < 2:
            raise ValueError("at least two calibration samples are required")
        names = features[0].names
        if any(item.names != names for item in features):
            raise ValueError("all feature vectors must have the same names and order")
        x = np.stack([item.values for item in features], axis=0)
        mean = x.mean(axis=0)
        centered = x - mean
        cov = (centered.T @ centered) / max(len(features) - 1, 1)
        cov = cov + self.regularization * np.eye(cov.shape[0])
        self._mean = mean
        self._precision = np.linalg.pinv(cov)
        self._names = names
        return self

    def squared_distance(self, features: FeatureVector) -> float:
        self._ensure_fitted(features)
        assert self._mean is not None
        assert self._precision is not None
        delta = features.values - self._mean
        return max(float(delta @ self._precision @ delta), 0.0)

    def support_score(self, features: FeatureVector) -> float:
        d2 = self.squared_distance(features)
        return float(np.exp(-0.5 * d2 / self.temperature))

    def save(self, path: str | Path) -> None:
        """Serialize fitted state as a versioned, pickle-free NPZ artifact."""

        self._ensure_fitted_schema()
        assert self._mean is not None
        assert self._precision is not None
        assert self._names is not None
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps(
            {
                "artifact_version": _SUPPORT_ARTIFACT_VERSION,
                "model_type": type(self).__name__,
                "regularization": self.regularization,
                "temperature": self.temperature,
                "feature_names": list(self._names),
            },
            sort_keys=True,
        )
        np.savez_compressed(
            destination,
            mean=self._mean,
            precision=self._precision,
            metadata=metadata,
        )

    @classmethod
    def load(cls, path: str | Path) -> GaussianSupportModel:
        """Load and validate a versioned NPZ artifact."""

        with np.load(Path(path), allow_pickle=False) as artifact:
            metadata = json.loads(str(artifact["metadata"].item()))
            if metadata.get("artifact_version") != _SUPPORT_ARTIFACT_VERSION:
                raise ValueError("unsupported support artifact version")
            if metadata.get("model_type") != cls.__name__:
                raise ValueError("support artifact model type mismatch")
            model = cls(
                regularization=float(metadata["regularization"]),
                temperature=float(metadata["temperature"]),
            )
            mean = np.asarray(artifact["mean"], dtype=np.float64)
            precision = np.asarray(artifact["precision"], dtype=np.float64)
            names = tuple(str(name) for name in metadata["feature_names"])
        if mean.ndim != 1 or precision.shape != (mean.size, mean.size) or len(names) != mean.size:
            raise ValueError("invalid support artifact dimensions")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(precision)):
            raise ValueError("support artifact contains non-finite values")
        model._mean = mean
        model._precision = precision
        model._names = names
        return model

    def _ensure_fitted(self, features: FeatureVector) -> None:
        self._ensure_fitted_schema()
        if features.names != self._names:
            raise ValueError("feature schema differs from fitted support model")

    def _ensure_fitted_schema(self) -> None:
        if self._mean is None or self._precision is None or self._names is None:
            raise RuntimeError("support model has not been fitted")


def support_threshold_from_calibration(
    scores: npt.ArrayLike,
    *,
    lower_quantile: float = 0.01,
) -> float:
    """Choose a support threshold from calibration scores only.

    The returned threshold is the requested lower quantile, clipped to [0, 1]. This utility
    does not confer a statistical coverage guarantee; it simply replaces arbitrary hand-picked
    thresholds with a reproducible calibration-partition rule.
    """

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)) or not np.all((values >= 0.0) & (values <= 1.0)):
        raise ValueError("support scores must be finite values in [0, 1]")
    if not 0.0 <= lower_quantile <= 1.0:
        raise ValueError("lower_quantile must be in [0, 1]")
    return float(np.quantile(values, lower_quantile))
