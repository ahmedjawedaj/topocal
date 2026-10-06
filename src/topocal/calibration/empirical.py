"""Transparent empirical failure-risk calibration baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from topocal.types import FeatureVector

Array = npt.NDArray[np.float64]
_RISK_ARTIFACT_VERSION = 1


@dataclass(slots=True)
class EmpiricalRiskCalibrator:
    """K-nearest empirical failure probability in standardized feature space.

    This is an intentionally interpretable Phase-0 baseline. Later experiments will compare
    it with conformal risk control, weighted conformal methods, logistic/boosted models, and
    topology-conditioned variants.
    """

    neighbors: int = 32
    smoothing: float = 1.0
    _x: Array | None = None
    _y: Array | None = None
    _mean: Array | None = None
    _scale: Array | None = None
    _names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.neighbors < 1:
            raise ValueError("neighbors must be at least 1")
        if self.smoothing < 0:
            raise ValueError("smoothing must be non-negative")

    def fit(
        self,
        features: list[FeatureVector],
        failures: npt.ArrayLike,
    ) -> EmpiricalRiskCalibrator:
        if not features:
            raise ValueError("calibration features cannot be empty")
        y = np.asarray(failures, dtype=np.float64)
        if y.ndim != 1 or y.shape[0] != len(features):
            raise ValueError("failures must have one value per feature vector")
        if not np.all((y == 0.0) | (y == 1.0)):
            raise ValueError("failures must be binary values")
        names = features[0].names
        if any(item.names != names for item in features):
            raise ValueError("all feature vectors must share the same schema")
        x = np.stack([item.values for item in features])
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(scale < 1e-12, 1.0, scale)
        self._x = (x - mean) / scale
        self._y = y
        self._mean = mean
        self._scale = scale
        self._names = names
        return self

    def predict_failure_probability(self, features: FeatureVector) -> float:
        self._ensure_fitted(features)
        assert self._x is not None
        assert self._y is not None
        assert self._mean is not None
        assert self._scale is not None
        query = (features.values - self._mean) / self._scale
        distances = np.linalg.norm(self._x - query, axis=1)
        k = min(self.neighbors, self._x.shape[0])
        idx = np.argpartition(distances, k - 1)[:k]
        failures = float(self._y[idx].sum())
        denominator = k + 2.0 * self.smoothing
        if denominator <= 0:
            raise RuntimeError("invalid empirical-risk denominator")
        # Beta(smoothing, smoothing)-style smoothing avoids exact 0/1 risk on small samples.
        return float((failures + self.smoothing) / denominator)

    def save(self, path: str | Path) -> None:
        """Serialize fitted state as a versioned, pickle-free NPZ artifact."""

        self._ensure_fitted_schema()
        assert self._x is not None
        assert self._y is not None
        assert self._mean is not None
        assert self._scale is not None
        assert self._names is not None
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps(
            {
                "artifact_version": _RISK_ARTIFACT_VERSION,
                "model_type": type(self).__name__,
                "neighbors": self.neighbors,
                "smoothing": self.smoothing,
                "feature_names": list(self._names),
            },
            sort_keys=True,
        )
        np.savez_compressed(
            destination,
            x=self._x,
            y=self._y,
            mean=self._mean,
            scale=self._scale,
            metadata=metadata,
        )

    @classmethod
    def load(cls, path: str | Path) -> EmpiricalRiskCalibrator:
        """Load and validate a versioned NPZ artifact."""

        with np.load(Path(path), allow_pickle=False) as artifact:
            metadata = json.loads(str(artifact["metadata"].item()))
            if metadata.get("artifact_version") != _RISK_ARTIFACT_VERSION:
                raise ValueError("unsupported risk artifact version")
            if metadata.get("model_type") != cls.__name__:
                raise ValueError("risk artifact model type mismatch")
            model = cls(
                neighbors=int(metadata["neighbors"]),
                smoothing=float(metadata["smoothing"]),
            )
            x = np.asarray(artifact["x"], dtype=np.float64)
            y = np.asarray(artifact["y"], dtype=np.float64)
            mean = np.asarray(artifact["mean"], dtype=np.float64)
            scale = np.asarray(artifact["scale"], dtype=np.float64)
            names = tuple(str(name) for name in metadata["feature_names"])
        if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.size:
            raise ValueError("invalid risk artifact sample dimensions")
        if (
            mean.ndim != 1
            or scale.ndim != 1
            or x.shape[1] != mean.size
            or mean.shape != scale.shape
        ):
            raise ValueError("invalid risk artifact feature dimensions")
        if len(names) != mean.size:
            raise ValueError("risk artifact feature schema mismatch")
        arrays = (x, y, mean, scale)
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("risk artifact contains non-finite values")
        if not np.all((y == 0.0) | (y == 1.0)) or np.any(scale <= 0):
            raise ValueError("risk artifact contains invalid fitted state")
        model._x = x
        model._y = y
        model._mean = mean
        model._scale = scale
        model._names = names
        return model

    def _ensure_fitted(self, features: FeatureVector) -> None:
        self._ensure_fitted_schema()
        if features.names != self._names:
            raise ValueError("feature schema differs from fitted calibrator")

    def _ensure_fitted_schema(self) -> None:
        if any(item is None for item in (self._x, self._y, self._mean, self._scale, self._names)):
            raise RuntimeError("risk calibrator has not been fitted")
