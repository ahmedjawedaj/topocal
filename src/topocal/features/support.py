"""Calibration-support estimators and threshold utilities."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from topocal.types import FeatureVector

Array = npt.NDArray[np.float64]
_SUPPORT_ARTIFACT_VERSION = 1

# Fitted-state tolerances. Both are relative to the matrix scale and sit about eight orders of
# magnitude above float64 roundoff, so legitimate pseudoinverse output always passes while any
# materially asymmetric or indefinite matrix does not.
#
# * asymmetry: max|P - P.T| must be at most _STATE_RTOL * max|P|. Accepted matrices are
#   replaced by their exact symmetric part.
# * negative curvature: the smallest eigenvalue must be at least -_STATE_RTOL * the largest.
#   Accepted tiny negative eigenvalues are projected to zero, so the stored matrix is exactly
#   positive semidefinite. Singular PSD matrices (from pseudoinverse truncation) are valid.
#
# An all-zero precision matrix is rejected: it is formally PSD but gives every query distance
# zero and therefore maximum support, which is the failure this validation exists to prevent.
_STATE_RTOL = 1e-8


def _positive_finite(name: str, value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite positive number, got {value!r}") from None
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return number


def _validate_names(names: tuple[str, ...], size: int) -> None:
    if not names:
        raise ValueError("support model must have at least one feature name")
    if not all(isinstance(name, str) and name.strip() for name in names):
        raise ValueError("support model feature names must be non-blank strings")
    if len(set(names)) != len(names):
        raise ValueError("support model feature names must be unique")
    if len(names) != size:
        raise ValueError("invalid support artifact dimensions")


def _validated_precision(precision: Array, mean: Array) -> Array:
    """Return an exactly symmetric PSD precision matrix, or raise ``ValueError``."""

    if mean.ndim != 1 or mean.size == 0 or precision.shape != (mean.size, mean.size):
        raise ValueError("invalid support artifact dimensions")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(precision)):
        raise ValueError("support artifact contains non-finite values")
    scale = float(np.max(np.abs(precision)))
    if scale == 0.0:
        raise ValueError(
            "precision matrix is all zeros, so every query would receive maximum support"
        )
    if float(np.max(np.abs(precision - precision.T))) > _STATE_RTOL * scale:
        raise ValueError("precision matrix is not symmetric")
    symmetric = 0.5 * (precision + precision.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if float(eigenvalues[0]) < -_STATE_RTOL * float(eigenvalues[-1]) or eigenvalues[-1] <= 0.0:
        raise ValueError("precision matrix is not positive semidefinite")
    if eigenvalues[0] < 0.0:
        clipped = np.clip(eigenvalues, 0.0, None)
        symmetric = (eigenvectors * clipped) @ eigenvectors.T
        symmetric = 0.5 * (symmetric + symmetric.T)
    return np.asarray(symmetric, dtype=np.float64)


@dataclass(slots=True)
class GaussianSupportModel:
    """Transparent regularized Mahalanobis support baseline.

    The score is exp(-0.5 * d^2 / temperature), where d is a regularized Mahalanobis
    distance. The score is a ranking/support heuristic, not a probability or formal coverage
    guarantee. Thresholds should be selected on a calibration partition, never on final test.
    """

    regularization: float = 1e-6
    temperature: float = 1.0
    _mean: Array | None = field(default=None, init=False, repr=False)
    _precision: Array | None = field(default=None, init=False, repr=False)
    _names: tuple[str, ...] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.regularization = _positive_finite("regularization", self.regularization)
        self.temperature = _positive_finite("temperature", self.temperature)

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
        precision = _validated_precision(np.linalg.pinv(cov), mean)
        self._mean = mean
        self._precision = precision
        self._names = names
        return self

    def squared_distance(self, features: FeatureVector) -> float:
        self._ensure_fitted(features)
        assert self._mean is not None
        assert self._precision is not None
        delta = features.values - self._mean
        distance = float(delta @ self._precision @ delta)
        magnitude = float(np.abs(delta) @ np.abs(self._precision) @ np.abs(delta))
        if math.isnan(distance) or distance < -_STATE_RTOL * magnitude:
            # A valid PSD state cannot produce this. Raising lets the router fail closed instead
            # of clamping a corrupted negative distance to maximum support.
            raise ValueError("support model state is not positive semidefinite")
        return max(distance, 0.0)

    def support_score(self, features: FeatureVector) -> float:
        d2 = self.squared_distance(features)
        return float(np.exp(-0.5 * d2 / self.temperature))

    def save(self, path: str | Path) -> None:
        """Serialize fitted state as a versioned, pickle-free NPZ artifact."""

        self._ensure_fitted_schema()
        assert self._mean is not None
        assert self._precision is not None
        assert self._names is not None
        _validated_precision(self._precision, self._mean)
        _validate_names(self._names, self._mean.size)
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
            try:
                metadata = json.loads(str(artifact["metadata"].item()))
                raw_mean = artifact["mean"]
                raw_precision = artifact["precision"]
            except KeyError as exc:
                raise ValueError(f"support artifact is missing {exc}") from None
        if not isinstance(metadata, dict):
            raise ValueError("support artifact metadata must be a JSON object")
        if metadata.get("artifact_version") != _SUPPORT_ARTIFACT_VERSION:
            raise ValueError("unsupported support artifact version")
        if metadata.get("model_type") != cls.__name__:
            raise ValueError("support artifact model type mismatch")
        for key in ("regularization", "temperature", "feature_names"):
            if key not in metadata:
                raise ValueError(f"support artifact metadata is missing {key!r}")
        raw_names = metadata["feature_names"]
        if not isinstance(raw_names, list):
            raise ValueError("support artifact feature_names must be a list")
        names = tuple(raw_names)
        model = cls(
            regularization=metadata["regularization"],
            temperature=metadata["temperature"],
        )
        mean = np.asarray(raw_mean, dtype=np.float64)
        precision = np.asarray(raw_precision, dtype=np.float64)
        _validate_names(names, mean.size)
        model._precision = _validated_precision(precision, mean)
        model._mean = mean
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
