"""Shared data structures and dependency-inversion protocols."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Problem:
    """A model-agnostic PDE query.

    `payload` can be a mesh, coefficient field, graph, tensor bundle, or adapter object.
    The core package deliberately does not interpret it.
    """

    problem_id: str
    payload: object


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """Features used by support and risk models.

    The initial implementation uses a dense numerical vector. Feature names are retained
    to keep experiment artifacts auditable and to prevent silent column-order mistakes.
    """

    values: Array
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("FeatureVector.values must be one-dimensional")
        if len(self.names) != values.shape[0]:
            raise ValueError("Feature names and values must have the same length")
        if len(set(self.names)) != len(self.names):
            raise ValueError("Feature names must be unique")
        if not all(name.strip() for name in self.names):
            raise ValueError("Feature names cannot be blank")
        if not np.all(np.isfinite(values)):
            raise ValueError("FeatureVector values must be finite")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class PredictionBundle:
    """Neural prediction and optional reliability features.

    `features` remains optional so production adapters can keep model inference separate from
    topology/spectral/physics feature extraction. Legacy adapters may still return features
    directly while the research code transitions to an explicit `FeatureExtractor`.
    """

    solution: Array
    features: FeatureVector | None = None

    def __post_init__(self) -> None:
        solution = np.asarray(self.solution, dtype=np.float64)
        if solution.size == 0:
            raise ValueError("PredictionBundle.solution cannot be empty")
        if not np.all(np.isfinite(solution)):
            raise ValueError("PredictionBundle.solution must be finite")
        object.__setattr__(self, "solution", solution)


class Decision(StrEnum):
    ACCEPT_NEURAL = "accept_neural"
    FALLBACK_SOLVER = "fallback_solver"


class RiskStatus(StrEnum):
    """Interpretation of the returned risk value."""

    CALIBRATED = "calibrated"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"


class SupportStatus(StrEnum):
    """Interpretation of the returned support score."""

    SUPPORTED = "supported"
    OUTSIDE_SUPPORT = "outside_support"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class RoutingResult:
    """End-to-end decision returned to a caller.

    `risk=None` is intentional whenever calibrated risk is unknown or invalid. Unsupported
    queries must never be represented as a fictitious 100% failure probability.
    """

    problem_id: str
    solution: Array
    decision: Decision
    risk: float | None
    risk_status: RiskStatus
    support_score: float | None
    support_status: SupportStatus
    reason: str


@runtime_checkable
class Surrogate(Protocol):
    """Contract for a neural/operator surrogate adapter."""

    def predict(self, problem: Problem) -> PredictionBundle: ...


@runtime_checkable
class FeatureExtractor(Protocol):
    """Contract for model-independent reliability feature extraction."""

    def extract(self, problem: Problem, prediction: PredictionBundle) -> FeatureVector: ...


@runtime_checkable
class FallbackSolver(Protocol):
    """Contract for a trusted numerical solver adapter."""

    def solve(self, problem: Problem) -> Array: ...


@runtime_checkable
class RiskModel(Protocol):
    """Contract for calibrated failure-probability estimators."""

    def predict_failure_probability(self, features: FeatureVector) -> float: ...


@runtime_checkable
class SupportModel(Protocol):
    """Contract for calibration-support estimators."""

    def support_score(self, features: FeatureVector) -> float: ...
