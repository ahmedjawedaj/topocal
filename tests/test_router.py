from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from topocal.demo import build_demo_router
from topocal.features.support import GaussianSupportModel
from topocal.routing.router import TopoCalRouter
from topocal.types import (
    Decision,
    FeatureVector,
    PredictionBundle,
    Problem,
    RiskStatus,
    SupportStatus,
)

NAMES = ("signal",)


@dataclass
class FixedSurrogate:
    solution: float = 1.0
    with_features: bool = True

    def predict(self, problem: Problem) -> PredictionBundle:
        features = FeatureVector(np.array([0.0]), NAMES) if self.with_features else None
        return PredictionBundle(np.array([self.solution]), features)


@dataclass
class FixedFallback:
    value: float = 99.0

    def solve(self, problem: Problem) -> np.ndarray:
        return np.array([self.value])


@dataclass
class FixedSupport:
    value: float

    def support_score(self, features: FeatureVector) -> float:
        return self.value


@dataclass
class FixedRisk:
    value: float

    def predict_failure_probability(self, features: FeatureVector) -> float:
        return self.value


@dataclass
class ExternalExtractor:
    def extract(self, problem: Problem, prediction: PredictionBundle) -> FeatureVector:
        return FeatureVector(np.array([0.25]), NAMES)


def router(*, support: float = 0.9, risk: float = 0.01, **kwargs: object) -> TopoCalRouter:
    return TopoCalRouter(
        surrogate=FixedSurrogate(),
        fallback=FixedFallback(),
        risk_model=FixedRisk(risk),
        support_model=FixedSupport(support),
        max_failure_risk=0.05,
        min_support_score=0.10,
        **kwargs,
    )


def test_router_accepts_low_risk_case() -> None:
    result = build_demo_router().solve(Problem("safe", np.array([0.1, -0.8, -1.0, -0.7])))
    assert result.decision == Decision.ACCEPT_NEURAL
    assert result.risk_status == RiskStatus.CALIBRATED
    assert result.support_status == SupportStatus.SUPPORTED


def test_router_falls_back_outside_support_with_unknown_risk() -> None:
    result = build_demo_router().solve(Problem("ood", np.array([8.0, 8.0, 8.0, 8.0])))
    assert result.decision == Decision.FALLBACK_SOLVER
    assert result.risk is None
    assert result.risk_status == RiskStatus.UNSUPPORTED
    assert result.support_status == SupportStatus.OUTSIDE_SUPPORT
    assert "unknown" in result.reason


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.01, 1.01])
def test_invalid_support_fails_closed(bad: float) -> None:
    result = router(support=bad).solve(Problem("x", None))
    assert result.decision == Decision.FALLBACK_SOLVER
    assert result.risk is None
    assert result.risk_status == RiskStatus.UNAVAILABLE
    assert result.support_score is None
    assert result.support_status == SupportStatus.INVALID


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.01, 1.01])
def test_invalid_risk_fails_closed(bad: float) -> None:
    result = router(risk=bad).solve(Problem("x", None))
    assert result.decision == Decision.FALLBACK_SOLVER
    assert result.risk is None
    assert result.risk_status == RiskStatus.INVALID
    assert result.support_status == SupportStatus.SUPPORTED


def test_external_feature_extractor_supports_featureless_surrogate() -> None:
    instance = router(feature_extractor=ExternalExtractor())
    instance.surrogate = FixedSurrogate(with_features=False)
    result = instance.solve(Problem("x", None))
    assert result.decision == Decision.ACCEPT_NEURAL


def test_missing_features_fail_closed() -> None:
    instance = router()
    instance.surrogate = FixedSurrogate(with_features=False)
    result = instance.solve(Problem("x", None))
    assert result.decision == Decision.FALLBACK_SOLVER
    assert result.risk_status == RiskStatus.UNAVAILABLE


def test_router_rejects_invalid_policy_threshold() -> None:
    with pytest.raises(ValueError, match="max_failure_risk"):
        TopoCalRouter(
            surrogate=FixedSurrogate(),
            fallback=FixedFallback(),
            risk_model=FixedRisk(0.1),
            support_model=FixedSupport(0.5),
            max_failure_risk=1.1,
        )

@dataclass
class RaisingSupport:
    def support_score(self, features: FeatureVector) -> float:
        raise RuntimeError("boom")


@dataclass
class RaisingRisk:
    def predict_failure_probability(self, features: FeatureVector) -> float:
        raise RuntimeError("boom")


def test_support_exception_fails_closed() -> None:
    instance = router()
    instance.support_model = RaisingSupport()
    result = instance.solve(Problem("x", None))
    assert result.decision == Decision.FALLBACK_SOLVER
    assert result.support_status == SupportStatus.UNAVAILABLE


def test_risk_exception_fails_closed() -> None:
    instance = router()
    instance.risk_model = RaisingRisk()
    result = instance.solve(Problem("x", None))
    assert result.decision == Decision.FALLBACK_SOLVER
    assert result.risk_status == RiskStatus.UNAVAILABLE


def test_high_calibrated_risk_routes_to_fallback() -> None:
    result = router(risk=0.9).solve(Problem("x", None))
    assert result.decision == Decision.FALLBACK_SOLVER
    assert result.risk == pytest.approx(0.9)
    assert result.risk_status == RiskStatus.CALIBRATED


def test_invalid_fallback_solution_raises() -> None:
    instance = router(support=0.0)
    instance.fallback = FixedFallback(np.nan)
    with pytest.raises(RuntimeError, match="fallback"):
        instance.solve(Problem("x", None))


@dataclass
class FarSurrogate:
    """Emits features far from the support model mean, so distance is strictly positive."""

    def predict(self, problem: Problem) -> PredictionBundle:
        return PredictionBundle(np.array([1.0]), FeatureVector(np.array([500.0]), NAMES))


def real_support() -> GaussianSupportModel:
    vectors = [FeatureVector(np.array([x]), NAMES) for x in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    return GaussianSupportModel().fit(vectors)


def test_router_falls_back_when_real_support_state_is_corrupted() -> None:
    """Regression: a negative precision used to yield support 1.0 and accept the neural output."""

    support = real_support()
    support._precision = -np.eye(1)
    instance = router()
    instance.support_model = support
    instance.surrogate = FarSurrogate()
    result = instance.solve(Problem("corrupt", None))
    assert result.decision == Decision.FALLBACK_SOLVER
    assert result.support_status == SupportStatus.UNAVAILABLE


def test_router_falls_back_for_unfitted_real_support_model() -> None:
    instance = router()
    instance.support_model = GaussianSupportModel()
    result = instance.solve(Problem("unfitted", None))
    assert result.decision == Decision.FALLBACK_SOLVER
    assert result.support_status == SupportStatus.UNAVAILABLE


def test_router_real_support_outside_support_leaves_risk_unknown() -> None:
    instance = router()
    instance.support_model = real_support()
    instance.surrogate = FarSurrogate()
    result = instance.solve(Problem("far", None))
    assert result.decision == Decision.FALLBACK_SOLVER
    assert result.support_status == SupportStatus.OUTSIDE_SUPPORT
    assert result.risk is None
