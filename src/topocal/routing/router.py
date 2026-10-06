"""Selective accept/fallback policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from topocal.types import (
    Decision,
    FallbackSolver,
    FeatureExtractor,
    FeatureVector,
    PredictionBundle,
    Problem,
    RiskModel,
    RiskStatus,
    RoutingResult,
    SupportModel,
    SupportStatus,
    Surrogate,
)


@dataclass(slots=True)
class TopoCalRouter:
    """Route a PDE query based on calibrated risk and calibration support.

    Reliability infrastructure is deliberately fail-closed. Any unavailable, non-finite, or
    out-of-range support/risk output triggers the trusted fallback solver rather than silently
    accepting the neural prediction.
    """

    surrogate: Surrogate
    fallback: FallbackSolver
    risk_model: RiskModel
    support_model: SupportModel
    max_failure_risk: float = 0.05
    min_support_score: float = 0.05
    feature_extractor: FeatureExtractor | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_failure_risk <= 1.0:
            raise ValueError("max_failure_risk must be in [0, 1]")
        if not 0.0 <= self.min_support_score <= 1.0:
            raise ValueError("min_support_score must be in [0, 1]")

    def solve(self, problem: Problem) -> RoutingResult:
        try:
            prediction = self.surrogate.predict(problem)
            features = self._extract_features(problem, prediction)
        except Exception as exc:  # reliability boundary: surrogate/feature failures fail closed
            return self._fallback_result(
                problem,
                risk=None,
                risk_status=RiskStatus.UNAVAILABLE,
                support_score=None,
                support_status=SupportStatus.UNAVAILABLE,
                reason=f"surrogate or feature extraction unavailable: {type(exc).__name__}",
            )

        try:
            support = float(self.support_model.support_score(features))
        except Exception as exc:
            return self._fallback_result(
                problem,
                risk=None,
                risk_status=RiskStatus.UNAVAILABLE,
                support_score=None,
                support_status=SupportStatus.UNAVAILABLE,
                reason=f"support assessment unavailable: {type(exc).__name__}",
            )

        if not _valid_probability(support):
            return self._fallback_result(
                problem,
                risk=None,
                risk_status=RiskStatus.UNAVAILABLE,
                support_score=None,
                support_status=SupportStatus.INVALID,
                reason="support assessment returned an invalid value",
            )

        if support < self.min_support_score:
            return self._fallback_result(
                problem,
                risk=None,
                risk_status=RiskStatus.UNSUPPORTED,
                support_score=support,
                support_status=SupportStatus.OUTSIDE_SUPPORT,
                reason="outside calibrated feature support; calibrated risk is unknown",
            )

        try:
            risk = float(self.risk_model.predict_failure_probability(features))
        except Exception as exc:
            return self._fallback_result(
                problem,
                risk=None,
                risk_status=RiskStatus.UNAVAILABLE,
                support_score=support,
                support_status=SupportStatus.SUPPORTED,
                reason=f"risk assessment unavailable: {type(exc).__name__}",
            )

        if not _valid_probability(risk):
            return self._fallback_result(
                problem,
                risk=None,
                risk_status=RiskStatus.INVALID,
                support_score=support,
                support_status=SupportStatus.SUPPORTED,
                reason="risk assessment returned an invalid value",
            )

        if risk > self.max_failure_risk:
            return self._fallback_result(
                problem,
                risk=risk,
                risk_status=RiskStatus.CALIBRATED,
                support_score=support,
                support_status=SupportStatus.SUPPORTED,
                reason="calibrated failure risk exceeds policy threshold",
            )

        return RoutingResult(
            problem_id=problem.problem_id,
            solution=prediction.solution,
            decision=Decision.ACCEPT_NEURAL,
            risk=risk,
            risk_status=RiskStatus.CALIBRATED,
            support_score=support,
            support_status=SupportStatus.SUPPORTED,
            reason="inside calibrated support and below failure-risk threshold",
        )

    def _extract_features(self, problem: Problem, prediction: PredictionBundle) -> FeatureVector:
        if self.feature_extractor is not None:
            return self.feature_extractor.extract(problem, prediction)
        if prediction.features is None:
            raise ValueError("no reliability features available")
        return prediction.features

    def _fallback_result(
        self,
        problem: Problem,
        *,
        risk: float | None,
        risk_status: RiskStatus,
        support_score: float | None,
        support_status: SupportStatus,
        reason: str,
    ) -> RoutingResult:
        solution = np.asarray(self.fallback.solve(problem), dtype=np.float64)
        if solution.size == 0 or not np.all(np.isfinite(solution)):
            raise RuntimeError("fallback solver returned an invalid solution")
        return RoutingResult(
            problem_id=problem.problem_id,
            solution=solution,
            decision=Decision.FALLBACK_SOLVER,
            risk=risk,
            risk_status=risk_status,
            support_score=support_score,
            support_status=support_status,
            reason=reason,
        )


def _valid_probability(value: float) -> bool:
    return bool(np.isfinite(value) and 0.0 <= value <= 1.0)
