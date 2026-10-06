"""Synthetic vertical slice used by CLI and smoke tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from topocal.calibration.empirical import EmpiricalRiskCalibrator
from topocal.features.support import GaussianSupportModel, support_threshold_from_calibration
from topocal.routing.router import TopoCalRouter
from topocal.types import FeatureVector, PredictionBundle, Problem

FEATURE_NAMES = ("rayleigh", "spectral_distance", "pde_residual", "model_uncertainty")


@dataclass(slots=True)
class SyntheticSurrogate:
    def predict(self, problem: Problem) -> PredictionBundle:
        x = np.asarray(problem.payload, dtype=np.float64)
        # Payload doubles as observable features for this smoke/demo path.
        return PredictionBundle(
            solution=np.array([x.sum()], dtype=np.float64),
            features=FeatureVector(values=x, names=FEATURE_NAMES),
        )


@dataclass(slots=True)
class SyntheticFallback:
    def solve(self, problem: Problem) -> np.ndarray:
        x = np.asarray(problem.payload, dtype=np.float64)
        return np.array([x.sum() + 1000.0], dtype=np.float64)


def build_demo_router(seed: int = 7) -> TopoCalRouter:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(600, len(FEATURE_NAMES)))
    # Synthetic failure mechanism: high spectral distance/residual/uncertainty => failure.
    logits = 1.2 * x[:, 1] + 1.8 * x[:, 2] + 1.3 * x[:, 3] - 2.0
    probability = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(x.shape[0]) < probability).astype(np.float64)
    vectors = [FeatureVector(row, FEATURE_NAMES) for row in x]
    support = GaussianSupportModel(temperature=4.0).fit(vectors)
    risk = EmpiricalRiskCalibrator(neighbors=48).fit(vectors, y)
    support_scores = np.array([support.support_score(vector) for vector in vectors])
    min_support = support_threshold_from_calibration(support_scores, lower_quantile=0.01)
    return TopoCalRouter(
        surrogate=SyntheticSurrogate(),
        fallback=SyntheticFallback(),
        risk_model=risk,
        support_model=support,
        max_failure_risk=0.20,
        min_support_score=min_support,
    )


def run_demo() -> list[dict[str, object]]:
    router = build_demo_router()
    cases = {
        "in_support_low_risk": np.array([0.1, -0.8, -1.0, -0.7]),
        "in_support_high_risk": np.array([0.2, 0.9, 1.3, 1.0]),
        "far_out_of_support": np.array([8.0, 8.0, 8.0, 8.0]),
    }
    output: list[dict[str, object]] = []
    for name, payload in cases.items():
        result = router.solve(Problem(problem_id=name, payload=payload))
        output.append(
            {
                "problem_id": result.problem_id,
                "decision": result.decision.value,
                "risk": None if result.risk is None else round(result.risk, 4),
                "risk_status": result.risk_status.value,
                "support_score": (
                    None if result.support_score is None else round(result.support_score, 4)
                ),
                "support_status": result.support_status.value,
                "reason": result.reason,
                "solution": result.solution.tolist(),
            }
        )
    return output
