"""Selective-prediction metrics used by TopoCal experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class RiskCoveragePoint:
    coverage: float
    selective_risk: float
    threshold: float
    accepted: int


def risk_coverage_curve(
    risks: npt.ArrayLike,
    failures: npt.ArrayLike,
) -> list[RiskCoveragePoint]:
    """Compute a threshold-realizable empirical risk-coverage curve.

    Every returned point corresponds exactly to the deterministic rule `accept risk <= threshold`.
    Equal risk scores are therefore accepted as a group. This avoids impossible intermediate
    operating points that arise when ties are broken sample-by-sample.

    `risks` may be calibrated probabilities or any monotonic risk score.
    `failures` are binary ground-truth unacceptable-error indicators.
    """

    r, y = _validate_inputs(risks, failures)
    order = np.argsort(r, kind="stable")
    sorted_risk = r[order]
    sorted_failure = y[order]
    cumulative_failures = np.cumsum(sorted_failure)
    n = r.size

    # A deterministic threshold cannot split a group of equal scores, so only emit group ends.
    group_ends = np.flatnonzero(np.r_[sorted_risk[1:] != sorted_risk[:-1], True])
    return [
        RiskCoveragePoint(
            coverage=float((idx + 1) / n),
            selective_risk=float(cumulative_failures[idx] / (idx + 1)),
            threshold=float(sorted_risk[idx]),
            accepted=int(idx + 1),
        )
        for idx in group_ends
    ]


def max_coverage_at_risk(
    risks: npt.ArrayLike,
    failures: npt.ArrayLike,
    *,
    target_risk: float,
) -> RiskCoveragePoint | None:
    """Return the highest realizable coverage whose accepted failure rate <= target."""

    if not 0.0 <= target_risk <= 1.0:
        raise ValueError("target_risk must be in [0, 1]")
    eligible = [
        point
        for point in risk_coverage_curve(risks, failures)
        if point.selective_risk <= target_risk
    ]
    return eligible[-1] if eligible else None


def _validate_inputs(
    risks: npt.ArrayLike,
    failures: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r = np.asarray(risks, dtype=np.float64)
    y = np.asarray(failures, dtype=np.float64)
    if r.ndim != 1 or y.ndim != 1 or r.shape != y.shape:
        raise ValueError("risks and failures must be one-dimensional arrays of equal length")
    if r.size == 0:
        raise ValueError("at least one sample is required")
    if not np.all(np.isfinite(r)):
        raise ValueError("risk values must be finite")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("failures must be binary")
    return r, y
