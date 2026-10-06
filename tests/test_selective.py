import numpy as np
import pytest

from topocal.evaluation.selective import max_coverage_at_risk, risk_coverage_curve


def test_risk_coverage_accepts_low_risk_first() -> None:
    risks = np.array([0.9, 0.1, 0.2, 0.8])
    failures = np.array([1, 0, 0, 1])
    curve = risk_coverage_curve(risks, failures)
    assert curve[1].coverage == pytest.approx(0.5)
    assert curve[1].selective_risk == pytest.approx(0.0)
    assert curve[-1].selective_risk == pytest.approx(0.5)


def test_risk_coverage_never_splits_equal_risk_ties() -> None:
    risks = np.array([0.1, 0.1])
    failures = np.array([0, 1])
    curve = risk_coverage_curve(risks, failures)
    assert len(curve) == 1
    assert curve[0].threshold == pytest.approx(0.1)
    assert curve[0].coverage == pytest.approx(1.0)
    assert curve[0].selective_risk == pytest.approx(0.5)


def test_max_coverage_at_target_risk() -> None:
    risks = np.array([0.1, 0.2, 0.8, 0.9])
    failures = np.array([0, 0, 1, 1])
    point = max_coverage_at_risk(risks, failures, target_risk=0.0)
    assert point is not None
    assert point.coverage == pytest.approx(0.5)


def test_max_coverage_returns_none_when_no_threshold_is_safe() -> None:
    assert max_coverage_at_risk([0.1, 0.2], [1, 1], target_risk=0.0) is None


@pytest.mark.parametrize(
    ("risks", "failures"),
    [([], []), ([np.nan], [0]), ([0.1], [2]), ([[0.1]], [0])],
)
def test_risk_coverage_rejects_invalid_inputs(risks: object, failures: object) -> None:
    with pytest.raises(ValueError):
        risk_coverage_curve(risks, failures)


def test_max_coverage_rejects_invalid_target() -> None:
    with pytest.raises(ValueError, match="target_risk"):
        max_coverage_at_risk([0.1], [0], target_risk=1.1)
