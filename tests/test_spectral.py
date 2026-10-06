import numpy as np
import pytest

from topocal.features.spectral import rayleigh_quotient


def test_rayleigh_quotient_for_eigenvector() -> None:
    operator = np.diag([2.0, 5.0])
    vector = np.array([0.0, 3.0])
    assert rayleigh_quotient(vector, operator) == pytest.approx(5.0)


def test_rayleigh_rejects_zero_vector() -> None:
    with pytest.raises(ValueError):
        rayleigh_quotient(np.zeros(2), np.eye(2))


def test_rayleigh_rejects_bad_shapes_and_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        rayleigh_quotient(np.zeros((2, 1)), np.eye(2))
    with pytest.raises(ValueError, match="square"):
        rayleigh_quotient(np.ones(2), np.ones((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        rayleigh_quotient(np.array([1.0, np.nan]), np.eye(2))
