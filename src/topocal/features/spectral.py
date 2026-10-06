"""Small, dependency-light spectral primitives.

Research adapters can later expose mesh/Hodge Laplacian construction. The core accepts an
already constructed operator so it remains independent of GUDHI/SciPy/FEniCSx.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def rayleigh_quotient(vector: Array, operator: Array, *, eps: float = 1e-12) -> float:
    """Return v^T A v / v^T v for a dense real-valued operator.

    Raises:
        ValueError: for incompatible shapes, non-finite inputs, or near-zero vectors.
    """

    v = np.asarray(vector, dtype=np.float64)
    a = np.asarray(operator, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError("vector must be one-dimensional")
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] != v.shape[0]:
        raise ValueError("operator must be square and compatible with vector")
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(a)):
        raise ValueError("inputs must be finite")
    denominator = float(v @ v)
    if denominator <= eps:
        raise ValueError("Rayleigh quotient is undefined for a near-zero vector")
    return float((v @ a @ v) / denominator)
