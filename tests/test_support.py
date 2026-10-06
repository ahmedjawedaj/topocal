import json
from pathlib import Path

import numpy as np
import pytest

from topocal.features.support import GaussianSupportModel, support_threshold_from_calibration
from topocal.types import FeatureVector

NAMES = ("a", "b")


def fv(a: float, b: float) -> FeatureVector:
    return FeatureVector(np.array([a, b]), NAMES)


def fitted_model() -> GaussianSupportModel:
    return GaussianSupportModel(temperature=2.0).fit(
        [fv(-1, 0), fv(0, -1), fv(1, 0), fv(0, 1), fv(0, 0)]
    )


def test_support_is_higher_near_calibration_cloud() -> None:
    model = fitted_model()
    assert model.support_score(fv(0.1, 0.1)) > model.support_score(fv(10, 10))


def test_support_artifact_roundtrip(tmp_path: Path) -> None:
    model = fitted_model()
    path = tmp_path / "support.npz"
    model.save(path)
    restored = GaussianSupportModel.load(path)
    assert restored.support_score(fv(0.5, 0.5)) == pytest.approx(model.support_score(fv(0.5, 0.5)))


def test_support_threshold_comes_from_calibration_scores() -> None:
    threshold = support_threshold_from_calibration([0.1, 0.2, 0.4, 0.8], lower_quantile=0.25)
    assert threshold == pytest.approx(0.175)


@pytest.mark.parametrize("scores", [[], [np.nan], [-0.1], [1.1]])
def test_support_threshold_rejects_invalid_scores(scores: list[float]) -> None:
    with pytest.raises(ValueError):
        support_threshold_from_calibration(scores)


def test_support_requires_fitted_model() -> None:
    with pytest.raises(RuntimeError, match="not been fitted"):
        GaussianSupportModel().support_score(fv(0, 0))


def test_support_configuration_and_fit_validation() -> None:
    with pytest.raises(ValueError, match="regularization"):
        GaussianSupportModel(regularization=0)
    with pytest.raises(ValueError, match="temperature"):
        GaussianSupportModel(temperature=0)
    with pytest.raises(ValueError, match="at least two"):
        GaussianSupportModel().fit([fv(0, 0)])
    with pytest.raises(ValueError, match="same names"):
        GaussianSupportModel().fit(
            [fv(0, 0), FeatureVector(np.array([1.0, 1.0]), ("x", "y"))]
        )


def test_support_rejects_schema_mismatch() -> None:
    model = fitted_model()
    with pytest.raises(ValueError, match="schema"):
        model.support_score(FeatureVector(np.array([0.0, 0.0]), ("x", "y")))


def test_support_threshold_rejects_invalid_quantile() -> None:
    with pytest.raises(ValueError, match="lower_quantile"):
        support_threshold_from_calibration([0.2, 0.4], lower_quantile=1.2)


# ---------------------------------------------------------------------------
# Fitted-state and artifact validation (Task C)
# ---------------------------------------------------------------------------


def write_artifact(
    path: Path,
    *,
    mean: object = (0.0,),
    precision: object = ((1.0,),),
    names: object = ("x",),
    regularization: object = 1e-6,
    temperature: object = 1.0,
    **metadata_overrides: object,
) -> Path:
    metadata: dict[str, object] = {
        "artifact_version": 1,
        "model_type": "GaussianSupportModel",
        "regularization": regularization,
        "temperature": temperature,
        "feature_names": names,
    }
    metadata.update(metadata_overrides)
    np.savez_compressed(
        path,
        mean=np.asarray(mean, dtype=np.float64),
        precision=np.asarray(precision, dtype=np.float64),
        metadata=json.dumps(metadata),
    )
    return path


def diag_artifact(path: Path, *eigenvalues: float) -> Path:
    n = len(eigenvalues)
    return write_artifact(
        path,
        mean=np.zeros(n),
        precision=np.diag(eigenvalues),
        names=tuple(f"f{i}" for i in range(n)),
    )


def test_valid_artifact_helper_loads(tmp_path: Path) -> None:
    model = GaussianSupportModel.load(write_artifact(tmp_path / "ok.npz"))
    assert model.support_score(FeatureVector(np.array([0.0]), ("x",))) == pytest.approx(1.0)


def test_negative_precision_artifact_is_rejected(tmp_path: Path) -> None:
    """Regression: a negative precision used to clamp distance to 0 and give support 1.0."""

    path = write_artifact(tmp_path / "negative.npz", precision=((-1.0,),))
    with pytest.raises(ValueError, match="positive semidefinite"):
        GaussianSupportModel.load(path)


@pytest.mark.parametrize(
    "precision",
    [
        ((1.0, 0.0), (0.0, -1.0)),
        ((1.0, 2.0), (2.0, 1.0)),
    ],
)
def test_indefinite_precision_artifact_is_rejected(
    tmp_path: Path, precision: tuple[tuple[float, float], ...]
) -> None:
    path = write_artifact(
        tmp_path / "indefinite.npz", mean=(0.0, 0.0), precision=precision, names=("a", "b")
    )
    with pytest.raises(ValueError, match="positive semidefinite"):
        GaussianSupportModel.load(path)


def test_materially_asymmetric_precision_artifact_is_rejected(tmp_path: Path) -> None:
    path = write_artifact(
        tmp_path / "asym.npz",
        mean=(0.0, 0.0),
        precision=((2.0, 0.5), (0.0, 2.0)),
        names=("a", "b"),
    )
    with pytest.raises(ValueError, match="symmetric"):
        GaussianSupportModel.load(path)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_state_is_rejected(tmp_path: Path, bad: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        GaussianSupportModel.load(write_artifact(tmp_path / "mean.npz", mean=(bad,)))
    with pytest.raises(ValueError, match="non-finite"):
        GaussianSupportModel.load(write_artifact(tmp_path / "prec.npz", precision=((bad,),)))


def test_all_zero_precision_is_rejected(tmp_path: Path) -> None:
    path = write_artifact(tmp_path / "zero.npz", precision=((0.0,),))
    with pytest.raises(ValueError, match="all zeros"):
        GaussianSupportModel.load(path)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 0.0, -1.0, "abc", None])
@pytest.mark.parametrize("field_name", ["regularization", "temperature"])
def test_invalid_hyperparameters_rejected_by_constructor(field_name: str, bad: object) -> None:
    with pytest.raises(ValueError, match=field_name):
        GaussianSupportModel(**{field_name: bad})  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0, "abc", None])
@pytest.mark.parametrize("field_name", ["regularization", "temperature"])
def test_invalid_hyperparameters_rejected_on_load(
    tmp_path: Path, field_name: str, bad: object
) -> None:
    path = write_artifact(tmp_path / "hyper.npz", **{field_name: bad})
    with pytest.raises(ValueError, match=field_name):
        GaussianSupportModel.load(path)


@pytest.mark.parametrize(
    ("names", "mean", "precision", "match"),
    [
        ([], (0.0,), ((1.0,),), "at least one"),
        (["x", "x"], (0.0, 0.0), ((1.0, 0.0), (0.0, 1.0)), "unique"),
        (["x", "  "], (0.0, 0.0), ((1.0, 0.0), (0.0, 1.0)), "non-blank"),
        (["x", ""], (0.0, 0.0), ((1.0, 0.0), (0.0, 1.0)), "non-blank"),
        (["x", 3], (0.0, 0.0), ((1.0, 0.0), (0.0, 1.0)), "non-blank"),
        (["x"], (0.0, 0.0), ((1.0, 0.0), (0.0, 1.0)), "dimensions"),
        (["x", "y"], (0.0,), ((1.0,),), "dimensions"),
        (["x"], (0.0,), ((1.0, 0.0), (0.0, 1.0)), "dimensions"),
    ],
)
def test_invalid_schema_is_rejected(
    tmp_path: Path, names: list[object], mean: object, precision: object, match: str
) -> None:
    path = write_artifact(tmp_path / "schema.npz", mean=mean, precision=precision, names=names)
    with pytest.raises(ValueError, match=match):
        GaussianSupportModel.load(path)


def test_feature_names_must_be_a_list(tmp_path: Path) -> None:
    path = write_artifact(tmp_path / "names.npz", names="x")
    with pytest.raises(ValueError, match="list"):
        GaussianSupportModel.load(path)


def test_missing_keys_and_bad_versions_raise_value_error(tmp_path: Path) -> None:
    np.savez_compressed(tmp_path / "nometa.npz", mean=np.zeros(1), precision=np.eye(1))
    with pytest.raises(ValueError, match="missing"):
        GaussianSupportModel.load(tmp_path / "nometa.npz")
    with pytest.raises(ValueError, match="version"):
        GaussianSupportModel.load(write_artifact(tmp_path / "v.npz", artifact_version=2))
    with pytest.raises(ValueError, match="model type"):
        GaussianSupportModel.load(write_artifact(tmp_path / "t.npz", model_type="Other"))
    path = write_artifact(tmp_path / "k.npz")
    with np.load(path) as artifact:
        metadata = json.loads(str(artifact["metadata"].item()))
    del metadata["temperature"]
    np.savez_compressed(
        tmp_path / "k2.npz", mean=np.zeros(1), precision=np.eye(1), metadata=json.dumps(metadata)
    )
    with pytest.raises(ValueError, match="temperature"):
        GaussianSupportModel.load(tmp_path / "k2.npz")


def test_trained_model_roundtrip_is_exact(tmp_path: Path) -> None:
    model = fitted_model()
    model.save(tmp_path / "m.npz")
    restored = GaussianSupportModel.load(tmp_path / "m.npz")
    assert restored.regularization == model.regularization
    assert restored.temperature == model.temperature
    for query in (fv(0, 0), fv(0.3, -0.7), fv(5, 5)):
        assert restored.support_score(query) == pytest.approx(model.support_score(query), abs=0)


def test_legitimate_singular_psd_is_preserved(tmp_path: Path) -> None:
    """A rank-deficient precision is valid. Its null direction is simply unconstrained."""

    path = write_artifact(
        tmp_path / "singular.npz",
        mean=(0.0, 0.0),
        precision=((1.0, 0.0), (0.0, 0.0)),
        names=("a", "b"),
    )
    model = GaussianSupportModel.load(path)
    free = model.support_score(fv(0.0, 1000.0))
    constrained = model.support_score(fv(3.0, 0.0))
    assert free == pytest.approx(1.0)
    assert constrained == pytest.approx(np.exp(-4.5))


def test_fit_on_degenerate_features_yields_valid_support() -> None:
    constant_second_feature = [fv(float(i), 1.0) for i in range(6)]
    model = GaussianSupportModel(regularization=1e-12).fit(constant_second_feature)
    assert 0.0 <= model.support_score(fv(2.5, 1.0)) <= 1.0
    assert model.support_score(fv(2.5, 1.0)) > model.support_score(fv(100.0, 1.0))


@pytest.mark.parametrize(
    ("eigenvalues", "accepted"),
    [
        ((1.0, -5e-9), True),
        ((1.0, -2e-8), False),
        ((1e6, -5e-3), True),
        ((1e6, -2e-2), False),
        ((1e-6, -5e-15), True),
        ((1e-6, -2e-14), False),
    ],
)
def test_psd_tolerance_is_scale_aware(
    tmp_path: Path, eigenvalues: tuple[float, float], accepted: bool
) -> None:
    path = diag_artifact(tmp_path / "tol.npz", *eigenvalues)
    if accepted:
        model = GaussianSupportModel.load(path)
        assert model._precision is not None
        assert np.linalg.eigvalsh(model._precision)[0] >= 0.0
    else:
        with pytest.raises(ValueError, match="positive semidefinite"):
            GaussianSupportModel.load(path)


@pytest.mark.parametrize(("off", "accepted"), [(5e-9, True), (2e-8, False)])
def test_asymmetry_tolerance_is_scale_aware(tmp_path: Path, off: float, accepted: bool) -> None:
    path = write_artifact(
        tmp_path / "asym.npz",
        mean=(0.0, 0.0),
        precision=((1.0, 0.1 + off), (0.1, 1.0)),
        names=("a", "b"),
    )
    if accepted:
        model = GaussianSupportModel.load(path)
        assert model._precision is not None
        assert np.array_equal(model._precision, model._precision.T)
    else:
        with pytest.raises(ValueError, match="symmetric"):
            GaussianSupportModel.load(path)


def test_runtime_guard_rejects_corrupted_in_memory_state() -> None:
    model = fitted_model()
    model._precision = -np.eye(2)
    with pytest.raises(ValueError, match="positive semidefinite"):
        model.support_score(fv(1000.0, 1000.0))


def test_save_refuses_corrupted_in_memory_state(tmp_path: Path) -> None:
    model = fitted_model()
    model._precision = -np.eye(2)
    with pytest.raises(ValueError, match="positive semidefinite"):
        model.save(tmp_path / "bad.npz")
    assert not (tmp_path / "bad.npz").exists()


def test_private_state_is_not_constructor_argument() -> None:
    with pytest.raises(TypeError):
        GaussianSupportModel(_precision=-np.eye(1))  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Overflow robustness and mutable hyperparameters (review items R1 and R2)
# ---------------------------------------------------------------------------


def one_feature_model() -> GaussianSupportModel:
    vectors = [FeatureVector(np.array([x]), ("x",)) for x in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    return GaussianSupportModel().fit(vectors)


def xq(value: float) -> FeatureVector:
    return FeatureVector(np.array([value]), ("x",))


@pytest.mark.parametrize("query", [1e200, 1e154, 1e10])
@pytest.mark.parametrize("precision", [-1.0, -1e300])
def test_negative_precision_cannot_hide_behind_distance_overflow(
    query: float, precision: float
) -> None:
    """Regression: distance -inf and magnitude inf made `-inf < -inf` false, giving support 1."""

    model = one_feature_model()
    model._precision = np.array([[precision]])
    with pytest.raises(ValueError, match="positive semidefinite"):
        model.support_score(xq(query))


@pytest.mark.parametrize("query", [1e200, 1e154, 1e10, -1e308])
def test_valid_precision_with_extreme_query_never_gains_support(query: float) -> None:
    model = one_feature_model()
    score = model.support_score(xq(query))
    assert score == 0.0
    assert score < model.support_score(xq(0.5))


def test_extreme_query_difference_overflow_never_gains_support(tmp_path: Path) -> None:
    path = write_artifact(tmp_path / "far.npz", mean=(-1e308,), precision=((1.0,),))
    model = GaussianSupportModel.load(path)
    assert model.support_score(xq(1e308)) == 0.0


def test_huge_finite_precision_is_kept_finite_and_valid(tmp_path: Path) -> None:
    """Regression: 0.5 * (P + P.T) overflowed, so loading produced an infinite matrix."""

    path = write_artifact(tmp_path / "huge.npz", precision=((1e308,),))
    model = GaussianSupportModel.load(path)
    assert model._precision is not None
    assert np.all(np.isfinite(model._precision))
    assert model.support_score(xq(0.0)) == pytest.approx(1.0)
    assert model.support_score(xq(1.0)) == 0.0
    assert model.support_score(xq(1e10)) == 0.0


def test_huge_finite_matrix_with_offdiagonal_mass_is_kept_valid(tmp_path: Path) -> None:
    path = write_artifact(
        tmp_path / "huge2.npz",
        mean=(0.0, 0.0),
        precision=((1e308, 5e307), (5e307, 1e308)),
        names=("a", "b"),
    )
    model = GaussianSupportModel.load(path)
    assert model._precision is not None
    assert np.all(np.isfinite(model._precision))
    assert np.array_equal(model._precision, model._precision.T)
    assert np.linalg.eigvalsh(model._precision / 1e308)[0] >= 0.0
    assert model.support_score(fv(0.0, 0.0)) == pytest.approx(1.0)
    # BLAS implementations can produce +inf or NaN when overflowing intermediate
    # products cancel. Either zero support or explicit failure is conservative.
    try:
        score = model.support_score(fv(3.0, -4.0))
    except ValueError as exc:
        assert "indeterminate distance" in str(exc)
    else:
        assert score == 0.0


def test_huge_scale_tiny_negative_eigenvalue_is_projected(tmp_path: Path) -> None:
    path = write_artifact(
        tmp_path / "huge3.npz",
        mean=(0.0, 0.0),
        precision=((1e308, 0.0), (0.0, -1e299)),
        names=("a", "b"),
    )
    model = GaussianSupportModel.load(path)
    assert model._precision is not None
    assert np.all(np.isfinite(model._precision))
    assert model._precision[1, 1] >= 0.0


def test_huge_scale_material_negative_eigenvalue_is_rejected(tmp_path: Path) -> None:
    path = write_artifact(
        tmp_path / "huge4.npz",
        mean=(0.0, 0.0),
        precision=((1e308, 0.0), (0.0, -1e301)),
        names=("a", "b"),
    )
    with pytest.raises(ValueError, match="positive semidefinite"):
        GaussianSupportModel.load(path)


def test_huge_scale_asymmetry_is_rejected(tmp_path: Path) -> None:
    path = write_artifact(
        tmp_path / "huge5.npz",
        mean=(0.0, 0.0),
        precision=((1e308, 1e308), (-1e308, 1e308)),
        names=("a", "b"),
    )
    with pytest.raises(ValueError, match="symmetric"):
        GaussianSupportModel.load(path)


def test_fit_rejects_overflowing_calibration_features() -> None:
    vectors = [fv(-1e200, 0.0), fv(1e200, 1.0), fv(0.0, 2.0)]
    with pytest.raises(ValueError, match="overflow|finite"):
        GaussianSupportModel().fit(vectors)


# R2: hyperparameters are public and mutable, so they are revalidated at use boundaries.

BAD_HYPERPARAMETERS = [float("inf"), float("-inf"), float("nan"), 0.0, -1.0, "abc", None, True]


@pytest.mark.parametrize("bad", BAD_HYPERPARAMETERS)
def test_invalid_temperature_after_construction_cannot_grant_support(bad: object) -> None:
    model = fitted_model()
    model.temperature = bad  # type: ignore[assignment]
    with pytest.raises(ValueError, match="temperature"):
        model.support_score(fv(1000.0, 1000.0))


@pytest.mark.parametrize("bad", BAD_HYPERPARAMETERS)
def test_invalid_regularization_after_construction_rejected_by_fit(bad: object) -> None:
    model = GaussianSupportModel()
    model.regularization = bad  # type: ignore[assignment]
    with pytest.raises(ValueError, match="regularization"):
        model.fit([fv(-1, 0), fv(0, -1), fv(1, 0), fv(0, 1)])


@pytest.mark.parametrize("field_name", ["temperature", "regularization"])
@pytest.mark.parametrize("bad", BAD_HYPERPARAMETERS)
def test_save_rejects_invalid_current_hyperparameters_without_touching_artifact(
    tmp_path: Path, field_name: str, bad: object
) -> None:
    model = fitted_model()
    path = tmp_path / "support.npz"
    model.save(path)
    original = path.read_bytes()
    setattr(model, field_name, bad)
    with pytest.raises(ValueError, match=field_name):
        model.save(path)
    assert path.read_bytes() == original
    fresh = tmp_path / "never.npz"
    with pytest.raises(ValueError, match=field_name):
        model.save(fresh)
    assert not fresh.exists()
    GaussianSupportModel.load(path)


def test_valid_hyperparameter_mutation_is_supported_and_roundtrips(tmp_path: Path) -> None:
    model = fitted_model()
    query = fv(1.0, 1.0)
    d2 = model.squared_distance(query)
    model.temperature = 4.0
    assert model.support_score(query) == pytest.approx(np.exp(-0.5 * d2 / 4.0))
    model.temperature = 8
    assert model.support_score(query) == pytest.approx(np.exp(-0.5 * d2 / 8.0))
    model.save(tmp_path / "m.npz")
    restored = GaussianSupportModel.load(tmp_path / "m.npz")
    assert restored.temperature == 8.0
    assert restored.support_score(query) == pytest.approx(model.support_score(query))
    model.regularization = 1e-3
    model.fit([fv(-1, 0), fv(0, -1), fv(1, 0), fv(0, 1), fv(0, 0)])
    assert 0.0 < model.support_score(query) <= 1.0


@pytest.mark.parametrize("bad", ["2.0", True, False, b"2"])
def test_hyperparameters_must_be_real_numbers(bad: object) -> None:
    with pytest.raises(ValueError, match="temperature"):
        GaussianSupportModel(temperature=bad)  # type: ignore[arg-type]


def test_nan_mean_in_memory_raises_instead_of_scoring() -> None:
    model = one_feature_model()
    model._mean = np.array([np.nan])
    with pytest.raises(ValueError, match="indeterminate"):
        model.support_score(xq(0.0))


def test_artifact_metadata_must_be_a_json_object(tmp_path: Path) -> None:
    np.savez_compressed(
        tmp_path / "list.npz", mean=np.zeros(1), precision=np.eye(1), metadata="[1, 2]"
    )
    with pytest.raises(ValueError, match="JSON object"):
        GaussianSupportModel.load(tmp_path / "list.npz")


def test_unusable_eigendecomposition_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_eigh(matrix: object) -> tuple[np.ndarray, np.ndarray]:
        return np.array([np.nan, 1.0]), np.eye(2)

    monkeypatch.setattr(np.linalg, "eigh", broken_eigh)
    path = write_artifact(
        tmp_path / "eigh.npz", mean=(0.0, 0.0), precision=np.eye(2), names=("a", "b")
    )
    with pytest.raises(ValueError, match="eigendecomposition"):
        GaussianSupportModel.load(path)


def test_projection_that_would_overflow_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defensive branch: a projected matrix is rescaled, and the result must still be finite."""

    def inflating_eigh(matrix: object) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-1e-9, 1.0]), 2.0 * np.eye(2)

    monkeypatch.setattr(np.linalg, "eigh", inflating_eigh)
    path = write_artifact(
        tmp_path / "inflate.npz",
        mean=(0.0, 0.0),
        precision=((1e308, 0.0), (0.0, 1e300)),
        names=("a", "b"),
    )
    with pytest.raises(ValueError, match="overflows"):
        GaussianSupportModel.load(path)
