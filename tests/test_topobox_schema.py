import importlib
import json
from typing import Any

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from topobox_fixture import FixtureTree  # noqa: E402

from topocal.datasets import TopoBoxDataset, TopoBoxSchemaError, inspect_shard  # noqa: E402
from topocal.datasets.topobox_schema import (  # noqa: E402
    require_h5py,
    validate_geometry_arrays,
    validate_hodge_arrays,
)
from topocal.experiments.manifest import file_sha256  # noqa: E402
from topocal.experiments.splits import ExperimentPartition  # noqa: E402

TRAIN_A = "PB_train_0000_b00"
TRAIN_B = "PB_train_0001_b10"


def packed(tree: FixtureTree, number: int = 0, split: str = "train") -> Any:
    return tree.geometry_root / "protocol_B" / split / f"shard_{number:04d}.h5"


def test_inspect_geometry_shard_matches_documented_layout(topobox_tree: FixtureTree) -> None:
    schema = inspect_shard(packed(topobox_tree), kind="geometry")
    assert schema.n_samples == 1 and schema.kind == "geometry"
    assert schema.missing_expected == () and schema.unexpected == ()
    assert schema.inconsistent_samples == ()
    assert schema.shared["regular_grid_normalized_xyz"]["shape"] == [32, 16, 16, 3]
    assert schema.sample_fields["points"].ndim == 2
    assert schema.sample_fields["points"].dtype == "float64"
    assert schema.sha256 is None
    assert len(schema.fingerprint) == 64
    other = inspect_shard(packed(topobox_tree, 1), kind="geometry")
    assert other.fingerprint == schema.fingerprint
    json.dumps(schema.to_dict())


def test_inspect_hodge_shard_and_hash(topobox_tree: FixtureTree) -> None:
    path = topobox_tree.solution_root / "protocol_B" / "train" / "shard_0000.h5"
    schema = inspect_shard(path, kind="hodge_heat", hash_file=True)
    assert schema.n_samples == 2
    assert schema.sha256 == file_sha256(path)
    assert schema.missing_expected == () and schema.unexpected == ()
    assert schema.sample_fields["k2/w0"].ndim == 2
    # Without a declared kind nothing is compared with the documented layout.
    untyped = inspect_shard(path)
    assert untyped.missing_expected == () and untyped.unexpected == ()


def test_inspect_reports_layout_deviations(topobox_tree: FixtureTree) -> None:
    path = packed(topobox_tree)
    with h5py.File(path, "r+") as handle:
        sample = handle["samples"][TRAIN_A]
        del sample["edge_lengths"]
        sample.create_dataset("surprise", data=np.zeros(3))
    schema = inspect_shard(path, kind="geometry")
    assert schema.missing_expected == ("edge_lengths",)
    assert schema.unexpected == ("surprise",)


def test_inspect_flags_samples_that_differ_from_the_reference(topobox_tree: FixtureTree) -> None:
    path = topobox_tree.solution_root / "protocol_B" / "train" / "shard_0000.h5"
    with h5py.File(path, "r+") as handle:
        del handle["samples"][TRAIN_B]["k0"]["seeds"]
    schema = inspect_shard(path, kind="hodge_heat")
    assert schema.inconsistent_samples == (TRAIN_B,)


def test_inspect_requires_samples_group(topobox_tree: FixtureTree) -> None:
    path = topobox_tree.root / "empty.h5"
    with h5py.File(path, "w") as handle:
        handle.create_group("shared")
    with pytest.raises(TopoBoxSchemaError, match="samples"):
        inspect_shard(path)


def test_inspect_handles_a_shard_without_shared_group(topobox_tree: FixtureTree) -> None:
    path = topobox_tree.root / "no_shared.h5"
    with h5py.File(path, "w") as handle:
        handle.create_group("samples")
    schema = inspect_shard(path)
    assert schema.n_samples == 0 and schema.shared == {} and schema.sample_fields == {}


def test_missing_h5py_gives_an_actionable_message(monkeypatch: pytest.MonkeyPatch) -> None:
    real = importlib.import_module

    def fake(name: str, *args: Any) -> Any:
        if name == "h5py":
            raise ImportError("no h5py")
        return real(name, *args)

    monkeypatch.setattr("topocal.datasets.topobox_schema.importlib.import_module", fake)
    with pytest.raises(ImportError, match=r"topocal\[data\]"):
        require_h5py()


@pytest.fixture
def geometry_arrays(topobox_tree: FixtureTree) -> dict[str, np.ndarray]:
    dataset = TopoBoxDataset(
        manifest=topobox_tree.manifest,
        split_manifest=topobox_tree.split_manifest,
        partition=ExperimentPartition.TRAIN,
        geometry_root=topobox_tree.geometry_root,
    )
    return {k: np.array(v) for k, v in dataset.load_geometry(TRAIN_A).arrays.items()}


def test_validators_accept_unknown_extra_fields(geometry_arrays: dict[str, np.ndarray]) -> None:
    geometry_arrays["brand_new_field"] = np.zeros(3)
    validate_geometry_arrays(geometry_arrays, context="test")
    validate_geometry_arrays({}, context="empty is fine")


def test_validators_reject_bad_text_fields(geometry_arrays: dict[str, np.ndarray]) -> None:
    geometry_arrays["geometry_feature_names"] = np.array([1, 2, 3, 4, 5])
    with pytest.raises(TopoBoxSchemaError, match="geometry_feature_names"):
        validate_geometry_arrays(geometry_arrays, context="test")


def test_hodge_validator_accepts_partial_arrays() -> None:
    validate_hodge_arrays({"mass": np.ones(3)}, context="test", expected_size=3)
    with pytest.raises(TopoBoxSchemaError, match="3 simplices"):
        validate_hodge_arrays({"mass": np.ones(4)}, context="test", expected_size=3)
