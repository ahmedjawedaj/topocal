import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from topobox_fixture import BETTI, FixtureTree, geometry_sample, hodge_sample  # noqa: E402

from topocal.datasets import (  # noqa: E402
    PDEInstanceKey,
    Sha256Sums,
    TopoBoxDataset,
    TopoBoxSchemaError,
)
from topocal.experiments.manifest import file_sha256  # noqa: E402
from topocal.experiments.splits import ExperimentPartition, SplitManifest  # noqa: E402

TRAIN_A = "PB_train_0000_b00"
TRAIN_B = "PB_train_0001_b10"


def make_dataset(
    tree: FixtureTree,
    partition: ExperimentPartition = ExperimentPartition.TRAIN,
    **kwargs: Any,
) -> TopoBoxDataset:
    return TopoBoxDataset(
        manifest=tree.manifest,
        split_manifest=tree.split_manifest,
        partition=partition,
        geometry_root=tree.geometry_root,
        solution_root=tree.solution_root,
        **kwargs,
    )


def geometry_shard(tree: FixtureTree, protocol: str, split: str, number: int = 0) -> Path:
    return tree.geometry_root / f"protocol_{protocol}" / split / f"shard_{number:04d}.h5"


def solution_shard(tree: FixtureTree, protocol: str, split: str, number: int = 0) -> Path:
    return tree.solution_root / f"protocol_{protocol}" / split / f"shard_{number:04d}.h5"


def mutate(path: Path, group_path: str, name: str, fn: Callable[[np.ndarray], Any]) -> None:
    with h5py.File(path, "r+") as handle:
        group = handle[group_path]
        value = fn(group[name][()])
        del group[name]
        group.create_dataset(name, data=value)


def test_importing_datasets_does_not_import_h5py() -> None:
    code = "import sys, topocal.datasets; assert 'h5py' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, env=dict(os.environ))


def test_construction_opens_no_files(
    topobox_tree: FixtureTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("construction must not open HDF5 files")

    monkeypatch.setattr(h5py, "File", forbidden)
    dataset = make_dataset(topobox_tree)
    assert len(dataset) == 2
    assert list(dataset) == [TRAIN_A, TRAIN_B]
    assert dataset.record(TRAIN_B).beta1 == 1
    assert len(dataset.instance_keys()) == 2 * 3 * 4
    with pytest.raises(AssertionError, match="must not open"):
        dataset.load_geometry(TRAIN_A)


def test_missing_roots_are_not_checked_until_first_read(topobox_tree: FixtureTree) -> None:
    dataset = TopoBoxDataset(
        manifest=topobox_tree.manifest,
        split_manifest=topobox_tree.split_manifest,
        partition=ExperimentPartition.TRAIN,
        geometry_root=topobox_tree.root / "nowhere",
    )
    assert dataset.partition is ExperimentPartition.TRAIN
    with pytest.raises(FileNotFoundError, match="directory not found"):
        dataset.load_geometry(TRAIN_A)
    with pytest.raises(ValueError, match="solution_root"):
        dataset.load_hodge_heat(TRAIN_A, 0)


def test_final_ood_test_is_locked(topobox_tree: FixtureTree) -> None:
    with pytest.raises(PermissionError, match="locked"):
        make_dataset(topobox_tree, ExperimentPartition.OOD_TEST)
    dataset = make_dataset(topobox_tree, ExperimentPartition.OOD_TEST, allow_final_test=True)
    assert dataset.ids == ("PC_test_ood_0000_b03",)
    geometry = dataset.load_geometry("PC_test_ood_0000_b03")
    assert geometry.protocol == "C" and geometry.split == "test_ood"


def test_logical_partition_differs_from_physical_split(topobox_tree: FixtureTree) -> None:
    dataset = make_dataset(topobox_tree, ExperimentPartition.CALIBRATION)
    geometry = dataset.load_geometry("PB_validation_0000_b00")
    assert geometry.split == "validation"
    dev = make_dataset(topobox_tree, ExperimentPartition.OOD_DEV)
    assert dev.load_geometry("PB_test_ood_0000_b30").split == "test_ood"


def test_ids_outside_the_partition_are_denied(topobox_tree: FixtureTree) -> None:
    dataset = make_dataset(topobox_tree)
    for forbidden in ("PB_validation_0000_b00", "PC_test_ood_0000_b03", "nonsense"):
        with pytest.raises(PermissionError, match="not part of partition"):
            dataset.load_geometry(forbidden)
        with pytest.raises(PermissionError):
            dataset.load_hodge_heat(forbidden, 0)
        with pytest.raises(PermissionError):
            dataset.record(forbidden)


def test_split_must_match_manifest_fingerprint(topobox_tree: FixtureTree) -> None:
    other = SplitManifest(
        dataset_fingerprint="0" * 64,
        train_ids=(TRAIN_A,),
        calibration_ids=(),
        ood_dev_ids=(),
        ood_test_ids=(),
        seed=1,
    )
    with pytest.raises(ValueError, match="different TopoBox manifest"):
        TopoBoxDataset(
            manifest=topobox_tree.manifest,
            split_manifest=other,
            partition=ExperimentPartition.TRAIN,
            geometry_root=topobox_tree.geometry_root,
        )


def _split_with(tree: FixtureTree, train_ids: tuple[str, ...]) -> SplitManifest:
    return SplitManifest(tree.manifest.sha256, train_ids, (), (), (), 1)


def test_unknown_and_malformed_ids_are_rejected(topobox_tree: FixtureTree) -> None:
    with pytest.raises(ValueError, match="not in the TopoBox manifest"):
        TopoBoxDataset(
            manifest=topobox_tree.manifest,
            split_manifest=_split_with(topobox_tree, (TRAIN_A, "ghost")),
            partition=ExperimentPartition.TRAIN,
            geometry_root=topobox_tree.geometry_root,
        )
    with pytest.raises(ValueError, match="cannot contain"):
        TopoBoxDataset(
            manifest=topobox_tree.manifest,
            split_manifest=_split_with(topobox_tree, ("a/b",)),
            partition=ExperimentPartition.TRAIN,
            geometry_root=topobox_tree.geometry_root,
        )


def test_ids_repeated_across_protocols_are_ambiguous(topobox_tree: FixtureTree) -> None:
    from topocal.datasets import TopoBoxManifest

    path = topobox_tree.root / "dup.csv"
    path.write_text(
        topobox_tree.manifest_path.read_text(encoding="utf-8") + "PB_train_0000_b00,C,train,"
        "false,0,0,C,9\n",
        encoding="utf-8",
    )
    manifest = TopoBoxManifest.from_csv(path)
    with pytest.raises(ValueError, match="ambiguous"):
        TopoBoxDataset(
            manifest=manifest,
            split_manifest=SplitManifest(manifest.sha256, (TRAIN_A,), (), (), (), 1),
            partition=ExperimentPartition.TRAIN,
            geometry_root=topobox_tree.geometry_root,
        )


def test_load_geometry_returns_validated_read_only_arrays(topobox_tree: FixtureTree) -> None:
    dataset = make_dataset(topobox_tree)
    geometry = dataset.load_geometry(TRAIN_A)
    expected = geometry_sample(TRAIN_A)
    assert geometry.protocol == "B"
    np.testing.assert_allclose(geometry["points"], expected["points"])
    assert geometry["tetra"].shape == (2, 4)
    assert list(geometry["geometry_feature_names"]) == [
        "x_normalized",
        "y_normalized",
        "z_normalized",
        "is_boundary",
        "analytic_domain_sdf",
    ]
    assert geometry["has_tunnel"].shape == () and not bool(geometry["has_tunnel"])
    assert geometry["regular_grid_features"].shape == (32, 16, 16, 5)
    assert "metadata" not in geometry.arrays
    assert geometry.metadata is None
    with pytest.raises(ValueError, match="read-only"):
        geometry["points"][0, 0] = 5.0
    assert [geometry.cochain_size(d) for d in (0, 1, 2)] == [5, 9, 7]
    with pytest.raises(ValueError, match="degree"):
        geometry.cochain_size(3)


def test_incidence_operators_form_a_chain_complex(topobox_tree: FixtureTree) -> None:
    geometry = make_dataset(topobox_tree).load_geometry(TRAIN_A)
    b1, b2, b3 = (geometry.incidence(k).to_dense() for k in (1, 2, 3))
    assert b1.shape == (5, 9) and b2.shape == (9, 7) and b3.shape == (7, 2)
    assert np.abs(b1 @ b2).max() == 0.0
    assert np.abs(b2 @ b3).max() == 0.0
    with pytest.raises(ValueError, match="incidence degree"):
        geometry.incidence(4)


def test_field_subset_and_missing_fields(topobox_tree: FixtureTree) -> None:
    dataset = make_dataset(topobox_tree)
    geometry = dataset.load_geometry(TRAIN_A, fields=("points", "edges"))
    assert sorted(geometry.arrays) == ["edges", "points"]
    with pytest.raises(KeyError, match="was not loaded"):
        geometry["tetra"]
    with pytest.raises(TopoBoxSchemaError, match=r"missing fields \['bogus'\].*available"):
        dataset.load_geometry(TRAIN_A, fields=("points", "bogus"))


def test_float_dtype_cast_only_touches_floats(topobox_tree: FixtureTree) -> None:
    geometry = make_dataset(topobox_tree, float_dtype=np.float32).load_geometry(TRAIN_A)
    assert geometry["points"].dtype == np.float32
    assert geometry["tetra"].dtype.kind == "i"


def test_metadata_is_opt_in_and_parsed(topobox_tree: FixtureTree) -> None:
    dataset = make_dataset(topobox_tree)
    geometry = dataset.load_geometry(TRAIN_A, include_metadata=True)
    assert geometry.metadata == {"geometry_id": TRAIN_A, "seed": 7}


def test_metadata_from_group_attributes(topobox_tree: FixtureTree) -> None:
    shard = geometry_shard(topobox_tree, "B", "train", 0)
    with h5py.File(shard, "r+") as handle:
        group = handle["samples"][TRAIN_A]
        del group["metadata"]
        node = group.create_group("metadata")
        node.attrs["seed"] = np.int64(7)
        node.attrs["tags"] = np.array([1, 2])
        node.attrs["name"] = "abc"
    geometry = make_dataset(topobox_tree).load_geometry(TRAIN_A, include_metadata=True)
    assert geometry.metadata == {"seed": 7, "tags": [1, 2], "name": "abc"}


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (None, "absent"),
        ("not json", "not valid JSON"),
        ("[1, 2]", "must be an object"),
        (["a", "b"], "scalar JSON string"),
    ],
)
def test_bad_metadata_is_reported(
    topobox_tree: FixtureTree, replacement: Any, message: str
) -> None:
    shard = geometry_shard(topobox_tree, "B", "train", 0)
    with h5py.File(shard, "r+") as handle:
        group = handle["samples"][TRAIN_A]
        del group["metadata"]
        if replacement is not None:
            group.create_dataset("metadata", data=replacement, dtype=h5py.string_dtype())
    with pytest.raises(TopoBoxSchemaError, match=message):
        make_dataset(topobox_tree).load_geometry(TRAIN_A, include_metadata=True)


GEOMETRY_CORRUPTIONS: list[tuple[str, Callable[[np.ndarray], Any], str]] = [
    ("tetra", lambda a: a[:, :3], "shape"),
    ("points", lambda a: np.where(np.arange(a.size).reshape(a.shape) == 0, np.nan, a), "non-fin"),
    ("points", lambda a: a.astype(np.int64), "dtype"),
    ("edges", lambda a: a + 99, "outside"),
    ("faces", lambda a: a - 1, "negative"),
    ("tetra_quality", lambda a: a[:-1], "inconsistent tetrahedra"),
    ("incidence_1_shape", lambda a: a + np.array([1, 0]), "disagrees"),
    ("incidence_2_row", lambda a: a[:-1], "differ in length"),
    ("incidence_2_row", lambda a: a + 50, "row index"),
    ("incidence_3_col", lambda a: a + 50, "column index"),
]


@pytest.mark.parametrize(("name", "corrupt", "message"), GEOMETRY_CORRUPTIONS)
def test_corrupt_geometry_fails_loudly(
    topobox_tree: FixtureTree, name: str, corrupt: Callable[[np.ndarray], Any], message: str
) -> None:
    shard = geometry_shard(topobox_tree, "B", "train", 0)
    mutate(shard, f"samples/{TRAIN_A}", name, corrupt)
    with pytest.raises(TopoBoxSchemaError, match=message):
        make_dataset(topobox_tree).load_geometry(TRAIN_A)


def test_hodge_heat_target_is_opt_in(topobox_tree: FixtureTree) -> None:
    dataset = make_dataset(topobox_tree)
    expected = hodge_sample(TRAIN_B)["k1"]
    plain = dataset.load_hodge_heat(TRAIN_B, 1)
    assert plain.target is None
    assert "relative_final_mass_norm" not in plain.aux
    assert plain.configs[2] == "balanced"
    np.testing.assert_allclose(plain.w0, expected["w0"])
    np.testing.assert_allclose(plain.mass, expected["mass"])
    assert plain.harmonic_basis.shape == (6, BETTI[TRAIN_B][0])
    assert plain.aux["seeds"].tolist() == [100, 101, 102, 103]
    full = dataset.load_hodge_heat(TRAIN_B, 1, include_target=True)
    assert full.target is not None
    np.testing.assert_allclose(full.target, expected["wT"])
    np.testing.assert_allclose(
        full.aux["relative_final_mass_norm"], expected["relative_final_mass_norm"]
    )
    with pytest.raises(ValueError, match="degree"):
        dataset.load_hodge_heat(TRAIN_B, 3)


def test_instances_split_inputs_from_ground_truth(topobox_tree: FixtureTree) -> None:
    dataset = make_dataset(topobox_tree)
    keys = dataset.instance_keys()
    assert keys[0] == PDEInstanceKey(TRAIN_A, 0, "non_harmonic")
    assert keys[-1] == PDEInstanceKey(TRAIN_B, 2, "strong_harmonic")
    key = PDEInstanceKey(TRAIN_A, 2, "balanced")
    assert key.problem_id == f"{TRAIN_A}:k2:balanced"
    inputs_only = dataset.load_instance(key)
    assert inputs_only.target is None
    assert inputs_only.w0.shape == (7,) == inputs_only.mass.shape
    scored = dataset.load_instance(key, include_target=True, geometry_fields=("points", "faces"))
    assert scored.target is not None and scored.target.shape == (7,)
    np.testing.assert_allclose(scored.target, hodge_sample(TRAIN_A)["k2"]["wT"][2])
    assert sorted(scored.geometry.arrays) == ["faces", "points"]
    problem = scored.to_problem()
    assert problem.problem_id == key.problem_id
    assert problem.payload.target is None  # type: ignore[attr-defined]
    np.testing.assert_allclose(problem.payload.w0, scored.w0)  # type: ignore[attr-defined]


def test_pde_instance_key_validation() -> None:
    with pytest.raises(ValueError, match="degree"):
        PDEInstanceKey("g", 3, "balanced")
    with pytest.raises(ValueError, match="config"):
        PDEInstanceKey("g", 0, "mystery")


def test_solution_geometry_size_mismatch_is_detected(topobox_tree: FixtureTree) -> None:
    shard = solution_shard(topobox_tree, "B", "train", 0)
    group = f"samples/{TRAIN_A}/k1"
    mutate(shard, group, "w0", lambda a: np.hstack([a, a[:, :1]]))
    with pytest.raises(TopoBoxSchemaError, match="inconsistent cochain size"):
        make_dataset(topobox_tree).load_hodge_heat(TRAIN_A, 1)
    mutate(shard, group, "mass", lambda a: np.append(a, 1.0))
    mutate(shard, group, "wT", lambda a: np.hstack([a, a[:, :1]]))
    mutate(shard, group, "harmonic_basis", lambda a: np.vstack([a, a[:1]]))
    with pytest.raises(TopoBoxSchemaError, match="geometry has 9"):
        make_dataset(topobox_tree).load_hodge_heat(TRAIN_A, 1)


def test_non_positive_mass_and_missing_required_fields(topobox_tree: FixtureTree) -> None:
    shard = solution_shard(topobox_tree, "B", "train", 0)
    mutate(shard, f"samples/{TRAIN_A}/k0", "mass", lambda a: a * 0.0)
    with pytest.raises(TopoBoxSchemaError, match="strictly positive"):
        make_dataset(topobox_tree).load_hodge_heat(TRAIN_A, 0)
    with h5py.File(shard, "r+") as handle:
        del handle["samples"][TRAIN_B]["k0"]["wT"]
    with pytest.raises(TopoBoxSchemaError, match="missing fields"):
        make_dataset(topobox_tree).load_hodge_heat(TRAIN_B, 0, include_target=True)
    with h5py.File(shard, "r+") as handle:
        del handle["samples"][TRAIN_B]["k0"]
    with pytest.raises(TopoBoxSchemaError, match="missing group 'k0'"):
        make_dataset(topobox_tree).load_hodge_heat(TRAIN_B, 0)


def test_geometry_without_cochain_field_is_reported(topobox_tree: FixtureTree) -> None:
    shard = geometry_shard(topobox_tree, "B", "train", 0)
    with h5py.File(shard, "r+") as handle:
        del handle["samples"][TRAIN_A]["edges"]
    with pytest.raises(TopoBoxSchemaError, match="missing field 'edges'"):
        make_dataset(topobox_tree).load_hodge_heat(TRAIN_A, 1)


def test_shard_layout_problems_are_reported(topobox_tree: FixtureTree) -> None:
    train = topobox_tree.geometry_root / "protocol_B" / "train"
    (train / "shard_0001.h5").unlink()
    dataset = make_dataset(topobox_tree)
    with pytest.raises(LookupError, match="in no geometry shard"):
        dataset.load_geometry(TRAIN_B)

    (train / "shard_0000.h5").unlink()
    with pytest.raises(FileNotFoundError, match="no shard"):
        make_dataset(topobox_tree).load_geometry(TRAIN_A)


def test_duplicate_geometry_across_shards_is_rejected(topobox_tree: FixtureTree) -> None:
    train = topobox_tree.geometry_root / "protocol_B" / "train"
    (train / "shard_0009.h5").write_bytes((train / "shard_0000.h5").read_bytes())
    with pytest.raises(TopoBoxSchemaError, match="appears in both"):
        make_dataset(topobox_tree).load_geometry(TRAIN_A)


def test_shard_without_samples_group_is_rejected(topobox_tree: FixtureTree) -> None:
    shard = geometry_shard(topobox_tree, "B", "train", 0)
    with h5py.File(shard, "w") as handle:
        handle.create_group("other")
    with pytest.raises(TopoBoxSchemaError, match="samples"):
        make_dataset(topobox_tree).load_geometry(TRAIN_A)


def test_provenance_lists_only_shards_that_were_read(topobox_tree: FixtureTree) -> None:
    dataset = make_dataset(topobox_tree)
    assert dataset.provenance().files_sha256 == {}
    dataset.load_geometry(TRAIN_A)
    dataset.load_hodge_heat(TRAIN_B, 0)
    provenance = dataset.provenance()
    geometry_key = "geometry/protocol_B/train/shard_0000.h5"
    solution_key = "solution/protocol_B/train/shard_0000.h5"
    assert set(provenance.files_sha256) == {
        geometry_key,
        "geometry/protocol_B/train/shard_0001.h5",
        solution_key,
    }
    assert provenance.files_sha256[geometry_key] == file_sha256(
        geometry_shard(topobox_tree, "B", "train", 0)
    )
    assert provenance.manifest_sha256 == topobox_tree.manifest.sha256
    assert provenance.split_fingerprint == topobox_tree.split_manifest.fingerprint
    assert provenance.partition == "train" and provenance.n_ids == 2
    assert len(provenance.ids_sha256) == 64
    for schema in provenance.schemas.values():
        assert schema["missing_expected"] == [] and schema["unexpected"] == []
    assert provenance.schemas[geometry_key]["kind"] == "geometry"
    assert provenance.schemas[solution_key]["kind"] == "hodge_heat"
    assert provenance.published_sha256_status == {}

    unhashed = dataset.provenance(hash_shards=False)
    assert set(unhashed.files_sha256.values()) == {None}

    destination = topobox_tree.root / "out" / "provenance.json"
    provenance.write_json(destination)
    assert json.loads(destination.read_text(encoding="utf-8")) == provenance.to_dict()


def test_provenance_compares_with_published_checksums(topobox_tree: FixtureTree) -> None:
    dataset = make_dataset(topobox_tree)
    dataset.load_geometry(TRAIN_A)
    dataset.load_geometry(TRAIN_B)
    dataset.load_hodge_heat(TRAIN_A, 0)
    good = file_sha256(geometry_shard(topobox_tree, "B", "train", 0))
    sums = topobox_tree.root / "SHA256SUMS.txt"
    sums.write_text(
        f"{good}  TopoBox-3D/packed/protocol_B/train/shard_0000.h5\n"
        f"{'0' * 64} *TopoBox-3D/packed/protocol_B/train/shard_0001.h5\n"
        "\n",
        encoding="utf-8",
    )
    status = dataset.provenance(published_sums=Sha256Sums.from_file(sums)).published_sha256_status
    assert status == {
        "geometry/protocol_B/train/shard_0000.h5": "match",
        "geometry/protocol_B/train/shard_0001.h5": "mismatch",
        "solution/protocol_B/train/shard_0000.h5": "unlisted",
    }


def test_sha256sums_parsing_and_ambiguity(tmp_path: Path) -> None:
    path = tmp_path / "sums.txt"
    path.write_text(f"{'a' * 64}  x/shard_0000.h5\n{'b' * 64}  y/shard_0000.h5\n", encoding="utf-8")
    sums = Sha256Sums.from_file(path)
    assert sums.status("shard_0000.h5", "a" * 64) == "ambiguous"
    assert sums.status("x/shard_0000.h5", "A" * 64) == "match"
    path.write_text("not a checksum line\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed checksum line 1"):
        Sha256Sums.from_file(path)


@pytest.mark.parametrize("partition", [ExperimentPartition.OOD_TEST, "ood_test"])
def test_final_test_is_blocked_before_any_file_opens(
    topobox_tree: FixtureTree, monkeypatch: pytest.MonkeyPatch, partition: Any
) -> None:
    opened: list[object] = []

    def spy(*args: Any, **kwargs: Any) -> None:
        opened.append(args)
        raise AssertionError("no HDF5 file may open before the guard")

    monkeypatch.setattr(h5py, "File", spy)
    with pytest.raises(PermissionError, match="locked"):
        make_dataset(topobox_tree, partition)
    with pytest.raises(PermissionError, match="locked"):
        make_dataset(topobox_tree, partition, allow_final_test="yes")  # type: ignore[arg-type]
    assert opened == []


def test_string_partitions_are_normalized_for_ordinary_use(topobox_tree: FixtureTree) -> None:
    dataset = make_dataset(topobox_tree, "train")
    assert dataset.partition is ExperimentPartition.TRAIN
    assert dataset.ids == (TRAIN_A, TRAIN_B)
    dataset.load_geometry(TRAIN_A)
    assert dataset.provenance().partition == "train"
    assert make_dataset(topobox_tree, "calibration").ids == ("PB_validation_0000_b00",)
    assert make_dataset(topobox_tree, "ood_dev").ids == ("PB_test_ood_0000_b30",)


def test_explicit_final_opt_in_with_string_input_on_synthetic_data(
    topobox_tree: FixtureTree,
) -> None:
    dataset = make_dataset(topobox_tree, "ood_test", allow_final_test=True)
    assert dataset.partition is ExperimentPartition.OOD_TEST
    assert dataset.ids == ("PC_test_ood_0000_b03",)
    assert dataset.load_geometry("PC_test_ood_0000_b03").protocol == "C"
    assert dataset.provenance().partition == "ood_test"


@pytest.mark.parametrize("partition", ["final", "OOD_TEST", None, 7])
def test_invalid_dataset_partitions_raise_value_error(
    topobox_tree: FixtureTree, partition: Any
) -> None:
    with pytest.raises(ValueError, match="unknown experiment partition"):
        make_dataset(topobox_tree, partition)
