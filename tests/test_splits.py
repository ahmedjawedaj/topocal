from pathlib import Path

import pytest

from topocal.experiments.splits import ExperimentPartition, SplitManifest


def split_manifest() -> SplitManifest:
    return SplitManifest(
        dataset_fingerprint="dataset123",
        train_ids=("g1", "g2"),
        calibration_ids=("g3",),
        ood_dev_ids=("g4",),
        ood_test_ids=("g5",),
        seed=7,
    )


def test_final_ood_test_is_locked_by_default() -> None:
    manifest = split_manifest()
    with pytest.raises(PermissionError, match="locked"):
        manifest.ids_for(ExperimentPartition.OOD_TEST)
    assert manifest.ids_for(ExperimentPartition.OOD_TEST, allow_final_test=True) == ("g5",)


def test_split_manifest_rejects_cross_partition_leakage() -> None:
    with pytest.raises(ValueError, match="both"):
        SplitManifest(
            dataset_fingerprint="x",
            train_ids=("g1",),
            calibration_ids=("g1",),
            ood_dev_ids=(),
            ood_test_ids=(),
            seed=1,
        )


def test_split_manifest_roundtrip_and_fingerprint(tmp_path: Path) -> None:
    manifest = split_manifest()
    path = tmp_path / "splits.json"
    manifest.write_json(path)
    restored = SplitManifest.from_json(path)
    assert restored == manifest
    assert len(restored.fingerprint) == 64


def test_split_manifest_rejects_blank_metadata_and_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="dataset_fingerprint"):
        SplitManifest("", (), (), (), (), 1)
    with pytest.raises(ValueError, match="duplicate IDs"):
        SplitManifest("x", ("g1", "g1"), (), (), (), 1)
    with pytest.raises(ValueError, match="blank ID"):
        SplitManifest("x", ("",), (), (), (), 1)


def test_regular_partitions_are_accessible() -> None:
    manifest = split_manifest()
    assert manifest.ids_for(ExperimentPartition.TRAIN) == ("g1", "g2")
    assert manifest.ids_for(ExperimentPartition.CALIBRATION) == ("g3",)
    assert manifest.ids_for(ExperimentPartition.OOD_DEV) == ("g4",)


FINAL_FORMS = [ExperimentPartition.OOD_TEST, "ood_test"]


def synthetic_manifest() -> SplitManifest:
    return SplitManifest("synthetic", ("train",), ("cal",), ("dev",), ("final",), 7)


@pytest.mark.parametrize("partition", FINAL_FORMS)
def test_final_ood_test_is_locked_for_enum_and_string_input(partition: object) -> None:
    manifest = synthetic_manifest()
    with pytest.raises(PermissionError, match="locked"):
        manifest.ids_for(partition)  # type: ignore[arg-type]
    with pytest.raises(PermissionError, match="locked"):
        manifest.ids_for(partition, allow_final_test=False)  # type: ignore[arg-type]


@pytest.mark.parametrize("partition", FINAL_FORMS)
def test_explicit_final_opt_in_works_for_enum_and_string_input(partition: object) -> None:
    ids = synthetic_manifest().ids_for(partition, allow_final_test=True)  # type: ignore[arg-type]
    assert ids == ("final",)


@pytest.mark.parametrize("flag", ["false", "no", 1, 0.5, "", None, object()])
def test_only_boolean_true_unlocks_the_final_test(flag: object) -> None:
    manifest = synthetic_manifest()
    for partition in FINAL_FORMS:
        with pytest.raises(PermissionError, match="locked"):
            manifest.ids_for(partition, allow_final_test=flag)  # type: ignore[arg-type]


def test_string_partitions_match_enum_partitions() -> None:
    manifest = synthetic_manifest()
    for partition in (
        ExperimentPartition.TRAIN,
        ExperimentPartition.CALIBRATION,
        ExperimentPartition.OOD_DEV,
    ):
        assert manifest.ids_for(partition.value) == manifest.ids_for(partition)
    assert manifest.ids_for("train") == ("train",)
    assert manifest.ids_for("calibration") == ("cal",)
    assert manifest.ids_for("ood_dev") == ("dev",)


@pytest.mark.parametrize(
    "partition", ["OOD_TEST", "Train", " train", "final", "", None, 3, b"train"]
)
def test_invalid_partitions_raise_a_clear_value_error(partition: object) -> None:
    with pytest.raises(ValueError, match="unknown experiment partition"):
        synthetic_manifest().ids_for(partition)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expected one of"):
        ExperimentPartition.parse(partition)


def test_parse_is_idempotent_on_members() -> None:
    for member in ExperimentPartition:
        assert ExperimentPartition.parse(member) is member
        assert ExperimentPartition.parse(member.value) is member
