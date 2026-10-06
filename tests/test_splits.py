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
