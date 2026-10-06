import json
from pathlib import Path

from topocal.experiments.manifest import ExperimentManifest, file_sha256


def test_experiment_manifest_captures_reproducibility_bundle(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"weights")
    config = {"seed": 7, "protocol": "B"}
    manifest = ExperimentManifest.create(
        experiment="phase1-smoke",
        dataset_fingerprint="dataset123",
        split_fingerprint="split123",
        config=config,
        repo_root=Path(__file__).parents[1],
        random_seeds={"numpy": 7},
        artifact_paths={"checkpoint": checkpoint},
        feature_schema=("rayleigh", "pde_residual"),
        solver={"name": "fenicsx", "version": "external"},
    )
    output = tmp_path / "run"
    manifest.write_bundle(output)
    data = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    exact_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert data["experiment"] == "phase1-smoke"
    assert data["config_sha256"]
    assert data["artifacts_sha256"]["checkpoint"] == file_sha256(checkpoint)
    assert data["feature_schema"] == ["rayleigh", "pde_residual"]
    assert exact_config == config


def test_experiment_manifest_records_dataset_provenance(tmp_path: Path) -> None:
    provenance = {"manifest_sha256": "abc", "files_sha256": {"geometry/x.h5": "def"}}
    manifest = ExperimentManifest.create(
        experiment="phase1-loader",
        dataset_fingerprint="dataset123",
        split_fingerprint="split123",
        config={"seed": 1},
        dataset_provenance=provenance,
    )
    manifest.write_bundle(tmp_path / "run")
    data = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert data["dataset_provenance"] == provenance
    default = ExperimentManifest.create(
        experiment="x", dataset_fingerprint="d", split_fingerprint="s", config={}
    )
    assert default.dataset_provenance == {}
