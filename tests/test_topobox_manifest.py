from pathlib import Path

import pytest

from topocal.datasets.topobox import TopoBoxManifest

CSV = """geometry_id,protocol,split,is_ood,beta1,beta2,geometry_family,seed
PB_train_0000_b00,B,train,false,0,0,B,1
PB_train_0001_b10,B,train,false,1,0,B,2
PB_test_ood_0000_b30,B,test_ood,true,3,0,B,3
PC_test_ood_0000_b03,C,test_ood,true,0,3,C,4
"""


def test_topobox_manifest_filters_and_fingerprints(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text(CSV, encoding="utf-8")
    manifest = TopoBoxManifest.from_csv(path)
    assert len(manifest) == 4
    assert manifest.geometry_ids(protocol="B", split="train") == (
        "PB_train_0000_b00",
        "PB_train_0001_b10",
    )
    assert manifest.geometry_ids(protocol="B", is_ood=True) == ("PB_test_ood_0000_b30",)
    assert len(manifest.sha256) == 64
    assert len(manifest.split_fingerprint(protocol="B", split="train")) == 64


def test_topobox_manifest_rejects_duplicate_protocol_ids(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text(CSV + "PB_train_0000_b00,B,train,false,0,0,B,5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        TopoBoxManifest.from_csv(path)


def test_topobox_manifest_rejects_invalid_fields(tmp_path: Path) -> None:
    header = "geometry_id,protocol,split,is_ood,beta1,beta2,geometry_family,seed\n"
    invalid_cases = [
        header + "g1,Z,train,false,0,0,B,1\n",
        header + "g1,B,bad,false,0,0,B,1\n",
        header + "g1,B,train,maybe,0,0,B,1\n",
        header + "g1,B,train,false,-1,0,B,1\n",
        header + ",B,train,false,0,0,B,1\n",
    ]
    for idx, content in enumerate(invalid_cases):
        path = tmp_path / f"invalid-{idx}.csv"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError):
            TopoBoxManifest.from_csv(path)


def test_topobox_manifest_select_validation_and_aliases(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text(CSV, encoding="utf-8")
    manifest = TopoBoxManifest.from_csv(path)
    assert manifest.select(protocol="B", split="testood")[0].is_ood is True
    with pytest.raises(ValueError, match="unknown"):
        manifest.select(protocol="Z")


def test_topobox_manifest_requires_columns(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text("geometry_id,protocol\ng1,B\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        TopoBoxManifest.from_csv(path)
