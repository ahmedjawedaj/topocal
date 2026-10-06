import hashlib
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


REAL_HEADER = (
    "geometry_id,protocol,split,is_ood,geometry_family,beta1,beta2,n_vertices,n_tetrahedra,"
    "min_clearance,quality_min,quality_mean,relative_path\n"
)
REAL_ROW = "PB_train_0000_b00,B,train,False,B,0,0,812,3900,0.04,0.31,0.62,packed/x.h5\n"


def write(tmp_path: Path, content: str, name: str = "manifest.csv") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_real_upstream_header_loads_with_unknown_seed(tmp_path: Path) -> None:
    """The published manifest has no ``seed`` column. Absence is None, never 0."""

    manifest = TopoBoxManifest.from_csv(write(tmp_path, REAL_HEADER + REAL_ROW))
    (record,) = manifest.records
    assert record.seed is None
    assert record.geometry_id == "PB_train_0000_b00"
    assert (record.protocol, record.split) == ("B", "train")
    assert record.is_ood is False
    assert (record.beta1, record.beta2, record.geometry_family) == (0, 0, "B")
    assert manifest.geometry_ids(protocol="B", split="train") == ("PB_train_0000_b00",)


def test_legacy_seeded_manifest_still_preserves_seeds(tmp_path: Path) -> None:
    manifest = TopoBoxManifest.from_csv(write(tmp_path, CSV))
    assert [record.seed for record in manifest.records] == [1, 2, 3, 4]


def test_seed_zero_is_a_real_seed_distinct_from_absent(tmp_path: Path) -> None:
    header = "geometry_id,protocol,split,is_ood,beta1,beta2,geometry_family,seed\n"
    rows = "a,B,train,false,0,0,B,0\nb,B,train,false,0,0,B,\nc,B,train,false,0,0,B,  7 \n"
    manifest = TopoBoxManifest.from_csv(write(tmp_path, header + rows))
    assert [record.seed for record in manifest.records] == [0, None, 7]


@pytest.mark.parametrize("bad", ["abc", "1.5", "7.0", "1e3", "0x10", "nan", "1 2"])
def test_malformed_supplied_seed_raises(tmp_path: Path, bad: str) -> None:
    header = "geometry_id,protocol,split,is_ood,beta1,beta2,geometry_family,seed\n"
    path = write(tmp_path, header + f"g1,B,train,false,0,0,B,{bad}\n")
    with pytest.raises(ValueError, match="seed"):
        TopoBoxManifest.from_csv(path)


@pytest.mark.parametrize(
    "column", ["geometry_id", "protocol", "split", "is_ood", "beta1", "beta2", "geometry_family"]
)
def test_mandatory_columns_stay_mandatory(tmp_path: Path, column: str) -> None:
    names = REAL_HEADER.strip().split(",")
    values = REAL_ROW.strip().split(",")
    index = names.index(column)
    header = ",".join(n for i, n in enumerate(names) if i != index) + "\n"
    row = ",".join(v for i, v in enumerate(values) if i != index) + "\n"
    with pytest.raises(ValueError, match="missing columns"):
        TopoBoxManifest.from_csv(write(tmp_path, header + row))


@pytest.mark.parametrize("bad_beta", ["", "x", "1.5"])
def test_blank_or_malformed_topology_values_are_not_optional(
    tmp_path: Path, bad_beta: str
) -> None:
    row = f"g1,B,train,false,B,{bad_beta},0,1,1,0.1,0.1,0.1,p\n"
    with pytest.raises(ValueError, match="beta1|integer"):
        TopoBoxManifest.from_csv(write(tmp_path, REAL_HEADER + row))


def test_short_rows_raise_value_error_not_attribute_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing a value"):
        TopoBoxManifest.from_csv(write(tmp_path, REAL_HEADER + "g1,B,train\n"))


def test_real_header_rejects_negative_betti_and_duplicates(tmp_path: Path) -> None:
    negative = "g1,B,train,false,B,-1,0,1,1,0.1,0.1,0.1,p\n"
    with pytest.raises(ValueError, match="non-negative"):
        TopoBoxManifest.from_csv(write(tmp_path, REAL_HEADER + negative))
    with pytest.raises(ValueError, match="duplicate"):
        TopoBoxManifest.from_csv(write(tmp_path, REAL_HEADER + REAL_ROW + REAL_ROW, "dup.csv"))


def test_real_header_filters_and_ood_flag(tmp_path: Path) -> None:
    content = (
        REAL_HEADER
        + REAL_ROW
        + "PB_test_ood_0000_b30,B,test_ood,True,B,3,0,9,9,0.1,0.1,0.1,p\n"
        + "PC_val_0000_b01,C,validation,False,C,0,1,9,9,0.1,0.1,0.1,p\n"
    )
    manifest = TopoBoxManifest.from_csv(write(tmp_path, content))
    assert manifest.geometry_ids(is_ood=True) == ("PB_test_ood_0000_b30",)
    assert manifest.geometry_ids(protocol="C", split="val") == ("PC_val_0000_b01",)
    assert manifest.select(protocol="B", split="test_ood")[0].beta1 == 3


def test_fingerprints_follow_exact_manifest_bytes(tmp_path: Path) -> None:
    content = REAL_HEADER + REAL_ROW
    path = write(tmp_path, content)
    manifest = TopoBoxManifest.from_csv(path)
    assert manifest.sha256 == hashlib.sha256(content.encode()).hexdigest()
    # A whitespace-only byte change keeps the parsed records but changes the file hash.
    other = TopoBoxManifest.from_csv(write(tmp_path, content + "\n", "other.csv"))
    assert other.records == manifest.records
    assert other.sha256 != manifest.sha256
    assert other.split_fingerprint(protocol="B", split="train") == manifest.split_fingerprint(
        protocol="B", split="train"
    )


def test_manifest_parse_does_not_touch_referenced_geometry_files(tmp_path: Path) -> None:
    """``relative_path`` is never opened while parsing, so a missing file is not an error."""

    content = REAL_HEADER + "g1,B,train,false,B,0,0,1,1,0.1,0.1,0.1,does/not/exist.h5\n"
    assert len(TopoBoxManifest.from_csv(write(tmp_path, content))) == 1
