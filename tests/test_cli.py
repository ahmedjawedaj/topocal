import json
import subprocess
import sys

from topocal import cli


def test_cli_demo_smoke() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "topocal.cli", "demo"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = json.loads(completed.stdout)
    assert len(rows) == 3
    assert any(row["risk"] is None for row in rows)


def test_cli_main_outputs_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["topocal", "demo"])
    cli.main()
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 3


def test_cli_inspect_shard_prints_observed_layout(topobox_tree, monkeypatch, capsys) -> None:
    shard = topobox_tree.geometry_root / "protocol_B" / "train" / "shard_0000.h5"
    monkeypatch.setattr(
        sys, "argv", ["topocal", "inspect-shard", str(shard), "--kind", "geometry", "--sha256"]
    )
    cli.main()
    schema = json.loads(capsys.readouterr().out)
    assert schema["n_samples"] == 1
    assert schema["missing_expected"] == []
    assert len(schema["sha256"]) == 64
    assert schema["sample_fields"]["points"] == {"dtype": "float64", "ndim": 2}
