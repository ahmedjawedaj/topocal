from pathlib import Path

import pytest


@pytest.fixture
def topobox_tree(tmp_path: Path):  # type: ignore[no-untyped-def]
    """A fresh synthetic TopoBox-3D release, safe for tests to mutate."""

    pytest.importorskip("h5py")
    from topobox_fixture import build_tree

    return build_tree(tmp_path / "release")
