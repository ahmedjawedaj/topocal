"""Tiny synthetic TopoBox-3D tree used by tests so CI never needs the real 27 GB release.

The layout mirrors the release's ``DATASET.md`` (packed geometry shards plus Hodge-heat shards,
sharded differently on purpose). Incidence operators are exact for the toy tetrahedral complexes,
so ``B1 @ B2 == 0`` and ``B2 @ B3 == 0`` hold. Harmonic bases and PDE values are structurally
faithful but not mathematically meaningful.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from topocal.datasets.topobox import TopoBoxManifest
from topocal.experiments.splits import SplitManifest

MANIFEST_CSV = """geometry_id,protocol,split,is_ood,beta1,beta2,geometry_family,seed
PB_train_0000_b00,B,train,false,0,0,B,1
PB_train_0001_b10,B,train,false,1,0,B,2
PB_validation_0000_b00,B,validation,false,0,0,B,3
PB_test_ood_0000_b30,B,test_ood,true,3,0,B,4
PC_test_ood_0000_b03,C,test_ood,true,0,3,C,5
"""

# geometry_id -> tetrahedra (vertex indices). Different sizes so array lengths vary per sample.
TETRA: dict[str, list[tuple[int, int, int, int]]] = {
    "PB_train_0000_b00": [(0, 1, 2, 3), (1, 2, 3, 4)],
    "PB_train_0001_b10": [(0, 1, 2, 3)],
    "PB_validation_0000_b00": [(0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5)],
    "PB_test_ood_0000_b30": [(0, 1, 2, 3), (0, 1, 2, 4)],
    "PC_test_ood_0000_b03": [(0, 1, 2, 3), (1, 2, 3, 4)],
}
BETTI = {
    "PB_train_0000_b00": (0, 0),
    "PB_train_0001_b10": (1, 0),
    "PB_validation_0000_b00": (0, 0),
    "PB_test_ood_0000_b30": (3, 0),
    "PC_test_ood_0000_b03": (0, 3),
}
# (protocol, physical split) -> list of shards, each a list of geometry IDs.
GEOMETRY_SHARDS: dict[tuple[str, str], list[list[str]]] = {
    ("B", "train"): [["PB_train_0000_b00"], ["PB_train_0001_b10"]],
    ("B", "validation"): [["PB_validation_0000_b00"]],
    ("B", "test_ood"): [["PB_test_ood_0000_b30"]],
    ("C", "test_ood"): [["PC_test_ood_0000_b03"]],
}
SOLUTION_SHARDS: dict[tuple[str, str], list[list[str]]] = {
    ("B", "train"): [["PB_train_0000_b00", "PB_train_0001_b10"]],
    ("B", "validation"): [["PB_validation_0000_b00"]],
    ("B", "test_ood"): [["PB_test_ood_0000_b30"]],
    ("C", "test_ood"): [["PC_test_ood_0000_b03"]],
}


@dataclass(frozen=True)
class FixtureTree:
    root: Path
    geometry_root: Path
    solution_root: Path
    manifest_path: Path
    manifest: TopoBoxManifest
    split_manifest: SplitManifest


def build_complex(tetra: list[tuple[int, int, int, int]]) -> dict[str, Any]:
    """Exact oriented simplicial complex (edges, faces, B1, B2, B3) of a list of tetrahedra."""

    tets = [tuple(sorted(t)) for t in tetra]
    n_vertices = max(max(t) for t in tets) + 1
    edges = sorted({pair for t in tets for pair in itertools.combinations(t, 2)})
    faces = sorted({tri for t in tets for tri in itertools.combinations(t, 3)})
    edge_index = {e: i for i, e in enumerate(edges)}
    face_index = {f: i for i, f in enumerate(faces)}

    def coo(entries: list[tuple[int, int, float]], shape: tuple[int, int]) -> dict[str, Any]:
        return {
            "row": np.array([e[0] for e in entries], dtype=np.int64),
            "col": np.array([e[1] for e in entries], dtype=np.int64),
            "value": np.array([e[2] for e in entries], dtype=np.float64),
            "shape": np.array(shape, dtype=np.int64),
        }

    b1 = [(e[0], edge_index[e], -1.0) for e in edges]
    b1 += [(e[1], edge_index[e], 1.0) for e in edges]
    b2: list[tuple[int, int, float]] = []
    for (a, b, c), j in face_index.items():
        b2 += [(edge_index[(b, c)], j, 1.0), (edge_index[(a, c)], j, -1.0)]
        b2 += [(edge_index[(a, b)], j, 1.0)]
    b3: list[tuple[int, int, float]] = []
    for k, (a, b, c, d) in enumerate(tets):
        b3 += [(face_index[(b, c, d)], k, 1.0), (face_index[(a, c, d)], k, -1.0)]
        b3 += [(face_index[(a, b, d)], k, 1.0), (face_index[(a, b, c)], k, -1.0)]
    counts = np.zeros(len(faces), dtype=int)
    for t in tets:
        for tri in itertools.combinations(t, 3):
            counts[face_index[tri]] += 1
    boundary = [f for f, c in zip(faces, counts, strict=True) if c == 1]
    return {
        "n_vertices": n_vertices,
        "tetra": np.array(tets, dtype=np.int64),
        "edges": np.array(edges, dtype=np.int64),
        "faces": np.array(faces, dtype=np.int64),
        "boundary_triangles": np.array(boundary, dtype=np.int64).reshape(-1, 3),
        "incidence": {
            1: coo(b1, (n_vertices, len(edges))),
            2: coo(b2, (len(edges), len(faces))),
            3: coo(b3, (len(faces), len(tets))),
        },
    }


def geometry_sample(geometry_id: str) -> dict[str, Any]:
    """All array datasets of one packed geometry sample, as documented in DATASET.md."""

    rng = np.random.default_rng(sum(geometry_id.encode()))
    cx = build_complex(TETRA[geometry_id])
    n, e, f, t = (cx["n_vertices"], len(cx["edges"]), len(cx["faces"]), len(cx["tetra"]))
    points = rng.uniform(0.0, 1.0, size=(n, 3)) * np.array([2.0, 1.0, 1.0])
    boundary_nodes = np.unique(cx["boundary_triangles"])
    is_boundary = np.zeros(n, dtype=np.int64)
    is_boundary[boundary_nodes] = 1
    sdf = rng.normal(size=n)
    normalized = points / np.array([2.0, 1.0, 1.0])
    edge_vectors = points[cx["edges"][:, 1]] - points[cx["edges"][:, 0]]
    beta1, beta2 = BETTI[geometry_id]
    sample: dict[str, Any] = {
        "points": points,
        "normalized_xyz": normalized,
        "tetra": cx["tetra"],
        "oriented_tetra": cx["tetra"].copy(),
        "boundary_triangles": cx["boundary_triangles"],
        "boundary_triangle_mask": np.ones(len(cx["boundary_triangles"]), dtype=np.int64),
        "boundary_mask": is_boundary.copy(),
        "is_boundary": is_boundary,
        "geometry_features": np.column_stack([normalized, is_boundary, sdf]),
        "geometry_feature_names": np.array(
            ["x_normalized", "y_normalized", "z_normalized", "is_boundary", "analytic_domain_sdf"],
            dtype=object,
        ),
        "analytic_domain_sdf": sdf,
        "analytic_internal_void_sdf": rng.normal(size=n),
        "discrete_sdf": rng.normal(size=n),
        "tetra_quality": rng.uniform(0.2, 1.0, size=t),
        "tetra_volumes": rng.uniform(0.01, 0.1, size=t),
        "edges": cx["edges"],
        "faces": cx["faces"],
        "edge_vectors": edge_vectors,
        "edge_lengths": np.linalg.norm(edge_vectors, axis=1),
        "face_area_vectors": rng.normal(size=(f, 3)),
        "face_areas": rng.uniform(0.01, 0.2, size=f),
        "vertex_lumped_volume": rng.uniform(0.01, 0.1, size=n),
        "harmonic_basis_0": np.ones((n, 1)) / np.sqrt(n),
        "harmonic_basis_1": rng.normal(size=(e, beta1)),
        "harmonic_basis_2": rng.normal(size=(f, beta2)),
        "regular_grid_features": rng.normal(size=(32, 16, 16, 5)),
        "has_tunnel": np.array(beta1 > 0),
        "has_cavity": np.array(beta2 > 0),
    }
    for degree, parts in cx["incidence"].items():
        for part, value in parts.items():
            sample[f"incidence_{degree}_{part}"] = value
    return sample


def _mass_norm(values: np.ndarray, mass: np.ndarray) -> np.ndarray:
    return np.sqrt((mass * values**2).sum(axis=1))


def hodge_sample(geometry_id: str) -> dict[str, dict[str, np.ndarray]]:
    """k0/k1/k2 groups of one Hodge-heat sample, sized from the same complex."""

    cx = build_complex(TETRA[geometry_id])
    sizes = [cx["n_vertices"], len(cx["edges"]), len(cx["faces"])]
    rng = np.random.default_rng(sum(geometry_id.encode()) + 1)
    betti = (1, *BETTI[geometry_id])
    out: dict[str, dict[str, np.ndarray]] = {}
    for degree, size in enumerate(sizes):
        mass = rng.uniform(0.5, 1.5, size=size)
        w0 = rng.normal(size=(4, size))
        w_t = w0 * 0.9
        fractions = rng.dirichlet(np.ones(3), size=4)
        out[f"k{degree}"] = {
            "w0": w0,
            "wT": w_t,
            "mass": mass,
            "harmonic_basis": rng.normal(size=(size, betti[degree])),
            "low_positive_eigenvalues": np.sort(rng.uniform(0.1, 2.0, size=min(6, size))),
            "requested_energy_fractions": fractions,
            "realized_energy_fractions": fractions,
            "relative_final_mass_norm": _mass_norm(w_t, mass) / _mass_norm(w0, mass),
            "seeds": np.arange(4, dtype=np.int64) + 100 * degree,
        }
    return out


def _write_value(group: h5py.Group, name: str, value: np.ndarray) -> None:
    if value.dtype == object:
        group.create_dataset(name, data=value, dtype=h5py.string_dtype())
    else:
        group.create_dataset(name, data=value)


def write_geometry_shard(path: Path, geometry_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        shared = handle.create_group("shared")
        shared.create_dataset("regular_grid_normalized_xyz", data=np.zeros((32, 16, 16, 3)))
        _write_value(
            shared,
            "regular_grid_feature_names",
            np.array(["a", "b", "c", "d", "e"], dtype=object),
        )
        samples = handle.create_group("samples")
        for geometry_id in geometry_ids:
            group = samples.create_group(geometry_id)
            for name, value in geometry_sample(geometry_id).items():
                _write_value(group, name, np.asarray(value))
            group.create_dataset(
                "metadata",
                data=json.dumps({"geometry_id": geometry_id, "seed": 7}),
                dtype=h5py.string_dtype(),
            )


def write_solution_shard(path: Path, geometry_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        samples = handle.create_group("samples")
        for geometry_id in geometry_ids:
            sample = samples.create_group(geometry_id)
            for degree_name, arrays in hodge_sample(geometry_id).items():
                group = sample.create_group(degree_name)
                for name, value in arrays.items():
                    group.create_dataset(name, data=value)


def build_tree(root: Path) -> FixtureTree:
    """Write the full synthetic release under ``root`` and return handles to it."""

    geometry_root = root / "TopoBox-3D" / "packed"
    solution_root = root / "TopoBox-3D-HodgeHeat"
    for (protocol, split), shards in GEOMETRY_SHARDS.items():
        for number, ids in enumerate(shards):
            write_geometry_shard(
                geometry_root / f"protocol_{protocol}" / split / f"shard_{number:04d}.h5", ids
            )
    for (protocol, split), shards in SOLUTION_SHARDS.items():
        for number, ids in enumerate(shards):
            write_solution_shard(
                solution_root / f"protocol_{protocol}" / split / f"shard_{number:04d}.h5", ids
            )
    manifest_path = root / "manifest.csv"
    manifest_path.write_text(MANIFEST_CSV, encoding="utf-8")
    manifest = TopoBoxManifest.from_csv(manifest_path)
    # Logical partitions deliberately differ from the release's physical splits.
    split_manifest = SplitManifest(
        dataset_fingerprint=manifest.sha256,
        train_ids=("PB_train_0000_b00", "PB_train_0001_b10"),
        calibration_ids=("PB_validation_0000_b00",),
        ood_dev_ids=("PB_test_ood_0000_b30",),
        ood_test_ids=("PC_test_ood_0000_b03",),
        seed=7,
    )
    return FixtureTree(
        root=root,
        geometry_root=geometry_root,
        solution_root=solution_root,
        manifest_path=manifest_path,
        manifest=manifest,
        split_manifest=split_manifest,
    )
