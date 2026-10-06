"""HDF5 layout knowledge, validation, and schema inspection for the TopoBox-3D release.

The expected layout below comes from the dataset's own ``DATASET.md`` (Hugging Face
``cppyyy/TopoBox-3D``). It has **not** yet been confirmed against real shards by this project.
Two design rules keep that uncertainty safe:

* validation fails loudly and lists the keys that were actually present
* :func:`inspect_shard` records the observed layout so a run artifact proves what was loaded

``h5py`` is an optional dependency and is imported only when a shard is actually opened.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from topocal.experiments.manifest import file_sha256

SHARD_KIND = Literal["geometry", "hodge_heat"]

HODGE_DEGREES: tuple[int, ...] = (0, 1, 2)
HODGE_HEAT_CONFIGS: tuple[str, ...] = (
    "non_harmonic",
    "weak_harmonic",
    "balanced",
    "strong_harmonic",
)

# Fields whose values come from the final-time solution. They are ground truth and must stay out
# of any online decision path, so the loader only returns them on explicit request.
TARGET_DERIVED_FIELDS: tuple[str, ...] = ("wT", "relative_final_mass_norm")

_SHAPE = tuple[int | None, ...]


class TopoBoxSchemaError(ValueError):
    """Raised when a shard or sample does not match the expected TopoBox-3D layout."""


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Expected dtype family and shape of one HDF5 dataset. ``None`` means any size."""

    kinds: str
    shape: _SHAPE


_FLOAT = "f"
_INT = "iu"
_BOOL = "b"
_TEXT = "US"


def _geometry_fields() -> dict[str, FieldSpec]:
    fields: dict[str, FieldSpec] = {
        "points": FieldSpec(_FLOAT, (None, 3)),
        "normalized_xyz": FieldSpec(_FLOAT, (None, 3)),
        "tetra": FieldSpec(_INT, (None, 4)),
        "oriented_tetra": FieldSpec(_INT, (None, 4)),
        "boundary_triangles": FieldSpec(_INT, (None, 3)),
        "boundary_triangle_mask": FieldSpec(_INT, (None,)),
        "boundary_mask": FieldSpec(_INT, (None,)),
        "is_boundary": FieldSpec(_INT, (None,)),
        "geometry_features": FieldSpec(_FLOAT, (None, 5)),
        "geometry_feature_names": FieldSpec(_TEXT, (5,)),
        "analytic_domain_sdf": FieldSpec(_FLOAT, (None,)),
        "analytic_internal_void_sdf": FieldSpec(_FLOAT, (None,)),
        "discrete_sdf": FieldSpec(_FLOAT, (None,)),
        "tetra_quality": FieldSpec(_FLOAT, (None,)),
        "tetra_volumes": FieldSpec(_FLOAT, (None,)),
        "edges": FieldSpec(_INT, (None, 2)),
        "faces": FieldSpec(_INT, (None, 3)),
        "edge_vectors": FieldSpec(_FLOAT, (None, 3)),
        "edge_lengths": FieldSpec(_FLOAT, (None,)),
        "face_area_vectors": FieldSpec(_FLOAT, (None, 3)),
        "face_areas": FieldSpec(_FLOAT, (None,)),
        "vertex_lumped_volume": FieldSpec(_FLOAT, (None,)),
        "harmonic_basis_0": FieldSpec(_FLOAT, (None, None)),
        "harmonic_basis_1": FieldSpec(_FLOAT, (None, None)),
        "harmonic_basis_2": FieldSpec(_FLOAT, (None, None)),
        "regular_grid_features": FieldSpec(_FLOAT, (32, 16, 16, 5)),
        "has_tunnel": FieldSpec(_BOOL, ()),
        "has_cavity": FieldSpec(_BOOL, ()),
    }
    for degree in (1, 2, 3):
        fields[f"incidence_{degree}_row"] = FieldSpec(_INT, (None,))
        fields[f"incidence_{degree}_col"] = FieldSpec(_INT, (None,))
        fields[f"incidence_{degree}_value"] = FieldSpec(_FLOAT, (None,))
        fields[f"incidence_{degree}_shape"] = FieldSpec(_INT, (2,))
    return fields


GEOMETRY_FIELDS: Mapping[str, FieldSpec] = _geometry_fields()

HODGE_FIELDS: Mapping[str, FieldSpec] = {
    "w0": FieldSpec(_FLOAT, (4, None)),
    "wT": FieldSpec(_FLOAT, (4, None)),
    "mass": FieldSpec(_FLOAT, (None,)),
    "harmonic_basis": FieldSpec(_FLOAT, (None, None)),
    "low_positive_eigenvalues": FieldSpec(_FLOAT, (None,)),
    "requested_energy_fractions": FieldSpec(_FLOAT, (4, 3)),
    "realized_energy_fractions": FieldSpec(_FLOAT, (4, 3)),
    "relative_final_mass_norm": FieldSpec(_FLOAT, (4,)),
    "seeds": FieldSpec(_INT, (4,)),
}

# Fields that share a leading axis, grouped by the simplex family they are indexed over.
_LEADING_AXIS_GROUPS: Mapping[str, tuple[str, ...]] = {
    "vertices": (
        "points",
        "normalized_xyz",
        "boundary_mask",
        "is_boundary",
        "geometry_features",
        "analytic_domain_sdf",
        "analytic_internal_void_sdf",
        "discrete_sdf",
        "vertex_lumped_volume",
        "harmonic_basis_0",
    ),
    "tetrahedra": ("tetra", "oriented_tetra", "tetra_quality", "tetra_volumes"),
    "edges": ("edges", "edge_vectors", "edge_lengths", "harmonic_basis_1"),
    "faces": ("faces", "face_area_vectors", "face_areas", "harmonic_basis_2"),
    "boundary_triangles": ("boundary_triangles", "boundary_triangle_mask"),
}

_INDEX_FIELDS: tuple[str, ...] = (
    "tetra",
    "oriented_tetra",
    "edges",
    "faces",
    "boundary_triangles",
)


def require_h5py() -> Any:
    """Import ``h5py`` on demand with an actionable error message."""

    try:
        return importlib.import_module("h5py")
    except ImportError as exc:
        raise ImportError(
            "h5py is required to read TopoBox-3D shards. "
            "Install it with: pip install 'topocal[data]'"
        ) from exc


def _check_spec(name: str, array: npt.NDArray[Any], spec: FieldSpec, context: str) -> None:
    if array.dtype.kind not in spec.kinds:
        raise TopoBoxSchemaError(
            f"{context}: field {name!r} has dtype {array.dtype}, expected kind in {spec.kinds!r}"
        )
    if array.ndim != len(spec.shape) or any(
        want is not None and have != want
        for have, want in zip(array.shape, spec.shape, strict=False)
    ):
        raise TopoBoxSchemaError(
            f"{context}: field {name!r} has shape {array.shape}, expected {spec.shape}"
        )
    if array.dtype.kind == "f" and not bool(np.all(np.isfinite(array))):
        raise TopoBoxSchemaError(f"{context}: field {name!r} contains non-finite values")


def validate_geometry_arrays(arrays: Mapping[str, npt.NDArray[Any]], *, context: str) -> None:
    """Validate whichever geometry fields are present, including cross-field consistency."""

    for name, array in arrays.items():
        spec = GEOMETRY_FIELDS.get(name)
        if spec is not None:
            _check_spec(name, array, spec, context)

    sizes: dict[str, int] = {}
    for family, names in _LEADING_AXIS_GROUPS.items():
        lengths = {name: int(arrays[name].shape[0]) for name in names if name in arrays}
        if len(set(lengths.values())) > 1:
            raise TopoBoxSchemaError(
                f"{context}: inconsistent {family} count across fields {lengths}"
            )
        if lengths:
            sizes[family] = next(iter(lengths.values()))

    for name in _INDEX_FIELDS:
        if name in arrays and arrays[name].size and int(arrays[name].min()) < 0:
            raise TopoBoxSchemaError(f"{context}: field {name!r} has negative indices")
    if "vertices" in sizes:
        limit = sizes["vertices"]
        for name in _INDEX_FIELDS:
            if name in arrays and arrays[name].size and int(arrays[name].max()) >= limit:
                raise TopoBoxSchemaError(
                    f"{context}: field {name!r} references a vertex outside 0..{limit - 1}"
                )

    families = ("vertices", "edges", "faces", "tetrahedra")
    for degree in (1, 2, 3):
        keys = [f"incidence_{degree}_{part}" for part in ("row", "col", "value", "shape")]
        if not all(key in arrays for key in keys):
            continue
        row, col, value, shape = (arrays[key] for key in keys)
        if not (row.shape == col.shape == value.shape):
            raise TopoBoxSchemaError(f"{context}: incidence_{degree} COO arrays differ in length")
        rows, cols = int(shape[0]), int(shape[1])
        if row.size and (int(row.min()) < 0 or int(row.max()) >= rows):
            raise TopoBoxSchemaError(f"{context}: incidence_{degree} row index out of range")
        if col.size and (int(col.min()) < 0 or int(col.max()) >= cols):
            raise TopoBoxSchemaError(f"{context}: incidence_{degree} column index out of range")
        expected = (sizes.get(families[degree - 1]), sizes.get(families[degree]))
        for have, want in zip((rows, cols), expected, strict=True):
            if want is not None and have != want:
                raise TopoBoxSchemaError(
                    f"{context}: incidence_{degree}_shape {(rows, cols)} disagrees with "
                    f"simplex counts {expected}"
                )


def validate_hodge_arrays(
    arrays: Mapping[str, npt.NDArray[Any]],
    *,
    context: str,
    expected_size: int | None = None,
) -> None:
    """Validate Hodge-heat degree arrays against their spec and the geometry cochain size."""

    for name, array in arrays.items():
        spec = HODGE_FIELDS.get(name)
        if spec is not None:
            _check_spec(name, array, spec, context)
    sizes = {
        name: int(arrays[name].shape[-1 if name in {"w0", "wT"} else 0])
        for name in ("w0", "wT", "mass", "harmonic_basis")
        if name in arrays
    }
    if len(set(sizes.values())) > 1:
        raise TopoBoxSchemaError(f"{context}: inconsistent cochain size across fields {sizes}")
    if expected_size is not None:
        for name, size in sizes.items():
            if size != expected_size:
                raise TopoBoxSchemaError(
                    f"{context}: field {name!r} has {size} entries but the geometry has "
                    f"{expected_size} simplices of this degree"
                )
    if "mass" in arrays and not bool(np.all(arrays["mass"] > 0)):
        raise TopoBoxSchemaError(f"{context}: mass must be strictly positive")


@dataclass(frozen=True, slots=True)
class FieldInfo:
    """Observed dtype and rank of one dataset. Per-sample sizes legitimately vary."""

    dtype: str
    ndim: int


@dataclass(frozen=True, slots=True)
class ShardSchema:
    """Observed structure of one HDF5 shard, recorded for run provenance."""

    name: str
    kind: str | None
    n_samples: int
    shared: dict[str, dict[str, Any]]
    sample_fields: dict[str, FieldInfo]
    missing_expected: tuple[str, ...]
    unexpected: tuple[str, ...]
    inconsistent_samples: tuple[str, ...]
    sha256: str | None = None
    fingerprint: str = field(default="")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "n_samples": self.n_samples,
            "shared": self.shared,
            "sample_fields": {
                key: {"dtype": info.dtype, "ndim": info.ndim}
                for key, info in sorted(self.sample_fields.items())
            },
            "missing_expected": list(self.missing_expected),
            "unexpected": list(self.unexpected),
            "inconsistent_samples": list(self.inconsistent_samples),
            "sha256": self.sha256,
            "fingerprint": self.fingerprint,
        }


def _flatten_datasets(h5py: Any, group: Any) -> dict[str, FieldInfo]:
    found: dict[str, FieldInfo] = {}

    def visit(name: str, node: Any) -> None:
        if isinstance(node, h5py.Dataset):
            found[name] = FieldInfo(dtype=str(node.dtype), ndim=len(node.shape))

    group.visititems(visit)
    return found


def _expected_names(kind: str | None) -> set[str]:
    if kind == "geometry":
        return set(GEOMETRY_FIELDS) | {"metadata"}
    if kind == "hodge_heat":
        return {f"k{degree}/{name}" for degree in HODGE_DEGREES for name in HODGE_FIELDS}
    return set()


def inspect_shard(
    path: str | Path,
    *,
    kind: SHARD_KIND | None = None,
    hash_file: bool = False,
) -> ShardSchema:
    """Record the observed layout of a shard without reading any sample data.

    The reference layout is the first sample (sorted by ID). Samples whose dataset names, dtypes
    or ranks differ from it are listed in ``inconsistent_samples``. When ``kind`` is given the
    reference layout is also compared with the documented one.
    """

    h5py = require_h5py()
    source = Path(path)
    with h5py.File(source, "r") as handle:
        if "samples" not in handle:
            raise TopoBoxSchemaError(f"{source.name}: missing top-level 'samples' group")
        samples = handle["samples"]
        ids = sorted(samples.keys())
        shared: dict[str, dict[str, Any]] = {}
        if "shared" in handle:
            shared = {
                name: {"dtype": info.dtype, "ndim": info.ndim}
                for name, info in _flatten_datasets(h5py, handle["shared"]).items()
            }
            for name in shared:
                shared[name]["shape"] = list(handle["shared"][name].shape)
        reference: dict[str, FieldInfo] = {}
        inconsistent: list[str] = []
        for index, geometry_id in enumerate(ids):
            observed = _flatten_datasets(h5py, samples[geometry_id])
            if index == 0:
                reference = observed
            elif observed != reference:
                inconsistent.append(geometry_id)

    expected = _expected_names(kind)
    present = set(reference)
    missing = tuple(sorted(expected - present - {"metadata"}))
    unexpected = tuple(sorted(present - expected)) if expected else ()
    canonical = json.dumps(
        {
            "shared": shared,
            "sample_fields": {k: [v.dtype, v.ndim] for k, v in sorted(reference.items())},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return ShardSchema(
        name=source.name,
        kind=kind,
        n_samples=len(ids),
        shared=shared,
        sample_fields=reference,
        missing_expected=missing,
        unexpected=unexpected,
        inconsistent_samples=tuple(inconsistent),
        sha256=file_sha256(source) if hash_file else None,
        fingerprint=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )
