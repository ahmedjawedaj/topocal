"""Lazy, partition-bound loading of TopoBox-3D geometry and Hodge-heat samples.

Design rules (see ADR 0008):

* constructing a :class:`TopoBoxDataset` opens no files and writes nothing
* IDs come only from a :class:`~topocal.experiments.splits.SplitManifest` partition, so the
  locked final OOD test cannot be reached without an explicit ``allow_final_test=True``
* every ID is resolved through the :class:`~topocal.datasets.topobox.TopoBoxManifest`
* shards are opened read-only, one sample at a time, and closed immediately
* final-time solutions (ground truth) are returned only when ``include_target=True``
* the shards actually read are recorded so a run can state exactly which bytes it used
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from topocal.datasets.topobox import TopoBoxGeometryRecord, TopoBoxManifest
from topocal.datasets.topobox_schema import (
    HODGE_DEGREES,
    HODGE_HEAT_CONFIGS,
    TARGET_DERIVED_FIELDS,
    TopoBoxSchemaError,
    inspect_shard,
    require_h5py,
    validate_geometry_arrays,
    validate_hodge_arrays,
)
from topocal.experiments.manifest import file_sha256
from topocal.experiments.splits import ExperimentPartition, SplitManifest
from topocal.types import Problem

_COCHAIN_FIELD = {0: "points", 1: "edges", 2: "faces"}
_SUM_LINE = re.compile(r"^([0-9a-fA-F]{64}) [ *](.+)$")


@dataclass(frozen=True, slots=True, eq=False)
class SparseCOO:
    """A COO sparse matrix. Used for the oriented incidence (boundary) operators."""

    row: npt.NDArray[Any]
    col: npt.NDArray[Any]
    value: npt.NDArray[Any]
    shape: tuple[int, int]

    def to_dense(self) -> npt.NDArray[np.float64]:
        dense = np.zeros(self.shape, dtype=np.float64)
        np.add.at(dense, (self.row, self.col), self.value)
        return dense


@dataclass(frozen=True, slots=True, eq=False)
class TopoBoxGeometry:
    """Geometry arrays of one TopoBox-3D domain. ``split`` is the release's physical split."""

    geometry_id: str
    protocol: str
    split: str
    arrays: Mapping[str, npt.NDArray[Any]]
    metadata: dict[str, Any] | None = None

    def __getitem__(self, name: str) -> npt.NDArray[Any]:
        try:
            return self.arrays[name]
        except KeyError:
            raise KeyError(
                f"field {name!r} was not loaded for {self.geometry_id}, "
                f"loaded fields: {sorted(self.arrays)}"
            ) from None

    def cochain_size(self, degree: int) -> int:
        """Number of vertices (0), edges (1), or faces (2)."""

        if degree not in _COCHAIN_FIELD:
            raise ValueError(f"degree must be one of {sorted(_COCHAIN_FIELD)}, got {degree}")
        return int(self[_COCHAIN_FIELD[degree]].shape[0])

    def incidence(self, degree: int) -> SparseCOO:
        """Oriented incidence operator ``B_degree`` (degree 1 to 3) as a COO matrix."""

        if degree not in (1, 2, 3):
            raise ValueError(f"incidence degree must be 1, 2 or 3, got {degree}")
        prefix = f"incidence_{degree}"
        shape = self[f"{prefix}_shape"]
        return SparseCOO(
            row=self[f"{prefix}_row"],
            col=self[f"{prefix}_col"],
            value=self[f"{prefix}_value"],
            shape=(int(shape[0]), int(shape[1])),
        )


@dataclass(frozen=True, slots=True)
class PDEInstanceKey:
    """One Hodge-heat instance: a geometry, a cochain degree, and an initial-condition type."""

    geometry_id: str
    degree: int
    config: str

    def __post_init__(self) -> None:
        if self.degree not in HODGE_DEGREES:
            raise ValueError(f"degree must be one of {HODGE_DEGREES}, got {self.degree}")
        if self.config not in HODGE_HEAT_CONFIGS:
            raise ValueError(f"config must be one of {HODGE_HEAT_CONFIGS}, got {self.config!r}")

    @property
    def problem_id(self) -> str:
        return f"{self.geometry_id}:k{self.degree}:{self.config}"


@dataclass(frozen=True, slots=True, eq=False)
class HodgeHeatDegree:
    """All four initial-condition configs of one geometry at one cochain degree.

    ``target`` (the final-time solution) is ``None`` unless it was explicitly requested.
    """

    geometry_id: str
    degree: int
    configs: tuple[str, ...]
    w0: npt.NDArray[Any]
    mass: npt.NDArray[Any]
    harmonic_basis: npt.NDArray[Any]
    target: npt.NDArray[Any] | None
    aux: Mapping[str, npt.NDArray[Any]]


@dataclass(frozen=True, slots=True, eq=False)
class TopoBoxPDEInstance:
    """A single PDE query, split into inputs and (optional) ground-truth target."""

    key: PDEInstanceKey
    w0: npt.NDArray[Any]
    mass: npt.NDArray[Any]
    harmonic_basis: npt.NDArray[Any]
    geometry: TopoBoxGeometry
    target: npt.NDArray[Any] | None = None

    def to_problem(self) -> Problem:
        """Opaque core ``Problem`` for the online path. The target is always stripped."""

        return Problem(problem_id=self.key.problem_id, payload=replace(self, target=None))


@dataclass(frozen=True, slots=True)
class Sha256Sums:
    """Parsed ``sha256sum``-style file, for example the release's ``SHA256SUMS.txt``."""

    entries: Mapping[str, str]

    @classmethod
    def from_file(cls, path: str | Path) -> Sha256Sums:
        entries: dict[str, str] = {}
        text = Path(path).read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            match = _SUM_LINE.match(line)
            if match is None:
                raise ValueError(f"malformed checksum line {number}: {line[:80]!r}")
            entries[match.group(2).strip()] = match.group(1).lower()
        return cls(entries=entries)

    def status(self, suffix: str, actual_sha256: str) -> str:
        """Compare a hash with the published one. Paths are matched by trailing components."""

        found = [p for p in self.entries if p == suffix or p.endswith("/" + suffix)]
        if not found:
            return "unlisted"
        if len(found) > 1:
            return "ambiguous"
        return "match" if self.entries[found[0]] == actual_sha256.lower() else "mismatch"


@dataclass(frozen=True, slots=True)
class TopoBoxProvenance:
    """What a run actually read: manifest hash, split identity, shard hashes and layouts."""

    manifest_sha256: str
    split_fingerprint: str
    partition: str
    n_ids: int
    ids_sha256: str
    files_sha256: dict[str, str | None]
    published_sha256_status: dict[str, str]
    schemas: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def _decode_text(value: Any) -> npt.NDArray[Any]:
    array = np.asarray(value)
    if array.dtype.kind in "OS":
        items = array.ravel().tolist()
        flat = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in items]
        return np.array(flat, dtype=str).reshape(array.shape)
    return array


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_metadata(h5py: Any, group: Any, context: str) -> dict[str, Any]:
    node = group.get("metadata")
    if node is None:
        raise TopoBoxSchemaError(f"{context}: metadata requested but absent")
    if isinstance(node, h5py.Dataset):
        text = _decode_text(node[()])
        if text.ndim != 0:
            raise TopoBoxSchemaError(f"{context}: metadata dataset must be a scalar JSON string")
        try:
            parsed = json.loads(str(text))
        except json.JSONDecodeError as exc:
            raise TopoBoxSchemaError(f"{context}: metadata is not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise TopoBoxSchemaError(f"{context}: metadata JSON must be an object")
        return parsed
    return {key: _jsonable(value) for key, value in node.attrs.items()}


class TopoBoxDataset:
    """One experiment partition of TopoBox-3D, resolved through the manifest and split guard.

    ``geometry_root`` is the release's ``TopoBox-3D/packed`` directory and ``solution_root`` is
    ``TopoBox-3D-HodgeHeat``. Both are expected to contain
    ``protocol_<P>/<split>/shard_<NNNN>.h5``. ``split_manifest.dataset_fingerprint`` must equal
    ``manifest.sha256`` so a split can never be applied to a different manifest than it was
    frozen against.

    ``partition`` accepts an :class:`ExperimentPartition` or its exact string value. It is
    normalized to the enum before the final-test guard runs, so ``"ood_test"`` is locked exactly
    like ``ExperimentPartition.OOD_TEST``. No file is opened before that check.
    """

    def __init__(
        self,
        *,
        manifest: TopoBoxManifest,
        split_manifest: SplitManifest,
        partition: ExperimentPartition | str,
        geometry_root: str | Path,
        solution_root: str | Path | None = None,
        allow_final_test: bool = False,
        float_dtype: npt.DTypeLike | None = None,
    ) -> None:
        partition = ExperimentPartition.parse(partition)
        ids = split_manifest.ids_for(partition, allow_final_test=allow_final_test)
        if split_manifest.dataset_fingerprint != manifest.sha256:
            raise ValueError(
                "split manifest was frozen against a different TopoBox manifest "
                f"({split_manifest.dataset_fingerprint[:12]} vs {manifest.sha256[:12]})"
            )
        by_id: dict[str, list[TopoBoxGeometryRecord]] = {}
        for record in manifest.records:
            by_id.setdefault(record.geometry_id, []).append(record)
        records: dict[str, TopoBoxGeometryRecord] = {}
        unknown: list[str] = []
        for geometry_id in ids:
            if "/" in geometry_id:
                raise ValueError(f"geometry ID cannot contain '/': {geometry_id!r}")
            candidates = by_id.get(geometry_id)
            if not candidates:
                unknown.append(geometry_id)
            elif len(candidates) > 1:
                raise ValueError(
                    f"geometry ID {geometry_id!r} is ambiguous across protocols "
                    f"{sorted(c.protocol for c in candidates)}"
                )
            else:
                records[geometry_id] = candidates[0]
        if unknown:
            raise ValueError(
                f"{len(unknown)} split IDs are not in the TopoBox manifest, e.g. {unknown[:3]}"
            )
        self._manifest = manifest
        self._split = split_manifest
        self._partition = partition
        self._ids = tuple(ids)
        self._records = records
        self._geometry_root = Path(geometry_root)
        self._solution_root = Path(solution_root) if solution_root is not None else None
        self._float_dtype = float_dtype
        self._shard_index: dict[tuple[str, str, str], dict[str, Path]] = {}
        self._touched: dict[str, tuple[str, Path]] = {}
        self._hash_cache: dict[Path, str] = {}

    @property
    def partition(self) -> ExperimentPartition:
        return self._partition

    @property
    def ids(self) -> tuple[str, ...]:
        return self._ids

    def record(self, geometry_id: str) -> TopoBoxGeometryRecord:
        """Manifest record of a geometry in this partition (topology labels live here)."""

        return self._member(geometry_id)

    def __len__(self) -> int:
        return len(self._ids)

    def __iter__(self) -> Iterator[str]:
        return iter(self._ids)

    def instance_keys(self) -> tuple[PDEInstanceKey, ...]:
        """Every (geometry, degree, config) instance, in deterministic order."""

        return tuple(
            PDEInstanceKey(geometry_id, degree, config)
            for geometry_id in self._ids
            for degree in HODGE_DEGREES
            for config in HODGE_HEAT_CONFIGS
        )

    def load_geometry(
        self,
        geometry_id: str,
        *,
        fields: Sequence[str] | None = None,
        include_metadata: bool = False,
    ) -> TopoBoxGeometry:
        """Read geometry arrays. ``fields=None`` loads every array dataset of the sample."""

        record = self._member(geometry_id)
        shard = self._shard_for("geometry", geometry_id)
        h5py = require_h5py()
        context = f"geometry {geometry_id} in {shard.name}"
        with h5py.File(shard, "r") as handle:
            group = handle["samples"][geometry_id]
            arrays = self._read_fields(h5py, group, fields, exclude=(), context=context)
            metadata = _read_metadata(h5py, group, context) if include_metadata else None
        validate_geometry_arrays(arrays, context=context)
        self._touch("geometry", shard)
        return TopoBoxGeometry(
            geometry_id=geometry_id,
            protocol=record.protocol,
            split=record.split,
            arrays=MappingProxyType(arrays),
            metadata=metadata,
        )

    def load_hodge_heat(
        self,
        geometry_id: str,
        degree: int,
        *,
        include_target: bool = False,
    ) -> HodgeHeatDegree:
        """Read the four initial-condition configs of one degree. Ground truth is opt-in."""

        if degree not in HODGE_DEGREES:
            raise ValueError(f"degree must be one of {HODGE_DEGREES}, got {degree}")
        self._member(geometry_id)
        shard = self._shard_for("solution", geometry_id)
        h5py = require_h5py()
        context = f"hodge-heat k{degree} for {geometry_id} in {shard.name}"
        exclude = () if include_target else TARGET_DERIVED_FIELDS
        with h5py.File(shard, "r") as handle:
            sample = handle["samples"][geometry_id]
            name = f"k{degree}"
            if name not in sample:
                raise TopoBoxSchemaError(f"{context}: missing group {name!r}")
            arrays = self._read_fields(h5py, sample[name], None, exclude=exclude, context=context)
        required = ["w0", "mass", "harmonic_basis", *(["wT"] if include_target else [])]
        missing = [key for key in required if key not in arrays]
        if missing:
            raise TopoBoxSchemaError(
                f"{context}: missing fields {missing}, present: {sorted(arrays)}"
            )
        expected = self._geometry_cochain_size(geometry_id, degree)
        validate_hodge_arrays(arrays, context=context, expected_size=expected)
        self._touch("solution", shard)
        main = {"w0", "mass", "harmonic_basis", "wT"}
        return HodgeHeatDegree(
            geometry_id=geometry_id,
            degree=degree,
            configs=HODGE_HEAT_CONFIGS,
            w0=arrays["w0"],
            mass=arrays["mass"],
            harmonic_basis=arrays["harmonic_basis"],
            target=arrays.get("wT") if include_target else None,
            aux=MappingProxyType({k: v for k, v in arrays.items() if k not in main}),
        )

    def load_instance(
        self,
        key: PDEInstanceKey,
        *,
        include_target: bool = False,
        geometry_fields: Sequence[str] | None = None,
    ) -> TopoBoxPDEInstance:
        """Load one PDE instance. Pass ``include_target=True`` only for training or scoring."""

        geometry = self.load_geometry(key.geometry_id, fields=geometry_fields)
        degree = self.load_hodge_heat(key.geometry_id, key.degree, include_target=include_target)
        index = HODGE_HEAT_CONFIGS.index(key.config)
        target = degree.target[index] if degree.target is not None else None
        return TopoBoxPDEInstance(
            key=key,
            w0=degree.w0[index],
            mass=degree.mass,
            harmonic_basis=degree.harmonic_basis,
            geometry=geometry,
            target=target,
        )

    def provenance(
        self,
        *,
        hash_shards: bool = True,
        published_sums: Sha256Sums | None = None,
    ) -> TopoBoxProvenance:
        """Describe the data actually read so far, for the experiment manifest.

        Only shards that were read are hashed, so a run never pays for the full 27 GB release.
        """

        files: dict[str, str | None] = {}
        status: dict[str, str] = {}
        schemas: dict[str, dict[str, Any]] = {}
        for key, (kind, path) in sorted(self._touched.items()):
            digest: str | None = None
            if hash_shards:
                digest = self._hash_cache.get(path)
                if digest is None:
                    digest = file_sha256(path)
                    self._hash_cache[path] = digest
            files[key] = digest
            if digest is not None and published_sums is not None:
                relative = key.split("/", 1)[1]
                suffix = f"packed/{relative}" if kind == "geometry" else (
                    f"TopoBox-3D-HodgeHeat/{relative}"
                )
                status[key] = published_sums.status(suffix, digest)
            shard_kind: Literal["geometry", "hodge_heat"] = (
                "geometry" if kind == "geometry" else "hodge_heat"
            )
            schemas[key] = inspect_shard(path, kind=shard_kind).to_dict()
        return TopoBoxProvenance(
            manifest_sha256=self._manifest.sha256,
            split_fingerprint=self._split.fingerprint,
            partition=self._partition.value,
            n_ids=len(self._ids),
            ids_sha256=hashlib.sha256("\n".join(sorted(self._ids)).encode("utf-8")).hexdigest(),
            files_sha256=files,
            published_sha256_status=status,
            schemas=schemas,
        )

    def _member(self, geometry_id: str) -> TopoBoxGeometryRecord:
        try:
            return self._records[geometry_id]
        except KeyError:
            raise PermissionError(
                f"geometry {geometry_id!r} is not part of partition {self._partition.value!r}"
            ) from None

    def _touch(self, kind: str, shard: Path) -> None:
        root = self._geometry_root if kind == "geometry" else self._solution_root
        assert root is not None
        self._touched[f"{kind}/{shard.relative_to(root).as_posix()}"] = (kind, shard)

    def _shard_for(self, kind: str, geometry_id: str) -> Path:
        record = self._records[geometry_id]
        root = self._geometry_root if kind == "geometry" else self._solution_root
        if root is None:
            raise ValueError("solution_root was not provided, so Hodge-heat data is unavailable")
        index_key = (kind, record.protocol, record.split)
        index = self._shard_index.get(index_key)
        directory = root / f"protocol_{record.protocol}" / record.split
        if index is None:
            index = self._scan_directory(kind, directory)
            self._shard_index[index_key] = index
        try:
            return index[geometry_id]
        except KeyError:
            raise LookupError(
                f"{geometry_id} is in the manifest but in no {kind} shard under {directory}"
            ) from None

    @staticmethod
    def _scan_directory(kind: str, directory: Path) -> dict[str, Path]:
        h5py = require_h5py()
        if not directory.is_dir():
            raise FileNotFoundError(f"TopoBox {kind} directory not found: {directory}")
        shards = sorted(directory.glob("shard_*.h5"))
        if not shards:
            raise FileNotFoundError(f"no shard_*.h5 files in {directory}")
        index: dict[str, Path] = {}
        for shard in shards:
            with h5py.File(shard, "r") as handle:
                if "samples" not in handle:
                    raise TopoBoxSchemaError(f"{shard.name}: missing top-level 'samples' group")
                for geometry_id in handle["samples"]:
                    if geometry_id in index:
                        raise TopoBoxSchemaError(
                            f"{geometry_id} appears in both {index[geometry_id].name} "
                            f"and {shard.name}"
                        )
                    index[geometry_id] = shard
        return index

    def _geometry_cochain_size(self, geometry_id: str, degree: int) -> int:
        shard = self._shard_for("geometry", geometry_id)
        h5py = require_h5py()
        name = _COCHAIN_FIELD[degree]
        with h5py.File(shard, "r") as handle:
            group = handle["samples"][geometry_id]
            if name not in group:
                raise TopoBoxSchemaError(
                    f"geometry {geometry_id} in {shard.name}: missing field {name!r}"
                )
            size = int(group[name].shape[0])
        self._touch("geometry", shard)
        return size

    def _read_fields(
        self,
        h5py: Any,
        group: Any,
        fields: Sequence[str] | None,
        *,
        exclude: Sequence[str],
        context: str,
    ) -> dict[str, npt.NDArray[Any]]:
        available = sorted(name for name, node in group.items() if isinstance(node, h5py.Dataset))
        if fields is None:
            names = [n for n in available if n not in exclude and n != "metadata"]
        else:
            names = list(fields)
            missing = [n for n in names if n not in available]
            if missing:
                raise TopoBoxSchemaError(
                    f"{context}: missing fields {missing}, available: {available}"
                )
        arrays: dict[str, npt.NDArray[Any]] = {}
        for name in names:
            array = _decode_text(group[name][()])
            if self._float_dtype is not None and array.dtype.kind == "f":
                array = array.astype(self._float_dtype)
            array.setflags(write=False)
            arrays[name] = array
        return arrays
