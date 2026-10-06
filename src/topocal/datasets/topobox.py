"""Manifest-level TopoBox-3D integration.

This module deliberately does not download or load 27GB research assets. It validates the
published geometry manifest and provides deterministic, auditable selection utilities.
Heavy HDF5/mesh loading will live in a separate adapter after baseline reproduction needs
are frozen.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

_VALID_PROTOCOLS = frozenset({"A", "B", "C", "D"})
_VALID_SPLITS = frozenset({"train", "validation", "test_iid", "test_ood"})


@dataclass(frozen=True, slots=True)
class TopoBoxGeometryRecord:
    geometry_id: str
    protocol: str
    split: str
    is_ood: bool
    beta1: int
    beta2: int
    geometry_family: str
    seed: int


@dataclass(frozen=True, slots=True)
class TopoBoxManifest:
    records: tuple[TopoBoxGeometryRecord, ...]
    source_path: Path
    sha256: str

    @classmethod
    def from_csv(cls, path: str | Path) -> "TopoBoxManifest":
        source = Path(path)
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        rows: list[TopoBoxGeometryRecord] = []
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "geometry_id",
                "protocol",
                "split",
                "is_ood",
                "beta1",
                "beta2",
                "geometry_family",
                "seed",
            }
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"TopoBox manifest missing columns: {sorted(missing)}")
            for row in reader:
                rows.append(_parse_record(row))

        _validate_unique_ids(rows)
        return cls(records=tuple(rows), source_path=source, sha256=digest)

    def select(
        self,
        *,
        protocol: str | None = None,
        split: str | None = None,
        is_ood: bool | None = None,
    ) -> tuple[TopoBoxGeometryRecord, ...]:
        if protocol is not None and protocol not in _VALID_PROTOCOLS:
            raise ValueError(f"unknown TopoBox protocol: {protocol}")
        normalized_split = _normalize_split(split) if split is not None else None
        return tuple(
            record
            for record in self.records
            if (protocol is None or record.protocol == protocol)
            and (normalized_split is None or record.split == normalized_split)
            and (is_ood is None or record.is_ood is is_ood)
        )

    def geometry_ids(self, **filters: object) -> tuple[str, ...]:
        return tuple(record.geometry_id for record in self.select(**filters))

    def split_fingerprint(self, **filters: object) -> str:
        """Stable hash of selected geometry IDs, independent of manifest row order."""

        ids = sorted(self.geometry_ids(**filters))
        payload = "\n".join(ids).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def __iter__(self) -> Iterator[TopoBoxGeometryRecord]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)


def _parse_record(row: dict[str, str]) -> TopoBoxGeometryRecord:
    protocol = row["protocol"].strip().upper()
    if protocol not in _VALID_PROTOCOLS:
        raise ValueError(f"invalid TopoBox protocol: {protocol}")
    split = _normalize_split(row["split"])
    is_ood = _parse_bool(row["is_ood"])
    beta1 = int(row["beta1"])
    beta2 = int(row["beta2"])
    if beta1 < 0 or beta2 < 0:
        raise ValueError("Betti numbers must be non-negative")
    geometry_id = row["geometry_id"].strip()
    if not geometry_id:
        raise ValueError("geometry_id cannot be empty")
    return TopoBoxGeometryRecord(
        geometry_id=geometry_id,
        protocol=protocol,
        split=split,
        is_ood=is_ood,
        beta1=beta1,
        beta2=beta2,
        geometry_family=row["geometry_family"].strip(),
        seed=int(row["seed"]),
    )


def _normalize_split(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {"val": "validation", "testiid": "test_iid", "testood": "test_ood"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in _VALID_SPLITS:
        raise ValueError(f"invalid TopoBox split: {value}")
    return normalized


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def _validate_unique_ids(records: Iterable[TopoBoxGeometryRecord]) -> None:
    seen: set[tuple[str, str]] = set()
    for record in records:
        # Geometry IDs may repeat across protocol views in derived manifests, so protocol is
        # part of identity. Within one protocol they must be unique.
        key = (record.protocol, record.geometry_id)
        if key in seen:
            raise ValueError(f"duplicate geometry record: {key}")
        seen.add(key)
