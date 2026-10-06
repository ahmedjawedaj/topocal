"""Experiment split manifests and final-test access guards."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path


class ExperimentPartition(StrEnum):
    """Logical partitions used by TopoCal research experiments."""

    TRAIN = "train"
    CALIBRATION = "calibration"
    OOD_DEV = "ood_dev"
    OOD_TEST = "ood_test"


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Immutable sample/geometry assignments with guarded final-test access.

    IDs should identify the independent resampling unit. For TopoBox primary experiments this
    should normally be the geometry ID, not individual PDE instances, to prevent leakage and
    pseudo-replication across partitions.
    """

    dataset_fingerprint: str
    train_ids: tuple[str, ...]
    calibration_ids: tuple[str, ...]
    ood_dev_ids: tuple[str, ...]
    ood_test_ids: tuple[str, ...]
    seed: int
    generation_version: str = "1"

    def __post_init__(self) -> None:
        if not self.dataset_fingerprint.strip():
            raise ValueError("dataset_fingerprint cannot be blank")
        if not self.generation_version.strip():
            raise ValueError("generation_version cannot be blank")
        groups = {
            ExperimentPartition.TRAIN: self.train_ids,
            ExperimentPartition.CALIBRATION: self.calibration_ids,
            ExperimentPartition.OOD_DEV: self.ood_dev_ids,
            ExperimentPartition.OOD_TEST: self.ood_test_ids,
        }
        seen: dict[str, ExperimentPartition] = {}
        for partition, ids in groups.items():
            if len(set(ids)) != len(ids):
                raise ValueError(f"duplicate IDs inside {partition.value}")
            for item in ids:
                if not item.strip():
                    raise ValueError(f"blank ID inside {partition.value}")
                previous = seen.get(item)
                if previous is not None:
                    raise ValueError(
                        f"ID {item!r} appears in both {previous.value} and {partition.value}"
                    )
                seen[item] = partition

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def ids_for(
        self,
        partition: ExperimentPartition,
        *,
        allow_final_test: bool = False,
    ) -> tuple[str, ...]:
        """Return IDs for a partition, guarding final OOD test by default."""

        if partition is ExperimentPartition.OOD_TEST and not allow_final_test:
            raise PermissionError(
                "final OOD test access is locked; pass allow_final_test=True "
                "only for frozen reporting"
            )
        return {
            ExperimentPartition.TRAIN: self.train_ids,
            ExperimentPartition.CALIBRATION: self.calibration_ids,
            ExperimentPartition.OOD_DEV: self.ood_dev_ids,
            ExperimentPartition.OOD_TEST: self.ood_test_ids,
        }[partition]

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "SplitManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            dataset_fingerprint=str(data["dataset_fingerprint"]),
            train_ids=tuple(str(item) for item in data["train_ids"]),
            calibration_ids=tuple(str(item) for item in data["calibration_ids"]),
            ood_dev_ids=tuple(str(item) for item in data["ood_dev_ids"]),
            ood_test_ids=tuple(str(item) for item in data["ood_test_ids"]),
            seed=int(data["seed"]),
            generation_version=str(data.get("generation_version", "1")),
        )
