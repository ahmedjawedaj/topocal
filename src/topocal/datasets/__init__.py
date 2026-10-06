"""Dataset adapters. Importing this package never imports h5py or touches the filesystem."""

from topocal.datasets.topobox import TopoBoxGeometryRecord, TopoBoxManifest
from topocal.datasets.topobox_loader import (
    HodgeHeatDegree,
    PDEInstanceKey,
    Sha256Sums,
    SparseCOO,
    TopoBoxDataset,
    TopoBoxGeometry,
    TopoBoxPDEInstance,
    TopoBoxProvenance,
)
from topocal.datasets.topobox_schema import ShardSchema, TopoBoxSchemaError, inspect_shard

__all__ = [
    "HodgeHeatDegree",
    "PDEInstanceKey",
    "ShardSchema",
    "Sha256Sums",
    "SparseCOO",
    "TopoBoxDataset",
    "TopoBoxGeometry",
    "TopoBoxGeometryRecord",
    "TopoBoxManifest",
    "TopoBoxPDEInstance",
    "TopoBoxProvenance",
    "TopoBoxSchemaError",
    "inspect_shard",
]
