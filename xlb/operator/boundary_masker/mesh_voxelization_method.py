# A class used to keep track of the available voxelization methods

from dataclasses import dataclass


# Registry
METHODS = {
    "AABB": 1,
    "RAY": 2,
    "AABB_CLOSE": 3,
    "WINDING": 4,
}


@dataclass
class VoxelizationMethod:
    id: int
    name: str
    options: dict


def MeshVoxelizationMethod(name: str, **options):
    assert name in METHODS.keys(), f"Unsupported voxelization method: {name}"
    return VoxelizationMethod(METHODS[name], name, options)
