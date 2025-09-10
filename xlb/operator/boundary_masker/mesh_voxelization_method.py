# Enum used to keep track of the available voxelization methods

from enum import Enum, auto


class MeshVoxelizationMethod(Enum):
    AABB = auto()
    RAY = auto()
    AABB_CLOSE = auto()
    WINDING = auto()
