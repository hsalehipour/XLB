"""
Mesh voxelization method registry.

Defines the available voxelization strategies (AABB, Ray, AABB-Close,
Winding) and provides a factory function to create the corresponding
:class:`VoxelizationMethod` data object.
"""

from dataclasses import dataclass


METHODS = {
    "AABB": 1,
    "RAY": 2,
    "AABB_CLOSE": 3,
    "WINDING": 4,
}


@dataclass
class VoxelizationMethod:
    """Describes a mesh voxelization strategy.

    Attributes
    ----------
    id : int
        Numeric identifier for the method.
    name : str
        Human-readable name (``"AABB"``, ``"RAY"``, etc.).
    options : dict
        Extra options (e.g. ``close_voxels`` for AABB_CLOSE).
    """

    id: int
    name: str
    options: dict


def MeshVoxelizationMethod(name: str, **options):
    """Create a :class:`VoxelizationMethod` by name.

    Parameters
    ----------
    name : str
        One of ``"AABB"``, ``"RAY"``, ``"AABB_CLOSE"``, ``"WINDING"``.
    **options
        Additional keyword arguments forwarded to
        ``VoxelizationMethod.options``.

    Returns
    -------
    VoxelizationMethod
    """
    assert name in METHODS.keys(), f"Unsupported voxelization method: {name}"
    return VoxelizationMethod(METHODS[name], name, options)
