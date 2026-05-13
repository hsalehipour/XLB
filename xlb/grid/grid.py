"""
Grid abstraction and factory functions for XLB.

Defines the :class:`Grid` abstract base class that every backend-specific
grid must implement, plus two factory helpers:

* :func:`grid_factory` — creates a single-resolution grid for any backend.
* :func:`multires_grid_factory` — creates a multi-resolution grid (Neon only).
"""

from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np

from xlb import DefaultConfig
from xlb.compute_backend import ComputeBackend


def grid_factory(
    shape: Tuple[int, ...],
    compute_backend: ComputeBackend = None,
    velocity_set=None,
    backend_config=None,
):
    """Create a single-resolution grid for the specified backend.

    Parameters
    ----------
    shape : tuple of int
        Domain dimensions, e.g. ``(nx, ny, nz)``.
    compute_backend : ComputeBackend, optional
        Backend to use.  Defaults to ``DefaultConfig.default_backend``.
    velocity_set : VelocitySet, optional
        Required for the Neon backend.
    backend_config : dict, optional
        Backend-specific configuration (Neon only).

    Returns
    -------
    Grid
        A backend-specific grid instance.
    """
    compute_backend = compute_backend or DefaultConfig.default_backend
    velocity_set = velocity_set or DefaultConfig.velocity_set
    if compute_backend == ComputeBackend.WARP:
        from xlb.grid.warp_grid import WarpGrid

        return WarpGrid(shape)
    elif compute_backend == ComputeBackend.NEON:
        from xlb.grid.neon_grid import NeonGrid

        return NeonGrid(shape=shape, velocity_set=velocity_set, backend_config=backend_config)
    elif compute_backend == ComputeBackend.JAX:
        from xlb.grid.jax_grid import JaxGrid

        return JaxGrid(shape)

    raise ValueError(f"Compute backend {compute_backend} is not supported")


def multires_grid_factory(
    shape: Tuple[int, ...],
    compute_backend: ComputeBackend = None,
    velocity_set=None,
    sparsity_pattern_list: List[np.ndarray] = None,
    sparsity_pattern_origins = None,
):
    import neon

    """Create a multi-resolution grid (Neon backend only).

    Parameters
    ----------
    shape : tuple of int
        Bounding-box dimensions at the finest level.
    compute_backend : ComputeBackend, optional
        Must be ``ComputeBackend.NEON``.
    velocity_set : VelocitySet, optional
        Lattice velocity set.
    sparsity_pattern_list : list of np.ndarray
        Active-voxel masks, one per level (finest first).
    sparsity_pattern_origins : list of neon.Index_3d
        Origin of each level's pattern in finest-level coordinates.

    Returns
    -------
    NeonMultiresGrid
        A multi-resolution Neon grid.
    """
    compute_backend = compute_backend or DefaultConfig.default_backend
    velocity_set = velocity_set or DefaultConfig.velocity_set
    if compute_backend == ComputeBackend.NEON:
        from xlb.grid.multires_grid import NeonMultiresGrid

        return NeonMultiresGrid(
            shape=shape, velocity_set=velocity_set, sparsity_pattern_list=sparsity_pattern_list, sparsity_pattern_origins=sparsity_pattern_origins
        )
    else:
        raise ValueError(f"Compute backend {compute_backend} is not supported for multires grid")


class Grid(ABC):
    """Abstract base class for all XLB computational grids.

    Subclasses must implement :meth:`_initialize_backend` to set up the
    backend-specific data structures and :meth:`create_field` (not
    enforced by ABC but expected by all operators).

    Parameters
    ----------
    shape : tuple of int
        Domain dimensions.
    compute_backend : ComputeBackend
        The compute backend this grid is associated with.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        compute_backend: ComputeBackend,
    ):
        self.shape = shape
        self.dim = len(shape)
        self.compute_backend = compute_backend
        self._initialize_backend()

    @abstractmethod
    def _initialize_backend(self):
        pass

    def get_compute_backend(self):
        """Return the compute backend associated with this grid."""
        return self.compute_backend

    def bounding_box_indices(self, shape=None, remove_edges=False):
        """
        This function calculates the indices of the bounding box of a 2D or 3D grid.
        The bounding box is defined as the set of grid points on the outer edge of the grid.

        Parameters
        ----------
        remove_edges : bool, optional
            If True, the nodes along the edges (not just the corners) are removed from the bounding box indices.
            Default is False.

        Returns
        -------
        boundingBox (dict): A dictionary where keys are the names of the bounding box faces
        ("bottom", "top", "left", "right" for 2D; additional "front", "back" for 3D), and values
        are numpy arrays of indices corresponding to each face.
        """

        # If shape is not give, use self.shape
        if shape is None:
            shape = self.shape

        # Get the shape of the grid
        origin = np.array([0, 0, 0])
        bounds = np.array(shape)
        if remove_edges:
            origin += 1
            bounds -= 1
        slice_x = slice(origin[0], bounds[0])
        slice_y = slice(origin[1], bounds[1])
        dim = len(bounds)

        # Generate bounding box indices for each face
        grid = np.indices(shape)
        boundingBoxIndices = {}

        if dim == 2:
            nx, ny = shape
            boundingBoxIndices = {
                "bottom": grid[:, slice_x, 0],
                "top": grid[:, slice_x, ny - 1],
                "left": grid[:, 0, slice_y],
                "right": grid[:, nx - 1, slice_y],
            }
        elif dim == 3:
            nx, ny, nz = shape
            slice_z = slice(origin[2], bounds[2])
            boundingBoxIndices = {
                "bottom": grid[:, slice_x, slice_y, 0].reshape(3, -1),
                "top": grid[:, slice_x, slice_y, nz - 1].reshape(3, -1),
                "left": grid[:, 0, slice_y, slice_z].reshape(3, -1),
                "right": grid[:, nx - 1, slice_y, slice_z].reshape(3, -1),
                "front": grid[:, slice_x, 0, slice_z].reshape(3, -1),
                "back": grid[:, slice_x, ny - 1, slice_z].reshape(3, -1),
            }

        return {k: v.tolist() for k, v in boundingBoxIndices.items()}
