"""
Multi-resolution sparse grid backed by the Neon ``mGrid`` runtime.

This module wraps ``neon.multires.mGrid`` and exposes it through the
:class:`Grid` interface.  The grid is hierarchical: level 0 is the finest
and level *N-1* is the coarsest.  Each coarser level has half the
resolution of the level below it (refinement factor 2).
"""

import numpy as np
import warp as wp
import neon
from .grid import Grid
from xlb.precision_policy import Precision
from xlb.compute_backend import ComputeBackend
from typing import Literal, List
from xlb import DefaultConfig


class NeonMultiresGrid(Grid):
    """Hierarchical multi-resolution grid on the Neon backend.

    Wraps ``neon.multires.mGrid``.  Each level is described by a boolean
    sparsity pattern (active-voxel mask) and an integer origin that
    places it within the finest-level coordinate system.

    Parameters
    ----------
    shape : tuple of int
        Bounding-box dimensions at the **finest** level ``(nx, ny, nz)``.
    velocity_set : VelocitySet
        Lattice velocity set defining neighbour connectivity.
    sparsity_pattern_list : list of np.ndarray
        One boolean/int array per level indicating which voxels are active.
        Index 0 = finest level, index *N-1* = coarsest.
    sparsity_pattern_origins : list of neon.Index_3d
        Origin offset for each level's pattern in the finest-level
        coordinate system.
    """

    def __init__(
        self,
        shape,
        velocity_set,
        sparsity_pattern_list: List[np.ndarray],
        sparsity_pattern_origins: List[neon.Index_3d],
    ):
        self.bk = None
        self.dim = None
        self.grid = None
        self.velocity_set = velocity_set
        self.sparsity_pattern_list = sparsity_pattern_list
        self.sparsity_pattern_origins = sparsity_pattern_origins
        self.count_levels = len(sparsity_pattern_list)
        self.refinement_factor = 2

        super().__init__(shape, ComputeBackend.NEON)

    def _get_velocity_set(self):
        return self.velocity_set

    def _initialize_backend(self):
        num_devs = 1
        dev_idx_list = list(range(num_devs))

        if len(self.shape) == 2:
            import py_neon

            self.dim = py_neon.Index_3d(self.shape[0], 1, self.shape[1])
            self.neon_stencil = []
            for q in range(self.velocity_set.q):
                xval, yval = self.velocity_set._c[:, q]
                self.neon_stencil.append([xval, 0, yval])

        else:
            self.dim = neon.Index_3d(self.shape[0], self.shape[1], self.shape[2])

            self.neon_stencil = []
            for q in range(self.velocity_set.q):
                xval, yval, zval = self.velocity_set._c[:, q]
                self.neon_stencil.append([xval, yval, zval])

        self.bk = neon.Backend(runtime=neon.Backend.Runtime.stream, dev_idx_list=dev_idx_list)

        self.grid = neon.multires.mGrid(
            backend=self.bk,
            dim=self.dim,
            sparsity_pattern_list=self.sparsity_pattern_list,
            sparsity_pattern_origins=self.sparsity_pattern_origins,
            stencil=self.neon_stencil,
        )
        # Print grid stats about voxel distribution between levels.
        self.grid.print_info()
        pass

    def create_field(
        self,
        cardinality: int,
        dtype: Literal[Precision.FP32, Precision.FP64, Precision.FP16] = None,
        fill_value=None,
        neon_memory_type: neon.MemoryType = neon.MemoryType.host_device(),
    ):
        """Allocate a new multi-resolution Neon field.

        The field spans all grid levels.  Each level is either zero-filled
        or filled with *fill_value*.

        Parameters
        ----------
        cardinality : int
            Number of components per voxel.
        dtype : Precision, optional
            Element precision.  Defaults to the store precision from the
            global config.
        fill_value : float, optional
            Value to fill every element with.  ``None`` means zero.
        neon_memory_type : neon.MemoryType
            Memory residency (host, device, or both).

        Returns
        -------
        neon.multires.mField
            The newly allocated multi-resolution field.
        """
        dtype = dtype.wp_dtype if dtype else DefaultConfig.default_precision_policy.store_precision.wp_dtype
        field = self.grid.new_field(
            cardinality=cardinality,
            dtype=dtype,
            memory_type=neon_memory_type,
        )
        for l in range(self.count_levels):
            if fill_value is None:
                field.zero_run(l, stream_idx=0)
            else:
                field.fill_run(level=l, value=fill_value, stream_idx=0)
        return field

    def get_neon_backend(self):
        """Return the underlying ``neon.Backend`` instance."""
        return self.bk

    def level_to_shape(self, level):
        """Return the bounding-box shape at the given grid level.

        Level 0 is the finest and has shape ``self.shape``.  Each subsequent
        level halves each dimension.
        """
        # level = 0 corresponds to the finest level
        return tuple(x // self.refinement_factor**level for x in self.shape)

    def boundary_indices_across_levels(self, level_data, box_side: str = "front", remove_edges: bool = False):
        """
        Get indices for creating a boundary condition on the specified box side that crosses multiples levels of a multiresolution grid.
        The indices are returned as a list of lists, where each sublist corresponds to a level

        Parameters
        ----------
        - level_data: Level data containing the origins and sparsity patterns for each level as prepared by mesher/make_cuboid_mesh function!
        - box_side: The side of the bounding box to get indices for (default is "front").
        returns:
        - A list of lists, where each sublist contains the indices for the boundary condition at that level.
        """
        num_levels = len(level_data)
        bc_indices_list = []
        d = self.velocity_set.d  # Dimensionality (2 or 3)

        # Define side configurations (adjust if your conventions differ)
        if d == 3:
            side_config = {
                "left": {"dim": 0, "value": 0},
                "right": {"dim": 0, "value": lambda s: s[0] - 1},
                "front": {"dim": 1, "value": 0},
                "back": {"dim": 1, "value": lambda s: s[1] - 1},
                "bottom": {"dim": 2, "value": 0},
                "top": {"dim": 2, "value": lambda s: s[2] - 1},
            }
        elif d == 2:
            side_config = {
                "left": {"dim": 0, "value": 0},
                "right": {"dim": 0, "value": lambda s: s[0] - 1},
                "bottom": {"dim": 1, "value": 0},
                "top": {"dim": 1, "value": lambda s: s[1] - 1},
            }
        else:
            raise ValueError(f"Unsupported dimensionality: {d}")

        if box_side not in side_config:
            raise ValueError(f"Unsupported box_side: {box_side}")

        # Compare in a single coordinate system: finest-level (global) units.
        # `origin` (from `make_cuboid_mesh`) is stored in **level-local cell units**,
        # so a level-`level` cell at local index `i` has finest-level lower corner
        # `(origin + i) * cell_size` and covers the interval
        # `[(origin + i) * cell_size, (origin + i + 1) * cell_size - 1]`.
        finest_shape = self.shape  # tuple of length d, in finest-level units

        conf = side_config[box_side]
        dim_idx = conf["dim"]
        # `global_bound` is the global box face position in finest-level units.
        global_bound = conf["value"](finest_shape) if callable(conf["value"]) else conf["value"]
        is_high_side = callable(conf["value"])  # high side uses s-1, low side uses 0

        for level in range(num_levels):
            mask = level_data[level][0]
            origin = level_data[level][2]  # level-local cell units, shape (d,)
            cell_size = self.refinement_factor**level

            # Get local indices of active voxels
            local_coords = np.nonzero(mask)  # tuple of d arrays in level-local cell indices
            if not local_coords[0].size:
                bc_indices_list.append([])
                continue

            # Cell lower corners in finest-level (global) units.
            global_coords = [(local_coords[i] + origin[i]) * cell_size for i in range(d)]

            # A cell is on the requested face if any of its finest-level cells
            # lies on that face. Lower corner is `gc`, upper corner is `gc + cell_size - 1`.
            if is_high_side:
                cond = (global_coords[dim_idx] + (cell_size - 1)) == global_bound
            else:
                cond = global_coords[dim_idx] == global_bound

            # If remove_edges, exclude perimeter of the face (in finest-level units).
            if remove_edges:
                for i in range(d):
                    if i != dim_idx:
                        cell_high_i = global_coords[i] + (cell_size - 1)
                        cond &= (global_coords[i] > 0) & (cell_high_i < finest_shape[i] - 1)

            # Collect filtered indices
            if np.any(cond):
                active_bc = [gc[cond] for gc in global_coords]
                bc_indices_list.append([arr.tolist() for arr in active_bc])
            else:
                bc_indices_list.append([])

        return bc_indices_list
