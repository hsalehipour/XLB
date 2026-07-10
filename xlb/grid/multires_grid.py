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
        self._sparse_pattern = (
            sparsity_pattern_list
            and sparsity_pattern_list[0].ndim == 2
            and sparsity_pattern_list[0].shape[1] == 3
        )
        self.domain_min_finest, self.domain_max_finest = self._domain_finest_bounds_from_sparsity(
            sparsity_pattern_list, sparsity_pattern_origins
        )

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

        if self._sparse_pattern:
            active_voxels = [
                np.ascontiguousarray(pattern, dtype=np.int32)
                for pattern in self.sparsity_pattern_list
            ]
            self.grid = neon.multires.mGrid.from_active_voxels(
                backend=self.bk,
                dim=self.dim,
                active_voxels_list=active_voxels,
                sparsity_pattern_origins=self.sparsity_pattern_origins,
                stencil=self.neon_stencil,
            )
        else:
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
        - A list of lists, where each sublist contains the **local** lattice indices for the boundary condition at that level.
          The Neon masker converts these to finest-level indices via ``(local + origin) << level``.
        """
        num_levels = len(level_data)
        bc_indices_list = []
        d = self.velocity_set.d  # Dimensionality (2 or 3)

        # Define side configurations (adjust if your conventions differ)
        if d == 3:
            side_config = {
                "left": {"dim": 0, "min": True},
                "right": {"dim": 0, "min": False},
                "front": {"dim": 1, "min": True},
                "back": {"dim": 1, "min": False},
                "bottom": {"dim": 2, "min": True},
                "top": {"dim": 2, "min": False},
            }
        elif d == 2:
            side_config = {
                "left": {"dim": 0, "min": True},
                "right": {"dim": 0, "min": False},
                "bottom": {"dim": 1, "min": True},
                "top": {"dim": 1, "min": False},
            }
        else:
            raise ValueError(f"Unsupported dimensionality: {d}")

        if box_side not in side_config:
            raise ValueError(f"Unsupported box_side: {box_side}")

        domain_min_finest, domain_max_finest = self._domain_finest_bounds(level_data, d)

        for level in range(num_levels):
            pattern = level_data[level][0]
            origin = np.asarray(level_data[level][2], dtype=np.int64)
            stride = 1 << level

            conf = side_config[box_side]
            dim_idx = conf["dim"]
            # Neon base indices are in [0, dim).  A cell at local L with origin O
            # and stride S lives at base_idx = (L + O) * S.  The leftmost
            # addressable local satisfies (L + O) * S >= 0, i.e. L >= -O.
            # Cells below that are phantom padding Neon cannot address.
            if conf["min"]:
                local_bound = int(-origin[dim_idx])
            else:
                local_bound = int(domain_max_finest[dim_idx] // stride - origin[dim_idx])

            if pattern.ndim == 2:
                local_coords = tuple(pattern[:, i].astype(np.int64) for i in range(d))
            else:
                local_coords = tuple(c.astype(np.int64) for c in np.nonzero(pattern))
            if not local_coords[0].size:
                bc_indices_list.append([])
                continue

            finest_coords = [(local_coords[i] + origin[i]) * stride for i in range(d)]

            cond = local_coords[dim_idx] == local_bound

            # If remove_edges, exclude perimeter of the face.
            # Only cells with virtual >= 0 are addressable; use 0 as effective min.
            if remove_edges:
                for i in range(d):
                    if i != dim_idx:
                        effective_min = max(0, int(domain_min_finest[i]))
                        cond &= (finest_coords[i] > effective_min) & (
                            finest_coords[i] < domain_max_finest[i]
                        )

            if np.any(cond):
                active_bc = [lc[cond].tolist() for lc in local_coords]
                bc_indices_list.append(active_bc)
            else:
                bc_indices_list.append([])

        return bc_indices_list

    @staticmethod
    def virtual_finest_to_neon_global(
        finest_virtual: np.ndarray,
        domain_min_finest: np.ndarray,
        domain_max_finest: np.ndarray | None = None,
        level: int = 0,
        num_levels: int = 1,
    ) -> np.ndarray:
        """Map virtual finest lattice indices to Neon base-index space.

        Neon's ``getGlobalIndex`` returns the base-grid index of a cell, which
        is identical to its virtual finest coordinate for all cells that exist
        in the grid (base indices are in ``[0, dim)``).  Cells at
        ``virtual < 0`` are phantom padding that Neon cannot address.

        Therefore this function is the **identity**: it returns the input
        unchanged.  It exists solely to document this contract and to keep call
        sites explicit about the coordinate system they operate in.
        """
        return np.asarray(finest_virtual, dtype=np.int32)

    @staticmethod
    def neon_global_to_virtual_finest(
        neon_global: np.ndarray,
        domain_min_finest: np.ndarray,
        domain_max_finest: np.ndarray,
        level: int = 0,
        num_levels: int = 1,
    ) -> np.ndarray:
        """Inverse of :meth:`virtual_finest_to_neon_global` — also identity."""
        return np.asarray(neon_global, dtype=np.int32)

    @staticmethod
    def _domain_finest_bounds_from_sparsity(sparsity_pattern_list, sparsity_pattern_origins, d: int = 3):
        """Return inclusive min/max finest lattice indices covered by active voxels."""
        level_data = []
        for level, pattern in enumerate(sparsity_pattern_list):
            origin_pt = sparsity_pattern_origins[level]
            origin = np.array([origin_pt.x, origin_pt.y, origin_pt.z], dtype=np.int64)
            level_data.append((pattern, None, origin, level))
        return NeonMultiresGrid._domain_finest_bounds(level_data, d)

    @staticmethod
    def _domain_finest_bounds(level_data, d: int):
        """Return inclusive min/max finest lattice indices covered by active voxels."""
        mins = np.full(d, np.iinfo(np.int64).max, dtype=np.int64)
        maxs = np.full(d, np.iinfo(np.int64).min, dtype=np.int64)
        for level in range(len(level_data)):
            pattern = level_data[level][0]
            origin = np.asarray(level_data[level][2], dtype=np.int64)
            stride = 1 << level
            if pattern.ndim == 2:
                coords = pattern.astype(np.int64)
            else:
                coords = np.stack(np.nonzero(pattern), axis=1).astype(np.int64)
            if coords.size == 0:
                continue
            finest = (coords + origin) * stride
            mins = np.minimum(mins, finest.min(axis=0))
            maxs = np.maximum(maxs, finest.max(axis=0))
        return mins, maxs
