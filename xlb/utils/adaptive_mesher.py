"""
Surface-adaptive multi-resolution mesh generation (Warp GPU backend).

This module generates strongly-balanced multi-resolution grids that conform
tightly to STL/OBJ surfaces. Fine cells are placed within a configurable
distance band of the geometry, and progressively coarser cells fill the
remaining domain volume using geometric expansion-ratio shells.

**Architecture:**

1. Distance computation: GPU-accelerated signed-distance queries via Warp BVH
   meshes, either in a single dense launch or tiled for large domains.
2. Level assignment: Each cell receives a target refinement level based on its
   distance to the surface (geometric-ratio shells).
3. Owner grid refinement: Strong-balance enforcement (ΔL ≤ 1), dyadic block
   alignment, and optional surface-targeted distance recomputation ensure
   artifact-free transitions.
4. Mask extraction: Greedy coarsest-first partitioning produces non-overlapping,
   fully-covering boolean masks.
5. Balance repair: A final subdivision pass guarantees the strong-balance
   invariant across mask boundaries.

**Requirements:** NVIDIA Warp (``pip install warp-lang``).

Output is compatible with :func:`xlb.utils.mesher.prepare_sparsity_pattern`.

CLI
---
::

    python -m xlb.utils.adaptive_mesher --stl path/to/model.stl [options]

See :func:`main` and :func:`build_parser` for all flags.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
import trimesh
import warp as wp

from xlb.utils.adaptive_mesher_kernels import (
    kernel_any_nonzero_uint8,
    kernel_any_uncovered,
    kernel_apply_balance_coarsen,
    kernel_apply_balance_refine,
    kernel_apply_subdivide_level,
    kernel_assign_levels,
    kernel_assign_levels_1d,
    kernel_batched_distances,
    kernel_conservative_coarse_targets,
    kernel_dense_distances,
    kernel_dense_distances_tiled,
    kernel_dilate_mask_uint8,
    kernel_dyadic_apply_flags,
    kernel_dyadic_flag_blocks,
    kernel_edt_init_from_mask,
    kernel_edt_pass_x,
    kernel_edt_pass_y,
    kernel_edt_pass_z,
    kernel_edt_sqrt,
    kernel_extract_level_mask,
    kernel_fill_coverage_gaps,
    kernel_greedy_activate_l0,
    kernel_greedy_check_block,
    kernel_greedy_fill_remaining,
    kernel_greedy_mark_covered,
    kernel_mark_coverage,
    kernel_mark_subdivide_offset,
    kernel_maximum_filter_3x3,
    kernel_minimum_filter_3x3,
    kernel_minimum_with_floor,
    kernel_owners_from_masks,
    kernel_paint_blocks_level,
    kernel_refine_transition,
    kernel_balance_need_coarsen,
    kernel_balance_need_refine,
)
from xlb.utils.mesher import (
    _align_domain_origin_for_dyadic,
    _domain_bbox_from_padding,
    _load_stl_mesh,
    _normalize_level_data,
    _stl_bounds,
    adjust_bbox,
)

# 26-neighbor offsets (excluding self) on a 3-D Cartesian grid.
_NEIGHBOR_OFFSETS_26 = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
]

_MAX_MASK_LEVELS = 8
_DUMMY_MASK_SHAPE = (1, 1, 1)


@dataclass
class AdaptiveMeshConfig:
    """Configuration for surface-adaptive multires meshing."""

    voxel_size: float
    num_levels: int
    expansion_ratio: float = 2.0
    finest_band_cells: int = 3
    domain_padding: Sequence[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    stl_filename: str = ""
    tile_size: int = 64
    max_dense_cells: int = 128**3

    def __post_init__(self):
        if self.num_levels < 1:
            raise ValueError("num_levels must be at least 1.")
        if self.expansion_ratio <= 1.0:
            raise ValueError("expansion_ratio must be greater than 1.")
        if self.finest_band_cells < 1:
            raise ValueError("finest_band_cells must be at least 1.")
        if len(self.domain_padding) != 6:
            raise ValueError("domain_padding must be a 6-tuple: [-x, +x, -y, +y, -z, +z].")


# ---------------------------------------------------------------------------
# Domain helpers
# ---------------------------------------------------------------------------

def _compute_domain(config: AdaptiveMeshConfig):
    """Compute the physical domain parameters for adaptive meshing.

    Loads the STL, computes bounding-box padding, and aligns the domain
    origin and grid shape for dyadic nesting.

    Args:
        config: Mesh configuration with STL path, voxel size, padding, etc.

    Returns:
        Tuple of (trimesh.Trimesh, origin ndarray, grid_shape tuple).
    """
    mesh = _load_stl_mesh(config.stl_filename)
    min_bound, max_bound, part_size = _stl_bounds(mesh)

    cuboid_min, cuboid_max = _domain_bbox_from_padding(min_bound, max_bound, part_size, config.domain_padding)
    adjusted_min, adjusted_max = adjust_bbox(cuboid_max, cuboid_min, config.voxel_size)
    origin, grid_shape = _align_domain_origin_for_dyadic(
        adjusted_min, adjusted_max, config.voxel_size, config.num_levels
    )

    return mesh, origin, grid_shape


def _shape_at_level(grid_shape_finest: Tuple[int, int, int], level: int) -> Tuple[int, int, int]:
    """Return the grid dimensions at a given refinement level.

    Each level halves each axis dimension relative to the finest (level 0).
    """
    stride = 2**level
    return (grid_shape_finest[0] // stride, grid_shape_finest[1] // stride, grid_shape_finest[2] // stride)


# ---------------------------------------------------------------------------
# Octree helpers
# ---------------------------------------------------------------------------

def _child_centers_and_keys(
    parent_refine: np.ndarray,
    origin: np.ndarray,
    parent_voxel: float,
    child_voxel: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized 2x2x2 subdivision: return (N*8, 3) centers and (N*8, 3) child indices."""
    if len(parent_refine) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=int)

    # Single (8, 3) offset table shared by both the physical child centers and
    # the integer child block coordinates, so the distance evaluated at a child
    # centre and the coordinate it is recorded under always refer to the same
    # child. (Previously the integer offsets were sliced with a stride that did
    # not match the centre ordering, scrambling child coordinates and scattering
    # active cells into diagonal stripes.)
    int_offsets = np.array(
        [
            [di, dj, dk]
            for di in range(2)
            for dj in range(2)
            for dk in range(2)
        ],
        dtype=np.int64,
    )

    bases = origin + parent_refine.astype(np.float64) * parent_voxel
    centers = (bases[:, None, :] + (int_offsets[None, :, :] + 0.5) * child_voxel).reshape(-1, 3)

    child_blocks = parent_refine.astype(np.int64)[:, None, :] * 2 + int_offsets[None, :, :]
    keys = child_blocks.reshape(-1, 3)
    return centers, keys


class LevelDataList(list):
    """Level data list with optional attached finest-grid shape metadata."""

    grid_shape_finest: Tuple[int, int, int] | None = None


def is_sparse_level_data(level_data: list) -> bool:
    """Return True when level_data stores (N, 3) sparse coords instead of dense masks."""
    if not level_data:
        return False
    pattern = level_data[0][0]
    return pattern.ndim == 2 and pattern.shape[1] == 3


def _assignments_to_active_coords(
    assignments: List[List[np.ndarray]],
    num_levels: int,
) -> List[np.ndarray]:
    """Convert octree per-level cell assignments into sparse (N, 3) coordinate arrays."""
    active_coords: List[np.ndarray] = []
    for target in range(num_levels):
        if assignments[target]:
            arr = np.unique(np.vstack(assignments[target]), axis=0).astype(np.int32)
        else:
            arr = np.empty((0, 3), dtype=np.int32)
        active_coords.append(arr)

    if all(len(a) == 0 for a in active_coords):
        raise RuntimeError("No cell assignments collected during octree refinement.")
    return active_coords


def _record_assignments(
    assignments: List[List[np.ndarray]],
    keys: np.ndarray,
    targets: np.ndarray,
    source_level: int,
    num_levels: int,
):
    """Record octree cell indices at each target level from child keys.

    Converts child indices on the ``source_level`` grid to equivalent block
    coordinates at the ``target`` level via bit-shifting (left-shift to go
    finer, right-shift to go coarser).

    Args:
        assignments: Mutable list-of-lists accumulating (N, 3) index arrays per level.
        keys: (N, 3) int array of child cell indices at source_level resolution.
        targets: (N,) int array of assigned target levels for each child.
        source_level: The refinement level at which keys are expressed.
        num_levels: Total number of refinement levels.
    """
    for target in range(num_levels):
        sel = targets == target
        if not np.any(sel):
            continue
        t_keys = keys[sel]
        if target < source_level:
            idx = t_keys << (source_level - target)
        elif target == source_level:
            idx = t_keys
        else:
            idx = t_keys >> (target - source_level)
        assignments[target].append(idx)


# ---------------------------------------------------------------------------
# Mask building (greedy coarsest-first)
# ---------------------------------------------------------------------------

def _build_masks_greedy_coarsest(owner: np.ndarray, num_levels: int) -> List[np.ndarray]:
    """Build non-overlapping, fully-covering masks using GPU-accelerated greedy strategy.

    The algorithm activates cells level-by-level:
      1. L0 is activated first wherever owner == 0 (tight, coherent surface band).
      2. Remaining volume is filled coarsest-first (L(n-1) down to L1): a block
         at level L is activated iff all its cells have owner >= L and none are
         already covered by a finer level.
      3. Any leftover uncovered cells are assigned to L0.

    All steps run on GPU via Warp kernels to avoid CPU bottlenecks on large grids.

    Args:
        owner: 3D int32 array of per-cell target levels (0=finest, num_levels-1=coarsest).
        num_levels: Total number of refinement levels.

    Returns:
        List of boolean mask arrays, one per level (mask[L] has shape grid//2^L).
    """
    nx, ny, nz = owner.shape
    owner_wp = wp.array(owner, dtype=wp.int32)
    covered = wp.zeros((nx, ny, nz), dtype=wp.uint8)

    masks_wp: List[wp.array] = []
    for level in range(num_levels):
        stride = 2**level
        sx, sy, sz = nx // stride, ny // stride, nz // stride
        masks_wp.append(wp.zeros((sx, sy, sz), dtype=wp.uint8))

    # Step 1: Activate L0 wherever owner == 0
    wp.launch(kernel_greedy_activate_l0, dim=(nx, ny, nz), inputs=[owner_wp, masks_wp[0], covered])

    # Step 2: Coarsest-first for levels > 0
    for level in range(num_levels - 1, 0, -1):
        stride = 2**level
        sx, sy, sz = nx // stride, ny // stride, nz // stride
        if sx == 0 or sy == 0 or sz == 0:
            continue
        block_ok = wp.zeros((sx, sy, sz), dtype=wp.uint8)
        wp.launch(
            kernel_greedy_check_block, dim=(sx, sy, sz),
            inputs=[owner_wp, covered, wp.int32(level), wp.int32(stride), block_ok],
        )
        wp.launch(
            kernel_greedy_mark_covered,
            dim=(sx * stride, sy * stride, sz * stride),
            inputs=[block_ok, wp.int32(stride), covered, masks_wp[level]],
        )

    # Step 3: Fill any remaining uncovered cells into L0
    wp.launch(kernel_greedy_fill_remaining, dim=(nx, ny, nz), inputs=[covered, masks_wp[0]])

    wp.synchronize()
    return [m.numpy().astype(bool) for m in masks_wp]


# ---------------------------------------------------------------------------
# Shift helper (used by validate_level_data)
# ---------------------------------------------------------------------------

def _shift_toward_offset(arr: np.ndarray, di: int, dj: int, dk: int, fill: int = -1) -> np.ndarray:
    """Return ``out[i,j,k] = arr[i+di,j+dj,k+dk]`` in bounds, else ``fill``."""
    out = np.full_like(arr, fill)
    nx, ny, nz = arr.shape
    d_i0, d_i1 = max(0, -di), min(nx, nx - di)
    d_j0, d_j1 = max(0, -dj), min(ny, ny - dj)
    d_k0, d_k1 = max(0, -dk), min(nz, nz - dk)
    if d_i0 < d_i1 and d_j0 < d_j1 and d_k0 < d_k1:
        out[d_i0:d_i1, d_j0:d_j1, d_k0:d_k1] = arr[
            d_i0 + di : d_i1 + di,
            d_j0 + dj : d_j1 + dj,
            d_k0 + dk : d_k1 + dk,
        ]
    return out


# ---------------------------------------------------------------------------
# Warp mesh conversion helpers
# ---------------------------------------------------------------------------

def trimesh_to_warp(mesh: trimesh.Trimesh) -> wp.Mesh:
    """Build a Warp BVH mesh from a trimesh surface."""
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces.reshape(-1), dtype=np.int32)
    return wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3),
        indices=wp.array(faces, dtype=wp.int32),
    )


def _max_query_distance(origin: np.ndarray, grid_shape: Tuple[int, int, int], voxel_size: float) -> float:
    """Compute a safe maximum query radius for BVH distance lookups.

    Returns twice the domain diagonal plus a margin, ensuring no cell
    center can be farther from the mesh than this distance.
    """
    extent = np.array(grid_shape, dtype=np.float64) * voxel_size
    return float(np.linalg.norm(extent) * 2.0 + 1.0)


def _pad_mask_list(masks: List[wp.array], num_levels: int) -> List[wp.array]:
    """Return exactly ``_MAX_MASK_LEVELS`` mask arrays (dummy 1³ for unused slots)."""
    dummy = wp.zeros(_DUMMY_MASK_SHAPE, dtype=wp.uint8, device=masks[0].device)
    out = list(masks)
    while len(out) < _MAX_MASK_LEVELS:
        out.append(dummy)
    return out[: _MAX_MASK_LEVELS]


def _edt_scratch_buffers(nx: int, ny: int, nz: int):
    """Allocate scratch buffers for the separable EDT algorithm.

    The Meijster EDT requires per-line envelope/intersection storage for
    each of the three axis passes (X, Y, Z). Returns six Warp arrays.
    """
    n_lines_xy = ny * nz
    n_lines_xz = nx * nz
    n_lines_yz = nx * ny
    max_dim = max(nx, ny, nz)
    return (
        wp.zeros((n_lines_xy, max_dim), dtype=wp.int32),
        wp.zeros((n_lines_xy, max_dim + 1), dtype=wp.float32),
        wp.zeros((n_lines_xz, max_dim), dtype=wp.int32),
        wp.zeros((n_lines_xz, max_dim + 1), dtype=wp.float32),
        wp.zeros((n_lines_yz, max_dim), dtype=wp.int32),
        wp.zeros((n_lines_yz, max_dim + 1), dtype=wp.float32),
    )


# ---------------------------------------------------------------------------
# Euclidean distance transform (Warp-native, SciPy parity)
# ---------------------------------------------------------------------------

def euclidean_edt_3d(background: np.ndarray) -> np.ndarray:
    """Exact Euclidean distance transform using Meijster's separable algorithm on GPU.

    Implements the three-pass separable EDT:
      1. X-pass: compute squared distances along rows.
      2. Y-pass: update with column contributions.
      3. Z-pass: update with depth contributions.
      4. Final sqrt pass to convert squared distances to Euclidean.

    Produces results matching ``scipy.ndimage.distance_transform_edt`` to
    floating-point precision.

    Args:
        background: 3D boolean/uint8 array where True/1 = background (compute
                    distance from foreground boundary).

    Returns:
        3D float32 array of Euclidean distances.
    """
    nx, ny, nz = background.shape
    mask_wp = wp.array(background.astype(np.uint8), dtype=wp.uint8)
    sq_wp = wp.zeros((nx, ny, nz), dtype=wp.float32)
    sv_xy, sz_xy, sv_xz, sz_xz, sv_yz, sz_yz = _edt_scratch_buffers(nx, ny, nz)
    wp.launch(kernel_edt_init_from_mask, dim=(nx, ny, nz), inputs=[mask_wp, sq_wp])
    wp.launch(kernel_edt_pass_x, dim=(ny, nz), inputs=[sq_wp, sv_xy, sz_xy])
    wp.launch(kernel_edt_pass_y, dim=(nx, nz), inputs=[sq_wp, sv_xz, sz_xz])
    wp.launch(kernel_edt_pass_z, dim=(nx, ny), inputs=[sq_wp, sv_yz, sz_yz])
    dist_wp = wp.zeros((nx, ny, nz), dtype=wp.float32)
    wp.launch(kernel_edt_sqrt, dim=(nx, ny, nz), inputs=[sq_wp, dist_wp])
    wp.synchronize()
    return dist_wp.numpy()


# ---------------------------------------------------------------------------
# WarpAdaptiveMesherOps – GPU-accelerated adaptive mesher operations
# ---------------------------------------------------------------------------

class WarpAdaptiveMesherOps:
    """GPU-accelerated operations for adaptive multi-resolution meshing.

    This class encapsulates all Warp-based mesh operations including:
    - Distance field computation (dense and tiled)
    - Level assignment from distance fields
    - Owner grid finalization (strong-balance enforcement)
    - Dyadic block alignment
    - Mask extraction and balance repair

    The class holds a reference to a Warp BVH mesh built from the input
    trimesh geometry. All GPU kernels are launched via Warp and operate on
    the same CUDA device.

    Thread safety: Instances are NOT thread-safe. Create one per thread if
    needed (each will allocate its own BVH mesh on the device).

    Example::

        ops = WarpAdaptiveMesherOps(trimesh_mesh)
        distances = ops.compute_distance_field(origin, shape, voxel_size, config)
        assigned = ops.assign_levels_from_distance(distances, config)
        masks = ops.build_masks_from_owner(assigned, num_levels)
    """

    def __init__(self, mesh: trimesh.Trimesh):
        """Initialize with a trimesh surface mesh.

        Args:
            mesh: Input triangle mesh (vertices + faces). A Warp BVH is
                  built immediately for GPU-accelerated distance queries.
        """
        self._mesh = mesh
        self._wp_mesh = trimesh_to_warp(mesh)
        self._mesh_id = wp.uint64(self._wp_mesh.id)

    @property
    def mesh_id(self) -> wp.uint64:
        return self._mesh_id

    def _sync_counter(self, counter: wp.array) -> int:
        wp.synchronize()
        return int(counter.numpy()[0])

    def euclidean_edt_3d(self, background: np.ndarray) -> np.ndarray:
        """Compute exact Euclidean distance transform on the GPU.

        Delegates to the module-level :func:`euclidean_edt_3d` which uses
        Meijster's separable algorithm implemented as Warp kernels.
        """
        return euclidean_edt_3d(background)

    def compute_distance_field_dense(
        self,
        origin: np.ndarray,
        grid_shape: Tuple[int, int, int],
        voxel_size: float,
    ) -> np.ndarray:
        """Compute unsigned distance field for the entire grid in a single GPU launch.

        Args:
            origin: Physical (x, y, z) origin of the domain.
            grid_shape: (nx, ny, nz) grid dimensions at finest level.
            voxel_size: Physical size of each cell.

        Returns:
            (nx, ny, nz) float64 array of unsigned distances to the STL surface.
        """
        nx, ny, nz = grid_shape
        max_dist = _max_query_distance(origin, grid_shape, voxel_size)
        distances = wp.zeros((nx, ny, nz), dtype=wp.float64)
        wp.launch(
            kernel_dense_distances,
            dim=(nx, ny, nz),
            inputs=[
                self._mesh_id,
                wp.vec3d(float(origin[0]), float(origin[1]), float(origin[2])),
                wp.float64(voxel_size),
                wp.float64(max_dist),
                distances,
            ],
        )
        wp.synchronize()
        return distances.numpy()

    def compute_distance_field(
        self,
        origin: np.ndarray,
        grid_shape: Tuple[int, int, int],
        voxel_size: float,
        config: AdaptiveMeshConfig,
    ) -> np.ndarray:
        """Compute unsigned distance field, choosing dense or tiled strategy.

        Uses a single dense GPU launch when the grid fits in ``max_dense_cells``,
        otherwise tiles the domain into chunks of size ``config.tile_size`` to
        limit per-launch memory usage.

        Args:
            origin: Physical domain origin.
            grid_shape: Finest-level grid dimensions.
            voxel_size: Finest cell size.
            config: Configuration (provides tile_size and max_dense_cells).

        Returns:
            (nx, ny, nz) float64 distance array.
        """
        nx, ny, nz = grid_shape
        if nx * ny * nz <= config.max_dense_cells:
            return self.compute_distance_field_dense(origin, grid_shape, voxel_size)
        distances = np.empty(grid_shape, dtype=np.float64)
        tile = config.tile_size
        max_dist = _max_query_distance(origin, grid_shape, voxel_size)
        origin_wp = wp.vec3d(float(origin[0]), float(origin[1]), float(origin[2]))
        for i0 in range(0, nx, tile):
            i1 = min(i0 + tile, nx)
            for j0 in range(0, ny, tile):
                j1 = min(j0 + tile, ny)
                for k0 in range(0, nz, tile):
                    k1 = min(k0 + tile, nz)
                    tile_shape = (i1 - i0, j1 - j0, k1 - k0)
                    tile_dist = wp.zeros(tile_shape, dtype=wp.float64)
                    wp.launch(
                        kernel_dense_distances_tiled,
                        dim=tile_shape,
                        inputs=[
                            self._mesh_id,
                            origin_wp,
                            wp.float64(voxel_size),
                            wp.float64(max_dist),
                            wp.int32(i0),
                            wp.int32(j0),
                            wp.int32(k0),
                            tile_dist,
                        ],
                    )
                    wp.synchronize()
                    distances[i0:i1, j0:j1, k0:k1] = tile_dist.numpy()
        return distances

    def assign_levels_from_distance(self, distances: np.ndarray, config: AdaptiveMeshConfig) -> np.ndarray:
        """Assign refinement levels from a 3D distance field using geometric shells.

        Level 0 (finest) is assigned to cells within ``finest_band_cells * voxel_size``.
        Each subsequent level extends by ``expansion_ratio`` multiplicatively.

        Args:
            distances: 3D float64 distance array.
            config: Configuration with shell parameters.

        Returns:
            3D int32 array of assigned levels (0=finest, num_levels-1=coarsest).
        """
        d_band = config.finest_band_cells * config.voxel_size
        log_ratio = math.log(config.expansion_ratio)
        assigned = wp.zeros(distances.shape, dtype=wp.int32)
        dist_wp = wp.array(distances, dtype=wp.float64)
        wp.launch(
            kernel_assign_levels,
            dim=distances.shape,
            inputs=[
                dist_wp,
                wp.float64(d_band),
                wp.float64(log_ratio),
                wp.int32(config.num_levels),
                assigned,
            ],
        )
        wp.synchronize()
        return assigned.numpy()

    def assign_levels_from_distances_1d(self, distances: np.ndarray, config: AdaptiveMeshConfig) -> np.ndarray:
        """Assign levels from a flat 1D array of distances (for batched point queries).

        Same geometric-shell logic as :meth:`assign_levels_from_distance` but
        operates on a 1D distance vector (e.g., from octree child centers).

        Args:
            distances: 1D float64 array of distances.
            config: Configuration with shell parameters.

        Returns:
            1D int32 array of assigned levels.
        """
        n = len(distances)
        if n == 0:
            return np.empty(0, dtype=np.int32)
        d_band = config.finest_band_cells * config.voxel_size
        log_ratio = math.log(config.expansion_ratio)
        dist_wp = wp.array(distances.astype(np.float64), dtype=wp.float64)
        assigned = wp.zeros(n, dtype=wp.int32)
        wp.launch(
            kernel_assign_levels_1d,
            dim=n,
            inputs=[
                dist_wp,
                wp.float64(d_band),
                wp.float64(log_ratio),
                wp.int32(config.num_levels),
                assigned,
            ],
        )
        wp.synchronize()
        return assigned.numpy()

    def batched_distances(self, points: np.ndarray, max_dist: float) -> np.ndarray:
        """Query unsigned distances from arbitrary points to the STL surface on GPU.

        Args:
            points: (N, 3) float64 array of query positions.
            max_dist: Maximum query radius for BVH traversal.

        Returns:
            (N,) float64 array of unsigned distances.
        """
        n = len(points)
        if n == 0:
            return np.empty(0, dtype=np.float64)
        pts = wp.array(points.astype(np.float64), dtype=wp.vec3d)
        dists = wp.zeros(n, dtype=wp.float64)
        wp.launch(
            kernel_batched_distances,
            dim=n,
            inputs=[self._mesh_id, pts, wp.float64(max_dist), dists],
        )
        wp.synchronize()
        return dists.numpy()

    def conservative_coarse_targets(
        self,
        origin: np.ndarray,
        grid_shape: Tuple[int, int, int],
        voxel_size: float,
        config: AdaptiveMeshConfig,
    ) -> np.ndarray:
        """Assign levels conservatively at a coarse grid resolution.

        Each coarse cell queries the BVH from its center and assigns a level
        from the distance. Used as the initial step of octree refinement.

        Args:
            origin: Physical origin of the coarse grid.
            grid_shape: Coarse grid dimensions.
            voxel_size: Coarse cell size.
            config: Configuration with shell parameters.

        Returns:
            3D int32 array of conservative target levels at coarse resolution.
        """
        nx, ny, nz = grid_shape
        d_band = config.finest_band_cells * config.voxel_size
        log_ratio = math.log(config.expansion_ratio)
        max_dist = _max_query_distance(origin, grid_shape, voxel_size)
        targets = wp.zeros((nx, ny, nz), dtype=wp.int32)
        wp.launch(
            kernel_conservative_coarse_targets,
            dim=(nx, ny, nz),
            inputs=[
                self._mesh_id,
                wp.vec3d(float(origin[0]), float(origin[1]), float(origin[2])),
                wp.float64(voxel_size),
                wp.float64(max_dist),
                wp.float64(d_band),
                wp.float64(log_ratio),
                wp.int32(config.num_levels),
                targets,
            ],
        )
        wp.synchronize()
        return targets.numpy()

    def _min_filter_3x3(self, field: wp.array) -> wp.array:
        """3x3x3 minimum filter on GPU. Does NOT synchronize (caller's responsibility)."""
        out = wp.zeros(field.shape, dtype=wp.int32)
        wp.launch(kernel_minimum_filter_3x3, dim=field.shape, inputs=[field, out])
        return out

    def _max_filter_3x3(self, field: wp.array) -> wp.array:
        """3x3x3 maximum filter on GPU. Does NOT synchronize (caller's responsibility)."""
        out = wp.zeros(field.shape, dtype=wp.int32)
        wp.launch(kernel_maximum_filter_3x3, dim=field.shape, inputs=[field, out])
        return out

    def _any_nonzero_uint8(self, flags: wp.array) -> bool:
        counter = wp.zeros(1, dtype=wp.int32)
        wp.launch(kernel_any_nonzero_uint8, dim=flags.shape, inputs=[flags, counter])
        return self._sync_counter(counter) > 0

    def enforce_strong_balance_bidirectional(self, assigned: np.ndarray, max_passes: int = 64) -> np.ndarray:
        """Enforce strong balance (ΔL ≤ 1) using both refinement and coarsening.

        Iteratively applies 3x3x3 min/max filters to detect violations, then
        refines or coarsens cells until convergence. Unlike :meth:`finalize_owner_grid`,
        this allows both directions of change.

        Args:
            assigned: 3D int32 owner grid.
            max_passes: Maximum iteration count (safety limit).

        Returns:
            Balanced 3D int32 owner grid.
        """
        shape = assigned.shape
        cur = wp.array(assigned, dtype=wp.int32)
        tmp = wp.zeros(shape, dtype=wp.int32)
        refine_flags = wp.zeros(shape, dtype=wp.uint8)
        coarsen_flags = wp.zeros(shape, dtype=wp.uint8)

        for _ in range(max_passes):
            nmin = self._min_filter_3x3(cur)
            nmax = self._max_filter_3x3(cur)
            wp.launch(kernel_balance_need_refine, dim=shape, inputs=[cur, nmin, refine_flags])
            wp.launch(kernel_balance_need_coarsen, dim=shape, inputs=[cur, nmax, coarsen_flags])
            need_refine = self._any_nonzero_uint8(refine_flags)
            need_coarsen = self._any_nonzero_uint8(coarsen_flags)
            if not need_refine and not need_coarsen:
                break
            if need_refine:
                wp.launch(kernel_apply_balance_refine, dim=shape, inputs=[cur, nmin, tmp])
                cur = tmp
                tmp = wp.zeros(shape, dtype=wp.int32)
            if need_coarsen:
                wp.launch(kernel_apply_balance_coarsen, dim=shape, inputs=[cur, nmax, tmp])
                cur = tmp
                tmp = wp.zeros(shape, dtype=wp.int32)
        wp.synchronize()
        return cur.numpy()

    def minimum_with_floor(self, a: np.ndarray, floor: np.ndarray) -> np.ndarray:
        """Element-wise minimum clamped by a floor: result = max(min(a, ...), floor).

        Used to prevent the owner grid from being coarsened below its
        distance-based assignment.
        """
        a_wp = wp.array(a, dtype=wp.int32)
        f_wp = wp.array(floor, dtype=wp.int32)
        out = wp.zeros(a.shape, dtype=wp.int32)
        wp.launch(kernel_minimum_with_floor, dim=a.shape, inputs=[a_wp, f_wp, out])
        wp.synchronize()
        return out.numpy()

    def refine_transition_layers(
        self, owner: np.ndarray, owner_floor: np.ndarray, num_levels: int
    ) -> np.ndarray:
        """Refine transition bands between levels using min-filter promotion.

        For each level from coarsest-1 down to 0, cells adjacent to finer
        regions are promoted (lowered in level number) to create smooth
        transition bands.

        Args:
            owner: 3D int32 owner grid.
            owner_floor: Per-cell minimum level (prevents excessive coarsening).
            num_levels: Total levels.

        Returns:
            Refined 3D int32 owner grid.
        """
        cur = wp.array(owner, dtype=wp.int32)
        tmp = wp.zeros(owner.shape, dtype=wp.int32)
        for level in range(num_levels - 2, -1, -1):
            nmin = self._min_filter_3x3(cur)
            wp.launch(
                kernel_refine_transition,
                dim=owner.shape,
                inputs=[cur, nmin, wp.int32(level), tmp],
            )
            cur = tmp
            tmp = wp.zeros(owner.shape, dtype=wp.int32)
        wp.synchronize()
        result = cur.numpy()
        return np.minimum(result, owner_floor)

    def finalize_owner_grid(
        self, owner: np.ndarray, owner_floor: np.ndarray, num_levels: int
    ) -> np.ndarray:
        """Enforce strong balance (ΔL ≤ 1) using refine-only passes."""
        shape = owner.shape
        cur = wp.array(owner, dtype=wp.int32)
        tmp = wp.zeros(shape, dtype=wp.int32)
        refine_flags = wp.zeros(shape, dtype=wp.uint8)

        for _ in range(64):
            nmin = self._min_filter_3x3(cur)
            wp.launch(kernel_balance_need_refine, dim=shape, inputs=[cur, nmin, refine_flags])
            if not self._any_nonzero_uint8(refine_flags):
                break
            wp.launch(kernel_apply_balance_refine, dim=shape, inputs=[cur, nmin, tmp])
            cur = tmp
            tmp = wp.zeros(shape, dtype=wp.int32)

        wp.synchronize()
        return cur.numpy()

    @staticmethod
    def _align_owner_dyadic(owner_wp: wp.array, num_levels: int) -> wp.array:
        """Enforce dyadic block alignment on GPU so mask extraction has no orphans.

        For each level L in [0, num_levels-2], any 2^(L+1)-aligned block that
        contains at least one cell with owner <= L is fully absorbed (all cells
        in the block are set to L). This guarantees clean, non-overlapping masks.

        Args:
            owner_wp: Warp int32 3D array of owner levels (modified in-place).
            num_levels: Total number of refinement levels.

        Returns:
            The same Warp array (modified in-place).
        """
        nx, ny, nz = owner_wp.shape
        for level in range(num_levels - 1):
            stride = 2 ** (level + 1)
            sx, sy, sz = nx // stride, ny // stride, nz // stride
            if sx == 0 or sy == 0 or sz == 0:
                continue
            flags = wp.zeros((sx, sy, sz), dtype=wp.int32)
            wp.launch(
                kernel_dyadic_flag_blocks,
                dim=(nx, ny, nz),
                inputs=[owner_wp, wp.int32(level), wp.int32(stride), flags],
            )
            wp.launch(
                kernel_dyadic_apply_flags,
                dim=(nx, ny, nz),
                inputs=[owner_wp, wp.int32(level), wp.int32(stride), flags],
            )
        return owner_wp

    def build_masks_from_owner(
        self, owner: np.ndarray, num_levels: int, owner_floor: np.ndarray | None = None,
        origin: np.ndarray | None = None,
        voxel_size: float = 1.0,
        config=None,
    ) -> List[np.ndarray]:
        """Build non-overlapping, fully-covering, strongly-balanced masks from an owner grid.

        Algorithm stages:
          1. Enforce strong balance (refine-only) on the initial owner grid.
          2. If origin/config provided, recompute true STL distances for cells
             near the surface and reassign their levels (removes octree artifacts).
          3. Iterate finalize + dyadic alignment to a fixed point (both on GPU).
          4. Extract masks using greedy coarsest-first partitioning.
          5. Repair any remaining balance violations by subdivision.

        Args:
            owner: 3D int32 array of per-cell target levels (0=finest).
            num_levels: Total refinement levels.
            owner_floor: Maximum coarseness per cell (distance-based ceiling).
            origin: Physical origin of the grid (for distance recomputation).
            voxel_size: Physical size of finest cell.
            config: AdaptiveMeshConfig for distance-based reassignment.

        Returns:
            List of boolean mask arrays, one per level.
        """
        floor = owner.copy() if owner_floor is None else owner_floor
        owner = self.finalize_owner_grid(owner, floor, num_levels)

        if origin is not None and config is not None:
            coarsest = num_levels - 1
            band_region = owner < coarsest
            # GPU-accelerated face-connected dilation for band growth
            radius = 2**coarsest
            band_wp = wp.array(band_region.astype(np.uint8), dtype=wp.uint8)
            tmp_wp = wp.zeros_like(band_wp)
            for _ in range(radius):
                wp.launch(kernel_dilate_mask_uint8, dim=band_wp.shape, inputs=[band_wp, tmp_wp])
                band_wp, tmp_wp = tmp_wp, band_wp
            band_region = band_wp.numpy().astype(bool)

            cells = np.argwhere(band_region)
            if len(cells) > 0:
                orig = origin.astype(np.float64)
                centers = orig + (cells.astype(np.float64) + 0.5) * voxel_size
                max_dist = float(np.linalg.norm(np.array(owner.shape) * voxel_size))
                dists = self.batched_distances(centers, max_dist)
                new_levels = self.assign_levels_from_distances_1d(dists, config)
                ci, cj, ck = cells[:, 0], cells[:, 1], cells[:, 2]
                owner[ci, cj, ck] = new_levels.astype(owner.dtype)

        # Fixed-point iteration: finalize (refine-only balance) + dyadic alignment.
        # Both are monotonically refining so convergence is guaranteed.
        for _ in range(num_levels + 2):
            owner = self.finalize_owner_grid(owner, floor, num_levels)
            owner_wp = wp.array(owner, dtype=wp.int32)
            owner_wp = self._align_owner_dyadic(owner_wp, num_levels)
            owner_new = owner_wp.numpy()
            if np.array_equal(owner, owner_new):
                break
            owner = owner_new

        masks = _build_masks_greedy_coarsest(owner, num_levels)
        masks = self.repair_balance_by_subdivision(masks, num_levels, owner.shape)
        return masks

    def build_non_overlapping_masks_vectorized(
        self, owner: np.ndarray, num_levels: int
    ) -> List[np.ndarray]:
        """Extract per-level masks where each block is uniform in owner (GPU)."""
        nx, ny, nz = owner.shape
        owner_wp = wp.array(owner, dtype=wp.int32)
        masks = []
        for level in range(num_levels):
            stride = 2**level
            sx, sy, sz = nx // stride, ny // stride, nz // stride
            mask_wp = wp.zeros((sx, sy, sz), dtype=wp.uint8)
            wp.launch(
                kernel_extract_level_mask,
                dim=(sx, sy, sz),
                inputs=[owner_wp, wp.int32(level), wp.int32(stride), mask_wp],
            )
            masks.append(mask_wp)
        wp.synchronize()
        return [m.numpy().astype(bool) for m in masks]

    def repair_balance_by_subdivision(
        self,
        masks: List[np.ndarray],
        num_levels: int,
        grid_shape: Tuple[int, int, int],
        max_passes: int = 96,
    ) -> List[np.ndarray]:
        """Repair strong-balance violations in extracted masks by subdividing coarse blocks.

        After greedy mask extraction, some block boundaries may violate the
        ΔL ≤ 1 constraint. This method iteratively identifies violating coarse
        blocks (via 26-neighbor owner checks) and subdivides them into the next
        finer level until the constraint is satisfied everywhere.

        Args:
            masks: List of boolean mask arrays (one per level).
            num_levels: Total levels.
            grid_shape: Finest-grid dimensions.
            max_passes: Safety limit on iterations.

        Returns:
            Repaired list of boolean mask arrays.
        """
        nx, ny, nz = grid_shape
        mask_wps = [wp.array(m.astype(np.uint8), dtype=wp.uint8) for m in masks]
        padded_masks = _pad_mask_list(mask_wps, num_levels)
        owners = wp.zeros((nx, ny, nz), dtype=wp.int32)
        subdivide = [wp.zeros(m.shape, dtype=wp.uint8) for m in mask_wps]
        padded_sub = _pad_mask_list(subdivide, num_levels)
        counter = wp.zeros(1, dtype=wp.int32)

        for _ in range(max_passes):
            wp.launch(
                kernel_owners_from_masks,
                dim=(nx, ny, nz),
                inputs=[wp.int32(num_levels), *padded_masks, owners],
            )
            for sub in subdivide:
                sub.zero_()
            for di, dj, dk in _NEIGHBOR_OFFSETS_26:
                wp.launch(
                    kernel_mark_subdivide_offset,
                    dim=(nx, ny, nz),
                    inputs=[owners, wp.int32(di), wp.int32(dj), wp.int32(dk), *padded_sub],
                )
            counter.zero_()
            for sub in subdivide:
                wp.launch(kernel_any_nonzero_uint8, dim=sub.shape, inputs=[sub, counter])
            if self._sync_counter(counter) == 0:
                break
            changed = False
            for level in range(1, num_levels):
                wp.launch(
                    kernel_apply_subdivide_level,
                    dim=mask_wps[level].shape,
                    inputs=[subdivide[level], mask_wps[level], mask_wps[level - 1]],
                )
                counter.zero_()
                wp.launch(kernel_any_nonzero_uint8, dim=subdivide[level].shape, inputs=[subdivide[level], counter])
                if self._sync_counter(counter) > 0:
                    changed = True
            if not changed:
                break

        wp.synchronize()
        return [m.numpy().astype(bool) for m in mask_wps]

    def fill_coverage_gaps(
        self,
        masks: List[np.ndarray],
        grid_shape: Tuple[int, int, int],
        num_levels: int,
    ) -> List[np.ndarray]:
        """Fill any uncovered finest-grid cells by activating them in the coarsest mask.

        This is a safety net to ensure the mesh is fully covering even if the
        greedy extraction missed edge cases (e.g., domain boundaries).

        Args:
            masks: List of boolean mask arrays.
            grid_shape: Finest-grid dimensions.
            num_levels: Total levels.

        Returns:
            Updated list of masks with coverage gaps filled.
        """
        nx, ny, nz = grid_shape
        mask_wps = [wp.array(m.astype(np.uint8), dtype=wp.uint8) for m in masks]
        padded_masks = _pad_mask_list(mask_wps, num_levels)
        covered = wp.zeros((nx, ny, nz), dtype=wp.uint8)
        wp.launch(
            kernel_mark_coverage,
            dim=(nx, ny, nz),
            inputs=[wp.int32(num_levels), *padded_masks, covered],
        )
        counter = wp.zeros(1, dtype=wp.int32)
        wp.launch(kernel_any_uncovered, dim=(nx, ny, nz), inputs=[covered, counter])
        if self._sync_counter(counter) == 0:
            return masks
        coarsest = num_levels - 1
        coarse_stride = 2**coarsest
        wp.launch(
            kernel_fill_coverage_gaps,
            dim=(nx, ny, nz),
            inputs=[covered, mask_wps[coarsest], wp.int32(coarse_stride)],
        )
        wp.synchronize()
        return [m.numpy().astype(bool) for m in mask_wps]

    def paint_owner_from_assignments(
        self,
        level_indices: List[np.ndarray],
        tight_shape: Tuple[int, int, int],
        fi_min: int,
        fj_min: int,
        fk_min: int,
        num_levels: int,
    ) -> np.ndarray:
        """GPU scatter of octree assignments into a tight owner grid."""
        owner = wp.full(tight_shape, wp.int32(num_levels - 1), dtype=wp.int32)
        for target in range(num_levels):
            cells = level_indices[target]
            n_blocks = len(cells)
            if n_blocks == 0:
                continue
            stride = 2**target
            block_i = wp.array(cells[:, 0].astype(np.int32), dtype=wp.int32)
            block_j = wp.array(cells[:, 1].astype(np.int32), dtype=wp.int32)
            block_k = wp.array(cells[:, 2].astype(np.int32), dtype=wp.int32)
            wp.launch(
                kernel_paint_blocks_level,
                dim=(n_blocks, stride, stride, stride),
                inputs=[
                    wp.int32(target),
                    block_i,
                    block_j,
                    block_k,
                    wp.int32(fi_min),
                    wp.int32(fj_min),
                    wp.int32(fk_min),
                    wp.int32(stride),
                    owner,
                ],
            )
        wp.synchronize()
        return owner.numpy()


# ---------------------------------------------------------------------------
# Dense and octree meshing paths
# ---------------------------------------------------------------------------

def _make_masks_dense(
    mesh: trimesh.Trimesh,
    origin_phys: np.ndarray,
    grid_shape: Tuple[int, int, int],
    config: AdaptiveMeshConfig,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Dense-path mesh generation: compute full finest-grid distances then build masks.

    Suitable for small-to-medium domains (below ``max_dense_cells``). Computes
    the distance field for every finest-grid cell, assigns levels, and builds
    masks via the owner-grid pipeline.

    Args:
        mesh: Input triangle mesh.
        origin_phys: Physical domain origin.
        grid_shape: Finest-grid dimensions.
        config: Mesh configuration.

    Returns:
        Tuple of (masks list, mask_origins list).
    """
    ops = WarpAdaptiveMesherOps(mesh)
    distances = ops.compute_distance_field(origin_phys, grid_shape, config.voxel_size, config)
    assigned = ops.assign_levels_from_distance(distances, config)
    owner_floor = assigned.copy()
    masks = ops.build_masks_from_owner(assigned, config.num_levels, owner_floor=owner_floor,
                                        origin=origin_phys, voxel_size=config.voxel_size,
                                        config=config)
    mask_origins = [np.zeros(3, dtype=int) for _ in range(config.num_levels)]
    return masks, mask_origins


def _make_masks_octree(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    grid_shape_finest: Tuple[int, int, int],
    config: AdaptiveMeshConfig,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Octree-path mesh generation: refine hierarchically from coarsest to finest.

    Avoids allocating the full finest-grid distance field. Starts at the
    coarsest level with conservative distance queries, then iteratively
    subdivides cells that need finer resolution (2x2x2 children at each step).

    Args:
        mesh: Input triangle mesh.
        origin: Physical domain origin.
        grid_shape_finest: Finest-level grid dimensions.
        config: Mesh configuration.

    Returns:
        Tuple of (masks list, mask_origins list).
    """
    ops = WarpAdaptiveMesherOps(mesh)
    num_levels = config.num_levels
    coarsest = num_levels - 1
    assignments: List[List[np.ndarray]] = [[] for _ in range(num_levels)]

    shape_c = _shape_at_level(grid_shape_finest, coarsest)
    voxel_c = config.voxel_size * (2**coarsest)
    n_c = int(np.prod(shape_c))
    print(
        f"  Level {coarsest} (coarsest): conservative distance on {shape_c} ({n_c:,} cells, voxel={voxel_c:.1f} m)",
        flush=True,
    )
    t0 = time.perf_counter()
    target_c = ops.conservative_coarse_targets(origin, shape_c, voxel_c, config)
    print(f"    distance done in {time.perf_counter() - t0:.1f}s", flush=True)
    coarse_keep = np.argwhere(target_c == coarsest)
    if len(coarse_keep):
        assignments[coarsest].append(coarse_keep)

    parent_refine = np.argwhere(target_c <= coarsest - 1) if num_levels > 1 else np.empty((0, 3), dtype=int)
    parent_level = coarsest
    max_dist = _max_query_distance(origin, grid_shape_finest, config.voxel_size)

    for level in range(coarsest - 1, -1, -1):
        n_refine = len(parent_refine)
        print(f"  Level {level}: refining {n_refine:,} parent cells", flush=True)
        if n_refine == 0:
            continue

        parent_voxel = config.voxel_size * (2**parent_level)
        child_voxel = config.voxel_size * (2**level)

        t0 = time.perf_counter()
        centers, keys = _child_centers_and_keys(parent_refine, origin, parent_voxel, child_voxel)
        dists = ops.batched_distances(centers, max_dist)
        child_targets = ops.assign_levels_from_distances_1d(dists, config)
        _record_assignments(assignments, keys, child_targets, level, num_levels)
        n_active = int(np.sum(child_targets == level))
        print(
            f"    {len(parent_refine):,} parents, {n_active:,} active at level {level} "
            f"in {time.perf_counter() - t0:.1f}s",
            flush=True,
        )

        parent_refine = keys[child_targets <= level - 1] if level > 0 else np.empty((0, 3), dtype=int)
        parent_level = level

    print("  Building sparse active-voxel coordinates from assignments...", flush=True)
    t0 = time.perf_counter()
    active_coords = _assignments_to_active_coords(assignments, num_levels)
    mask_origins = [np.zeros(3, dtype=int) for _ in range(num_levels)]
    print(f"    sparse coords built in {time.perf_counter() - t0:.1f}s", flush=True)
    for level, coords in enumerate(active_coords):
        print(f"    level {level}: {len(coords):,} active voxels", flush=True)
    return active_coords, mask_origins


def _build_masks_from_assignments(
    ops: WarpAdaptiveMesherOps,
    assignments: List[List[np.ndarray]],
    grid_shape_finest: Tuple[int, int, int],
    num_levels: int,
    max_dense_cells: int,
    origin: np.ndarray | None = None,
    config=None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Convert octree cell assignments into non-overlapping masks via GPU paint + build.

    Takes accumulated per-level cell indices from the octree traversal, paints
    them onto a full-resolution owner grid using GPU scatter, and then runs the
    standard mask-building pipeline (balance, alignment, greedy extraction).

    Args:
        ops: WarpAdaptiveMesherOps instance with loaded BVH mesh.
        assignments: Per-level lists of (N, 3) cell-index arrays.
        grid_shape_finest: Finest grid dimensions.
        num_levels: Total levels.
        max_dense_cells: Unused (kept for API compatibility).
        origin: Physical domain origin (for distance recomputation).
        config: Mesh configuration.

    Returns:
        Tuple of (masks list, mask_origins list).
    """
    nx, ny, nz = grid_shape_finest

    level_indices: List[np.ndarray] = []
    for target in range(num_levels):
        if assignments[target]:
            arr = np.unique(np.vstack(assignments[target]), axis=0)
        else:
            arr = np.empty((0, 3), dtype=int)
        level_indices.append(arr)

    if all(len(a) == 0 for a in level_indices):
        raise RuntimeError("No cell assignments collected during octree refinement.")

    owner = ops.paint_owner_from_assignments(
        level_indices, (nx, ny, nz), 0, 0, 0, num_levels
    )
    owner_floor = owner.copy()
    voxel_size = config.voxel_size if config is not None else 1.0
    masks = ops.build_masks_from_owner(owner, num_levels, owner_floor=owner_floor,
                                       origin=origin, voxel_size=voxel_size,
                                       config=config)
    mask_origins = [np.zeros(3, dtype=int) for _ in range(num_levels)]
    return masks, mask_origins


# ---------------------------------------------------------------------------
# Level-data packing
# ---------------------------------------------------------------------------

def _pack_level_data(
    patterns: List[np.ndarray],
    mask_origins: List[np.ndarray],
    origin_phys: np.ndarray,
    voxel_size_finest: float,
    num_levels: int,
    *,
    sparse: bool = False,
) -> list:
    """Pack per-level masks or sparse coords into level_data tuples.

    Output ordering is coarsest-first (reversed from the internal finest-first
    convention) to match the format expected by :func:`_normalize_level_data`.

    Each entry is ``(mask_or_coords, voxel_size_at_level, physical_origin, build_level)``.
    When ``sparse=True``, the first element is an (N, 3) int32 coordinate array.

    Args:
        patterns: Per-level boolean masks or sparse coordinate arrays (finest-first).
        mask_origins: Per-level grid-index origins.
        origin_phys: Physical domain origin.
        voxel_size_finest: Finest cell physical size.
        num_levels: Total levels.
        sparse: When True, ``patterns`` holds (N, 3) coordinate arrays.

    Returns:
        List of tuples in coarsest-first order.
    """
    raw_level_data = []
    for finest_level in range(num_levels):
        pattern = patterns[finest_level]
        voxel_size_level = voxel_size_finest * (2**finest_level)
        offset = mask_origins[finest_level]
        sub_origin_phys = origin_phys + offset.astype(float) * voxel_size_finest
        cuboid_build_level = num_levels - 1 - finest_level
        if sparse:
            active = int(pattern.shape[0])
            shape_desc = f"coords ({active:,} x 3)"
            packed = np.ascontiguousarray(pattern, dtype=np.int32)
        else:
            active = int(np.count_nonzero(pattern))
            shape_desc = f"shape {pattern.shape}"
            packed = pattern.copy()

        print(
            f"Level {finest_level}: {shape_desc}, active {active:,}, "
            f"origin offset {offset}, voxel_size {voxel_size_level}"
        )
        if active == 0:
            print(f"  (level {finest_level} unused: finer levels fully cover the domain)")
        raw_level_data.append((packed, voxel_size_level, sub_origin_phys, cuboid_build_level))

    return list(reversed(raw_level_data))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_adaptive_surface_mesh(
    voxel_size: float,
    num_levels: int,
    stl_filename: str,
    domain_padding: Sequence[float] = None,
    expansion_ratio: float = 2.0,
    finest_band_cells: int = 3,
    tile_size: int = 64,
    max_dense_cells: int = 128**3,
) -> list:
    """
    Create a strongly-balanced surface-adaptive multires mesh from an STL file.

    Fine cells are placed within ``finest_band_cells * voxel_size`` of the
    surface; coarser levels extend outward with geometric ``expansion_ratio``
    growth between consecutive shells.

    Args:
        voxel_size: Physical size of the finest lattice cell.
        num_levels: Number of refinement levels (finest = 0).
        stl_filename: Path to the STL/OBJ geometry file.
        domain_padding: 6-tuple ``[-x, +x, -y, +y, -z, +z]`` padding multipliers
            for the coarsest outer domain (relative to geometry extent).
        expansion_ratio: Geometric ratio between consecutive distance shells.
        finest_band_cells: Thickness of the finest shell in finest-cell counts.
        tile_size: Tile edge length for chunked distance queries on large domains.
        max_dense_cells: Use dense distance computation when the domain is smaller.

    Returns:
        ``level_data`` list (finest-first), compatible with
        :func:`xlb.utils.mesher.prepare_sparsity_pattern`.
    """
    if domain_padding is None:
        domain_padding = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    config = AdaptiveMeshConfig(
        voxel_size=voxel_size,
        num_levels=num_levels,
        expansion_ratio=expansion_ratio,
        finest_band_cells=finest_band_cells,
        domain_padding=list(domain_padding),
        stl_filename=stl_filename,
        tile_size=tile_size,
        max_dense_cells=max_dense_cells,
    )

    mesh, origin_phys, grid_shape = _compute_domain(config)
    nx, ny, nz = grid_shape
    n_finest = nx * ny * nz
    print(f"Adaptive mesh domain (finest): {grid_shape}, origin {origin_phys}, voxel_size {voxel_size}")
    print(f"  Finest-grid cell count: {n_finest:,} (dense limit: {max_dense_cells:,})")

    factor = 2 ** (num_levels - 1)
    pad_x = (factor - nx % factor) % factor
    pad_y = (factor - ny % factor) % factor
    pad_z = (factor - nz % factor) % factor
    if pad_x or pad_y or pad_z:
        nx += pad_x
        ny += pad_y
        nz += pad_z
        grid_shape = (nx, ny, nz)
        n_finest = nx * ny * nz
        print(f"Padded domain to {grid_shape} for dyadic nesting (factor {factor})")

    align = 2**num_levels
    aligned_shape = tuple(((n + align - 1) // align) * align for n in grid_shape)
    if aligned_shape != grid_shape:
        grid_shape = aligned_shape
        n_finest = int(np.prod(grid_shape))
        print(f"Aligned domain to {grid_shape} for adaptive partitioning (align={align})")

    if n_finest > max_dense_cells:
        print("Using octree surface meshing (avoids full finest-grid allocation).", flush=True)
        patterns, mask_origins = _make_masks_octree(mesh, origin_phys, grid_shape, config)
        raw_level_data = _pack_level_data(
            patterns, mask_origins, origin_phys, voxel_size, num_levels, sparse=True
        )
    else:
        print("Using dense finest-grid meshing.", flush=True)
        masks, mask_origins = _make_masks_dense(mesh, origin_phys, grid_shape, config)
        raw_level_data = _pack_level_data(masks, mask_origins, origin_phys, voxel_size, num_levels)

    level_data = LevelDataList(_normalize_level_data(raw_level_data, voxel_size))
    level_data.grid_shape_finest = grid_shape
    return level_data


# ---------------------------------------------------------------------------
# Validation and inspection utilities
# ---------------------------------------------------------------------------

def _embed_level_mask_on_finest(
    mask: np.ndarray,
    stride: int,
    origin_native: np.ndarray,
    base_origin: np.ndarray,
    grid_shape: Tuple[int, int, int],
) -> Tuple[Tuple[int, int, int, int, int, int], np.ndarray]:
    """Expand a level mask into a finest-grid slab (boolean, clipped to domain)."""
    nx, ny, nz = grid_shape
    origin_finest = np.asarray(origin_native, dtype=int) * stride - base_origin
    ox, oy, oz = int(origin_finest[0]), int(origin_finest[1]), int(origin_finest[2])
    sx, sy, sz = mask.shape
    expanded = np.repeat(
        np.repeat(np.repeat(mask, stride, axis=0), stride, axis=1),
        stride,
        axis=2,
    )
    x0, y0, z0 = max(0, ox), max(0, oy), max(0, oz)
    x1 = min(nx, ox + sx * stride)
    y1 = min(ny, oy + sy * stride)
    z1 = min(nz, oz + sz * stride)
    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        return (0, 0, 0, 0, 0, 0), np.zeros((0, 0, 0), dtype=bool)
    ex0, ey0, ez0 = x0 - ox, y0 - oy, z0 - oz
    ex1, ey1, ez1 = ex0 + (x1 - x0), ey0 + (y1 - y0), ez0 + (z1 - z0)
    return (x0, x1, y0, y1, z0, z1), expanded[ex0:ex1, ey0:ey1, ez0:ez1]


def validate_level_data(level_data: list, grid_shape_finest: Tuple[int, int, int]) -> dict:
    """
    Validate ``level_data`` for non-overlap and strong balance (ΔL ≤ 1).

    Returns a statistics dictionary with per-level active counts and validation flags.
    """
    stats = {"num_levels": len(level_data), "active_counts": [], "strongly_balanced": True, "non_overlapping": True}

    nx, ny, nz = grid_shape_finest
    coverage = np.zeros((nx, ny, nz), dtype=np.int16)
    finest_owners = np.full((nx, ny, nz), -1, dtype=np.int32)

    finest_origins = [np.asarray(entry[2], dtype=int) * int(entry[1]) for entry in level_data]
    base_origin = np.min(finest_origins, axis=0)

    for mask, stride_lattice, origin, level_id in level_data:
        stats["active_counts"].append(int(np.count_nonzero(mask)))
        if not np.any(mask):
            continue
        stride = int(stride_lattice)
        bounds, slab = _embed_level_mask_on_finest(mask, stride, origin, base_origin, grid_shape_finest)
        if slab.size == 0:
            continue
        x0, x1, y0, y1, z0, z1 = bounds
        region_cov = coverage[x0:x1, y0:y1, z0:z1]
        if np.any(region_cov[slab] > 0):
            stats["non_overlapping"] = False
        region_cov[slab] += 1
        finest_owners[x0:x1, y0:y1, z0:z1][slab] = level_id

    stats["fully_covering"] = not np.any(coverage == 0)

    marked = finest_owners >= 0
    if not stats["fully_covering"]:
        stats["strongly_balanced"] = False
    elif np.any(marked):
        for di, dj, dk in _NEIGHBOR_OFFSETS_26:
            n_o = _shift_toward_offset(finest_owners, di, dj, dk, -1)
            n_m = _shift_toward_offset(marked.astype(np.int8), di, dj, dk, 0).astype(bool)
            valid = marked & n_m & (n_o >= 0)
            if np.any(valid & (np.abs(finest_owners - n_o) > 1)):
                stats["strongly_balanced"] = False
                break

    return stats


def grid_shape_finest(level_data: list) -> Tuple[int, int, int]:
    """Finest lattice shape implied by level_data or attached metadata."""
    stored = getattr(level_data, "grid_shape_finest", None)
    if stored is not None:
        return tuple(int(x) for x in stored)
    if is_sparse_level_data(level_data):
        raise ValueError(
            "Sparse level_data has no attached grid_shape_finest; "
            "call make_adaptive_surface_mesh or set level_data.grid_shape_finest."
        )
    num_levels = len(level_data)
    return tuple(int(level_data[-1][0].shape[i] * 2 ** (num_levels - 1)) for i in range(3))


def default_cuboid_multipliers(domain_padding: Sequence[float], num_levels: int) -> List[List[float]]:
    """Build nested cuboid domain multipliers from the outer padding."""
    if num_levels < 1:
        raise ValueError("num_levels must be at least 1.")
    if num_levels == 1:
        return [list(domain_padding)]

    scale_steps = [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.7, 0.7, 0.7, 0.7, 0.8, 0.8],
        [0.4, 0.4, 0.4, 0.4, 0.6, 0.6],
        [0.2, 0.2, 0.2, 0.2, 0.4, 0.4],
    ]
    multipliers: List[List[float]] = []
    for level in range(num_levels):
        if level == 0:
            multipliers.append(list(domain_padding))
            continue
        step = scale_steps[min(level, len(scale_steps) - 1)]
        multipliers.append([domain_padding[i] * step[i] for i in range(6)])
    return multipliers


def load_and_shift_stl(stl_path: str, domain_padding: Sequence[float]) -> Tuple[str, np.ndarray]:
    """
    Load geometry and translate it into the mesh-domain frame.

    Returns a temporary STL path and the translation applied (for export offset).
    """
    loaded = trimesh.load(stl_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            raise ValueError(f"Loaded mesh is empty: {stl_path}")
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    else:
        mesh = loaded

    min_bound = mesh.vertices.min(axis=0)
    max_bound = mesh.vertices.max(axis=0)
    part_size = max_bound - min_bound

    stl_shift = np.array(
        [
            domain_padding[0] * part_size[0] - min_bound[0],
            domain_padding[2] * part_size[1] - min_bound[1],
            domain_padding[4] * part_size[2] - min_bound[2],
        ],
        dtype=float,
    )
    mesh.apply_translation(stl_shift)
    _ = mesh.vertex_normals

    fd, temp_path = tempfile.mkstemp(suffix=".stl", prefix="adaptive_mesh_")
    os.close(fd)
    mesh.export(temp_path)
    return temp_path, stl_shift


def print_mesh_statistics(
    label: str, level_data: list, grid_shape_finest_grid: Tuple[int, int, int]
) -> Tuple[int, int]:
    """Print per-level counts, validation flags; return total active / equiv. finest."""
    from xlb.utils.mesher import prepare_sparsity_pattern

    num_levels = len(level_data)
    sparsity_pattern, level_origins = prepare_sparsity_pattern(level_data)

    print(f"\n{label}")
    print("=" * len(label))
    print(f"Finest grid shape: {grid_shape_finest_grid}")
    sparse = is_sparse_level_data(level_data)

    def _active_count(pattern: np.ndarray) -> int:
        if pattern.ndim == 2:
            return int(pattern.shape[0])
        return int(np.count_nonzero(pattern))

    for lvl in range(num_levels):
        active = _active_count(sparsity_pattern[lvl])
        equiv_finest = active * (2 ** (num_levels - 1 - lvl))
        shape_desc = (
            f"coords {sparsity_pattern[lvl].shape}"
            if sparse
            else f"mask shape={sparsity_pattern[lvl].shape}"
        )
        print(
            f"  Level {lvl}: active={active:,}, "
            f"{shape_desc}, "
            f"origin={level_origins[lvl]}, "
            f"equiv. finest cells={equiv_finest:,}"
        )

    total_active = sum(_active_count(m) for m in sparsity_pattern)
    total_equiv_finest = sum(
        _active_count(sparsity_pattern[lvl]) * (2 ** (num_levels - 1 - lvl))
        for lvl in range(num_levels)
    )
    print(f"  Total active cells: {total_active:,}")
    print(f"  Total equivalent finest cells: {total_equiv_finest:,}")

    if sparse:
        print("  Validation: skipped (sparse level_data; dense validation not applicable)")
        stats = {
            "non_overlapping": None,
            "fully_covering": None,
            "strongly_balanced": None,
        }
    else:
        stats = validate_level_data(level_data, grid_shape_finest_grid)
        print(
            f"  Validation: non_overlapping={stats['non_overlapping']}, "
            f"fully_covering={stats['fully_covering']}, "
            f"strongly_balanced={stats['strongly_balanced']}"
        )
    return total_active, total_equiv_finest


def export_mesh_xdmf(
    level_data: list,
    voxel_size: float,
    output_basename: str,
    export_offset: Tuple[float, float, float],
    original_stl: str,
) -> None:
    """Write HDF5/XDMF geometry for ParaView inspection."""
    from xlb.utils.mesher import MultiresIO

    exporter = MultiresIO.__new__(MultiresIO)
    exporter.unit_convertor = None
    coords, conn, level_ids, n_cells = MultiresIO.process_geometry(exporter, level_data)
    coords, conn = MultiresIO._merge_duplicates(exporter, coords, conn, level_data)
    coords = MultiresIO._transform_coordinates(exporter, coords * voxel_size, export_offset)
    MultiresIO.save_xdmf(exporter, f"{output_basename}.h5", f"{output_basename}.xmf", n_cells, len(coords), fields={})
    MultiresIO.save_hdf5_file(exporter, output_basename, coords, conn, level_ids, fields_data={})
    print(f"\nExported {n_cells:,} cells to {output_basename}.xmf")
    if export_offset != (0.0, 0.0, 0.0):
        print(f"  Coordinates in original STL frame (offset applied: {export_offset})")
    print(f"  Overlay in ParaView with: {original_stl}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for surface-adaptive mesh generation."""
    parser = argparse.ArgumentParser(
        description="Build and inspect a surface-adaptive multires mesh from an STL/OBJ file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Level shells: finest band width = finest_band_cells * voxel_size; "
            "each outer shell grows by expansion_ratio."
        ),
    )
    parser.add_argument("--stl", required=True, help="Path to input STL/OBJ geometry.")
    parser.add_argument("--voxel-size", type=float, default=4.0, help="Finest cell size (default: 4.0).")
    parser.add_argument("--num-levels", type=int, default=4, help="Refinement levels; 0 is finest (default: 4).")
    parser.add_argument(
        "--expansion-ratio", type=float, default=2.0, help="Shell growth ratio (default: 2.0)."
    )
    parser.add_argument(
        "--finest-band-cells", type=int, default=3, help="Finest shell thickness in cells (default: 3)."
    )
    parser.add_argument(
        "--domain-padding",
        type=float,
        nargs=6,
        metavar=("MX", "PX", "MY", "PY", "MZ", "PZ"),
        default=[0.5, 0.5, 0.5, 0.5, 0.25, 1.0],
        help="Padding [-x,+x,-y,+y,-z,+z] x geometry extent.",
    )
    parser.add_argument(
        "--shift-stl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Translate STL into mesh-domain frame before meshing (default: on).",
    )
    parser.add_argument("--compare-cuboid", action="store_true", help="Compare vs nested cuboid mesh.")
    parser.add_argument("--export-mesh", action="store_true", help="Export HDF5/XDMF for ParaView.")
    parser.add_argument("-o", "--output", default=None, help="Export basename (default: <stl_stem>_adaptive_mesh).")
    parser.add_argument(
        "--max-dense-cells",
        type=int,
        default=128**3,
        help="Dense distance field limit (default: 128^3).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for building and inspecting adaptive surface meshes."""
    from xlb.utils.mesher import make_cuboid_mesh

    args = build_parser().parse_args(argv)

    stl_path = os.path.abspath(args.stl)
    if not os.path.isfile(stl_path):
        print(f"STL not found: {stl_path}", file=sys.stderr)
        return 1

    domain_padding = list(args.domain_padding)
    output_basename = args.output or f"{os.path.splitext(os.path.basename(stl_path))[0]}_adaptive_mesh"

    print("Surface-adaptive mesh generation")
    print(f"  STL: {stl_path}")
    print(f"  voxel_size={args.voxel_size}, num_levels={args.num_levels}")
    print(f"  expansion_ratio={args.expansion_ratio}, finest_band_cells={args.finest_band_cells}")
    print(f"  domain_padding={domain_padding}")
    print(f"  shift_stl={args.shift_stl}")

    temp_stl: str | None = None
    stl_shift = np.zeros(3, dtype=float)
    mesh_stl = stl_path

    try:
        if args.shift_stl:
            temp_stl, stl_shift = load_and_shift_stl(stl_path, domain_padding)
            mesh_stl = temp_stl
            print(f"  STL shift applied: {stl_shift}")

        t0 = time.perf_counter()
        level_data = make_adaptive_surface_mesh(
            voxel_size=args.voxel_size,
            num_levels=args.num_levels,
            stl_filename=mesh_stl,
            domain_padding=domain_padding,
            expansion_ratio=args.expansion_ratio,
            finest_band_cells=args.finest_band_cells,
            max_dense_cells=args.max_dense_cells,
        )
        print(f"\nAdaptive mesh built in {time.perf_counter() - t0:.1f} s")

        gs = grid_shape_finest(level_data)
        adaptive_total, adaptive_equiv = print_mesh_statistics("Adaptive surface mesh", level_data, gs)

        if args.compare_cuboid:
            cuboid_multipliers = default_cuboid_multipliers(domain_padding, args.num_levels)
            t0 = time.perf_counter()
            cuboid_data = make_cuboid_mesh(args.voxel_size, cuboid_multipliers, mesh_stl)
            print(f"\nCuboid mesh built in {time.perf_counter() - t0:.1f} s")

            cuboid_gs = grid_shape_finest(cuboid_data)
            cuboid_total, cuboid_equiv = print_mesh_statistics("Cuboid mesh (comparison)", cuboid_data, cuboid_gs)

            print("\nComparison")
            print("==========")
            print(f"  Adaptive finest-level cells: {int(np.count_nonzero(level_data[0][0])):,}")
            print(f"  Cuboid finest-level cells:   {int(np.count_nonzero(cuboid_data[0][0])):,}")
            print(f"  Adaptive total active:       {adaptive_total:,}")
            print(f"  Cuboid total active:         {cuboid_total:,}")
            print(f"  Adaptive equiv. finest:      {adaptive_equiv:,}")
            print(f"  Cuboid equiv. finest:        {cuboid_equiv:,}")
            if cuboid_equiv > 0:
                savings = 100.0 * (1.0 - adaptive_equiv / cuboid_equiv)
                print(f"  Equivalent-finest savings:   {savings:.1f}%")

        if args.export_mesh:
            export_offset = tuple(-stl_shift) if args.shift_stl else (0.0, 0.0, 0.0)
            export_mesh_xdmf(level_data, args.voxel_size, output_basename, export_offset, stl_path)

    finally:
        if temp_stl is not None and os.path.isfile(temp_stl):
            os.remove(temp_stl)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
