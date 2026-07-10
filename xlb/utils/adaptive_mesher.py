"""
Surface-adaptive multi-resolution mesh generation (Warp GPU backend).

This module generates graded-octree multi-resolution grids that conform
tightly to STL/OBJ surfaces with a non-overlap guarantee across levels.
Fine cells are placed within a configurable distance band of the geometry,
and progressively coarser cells fill the remaining domain volume using
geometric expansion-ratio shells.

Note: Final strong-balance enforcement (delta-L <= 1 between face-adjacent
cells) is performed by Neon's internal ``mGrid`` construction pass.  The
octree produced here is *graded* but may not be strictly strongly-balanced
until Neon processes it.

**Architecture:**

1. Octree refinement: GPU BVH distance queries at coarse levels, subdividing
   only where finer resolution is needed near the STL surface.
2. Level assignment: Each cell receives a target refinement level from its
   distance to the surface (geometric-ratio shells).
3. Sparse output: Per-level active-voxel coordinates ``(N, 3)`` — no dense
   finest-grid allocation at any stage.

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
    kernel_assign_levels_1d,
    kernel_batched_distances,
    kernel_conservative_coarse_targets,
    kernel_edt_init_from_mask,
    kernel_edt_pass_x,
    kernel_edt_pass_y,
    kernel_edt_pass_z,
    kernel_edt_sqrt,
    kernel_maximum_filter_3x3,
    kernel_minimum_filter_3x3,
)
from xlb.utils.mesher import (
    _align_domain_origin_for_dyadic,
    _domain_bbox_from_padding,
    _load_stl_mesh,
    _normalize_level_data,
    _stl_bounds,
    adjust_bbox,
    grid_shape_finest_from_level_data,
    is_sparse_level_data,
)

# 26-neighbor offsets (excluding self) on a 3-D Cartesian grid.
_NEIGHBOR_OFFSETS_26 = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if not (dx == 0 and dy == 0 and dz == 0)]


@dataclass
class AdaptiveMeshConfig:
    """Configuration for surface-adaptive multires meshing."""

    voxel_size: float
    num_levels: int
    expansion_ratio: float = 2.0
    finest_band_cells: int = 3
    domain_padding: Sequence[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    stl_filename: str = ""

    def __post_init__(self):
        if self.num_levels < 1:
            raise ValueError("num_levels must be at least 1.")
        if self.num_levels > 8:
            raise ValueError("num_levels must be <= 8 (internal Warp kernels use fixed-size mask arrays that cannot exceed 8 levels).")
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
    origin, grid_shape = _align_domain_origin_for_dyadic(adjusted_min, adjusted_max, config.voxel_size, config.num_levels)

    return mesh, origin, grid_shape


def _shape_at_level(grid_shape_finest: Tuple[int, int, int], level: int) -> Tuple[int, int, int]:
    """Return the grid dimensions at a given refinement level.

    Each level halves each axis dimension relative to the finest (level 0).
    """
    stride = 2**level
    return (grid_shape_finest[0] // stride, grid_shape_finest[1] // stride, grid_shape_finest[2] // stride)


def _pad_domain_for_dyadic(
    grid_shape: Tuple[int, int, int],
    origin_phys: np.ndarray,
    voxel_size: float,
    num_levels: int,
    domain_padding: Sequence[float],
) -> Tuple[Tuple[int, int, int], np.ndarray]:
    """Grow the finest grid to a multiple of ``2**num_levels`` cells per axis.

    The extra alignment cells are split between the low (-) and high (+) side of
    each axis in proportion to the requested ``domain_padding``, so a symmetric
    padding (e.g. ``0.5 / 0.5``) keeps the geometry centred, while an asymmetric
    one (e.g. a tight ground plane ``0.05 / 3.0``) preserves its intended offset.

    The low-side growth is quantised to ``2**(num_levels-1)`` cells so the origin
    stays divisible by every level stride — a requirement of
    :func:`xlb.utils.mesher._normalize_level_data`, which maps each level's origin
    to a common finest-grid corner.

    Args:
        grid_shape: Finest-grid dimensions before alignment.
        origin_phys: Physical domain origin (modified copy is returned).
        voxel_size: Finest cell size.
        num_levels: Number of refinement levels.
        domain_padding: 6-tuple ``[-x, +x, -y, +y, -z, +z]`` padding multipliers.

    Returns:
        Tuple of (aligned grid_shape, shifted origin_phys).
    """
    align = 2**num_levels
    factor = 2 ** (num_levels - 1)
    origin = origin_phys.astype(np.float64).copy()
    new_shape = list(grid_shape)
    for axis in range(3):
        n = int(grid_shape[axis])
        total = (align - n % align) % align
        if total == 0:
            continue
        p_lo = float(domain_padding[2 * axis])
        p_hi = float(domain_padding[2 * axis + 1])
        denom = p_lo + p_hi
        frac_lo = 0.5 if denom <= 0.0 else p_lo / denom
        # Quantise the low-side growth to whole coarse cells so the origin stays
        # dyadic-aligned; the remainder goes to the high side.
        low_extra = int(round((total * frac_lo) / factor)) * factor
        low_extra = max(0, min(low_extra, (total // factor) * factor))
        origin[axis] -= low_extra * voxel_size
        new_shape[axis] = n + total
    return (new_shape[0], new_shape[1], new_shape[2]), origin


# ---------------------------------------------------------------------------
# Octree helpers
# ---------------------------------------------------------------------------

_CHILD_OFFSETS_2x2x2 = np.array(
    [[di, dj, dk] for di in range(2) for dj in range(2) for dk in range(2)],
    dtype=np.int64,
)


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
    int_offsets = _CHILD_OFFSETS_2x2x2

    bases = origin + parent_refine.astype(np.float64) * parent_voxel
    centers = (bases[:, None, :] + (int_offsets[None, :, :] + 0.5) * child_voxel).reshape(-1, 3)

    child_blocks = parent_refine.astype(np.int64)[:, None, :] * 2 + int_offsets[None, :, :]
    keys = child_blocks.reshape(-1, 3)
    return centers, keys


class LevelDataList(list):
    """Level data list with optional attached finest-grid shape metadata."""

    grid_shape_finest: Tuple[int, int, int] | None = None


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

    Encapsulates Warp BVH distance queries and morphology filters used by the
    octree refinement path. All GPU kernels run on the same CUDA device.

    Thread safety: Instances are NOT thread-safe. Create one per thread if needed.

    Example::

        ops = WarpAdaptiveMesherOps(trimesh_mesh)
        dists = ops.batched_distances(centers, max_dist)
        levels = ops.assign_levels_from_distances_1d(dists, config)
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

    def euclidean_edt_3d(self, background: np.ndarray) -> np.ndarray:
        """Compute exact Euclidean distance transform on the GPU.

        Delegates to the module-level :func:`euclidean_edt_3d` which uses
        Meijster's separable algorithm implemented as Warp kernels.
        """
        return euclidean_edt_3d(background)

    def assign_levels_from_distances_1d(self, distances: np.ndarray, config: AdaptiveMeshConfig) -> np.ndarray:
        """Assign levels from a flat 1D array of distances (for batched point queries).

        Same geometric-shell logic as the octree distance assignment but
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


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Octree meshing path
# ---------------------------------------------------------------------------


def _make_masks_octree(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    grid_shape_finest: Tuple[int, int, int],
    config: AdaptiveMeshConfig,
    verbose: bool = True,
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
        Tuple of (sparse coordinate arrays per level, mask_origins list).
    """
    ops = WarpAdaptiveMesherOps(mesh)
    num_levels = config.num_levels
    coarsest = num_levels - 1
    assignments: List[List[np.ndarray]] = [[] for _ in range(num_levels)]

    shape_c = _shape_at_level(grid_shape_finest, coarsest)
    voxel_c = config.voxel_size * (2**coarsest)
    n_c = int(np.prod(shape_c))
    if verbose:
        print(
            f"  Level {coarsest} (coarsest): conservative distance on {shape_c} ({n_c:,} cells, voxel={voxel_c:.1f} m)",
            flush=True,
        )
    t0 = time.perf_counter()
    target_c = ops.conservative_coarse_targets(origin, shape_c, voxel_c, config)
    if verbose:
        print(f"    distance done in {time.perf_counter() - t0:.1f}s", flush=True)
    coarse_keep = np.argwhere(target_c == coarsest)
    if len(coarse_keep):
        assignments[coarsest].append(coarse_keep)

    parent_refine = np.argwhere(target_c <= coarsest - 1) if num_levels > 1 else np.empty((0, 3), dtype=int)
    parent_level = coarsest
    max_dist = _max_query_distance(origin, grid_shape_finest, config.voxel_size)

    for level in range(coarsest - 1, -1, -1):
        n_refine = len(parent_refine)
        if verbose:
            print(f"  Level {level}: refining {n_refine:,} parent cells", flush=True)
        if n_refine == 0:
            continue

        parent_voxel = config.voxel_size * (2**parent_level)
        child_voxel = config.voxel_size * (2**level)

        t0 = time.perf_counter()
        centers, keys = _child_centers_and_keys(parent_refine, origin, parent_voxel, child_voxel)
        dists = ops.batched_distances(centers, max_dist)
        child_targets = ops.assign_levels_from_distances_1d(dists, config)

        # Graded-octree leaf partition: a child is a leaf at this level when it
        # is not refined further (target >= level). Its 8 siblings exactly tile
        # the subdivided parent, so recording leaves at the *current* level (not
        # at their distance-derived target) yields a strictly non-overlapping,
        # fully-covering set. Children with target < level are subdivided next.
        # (Recording at the raw target instead makes coarse cells overlap their
        # own refined siblings.)
        is_leaf = child_targets >= level
        leaf_keys = keys[is_leaf]
        if len(leaf_keys):
            assignments[level].append(leaf_keys)
        n_active = int(len(leaf_keys))
        if verbose:
            print(
                f"    {len(parent_refine):,} parents, {n_active:,} active at level {level} in {time.perf_counter() - t0:.1f}s",
                flush=True,
            )

        parent_refine = keys[~is_leaf] if level > 0 else np.empty((0, 3), dtype=int)
        parent_level = level

    if verbose:
        print("  Building sparse active-voxel coordinates from assignments...", flush=True)
    t0 = time.perf_counter()
    active_coords = _assignments_to_active_coords(assignments, num_levels)
    mask_origins = [np.zeros(3, dtype=int) for _ in range(num_levels)]
    if verbose:
        print(f"    sparse coords built in {time.perf_counter() - t0:.1f}s", flush=True)
        for level, coords in enumerate(active_coords):
            print(f"    level {level}: {len(coords):,} active voxels", flush=True)
    return active_coords, mask_origins


# ---------------------------------------------------------------------------
# Level-data packing
# ---------------------------------------------------------------------------


def _pack_level_data(
    patterns: List[np.ndarray],
    mask_origins: List[np.ndarray],
    origin_phys: np.ndarray,
    voxel_size_finest: float,
    num_levels: int,
    verbose: bool = True,
) -> list:
    """Pack per-level sparse active-voxel coordinates into level_data tuples.

    Output ordering is coarsest-first (reversed from the internal finest-first
    convention) to match the format expected by :func:`_normalize_level_data`.

    Each entry is ``(coords, voxel_size_at_level, physical_origin, build_level)`` where
    ``coords`` is an (N, 3) int32 array of active lattice indices.

    Args:
        patterns: Per-level sparse coordinate arrays (finest-first).
        mask_origins: Per-level grid-index origins.
        origin_phys: Physical domain origin.
        voxel_size_finest: Finest cell physical size.
        num_levels: Total levels.

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
        active = int(pattern.shape[0])
        shape_desc = f"coords ({active:,} x 3)"
        packed = np.ascontiguousarray(pattern, dtype=np.int32)

        if verbose:
            print(f"Level {finest_level}: {shape_desc}, active {active:,}, origin offset {offset}, voxel_size {voxel_size_level}")
            if active == 0:
                print(f"  (level {finest_level} unused: finer levels fully cover the domain)")
        raw_level_data.append((packed, voxel_size_level, sub_origin_phys, cuboid_build_level))

    return list(reversed(raw_level_data))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _filter_phantom_cells(level_data):
    """Remove cells that Neon cannot address (virtual finest coord < 0 on any axis).

    Dyadic padding can produce negative origins.  A cell at local coord ``l``
    with origin ``o`` and stride ``s = 2^level`` has virtual finest coord
    ``(l + o) * s``.  Neon's base grid iterates ``[0, dim)`` so any cell with
    a negative virtual component is unreachable phantom padding.
    """
    filtered = []
    for lvl in range(len(level_data)):
        pattern, stride_or_voxel, origin_arr, level_idx = level_data[lvl]
        origin = np.asarray(origin_arr, dtype=np.int64)
        stride = 1 << lvl

        if pattern.ndim == 2:
            coords = np.asarray(pattern, dtype=np.int64)
            virtual = (coords + origin) * stride
            valid = np.all(virtual >= 0, axis=1)
            new_pattern = np.ascontiguousarray(coords[valid], dtype=pattern.dtype)
        else:
            new_pattern = pattern

        filtered.append((new_pattern, stride_or_voxel, origin_arr, level_idx))

    grid_shape = getattr(level_data, "grid_shape_finest", None)
    if grid_shape is not None:
        result = LevelDataList(filtered)
        result.grid_shape_finest = grid_shape
        return result
    return filtered


def make_adaptive_surface_mesh(
    voxel_size: float,
    num_levels: int,
    stl_filename: str,
    domain_padding: Sequence[float] = None,
    expansion_ratio: float = 2.0,
    finest_band_cells: int = 3,
    verbose: bool = True,
) -> list:
    """
    Create a graded-octree surface-adaptive multires mesh from an STL file.

    Fine cells are placed within ``finest_band_cells * voxel_size`` of the
    surface; coarser levels extend outward with geometric ``expansion_ratio``
    growth between consecutive shells.  The output guarantees non-overlap
    across levels; final strong-balance enforcement (delta-L <= 1) is
    delegated to Neon's ``mGrid`` construction.

    Uses octree refinement with sparse ``(N, 3)`` active-voxel output at every
    scale — no dense finest-grid allocation.

    Args:
        voxel_size: Physical size of the finest lattice cell.
        num_levels: Number of refinement levels (finest = 0).
        stl_filename: Path to the STL/OBJ geometry file.
        domain_padding: 6-tuple ``[-x, +x, -y, +y, -z, +z]`` padding multipliers
            for the coarsest outer domain (relative to geometry extent).
        expansion_ratio: Geometric ratio between consecutive distance shells.
        finest_band_cells: Thickness of the finest shell in finest-cell counts.
        verbose: If True (default), print progress and statistics to stdout.

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
    )

    mesh, origin_phys, grid_shape = _compute_domain(config)
    n_finest = int(np.prod(grid_shape))
    if verbose:
        print(f"Adaptive mesh domain (finest): {grid_shape}, origin {origin_phys}, voxel_size {voxel_size}")
        print(f"  Finest-grid cell count: {n_finest:,}")

    align = 2**num_levels
    aligned_shape, origin_phys = _pad_domain_for_dyadic(grid_shape, origin_phys, voxel_size, num_levels, config.domain_padding)
    if aligned_shape != grid_shape:
        grid_shape = aligned_shape
        n_finest = int(np.prod(grid_shape))
        if verbose:
            print(f"Aligned domain to {grid_shape}, origin {origin_phys} (align={align}, padding distributed by domain_padding ratio)")

    if verbose:
        print("Using octree surface meshing (sparse active-voxel output).", flush=True)
    patterns, mask_origins = _make_masks_octree(mesh, origin_phys, grid_shape, config, verbose=verbose)
    raw_level_data = _pack_level_data(patterns, mask_origins, origin_phys, voxel_size, num_levels, verbose=verbose)

    level_data = LevelDataList(_normalize_level_data(raw_level_data, voxel_size))
    level_data.grid_shape_finest = grid_shape

    level_data = _filter_phantom_cells(level_data)
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
    Validate ``level_data`` for non-overlap and grading (ΔL ≤ 1 between 26-neighbors).

    Returns a statistics dictionary with per-level active counts and validation flags.
    Note: this validation operates on dense level_data only; sparse data is skipped.
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


def validate_sparse_level_data(
    level_data: list,
    grid_shape_finest: Tuple[int, int, int],
    check_balance: bool = False,
) -> dict:
    """Validate sparse ``level_data`` for non-overlap and coverage.

    Unlike :func:`validate_level_data` (which requires dense masks), this
    function works on sparse ``(N, 3)`` coordinate arrays by projecting each
    cell's finest-grid footprint into a hash set.

    Args:
        level_data: Sparse level_data list (finest-first).
        grid_shape_finest: Expected finest lattice shape ``(nx, ny, nz)``.
        check_balance: If True, also verify grading (ΔL ≤ 1 between
            face-adjacent finest cells).  This is expensive for large meshes.

    Returns:
        Dictionary with keys ``non_overlapping``, ``fully_covering``,
        ``strongly_balanced`` (None if ``check_balance`` is False), and
        ``overlap_count``.
    """
    num_levels = len(level_data)
    nx, ny, nz = grid_shape_finest
    total_finest_cells = nx * ny * nz

    covered: set = set()
    overlap_count = 0
    non_overlapping = True

    owners: dict = {} if check_balance else None

    for lvl in range(num_levels):
        pattern = level_data[lvl][0]
        if pattern.ndim != 2 or pattern.shape[1] != 3:
            continue
        stride = 2**lvl
        origin = np.asarray(level_data[lvl][2], dtype=np.float64)
        voxel_size_level = level_data[lvl][1]
        if isinstance(voxel_size_level, (int, float)) and voxel_size_level > 0:
            origin_idx = np.round(origin / voxel_size_level).astype(np.int64) if np.any(origin != 0) else np.zeros(3, dtype=np.int64)
        else:
            origin_idx = np.zeros(3, dtype=np.int64)

        coords = np.asarray(pattern, dtype=np.int64)
        for row_idx in range(coords.shape[0]):
            cx, cy, cz = int(coords[row_idx, 0]), int(coords[row_idx, 1]), int(coords[row_idx, 2])
            fx = (cx + int(origin_idx[0])) * stride
            fy = (cy + int(origin_idx[1])) * stride
            fz = (cz + int(origin_idx[2])) * stride
            for di in range(stride):
                for dj in range(stride):
                    for dk in range(stride):
                        key = (fx + di, fy + dj, fz + dk)
                        if key[0] < 0 or key[1] < 0 or key[2] < 0:
                            continue
                        if key[0] >= nx or key[1] >= ny or key[2] >= nz:
                            continue
                        if key in covered:
                            overlap_count += 1
                            non_overlapping = False
                        else:
                            covered.add(key)
                            if owners is not None:
                                owners[key] = lvl

    fully_covering = len(covered) == total_finest_cells

    strongly_balanced = None
    if check_balance and owners:
        strongly_balanced = True
        face_offsets = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        for (x, y, z), lvl in owners.items():
            for dx, dy, dz in face_offsets:
                nb = (x + dx, y + dy, z + dz)
                nb_lvl = owners.get(nb)
                if nb_lvl is not None and abs(lvl - nb_lvl) > 1:
                    strongly_balanced = False
                    break
            if strongly_balanced is False:
                break

    return {
        "non_overlapping": non_overlapping,
        "fully_covering": fully_covering,
        "strongly_balanced": strongly_balanced,
        "overlap_count": overlap_count,
    }


def grid_shape_finest(level_data: list) -> Tuple[int, int, int]:
    """Finest lattice shape implied by level_data or attached metadata.

    Thin wrapper around :func:`xlb.utils.mesher.grid_shape_finest_from_level_data`
    that returns a tuple instead of a numpy array.
    """
    arr = grid_shape_finest_from_level_data(level_data)
    return (int(arr[0]), int(arr[1]), int(arr[2]))


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


def print_mesh_statistics(label: str, level_data: list, grid_shape_finest_grid: Tuple[int, int, int]) -> Tuple[int, int]:
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
        shape_desc = f"coords {sparsity_pattern[lvl].shape}" if sparse else f"mask shape={sparsity_pattern[lvl].shape}"
        print(f"  Level {lvl}: active={active:,}, {shape_desc}, origin={level_origins[lvl]}, equiv. finest cells={equiv_finest:,}")

    total_active = sum(_active_count(m) for m in sparsity_pattern)
    total_equiv_finest = sum(_active_count(sparsity_pattern[lvl]) * (2 ** (num_levels - 1 - lvl)) for lvl in range(num_levels))
    print(f"  Total active cells: {total_active:,}")
    print(f"  Total equivalent finest cells: {total_equiv_finest:,}")

    if sparse:
        grid_shape_for_validation = getattr(level_data, "grid_shape_finest", grid_shape_finest_grid)
        stats = validate_sparse_level_data(level_data, grid_shape_for_validation)
        print(
            f"  Validation (sparse): non_overlapping={stats['non_overlapping']}, "
            f"fully_covering={stats['fully_covering']}, "
            f"overlap_count={stats['overlap_count']}"
        )
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
        epilog=("Level shells: finest band width = finest_band_cells * voxel_size; each outer shell grows by expansion_ratio."),
    )
    parser.add_argument("--stl", required=True, help="Path to input STL/OBJ geometry.")
    parser.add_argument("--voxel-size", type=float, default=4.0, help="Finest cell size (default: 4.0).")
    parser.add_argument("--num-levels", type=int, default=4, help="Refinement levels; 0 is finest (default: 4).")
    parser.add_argument("--expansion-ratio", type=float, default=2.0, help="Shell growth ratio (default: 2.0).")
    parser.add_argument("--finest-band-cells", type=int, default=3, help="Finest shell thickness in cells (default: 3).")
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
            adaptive_finest = int(level_data[0][0].shape[0])
            print(f"  Adaptive finest-level cells: {adaptive_finest:,}")
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
