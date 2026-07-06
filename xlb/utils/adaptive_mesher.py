"""
Surface-adaptive multi-resolution mesh generation.

Builds strongly-balanced multires domains with fine cells near STL surfaces
and progressively coarser cells farther away, using geometric expansion-ratio
bands.  Output matches :func:`xlb.utils.mesher.make_cuboid_mesh` ``level_data``
format.

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
from typing import Dict, List, Literal, Sequence, Tuple, Union

import numpy as np
import trimesh
import warp as wp

from xlb.utils.adaptive_mesher_kernels import (
    kernel_any_changed,
    kernel_any_nonzero_uint8,
    kernel_any_uncovered,
    kernel_apply_balance_coarsen,
    kernel_apply_balance_refine,
    kernel_apply_subdivide_level,
    kernel_assign_levels,
    kernel_assign_levels_1d,
    kernel_batched_distances,
    kernel_binary_dilate_cube,
    kernel_block_uniformity_level,
    kernel_build_fine_mask,
    kernel_build_shell_mask,
    kernel_conservative_coarse_targets,
    kernel_dense_distances,
    kernel_dense_distances_tiled,
    kernel_edt_init_from_mask,
    kernel_edt_pass_x,
    kernel_edt_pass_y,
    kernel_edt_pass_z,
    kernel_edt_sqrt,
    kernel_extract_level_mask,
    kernel_fill_coverage_gaps,
    kernel_mark_coverage,
    kernel_mark_subdivide_offset,
    kernel_maximum_filter_3x3,
    kernel_minimum_filter_3x3,
    kernel_minimum_with_floor,
    kernel_owners_from_masks,
    kernel_paint_blocks_level,
    kernel_promote_near_fine,
    kernel_refine_transition,
    kernel_widen_shell,
    kernel_balance_need_coarsen,
    kernel_balance_need_refine,
)
from xlb.utils.mesher import (
    _align_domain_origin_for_dyadic,
    _domain_bbox_from_padding,
    _finest_grid_from_bbox,
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
    """Return physical origin, finest grid shape, and loaded mesh."""
    mesh = _load_stl_mesh(config.stl_filename)
    min_bound, max_bound, part_size = _stl_bounds(mesh)

    cuboid_min, cuboid_max = _domain_bbox_from_padding(min_bound, max_bound, part_size, config.domain_padding)
    adjusted_min, adjusted_max = adjust_bbox(cuboid_max, cuboid_min, config.voxel_size)
    origin, grid_shape = _align_domain_origin_for_dyadic(
        adjusted_min, adjusted_max, config.voxel_size, config.num_levels
    )

    return mesh, origin, grid_shape


def _shape_at_level(grid_shape_finest: Tuple[int, int, int], level: int) -> Tuple[int, int, int]:
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

    bases = origin + parent_refine.astype(np.float64) * parent_voxel
    offsets = np.array(
        [
            [di + 0.5, dj + 0.5, dk + 0.5]
            for di in range(2)
            for dj in range(2)
            for dk in range(2)
        ],
        dtype=np.float64,
    )
    centers = (bases[:, None, :] + offsets[None, :, :] * child_voxel).reshape(-1, 3)

    pi = np.repeat(parent_refine[:, 0], 8)
    pj = np.repeat(parent_refine[:, 1], 8)
    pk = np.repeat(parent_refine[:, 2], 8)
    child_off = np.tile(np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], dtype=int), len(parent_refine))
    di = child_off[0::3]
    dj = child_off[1::3]
    dk = child_off[2::3]
    keys = np.stack([pi * 2 + di, pj * 2 + dj, pk * 2 + dk], axis=1)
    return centers, keys


def _record_assignments(
    assignments: List[List[np.ndarray]],
    keys: np.ndarray,
    targets: np.ndarray,
    source_level: int,
    num_levels: int,
):
    """Record (i,j,k) cell indices at each target level from child keys on source_level grid."""
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
    """Build non-overlapping, fully-covering masks with coherent finest band.

    1. L0 is activated first wherever owner == 0 (tight, coherent surface band).
    2. Remaining volume is filled coarsest-first (L(n-1) down to L1) to maximize
       coarse block usage.
    3. Any leftover uncovered cells go to L0.
    """
    nx, ny, nz = owner.shape
    covered = np.zeros((nx, ny, nz), dtype=bool)
    masks: List[np.ndarray] = []
    for level in range(num_levels):
        stride = 2**level
        sx, sy, sz = nx // stride, ny // stride, nz // stride
        masks.append(np.zeros((sx, sy, sz), dtype=bool))

    l0_activate = (owner[:nx, :ny, :nz] == 0)
    masks[0] = l0_activate
    covered[:nx, :ny, :nz] |= l0_activate

    for level in range(num_levels - 1, 0, -1):
        stride = 2**level
        sx, sy, sz = nx // stride, ny // stride, nz // stride

        owner_blocks = owner[: sx * stride, : sy * stride, : sz * stride].reshape(
            sx, stride, sy, stride, sz, stride
        )
        block_min = owner_blocks.min(axis=(1, 3, 5))

        covered_blocks = covered[: sx * stride, : sy * stride, : sz * stride].reshape(
            sx, stride, sy, stride, sz, stride
        )
        block_uncovered = ~covered_blocks.any(axis=(1, 3, 5))

        activate = (block_min >= level) & block_uncovered
        masks[level] = activate

        if np.any(activate):
            expanded = np.repeat(
                np.repeat(np.repeat(activate, stride, axis=0), stride, axis=1),
                stride,
                axis=2,
            )
            covered[: sx * stride, : sy * stride, : sz * stride] |= expanded

    remaining = ~covered
    if np.any(remaining):
        masks[0] |= remaining[:nx, :ny, :nz]

    return masks


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
    """Warp-native exact Euclidean distance transform (SciPy parity)."""
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
    """GPU-accelerated adaptive mesher operations."""

    def __init__(self, mesh: trimesh.Trimesh):
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
        return euclidean_edt_3d(background)

    def compute_distance_field_dense(
        self,
        origin: np.ndarray,
        grid_shape: Tuple[int, int, int],
        voxel_size: float,
    ) -> np.ndarray:
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
        out = wp.zeros(field.shape, dtype=wp.int32)
        wp.launch(kernel_minimum_filter_3x3, dim=field.shape, inputs=[field, out])
        wp.synchronize()
        return out

    def _max_filter_3x3(self, field: wp.array) -> wp.array:
        out = wp.zeros(field.shape, dtype=wp.int32)
        wp.launch(kernel_maximum_filter_3x3, dim=field.shape, inputs=[field, out])
        wp.synchronize()
        return out

    def _any_nonzero_uint8(self, flags: wp.array) -> bool:
        counter = wp.zeros(1, dtype=wp.int32)
        wp.launch(kernel_any_nonzero_uint8, dim=flags.shape, inputs=[flags, counter])
        return self._sync_counter(counter) > 0

    def enforce_strong_balance_bidirectional(self, assigned: np.ndarray, max_passes: int = 64) -> np.ndarray:
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
                wp.synchronize()
                cur = tmp
                tmp = wp.zeros(shape, dtype=wp.int32)
            if need_coarsen:
                wp.launch(kernel_apply_balance_coarsen, dim=shape, inputs=[cur, nmax, tmp])
                wp.synchronize()
                cur = tmp
                tmp = wp.zeros(shape, dtype=wp.int32)
        return cur.numpy()

    def minimum_with_floor(self, a: np.ndarray, floor: np.ndarray) -> np.ndarray:
        a_wp = wp.array(a, dtype=wp.int32)
        f_wp = wp.array(floor, dtype=wp.int32)
        out = wp.zeros(a.shape, dtype=wp.int32)
        wp.launch(kernel_minimum_with_floor, dim=a.shape, inputs=[a_wp, f_wp, out])
        wp.synchronize()
        return out.numpy()

    def refine_transition_layers(
        self, owner: np.ndarray, owner_floor: np.ndarray, num_levels: int
    ) -> np.ndarray:
        cur = wp.array(owner, dtype=wp.int32)
        tmp = wp.zeros(owner.shape, dtype=wp.int32)
        for level in range(num_levels - 2, -1, -1):
            nmin = self._min_filter_3x3(cur)
            wp.launch(
                kernel_refine_transition,
                dim=owner.shape,
                inputs=[cur, nmin, wp.int32(level), tmp],
            )
            wp.synchronize()
            cur = tmp
            tmp = wp.zeros(owner.shape, dtype=wp.int32)
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
            wp.synchronize()
            cur = tmp
            tmp = wp.zeros(shape, dtype=wp.int32)

        return cur.numpy()

    def ensure_tileable_transition_bands(self, owner: np.ndarray, num_levels: int) -> np.ndarray:
        """Thicken transition bands using Warp EDT + cube dilation."""
        nx, ny, nz = owner.shape
        owner_wp = wp.array(owner, dtype=wp.int32)
        cur = owner_wp
        tmp = wp.zeros((nx, ny, nz), dtype=wp.int32)
        fine = wp.zeros((nx, ny, nz), dtype=wp.uint8)
        shell = wp.zeros((nx, ny, nz), dtype=wp.uint8)
        dilated = wp.zeros((nx, ny, nz), dtype=wp.uint8)
        dist_sq = wp.zeros((nx, ny, nz), dtype=wp.float32)
        dist = wp.zeros((nx, ny, nz), dtype=wp.float32)
        sv_xy, sz_xy, sv_xz, sz_xz, sv_yz, sz_yz = _edt_scratch_buffers(nx, ny, nz)

        for level in range(num_levels - 2, -1, -1):
            transition = level + 1
            band_width = 2**transition
            wp.launch(kernel_build_fine_mask, dim=(nx, ny, nz), inputs=[cur, wp.int32(level), fine])
            wp.launch(kernel_edt_init_from_mask, dim=(nx, ny, nz), inputs=[fine, dist_sq])
            wp.launch(kernel_edt_pass_x, dim=(ny, nz), inputs=[dist_sq, sv_xy, sz_xy])
            wp.launch(kernel_edt_pass_y, dim=(nx, nz), inputs=[dist_sq, sv_xz, sz_xz])
            wp.launch(kernel_edt_pass_z, dim=(nx, ny), inputs=[dist_sq, sv_yz, sz_yz])
            wp.launch(kernel_edt_sqrt, dim=(nx, ny, nz), inputs=[dist_sq, dist])
            wp.launch(
                kernel_promote_near_fine,
                dim=(nx, ny, nz),
                inputs=[cur, dist, wp.int32(level), wp.int32(transition), wp.float32(band_width), tmp],
            )
            cur = tmp
            tmp = wp.zeros((nx, ny, nz), dtype=wp.int32)

            wp.launch(
                kernel_build_shell_mask,
                dim=(nx, ny, nz),
                inputs=[cur, wp.int32(transition), shell],
            )
            counter = wp.zeros(1, dtype=wp.int32)
            wp.launch(kernel_any_nonzero_uint8, dim=(nx, ny, nz), inputs=[shell, counter])
            if self._sync_counter(counter) == 0:
                continue
            wp.launch(
                kernel_binary_dilate_cube,
                dim=(nx, ny, nz),
                inputs=[shell, dilated, wp.int32(band_width)],
            )
            wp.launch(
                kernel_widen_shell,
                dim=(nx, ny, nz),
                inputs=[cur, dilated, wp.int32(level), wp.int32(transition), tmp],
            )
            cur = tmp
            tmp = wp.zeros((nx, ny, nz), dtype=wp.int32)

        wp.synchronize()
        return cur.numpy()

    def enforce_owner_block_uniformity(
        self,
        owner: np.ndarray,
        num_levels: int,
        max_level: int | None = None,
        max_passes: int = 24,
    ) -> np.ndarray:
        """Refine owner levels within dyadic blocks up to ``max_level`` (inclusive)."""
        if max_level is None:
            max_level = num_levels - 1
        max_level = min(max_level, num_levels - 1)
        nx, ny, nz = owner.shape
        cur = wp.array(owner, dtype=wp.int32)
        out = wp.zeros((nx, ny, nz), dtype=wp.int32)
        changed_flag = wp.zeros(1, dtype=wp.int32)

        for _ in range(max_passes):
            pass_changed = False
            for level in range(max_level, 0, -1):
                stride = 2**level
                sx, sy, sz = nx // stride, ny // stride, nz // stride
                if sx == 0 or sy == 0 or sz == 0:
                    continue
                wp.launch(
                    kernel_block_uniformity_level,
                    dim=(sx, sy, sz),
                    inputs=[cur, wp.int32(stride), out],
                )
                wp.synchronize()
                changed_flag.zero_()
                wp.launch(kernel_any_changed, dim=(nx, ny, nz), inputs=[cur, out, changed_flag])
                if self._sync_counter(changed_flag) > 0:
                    pass_changed = True
                    cur = out
                    out = wp.zeros((nx, ny, nz), dtype=wp.int32)
            if not pass_changed:
                break
        return cur.numpy()

    @staticmethod
    def _align_owner_dyadic(owner: np.ndarray, num_levels: int) -> np.ndarray:
        """Enforce dyadic block alignment so mask extraction has no orphans.

        Invariant: for each level L in [0, num_levels-2], the region
        {owner <= L} is aligned to 2^(L+1) blocks. Any 2^(L+1)-aligned block
        that touches the region is fully absorbed into it. This guarantees
        every dyadic block is uniform, so _build_masks_greedy_coarsest emits
        clean, non-overlapping, fully-covering masks with no fallback-to-L0
        orphan cells (the source of staircase protrusions).
        """
        nx, ny, nz = owner.shape
        for level in range(num_levels - 1):
            stride = 2 ** (level + 1)
            sx, sy, sz = nx // stride, ny // stride, nz // stride
            if sx == 0 or sy == 0 or sz == 0:
                continue
            ex, ey, ez = sx * stride, sy * stride, sz * stride
            region = owner[:ex, :ey, :ez] <= level
            block_has = region.reshape(sx, stride, sy, stride, sz, stride).any(axis=(1, 3, 5))
            expanded = np.repeat(
                np.repeat(np.repeat(block_has, stride, axis=0), stride, axis=1),
                stride,
                axis=2,
            )
            sub = owner[:ex, :ey, :ez]
            owner[:ex, :ey, :ez] = np.where(expanded & (sub > level), level, sub)
        return owner

    def build_masks_from_owner(
        self, owner: np.ndarray, num_levels: int, owner_floor: np.ndarray | None = None,
        origin: np.ndarray | None = None,
        voxel_size: float = 1.0,
        config=None,
    ) -> List[np.ndarray]:
        floor = owner.copy() if owner_floor is None else owner_floor
        owner = self.finalize_owner_grid(owner, floor, num_levels)

        if origin is not None and config is not None:
            coarsest = num_levels - 1
            band_region = owner < coarsest
            for _ in range(2**coarsest):
                grown = band_region.copy()
                grown[1:] |= band_region[:-1]
                grown[:-1] |= band_region[1:]
                grown[:, 1:] |= band_region[:, :-1]
                grown[:, :-1] |= band_region[:, 1:]
                grown[:, :, 1:] |= band_region[:, :, :-1]
                grown[:, :, :-1] |= band_region[:, :, 1:]
                band_region = grown

            cells = np.argwhere(band_region)
            if len(cells) > 0:
                orig = origin.astype(np.float64)
                centers = orig + (cells.astype(np.float64) + 0.5) * voxel_size
                max_dist = float(np.linalg.norm(np.array(owner.shape) * voxel_size))
                dists = self.batched_distances(centers, max_dist)
                new_levels = self.assign_levels_from_distances_1d(dists, config)
                ci, cj, ck = cells[:, 0], cells[:, 1], cells[:, 2]
                owner[ci, cj, ck] = new_levels.astype(owner.dtype)

        for _ in range(num_levels + 2):
            prev = owner.copy()
            owner = self.finalize_owner_grid(owner, floor, num_levels)
            owner = self._align_owner_dyadic(owner, num_levels)
            if np.array_equal(owner, prev):
                break

        masks = _build_masks_greedy_coarsest(owner, num_levels)
        masks = self.repair_balance_by_subdivision(masks, num_levels, owner.shape)
        return masks

    def build_non_overlapping_masks_vectorized(
        self, owner: np.ndarray, num_levels: int
    ) -> List[np.ndarray]:
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
            wp.synchronize()
            masks.append(mask_wp.numpy().astype(bool))
        return masks

    def repair_balance_by_subdivision(
        self,
        masks: List[np.ndarray],
        num_levels: int,
        grid_shape: Tuple[int, int, int],
        max_passes: int = 96,
    ) -> List[np.ndarray]:
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
    """Dense-path mesh generation using Warp."""
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
    """Octree-path mesh generation with Warp distance queries."""
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

    print("  Building non-overlapping masks from assignments...", flush=True)
    t0 = time.perf_counter()
    masks, mask_origins = _build_masks_from_assignments(
        ops, assignments, grid_shape_finest, num_levels, config.max_dense_cells,
        origin, config,
    )
    print(f"    masks built in {time.perf_counter() - t0:.1f}s", flush=True)
    return masks, mask_origins


def _build_masks_from_assignments(
    ops: WarpAdaptiveMesherOps,
    assignments: List[List[np.ndarray]],
    grid_shape_finest: Tuple[int, int, int],
    num_levels: int,
    max_dense_cells: int,
    origin: np.ndarray | None = None,
    config=None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Warp-accelerated owner paint + mask build from octree assignments."""
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
    masks: List[np.ndarray],
    mask_origins: List[np.ndarray],
    origin_phys: np.ndarray,
    voxel_size_finest: float,
    num_levels: int,
) -> list:
    """Pack boolean masks into level_data tuples (coarsest-first, cuboid convention)."""
    raw_level_data = []
    for finest_level in range(num_levels):
        mask = masks[finest_level]
        voxel_size_level = voxel_size_finest * (2**finest_level)
        offset = mask_origins[finest_level]
        sub_origin_phys = origin_phys + offset.astype(float) * voxel_size_finest
        cuboid_build_level = num_levels - 1 - finest_level
        active = int(np.count_nonzero(mask))

        print(
            f"Level {finest_level}: shape {mask.shape}, active {active:,}, "
            f"origin offset {offset}, voxel_size {voxel_size_level}"
        )
        if active == 0:
            print(f"  (level {finest_level} unused: finer levels fully cover the domain)")
        raw_level_data.append((mask.copy(), voxel_size_level, sub_origin_phys, cuboid_build_level))

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
        masks, mask_origins = _make_masks_octree(mesh, origin_phys, grid_shape, config)
    else:
        print("Using dense finest-grid meshing.", flush=True)
        masks, mask_origins = _make_masks_dense(mesh, origin_phys, grid_shape, config)

    raw_level_data = _pack_level_data(masks, mask_origins, origin_phys, voxel_size, num_levels)

    return _normalize_level_data(raw_level_data, voxel_size)


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
    """Finest lattice shape implied by coarsest-level mask and level count."""
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
    for lvl in range(num_levels):
        active = int(np.count_nonzero(sparsity_pattern[lvl]))
        equiv_finest = active * (2 ** (num_levels - 1 - lvl))
        print(
            f"  Level {lvl}: active={active:,}, "
            f"mask shape={sparsity_pattern[lvl].shape}, "
            f"origin={level_origins[lvl]}, "
            f"equiv. finest cells={equiv_finest:,}"
        )

    total_active = sum(int(np.count_nonzero(m)) for m in sparsity_pattern)
    total_equiv_finest = sum(
        int(np.count_nonzero(sparsity_pattern[lvl])) * (2 ** (num_levels - 1 - lvl))
        for lvl in range(num_levels)
    )
    print(f"  Total active cells: {total_active:,}")
    print(f"  Total equivalent finest cells: {total_equiv_finest:,}")

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
