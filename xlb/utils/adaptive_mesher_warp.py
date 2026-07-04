"""
Warp-backed operations for surface-adaptive mesh generation.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
import trimesh
import warp as wp

from xlb.utils.adaptive_mesher import (
    AdaptiveMeshConfig,
    _child_centers_and_keys,
    _record_assignments,
    _shape_at_level,
)
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


def euclidean_edt_3d_warp(background: np.ndarray) -> np.ndarray:
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
        return euclidean_edt_3d_warp(background)

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
        d0 = config.finest_band_cells * config.voxel_size
        log_ratio = math.log(config.expansion_ratio)
        assigned = wp.zeros(distances.shape, dtype=wp.int32)
        dist_wp = wp.array(distances, dtype=wp.float64)
        wp.launch(
            kernel_assign_levels,
            dim=distances.shape,
            inputs=[
                dist_wp,
                wp.float64(d0),
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
        d0 = config.finest_band_cells * config.voxel_size
        log_ratio = math.log(config.expansion_ratio)
        dist_wp = wp.array(distances.astype(np.float64), dtype=wp.float64)
        assigned = wp.zeros(n, dtype=wp.int32)
        wp.launch(
            kernel_assign_levels_1d,
            dim=n,
            inputs=[
                dist_wp,
                wp.float64(d0),
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
        d0 = config.finest_band_cells * config.voxel_size
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
                wp.float64(d0),
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
        owner = self.enforce_strong_balance_bidirectional(owner)
        owner = self.minimum_with_floor(owner, owner_floor)
        owner = self.refine_transition_layers(owner, owner_floor, num_levels)
        owner = self.enforce_strong_balance_bidirectional(owner)
        owner = self.minimum_with_floor(owner, owner_floor)
        owner = self.refine_transition_layers(owner, owner_floor, num_levels)
        owner = self.minimum_with_floor(owner, owner_floor)
        return owner

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

    def enforce_owner_block_uniformity(self, owner: np.ndarray, num_levels: int, max_passes: int = 24) -> np.ndarray:
        nx, ny, nz = owner.shape
        cur = wp.array(owner, dtype=wp.int32)
        out = wp.zeros((nx, ny, nz), dtype=wp.int32)
        changed_flag = wp.zeros(1, dtype=wp.int32)

        for _ in range(max_passes):
            pass_changed = False
            for level in range(num_levels - 1, 0, -1):
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

    def build_masks_from_owner(
        self, owner: np.ndarray, num_levels: int, owner_floor: np.ndarray | None = None
    ) -> List[np.ndarray]:
        floor = owner.copy() if owner_floor is None else owner_floor
        owner = self.finalize_owner_grid(owner, floor, num_levels)
        owner = self.ensure_tileable_transition_bands(owner, num_levels)
        owner = self.finalize_owner_grid(owner, floor, num_levels)
        owner = self.enforce_owner_block_uniformity(owner, num_levels)
        masks = self.build_non_overlapping_masks_vectorized(owner, num_levels)
        masks = self.repair_balance_by_subdivision(masks, num_levels, owner.shape)
        masks = self.fill_coverage_gaps(masks, owner.shape, num_levels)
        return masks


def euclidean_edt_3d(background: np.ndarray) -> np.ndarray:
    """Public EDT helper (Warp-native, SciPy-parity)."""
    return euclidean_edt_3d_warp(background)


def make_masks_dense_warp(
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
    masks = ops.build_masks_from_owner(assigned, config.num_levels, owner_floor=owner_floor)
    mask_origins = [np.zeros(3, dtype=int) for _ in range(config.num_levels)]
    return masks, mask_origins


def make_masks_octree_warp(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    grid_shape_finest: Tuple[int, int, int],
    config: AdaptiveMeshConfig,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Octree-path mesh generation with Warp distance queries."""
    import time

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
    masks, mask_origins = _build_masks_from_assignments_warp(
        ops, assignments, grid_shape_finest, num_levels, config.max_dense_cells
    )
    print(f"    masks built in {time.perf_counter() - t0:.1f}s", flush=True)
    return masks, mask_origins


def _build_masks_from_assignments_warp(
    ops: WarpAdaptiveMesherOps,
    assignments: List[List[np.ndarray]],
    grid_shape_finest: Tuple[int, int, int],
    num_levels: int,
    max_dense_cells: int,
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

    fi_min, fj_min, fk_min = nx, ny, nz
    fi_max, fj_max, fk_max = 0, 0, 0
    for target, arr in enumerate(level_indices):
        if len(arr) == 0:
            continue
        stride = 2**target
        fi_min = min(fi_min, int(arr[:, 0].min()) * stride)
        fj_min = min(fj_min, int(arr[:, 1].min()) * stride)
        fk_min = min(fk_min, int(arr[:, 2].min()) * stride)
        fi_max = max(fi_max, (int(arr[:, 0].max()) + 1) * stride)
        fj_max = max(fj_max, (int(arr[:, 1].max()) + 1) * stride)
        fk_max = max(fk_max, (int(arr[:, 2].max()) + 1) * stride)

    if fi_max <= fi_min:
        raise RuntimeError("No cell assignments collected during octree refinement.")

    tight_nx, tight_ny, tight_nz = fi_max - fi_min, fj_max - fj_min, fk_max - fk_min
    max_owner_cells = max(max_dense_cells * 8, 512**3)
    if tight_nx * tight_ny * tight_nz > max_owner_cells:
        raise RuntimeError(
            f"Tight ownership grid ({tight_nx}, {tight_ny}, {tight_nz}) is too large; "
            "increase voxel_size or reduce num_levels."
        )

    owner = ops.paint_owner_from_assignments(
        level_indices, (tight_nx, tight_ny, tight_nz), fi_min, fj_min, fk_min, num_levels
    )
    owner_floor = owner.copy()
    full_owner = np.full((nx, ny, nz), num_levels - 1, dtype=np.int32)
    full_floor = np.full((nx, ny, nz), num_levels - 1, dtype=np.int32)
    full_owner[fi_min:fi_max, fj_min:fj_max, fk_min:fk_max] = owner
    full_floor[fi_min:fi_max, fj_min:fj_max, fk_min:fk_max] = owner_floor

    masks = ops.build_masks_from_owner(full_owner, num_levels, owner_floor=full_floor)
    mask_origins = [np.zeros(3, dtype=int) for _ in range(num_levels)]
    return masks, mask_origins
