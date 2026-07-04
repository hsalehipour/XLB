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
    _assign_levels_from_distances_1d,
    _build_masks_from_assignments,
    _child_centers_and_keys,
    _record_assignments,
    _shape_at_level,
)
from xlb.utils.adaptive_mesher_kernels import (
    kernel_apply_balance_coarsen,
    kernel_apply_balance_refine,
    kernel_assign_levels,
    kernel_batched_distances,
    kernel_block_uniformity_level,
    kernel_conservative_coarse_targets,
    kernel_dense_distances,
    kernel_dense_distances_tiled,
    kernel_extract_level_mask,
    kernel_maximum_filter_3x3,
    kernel_minimum_filter_3x3,
    kernel_minimum_with_floor,
    kernel_refine_transition,
)


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


def euclidean_edt_3d(background: np.ndarray) -> np.ndarray:
    """
    Exact Euclidean distance transform for ``background==True`` voxels.

    Matches ``scipy.ndimage.distance_transform_edt(background)``.
    Used for validation; tile-band thickening delegates to the SciPy reference path.
    """
    from scipy import ndimage

    return ndimage.distance_transform_edt(background).astype(np.float32)


class WarpAdaptiveMesherOps:
    """GPU-accelerated adaptive mesher operations."""

    def __init__(self, mesh: trimesh.Trimesh):
        self._mesh = mesh
        self._wp_mesh = trimesh_to_warp(mesh)
        self._mesh_id = wp.uint64(self._wp_mesh.id)

    @property
    def mesh_id(self) -> wp.uint64:
        return self._mesh_id

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
        # Tiled warp distance for large dense grids
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

    def enforce_strong_balance_bidirectional(self, assigned: np.ndarray, max_passes: int = 64) -> np.ndarray:
        result = assigned.copy()
        shape = result.shape
        cur = wp.array(result, dtype=wp.int32)
        tmp = wp.zeros(shape, dtype=wp.int32)
        nmin = wp.zeros(shape, dtype=wp.int32)
        nmax = wp.zeros(shape, dtype=wp.int32)

        for _ in range(max_passes):
            nmin = self._min_filter_3x3(cur)
            nmax = self._max_filter_3x3(cur)
            nmin_np = nmin.numpy()
            nmax_np = nmax.numpy()
            cur_np = cur.numpy()
            refine = cur_np > (nmin_np + 1)
            coarsen = cur_np < (nmax_np - 1)
            if not np.any(refine) and not np.any(coarsen):
                break
            if np.any(refine):
                wp.launch(kernel_apply_balance_refine, dim=shape, inputs=[cur, nmin, tmp])
                wp.synchronize()
                cur = tmp
                tmp = wp.zeros(shape, dtype=wp.int32)
            if np.any(coarsen):
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
        """Match SciPy EDT + cube dilation exactly (reference implementation)."""
        from xlb.utils.adaptive_mesher import _ensure_tileable_transition_bands

        return _ensure_tileable_transition_bands(owner, num_levels)

    def enforce_owner_block_uniformity(self, owner: np.ndarray, num_levels: int, max_passes: int = 24) -> np.ndarray:
        result = owner.copy()
        nx, ny, nz = result.shape
        for _ in range(max_passes):
            changed = False
            cur = wp.array(result, dtype=wp.int32)
            out = wp.zeros((nx, ny, nz), dtype=wp.int32)
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
                out_np = out.numpy()
                if not np.array_equal(out_np, result):
                    changed = True
                    result = out_np
                    cur = wp.array(result, dtype=wp.int32)
            if not changed:
                break
        return result

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

    def build_masks_from_owner(
        self, owner: np.ndarray, num_levels: int, owner_floor: np.ndarray | None = None
    ) -> List[np.ndarray]:
        floor = owner.copy() if owner_floor is None else owner_floor
        owner = self.finalize_owner_grid(owner, floor, num_levels)
        owner = self.ensure_tileable_transition_bands(owner, num_levels)
        owner = self.finalize_owner_grid(owner, floor, num_levels)
        owner = self.enforce_owner_block_uniformity(owner, num_levels)
        masks = self.build_non_overlapping_masks_vectorized(owner, num_levels)
        # Subdivision repair and gap fill use numpy (identical algorithms)
        from xlb.utils.adaptive_mesher import (
            _fill_coverage_gaps,
            _finest_coverage,
            _repair_balance_by_subdivision,
        )

        masks = _repair_balance_by_subdivision(masks, num_levels, owner.shape)
        if np.any(~_finest_coverage(masks, num_levels, owner.shape)):
            masks = _fill_coverage_gaps(masks, owner)
        return masks


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
        child_targets = _assign_levels_from_distances_1d(dists, config)
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

    owner = np.full((tight_nx, tight_ny, tight_nz), num_levels - 1, dtype=np.int32)
    for target in range(num_levels):
        stride = 2**target
        for i, j, k in level_indices[target]:
            i0, i1 = i * stride - fi_min, (i + 1) * stride - fi_min
            j0, j1 = j * stride - fj_min, (j + 1) * stride - fj_min
            k0, k1 = k * stride - fk_min, (k + 1) * stride - fk_min
            owner[i0:i1, j0:j1, k0:k1] = np.minimum(owner[i0:i1, j0:j1, k0:k1], target)

    owner_floor = owner.copy()
    full_owner = np.full((nx, ny, nz), num_levels - 1, dtype=np.int32)
    full_floor = np.full((nx, ny, nz), num_levels - 1, dtype=np.int32)
    full_owner[fi_min:fi_max, fj_min:fj_max, fk_min:fk_max] = owner
    full_floor[fi_min:fi_max, fj_min:fj_max, fk_min:fk_max] = owner_floor

    masks = ops.build_masks_from_owner(full_owner, num_levels, owner_floor=full_floor)
    mask_origins = [np.zeros(3, dtype=int) for _ in range(num_levels)]
    return masks, mask_origins
