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
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Sequence, Tuple, Union

import numpy as np
import trimesh
from trimesh.proximity import ProximityQuery

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


def _cell_centers(origin: np.ndarray, grid_shape: Tuple[int, int, int], voxel_size: float) -> np.ndarray:
    """Return (N, 3) cell-center coordinates for a uniform grid."""
    nx, ny, nz = grid_shape
    ii, jj, kk = np.mgrid[0:nx, 0:ny, 0:nz]
    centers = np.stack(
        [
            origin[0] + (ii + 0.5) * voxel_size,
            origin[1] + (jj + 0.5) * voxel_size,
            origin[2] + (kk + 0.5) * voxel_size,
        ],
        axis=-1,
    )
    return centers.reshape(-1, 3)


def _compute_distance_field_dense(mesh: trimesh.Trimesh, origin: np.ndarray, grid_shape: Tuple[int, int, int], voxel_size: float) -> np.ndarray:
    """Compute unsigned distance to the mesh surface at finest cell centers."""
    centers = _cell_centers(origin, grid_shape, voxel_size)
    pq = ProximityQuery(mesh)
    _, distances, _ = pq.on_surface(centers)
    return distances.reshape(grid_shape)


def _compute_distance_field_tiled(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    grid_shape: Tuple[int, int, int],
    voxel_size: float,
    tile_size: int,
) -> np.ndarray:
    """Tile-based distance computation for large domains."""
    nx, ny, nz = grid_shape
    distances = np.empty(grid_shape, dtype=np.float64)
    pq = ProximityQuery(mesh)

    for i0 in range(0, nx, tile_size):
        i1 = min(i0 + tile_size, nx)
        for j0 in range(0, ny, tile_size):
            j1 = min(j0 + tile_size, ny)
            for k0 in range(0, nz, tile_size):
                k1 = min(k0 + tile_size, nz)
                ii, jj, kk = np.mgrid[i0:i1, j0:j1, k0:k1]
                tile_centers = np.stack(
                    [
                        origin[0] + (ii + 0.5) * voxel_size,
                        origin[1] + (jj + 0.5) * voxel_size,
                        origin[2] + (kk + 0.5) * voxel_size,
                    ],
                    axis=-1,
                ).reshape(-1, 3)
                _, tile_dist, _ = pq.on_surface(tile_centers)
                distances[i0:i1, j0:j1, k0:k1] = tile_dist.reshape(i1 - i0, j1 - j0, k1 - k0)

    return distances


def _compute_distance_field(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    grid_shape: Tuple[int, int, int],
    voxel_size: float,
    config: AdaptiveMeshConfig,
) -> np.ndarray:
    """Choose dense or tiled distance computation based on domain size."""
    nx, ny, nz = grid_shape
    if nx * ny * nz <= config.max_dense_cells:
        return _compute_distance_field_dense(mesh, origin, grid_shape, voxel_size)
    return _compute_distance_field_tiled(mesh, origin, grid_shape, voxel_size, config.tile_size)


def _assign_levels_from_distances_1d(distances: np.ndarray, config: AdaptiveMeshConfig) -> np.ndarray:
    """Vectorized level assignment for a 1-D distance array."""
    d0 = config.finest_band_cells * config.voxel_size
    log_ratio = np.log(config.expansion_ratio)
    num_levels = config.num_levels

    assigned = np.full(distances.shape, num_levels - 1, dtype=np.int32)
    near = distances < d0
    assigned[near] = 0
    far = ~near
    if np.any(far):
        level_far = np.floor(np.log(distances[far] / d0) / log_ratio).astype(np.int32)
        assigned[far] = np.clip(level_far, 0, num_levels - 1)
    return assigned


def _assign_levels_from_distance(distances: np.ndarray, config: AdaptiveMeshConfig) -> np.ndarray:
    """Map unsigned surface distance to target level (0 = finest)."""
    return _assign_levels_from_distances_1d(distances.ravel(), config).reshape(distances.shape)


def _batched_on_surface(mesh: trimesh.Trimesh, centers: np.ndarray, batch_size: int = 500_000) -> np.ndarray:
    """Query surface distances in batches to limit peak memory."""
    pq = ProximityQuery(mesh)
    distances = np.empty(len(centers), dtype=np.float64)
    for start in range(0, len(centers), batch_size):
        end = min(start + batch_size, len(centers))
        _, distances[start:end], _ = pq.on_surface(centers[start:end])
    return distances


def _shape_at_level(grid_shape_finest: Tuple[int, int, int], level: int) -> Tuple[int, int, int]:
    stride = 2**level
    return (grid_shape_finest[0] // stride, grid_shape_finest[1] // stride, grid_shape_finest[2] // stride)


def _active_to_tight_mask(
    active_indices: np.ndarray,
    full_shape: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert sparse cell indices to a tight dense mask and finest-grid origin offset."""
    if active_indices.size == 0:
        return np.zeros((0, 0, 0), dtype=bool), np.zeros(3, dtype=int)

    arr = np.asarray(active_indices, dtype=int)
    imin = arr.min(axis=0)
    imax = arr.max(axis=0) + 1
    tight_shape = tuple(imax - imin)
    mask = np.zeros(tight_shape, dtype=bool)
    rel = arr - imin
    mask[rel[:, 0], rel[:, 1], rel[:, 2]] = True
    return mask, imin


def _carve_coarse_masks(
    masks: List[np.ndarray],
    mask_origins: List[np.ndarray],
    level: int,
    active_indices: np.ndarray,
    num_levels: int,
):
    """Clear coarser-level cells overlapped by finer active cells."""
    if len(active_indices) == 0:
        return
    for coarse in range(level + 1, num_levels):
        coarse_mask = masks[coarse]
        if coarse_mask.size == 0:
            continue
        scale = 2 ** (coarse - level)
        o = mask_origins[coarse]
        ii = active_indices[:, 0] // scale - o[0]
        jj = active_indices[:, 1] // scale - o[1]
        kk = active_indices[:, 2] // scale - o[2]
        valid = (
            (ii >= 0)
            & (jj >= 0)
            & (kk >= 0)
            & (ii < coarse_mask.shape[0])
            & (jj < coarse_mask.shape[1])
            & (kk < coarse_mask.shape[2])
        )
        if np.any(valid):
            coords = np.stack([ii[valid], jj[valid], kk[valid]], axis=1)
            coords = np.unique(coords, axis=0)
            coarse_mask[coords[:, 0], coords[:, 1], coords[:, 2]] = False


def _parent_centers(parent_refine: np.ndarray, origin: np.ndarray, parent_voxel: float) -> np.ndarray:
    """Cell-center coordinates for parent indices on the parent grid."""
    return origin + (parent_refine.astype(np.float64) + 0.5) * parent_voxel


def _conservative_coarse_targets(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    grid_shape: Tuple[int, int, int],
    voxel_size: float,
    config: AdaptiveMeshConfig,
    batch_cells: int = 50_000,
) -> np.ndarray:
    """
    Assign coarse-cell targets from the minimum surface distance over corners + center.

    Center-only distance can miss geometry inside large coarse cells (e.g. ground
    and buildings under a far-away cell center), leaving surface voxels at coarse
    resolution.  Sampling the cell extrema is conservative and cheap at coarse scale.
    """
    nx, ny, nz = grid_shape
    indices = np.stack(
        np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    corner_offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )

    bases = origin + indices.astype(np.float64) * voxel_size
    targets = np.empty(len(indices), dtype=np.int32)

    for start in range(0, len(indices), batch_cells):
        end = min(start + batch_cells, len(indices))
        sample_pts = (bases[start:end, None, :] + corner_offsets[None, :, :] * voxel_size).reshape(-1, 3)
        dists = _batched_on_surface(mesh, sample_pts).reshape(end - start, len(corner_offsets))
        targets[start:end] = _assign_levels_from_distances_1d(dists.min(axis=1), config)

    return targets.reshape(nx, ny, nz)


def _assign_children_from_parents(
    parent_refine: np.ndarray,
    origin: np.ndarray,
    parent_voxel: float,
    child_voxel: float,
    mesh: trimesh.Trimesh,
    level: int,
    config: AdaptiveMeshConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Subdivide parent cells and assign each child from its own surface distance."""
    if len(parent_refine) == 0:
        return np.empty(0, dtype=np.int32), np.empty((0, 3), dtype=int)

    centers, keys = _child_centers_and_keys(parent_refine, origin, parent_voxel, child_voxel)
    dists = _batched_on_surface(mesh, centers)
    child_targets = _assign_levels_from_distances_1d(dists, config)
    return child_targets, keys


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
        if target <= source_level:
            idx = t_keys >> (source_level - target) if source_level > target else t_keys
        else:
            idx = t_keys >> (target - source_level)
        assignments[target].append(idx)


def _enforce_owner_block_uniformity(owner: np.ndarray, num_levels: int, max_passes: int = 24) -> np.ndarray:
    """Refine owner levels within each dyadic block to the finest level present."""
    result = owner.copy()
    nx, ny, nz = result.shape

    for _ in range(max_passes):
        changed = False
        for level in range(num_levels - 1, 0, -1):
            stride = 2**level
            sx, sy, sz = nx // stride, ny // stride, nz // stride
            if sx == 0 or sy == 0 or sz == 0:
                continue
            trimmed = result[: sx * stride, : sy * stride, : sz * stride]
            blocks = trimmed.reshape(sx, stride, sy, stride, sz, stride)
            block_min = blocks.min(axis=(1, 3, 5))
            if np.any(blocks.max(axis=(1, 3, 5)) > block_min):
                blocks[...] = block_min[:, None, :, None, :, None]
                result[: sx * stride, : sy * stride, : sz * stride] = blocks.reshape(
                    sx * stride, sy * stride, sz * stride
                )
                changed = True
        if not changed:
            break

    return result


def _ensure_tileable_transition_bands(owner: np.ndarray, num_levels: int) -> np.ndarray:
    """
    Thicken transition bands so intermediate levels can be emitted as dyadic tiles.

    Strong balance on the finest grid requires owner==L+1 between L and L+2.
    Dyadic level-(L+1) cells need a shell at least ``2 ** (L + 1)`` finest cells
    wide; this pass promotes coarse cells near finer regions, then dilates each
    transition shell so 2x2x2 (etc.) tiles can be formed.
    """
    from scipy import ndimage

    result = owner.copy()
    for level in range(num_levels - 2, -1, -1):
        transition = level + 1
        band_width = 2**transition
        fine = result <= level
        if not np.any(fine):
            continue
        dist = ndimage.distance_transform_edt(~fine)
        promote = (result >= level + 2) & (dist > 0) & (dist <= band_width)
        if np.any(promote):
            result[promote] = transition

        shell = result == transition
        if not np.any(shell):
            continue
        structure = np.ones((band_width, band_width, band_width), dtype=bool)
        dilated = ndimage.binary_dilation(shell, structure=structure)
        widen = dilated & (result > level) & (result >= transition)
        if np.any(widen):
            result[widen] = transition
    return result


def _emit_uniform_owner_block(
    masks: List[np.ndarray],
    lev: int,
    i0: int,
    j0: int,
    k0: int,
    ei: int,
    ej: int,
    ek: int,
) -> bool:
    """Tile a uniform owner block into dyadic cells at the largest fitting level."""
    emit = lev
    stride = 2**emit
    while emit > 0 and (
        min(ei, ej, ek) < stride or ei % stride != 0 or ej % stride != 0 or ek % stride != 0
    ):
        emit -= 1
        stride = 2**emit

    if min(ei, ej, ek) < stride:
        return False

    for ii in range(i0, i0 + ei, stride):
        for jj in range(j0, j0 + ej, stride):
            for kk in range(k0, k0 + ek, stride):
                masks[emit][ii // stride, jj // stride, kk // stride] = True
    return True


def _refine_transition_layers(owner: np.ndarray, owner_floor: np.ndarray, num_levels: int) -> np.ndarray:
    """Refine coarse cells that sit across a gap from a much finer neighbor."""
    from scipy import ndimage

    result = owner.copy()
    structure = np.ones((3, 3, 3), dtype=bool)
    for level in range(num_levels - 2, -1, -1):
        neighbor_min = ndimage.minimum_filter(result, footprint=structure, mode="nearest")
        refine = (result >= level + 2) & (neighbor_min <= level)
        if np.any(refine):
            result[refine] = level + 1
    return np.minimum(result, owner_floor)


def _finalize_owner_grid(owner: np.ndarray, owner_floor: np.ndarray, num_levels: int) -> np.ndarray:
    """
    Enforce strong balance (ΔL ≤ 1) on the owner field.

    ``owner_floor`` is the distance-assigned maximum coarseness (level index); cells
    are never coarsened beyond that assignment (lower index = finer).
    """
    owner = _enforce_strong_balance_bidirectional(owner)
    owner = np.minimum(owner, owner_floor)
    owner = _refine_transition_layers(owner, owner_floor, num_levels)
    owner = _enforce_strong_balance_bidirectional(owner)
    owner = np.minimum(owner, owner_floor)
    owner = _refine_transition_layers(owner, owner_floor, num_levels)
    owner = np.minimum(owner, owner_floor)
    return owner


def _build_masks_from_assignments(
    assignments: List[List[np.ndarray]],
    grid_shape_finest: Tuple[int, int, int],
    num_levels: int,
    max_dense_cells: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Build non-overlapping per-level masks from collected cell-level assignments.

    Uses finest-wins ownership on a tight finest-grid bbox so partial parent
    refinement does not delete entire coarse cells (the source of checkerboard gaps).
    """
    nx, ny, nz = grid_shape_finest

    # Concatenate and deduplicate assignment indices per level.
    level_indices: List[np.ndarray] = []
    for target in range(num_levels):
        if assignments[target]:
            arr = np.unique(np.vstack(assignments[target]), axis=0)
        else:
            arr = np.empty((0, 3), dtype=int)
        level_indices.append(arr)

    # Tight finest-grid bbox covering all assigned cells.
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

    # Embed into the full finest grid, then enforce strong balance before packing masks.
    full_owner = np.full((nx, ny, nz), num_levels - 1, dtype=np.int32)
    full_floor = np.full((nx, ny, nz), num_levels - 1, dtype=np.int32)
    full_owner[fi_min:fi_max, fj_min:fj_max, fk_min:fk_max] = owner
    full_floor[fi_min:fi_max, fj_min:fj_max, fk_min:fk_max] = owner_floor
    owner = _finalize_owner_grid(full_owner, full_floor, num_levels)

    masks = _build_masks_from_owner(owner, num_levels, owner_floor=full_floor)
    mask_origins = [np.zeros(3, dtype=int) for _ in range(num_levels)]

    return masks, mask_origins


def _make_masks_octree_refine(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    grid_shape_finest: Tuple[int, int, int],
    config: AdaptiveMeshConfig,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Octree-style refinement: distance is evaluated only on the coarsest grid,
    then refined locally toward the surface.  Never allocates a full finest grid.
    """
    num_levels = config.num_levels
    coarsest = num_levels - 1

    assignments: List[List[np.ndarray]] = [[] for _ in range(num_levels)]

    shape_c = _shape_at_level(grid_shape_finest, coarsest)
    voxel_c = config.voxel_size * (2**coarsest)
    n_c = int(np.prod(shape_c))
    print(f"  Level {coarsest} (coarsest): conservative distance on {shape_c} ({n_c:,} cells, voxel={voxel_c:.1f} m)", flush=True)
    t0 = time.perf_counter()
    target_c = _conservative_coarse_targets(mesh, origin, shape_c, voxel_c, config)
    print(f"    distance done in {time.perf_counter() - t0:.1f}s", flush=True)
    coarse_keep = np.argwhere(target_c == coarsest)
    if len(coarse_keep):
        assignments[coarsest].append(coarse_keep)

    parent_refine = np.argwhere(target_c <= coarsest - 1) if num_levels > 1 else np.empty((0, 3), dtype=int)
    parent_level = coarsest

    for level in range(coarsest - 1, -1, -1):
        n_refine = len(parent_refine)
        print(f"  Level {level}: refining {n_refine:,} parent cells", flush=True)
        if n_refine == 0:
            continue

        parent_voxel = config.voxel_size * (2**parent_level)
        child_voxel = config.voxel_size * (2**level)

        t0 = time.perf_counter()
        child_targets, keys = _assign_children_from_parents(
            parent_refine, origin, parent_voxel, child_voxel, mesh, level, config
        )
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
        assignments, grid_shape_finest, num_levels, config.max_dense_cells
    )
    print(f"    masks built in {time.perf_counter() - t0:.1f}s", flush=True)
    return masks, mask_origins


def _enforce_strong_balance_on_grid(assigned: np.ndarray, max_passes: int = 64) -> np.ndarray:
    """Enforce ΔL ≤ 1 by refining cells that are more than one level too coarse."""
    from scipy import ndimage

    result = assigned.copy()
    structure = np.ones((3, 3, 3), dtype=bool)

    for _ in range(max_passes):
        neighbor_min = ndimage.minimum_filter(result, footprint=structure, mode="nearest")
        refine = result > (neighbor_min + 1)
        if not np.any(refine):
            break
        result[refine] = neighbor_min[refine] + 1

    return result


def _enforce_strong_balance_bidirectional(assigned: np.ndarray, max_passes: int = 64) -> np.ndarray:
    """Enforce ΔL ≤ 1 by refining and coarsening until neighbor differences are ≤ 1."""
    from scipy import ndimage

    result = assigned.copy()
    structure = np.ones((3, 3, 3), dtype=bool)

    for _ in range(max_passes):
        neighbor_min = ndimage.minimum_filter(result, footprint=structure, mode="nearest")
        neighbor_max = ndimage.maximum_filter(result, footprint=structure, mode="nearest")
        refine = result > (neighbor_min + 1)
        coarsen = result < (neighbor_max - 1)
        if not np.any(refine) and not np.any(coarsen):
            break
        if np.any(refine):
            result[refine] = neighbor_min[refine] + 1
        if np.any(coarsen):
            result[coarsen] = neighbor_max[coarsen] - 1

    return result


def _finest_coverage(masks: List[np.ndarray], num_levels: int, grid_shape: Tuple[int, int, int]) -> np.ndarray:
    """Return a boolean array marking which finest-grid slots are covered by any mask."""
    nx, ny, nz = grid_shape
    covered = np.zeros((nx, ny, nz), dtype=bool)
    for level, mask in enumerate(masks):
        stride = 2**level
        sx, sy, sz = nx // stride, ny // stride, nz // stride
        active = mask[:sx, :sy, :sz]
        expanded = np.repeat(
            np.repeat(np.repeat(active, stride, axis=0), stride, axis=1),
            stride,
            axis=2,
        )
        covered[: sx * stride, : sy * stride, : sz * stride] |= expanded
    return covered


def _owners_from_masks(masks: List[np.ndarray], num_levels: int, grid_shape: Tuple[int, int, int]) -> np.ndarray:
    """Map active mask cells to a per-finest-slot level label (finest wins)."""
    nx, ny, nz = grid_shape
    owners = np.full((nx, ny, nz), -1, dtype=np.int32)
    for level in range(num_levels - 1, -1, -1):
        stride = 2**level
        mask = masks[level]
        sx, sy, sz = nx // stride, ny // stride, nz // stride
        active = mask[:sx, :sy, :sz]
        if not np.any(active):
            continue
        expanded = np.repeat(
            np.repeat(np.repeat(active, stride, axis=0), stride, axis=1),
            stride,
            axis=2,
        )
        region = owners[: sx * stride, : sy * stride, : sz * stride]
        region[expanded] = level
        owners[: sx * stride, : sy * stride, : sz * stride] = region
    return owners


def _batch_activate_mask_cells(
    masks: List[np.ndarray],
    level: int,
    finest_points: np.ndarray,
) -> bool:
    """Activate all dyadic cells at ``level`` covering ``finest_points`` (N, 3)."""
    if len(finest_points) == 0:
        return False
    stride = 2**level
    cells = np.unique(finest_points // stride, axis=0)
    mask = masks[level]
    before = int(np.count_nonzero(mask))
    mask[cells[:, 0], cells[:, 1], cells[:, 2]] = True
    return int(np.count_nonzero(mask)) != before


def _activate_transition_mask_cell(
    masks: List[np.ndarray],
    level: int,
    fi: int,
    fj: int,
    fk: int,
    num_levels: int,
) -> bool:
    """Mark the dyadic cell at ``level`` covering finest slot (fi, fj, fk)."""
    stride = 2**level
    ci, cj, ck = fi // stride, fj // stride, fk // stride
    if (
        ci < 0
        or cj < 0
        or ck < 0
        or ci >= masks[level].shape[0]
        or cj >= masks[level].shape[1]
        or ck >= masks[level].shape[2]
    ):
        return False
    if masks[level][ci, cj, ck]:
        return False
    masks[level][ci, cj, ck] = True
    return True


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


def _parent_cells_for_finest_points(
    owners: np.ndarray,
    points: np.ndarray,
) -> set[Tuple[int, int, int, int]]:
    """Map finest-grid points to active parent cells ``(level, ci, cj, ck)``."""
    cells: set[Tuple[int, int, int, int]] = set()
    for fi, fj, fk in points:
        lv = int(owners[fi, fj, fk])
        stride = 2**lv
        cells.add((lv, fi // stride, fj // stride, fk // stride))
    return cells


def _subdivide_mask_cell(masks: List[np.ndarray], level: int, ci: int, cj: int, ck: int) -> bool:
    """Replace one active level-``level`` cell by its 2x2x2 children at ``level - 1``."""
    if level <= 0:
        return False
    if not masks[level][ci, cj, ck]:
        return False
    masks[level][ci, cj, ck] = False
    child_level = level - 1
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                masks[child_level][ci * 2 + di, cj * 2 + dj, ck * 2 + dk] = True
    return True


def _repair_balance_by_subdivision(
    masks: List[np.ndarray],
    num_levels: int,
    grid_shape: Tuple[int, int, int],
    max_passes: int = 96,
) -> List[np.ndarray]:
    """Subdivide coarse cells wherever finest labels jump by >1 (26-connected)."""
    masks = [m.copy() for m in masks]

    for _ in range(max_passes):
        owners = _owners_from_masks(masks, num_levels, grid_shape)
        marked = owners >= 0
        subdivide: set[Tuple[int, int, int, int]] = set()

        for di, dj, dk in _NEIGHBOR_OFFSETS_26:
            n_o = _shift_toward_offset(owners, di, dj, dk, -1)
            n_m = _shift_toward_offset(marked.astype(np.int8), di, dj, dk, 0).astype(bool)
            valid = marked & n_m & (n_o >= 0)

            neighbor_coarser = valid & (n_o - owners > 1)
            if np.any(neighbor_coarser):
                centers = np.argwhere(neighbor_coarser)
                neighbor_pts = centers + np.array([di, dj, dk], dtype=int)
                subdivide |= _parent_cells_for_finest_points(owners, neighbor_pts)

            center_coarser = valid & (owners - n_o > 1)
            if np.any(center_coarser):
                subdivide |= _parent_cells_for_finest_points(owners, np.argwhere(center_coarser))

        if not subdivide:
            break

        changed = False
        for lv, ci, cj, ck in subdivide:
            changed |= _subdivide_mask_cell(masks, lv, ci, cj, ck)
        if not changed:
            break

    return masks


def _build_non_overlapping_masks_vectorized(owner: np.ndarray, num_levels: int) -> List[np.ndarray]:
    """Build per-level masks where a cell is active iff its dyadic block is uniform in owner."""
    nx, ny, nz = owner.shape
    masks = []
    for level in range(num_levels):
        stride = 2**level
        sx, sy, sz = nx // stride, ny // stride, nz // stride
        trimmed = owner[: sx * stride, : sy * stride, : sz * stride]
        blocks = trimmed.reshape(sx, stride, sy, stride, sz, stride)
        masks.append(np.all(blocks == level, axis=(1, 3, 5)))
    return masks


def _build_masks_adaptive_partition(owner: np.ndarray, num_levels: int) -> List[np.ndarray]:
    """
    Partition ``owner`` into non-overlapping dyadic cells.

    Recursively subdivide mixed blocks; emit a cell only when the block is
    uniform and exactly tileable at the representative level.
    """
    nx, ny, nz = owner.shape
    masks = [
        np.zeros((nx // 2**level, ny // 2**level, nz // 2**level), dtype=bool)
        for level in range(num_levels)
    ]

    stack: List[Tuple[int, int, int, int, int, int]] = [(0, 0, 0, nx, ny, nz)]

    while stack:
        i0, j0, k0, ei, ej, ek = stack.pop()
        block = owner[i0 : i0 + ei, j0 : j0 + ej, k0 : k0 + ek]
        o_min = int(block.min())
        o_max = int(block.max())

        if o_min == o_max:
            if _emit_uniform_owner_block(masks, o_min, i0, j0, k0, ei, ej, ek):
                continue

        if ei == 1 and ej == 1 and ek == 1:
            masks[0][i0, j0, k0] = True
            continue

        hi = ei // 2 if ei > 1 else 1
        hj = ej // 2 if ej > 1 else 1
        hk = ek // 2 if ek > 1 else 1
        for di in (0, hi) if ei > 1 else (0,):
            for dj in (0, hj) if ej > 1 else (0,):
                for dk in (0, hk) if ek > 1 else (0,):
                    ni = hi if di == 0 and ei > 1 else ei - hi
                    nj = hj if dj == 0 and ej > 1 else ej - hj
                    nk = hk if dk == 0 and ek > 1 else ek - hk
                    if ni > 0 and nj > 0 and nk > 0:
                        stack.append((i0 + di, j0 + dj, k0 + dk, ni, nj, nk))

    return masks


def _fill_mask_coverage_gaps(
    masks: List[np.ndarray],
    owner: np.ndarray,
    num_levels: int,
    grid_shape: Tuple[int, int, int],
    max_passes: int = 8,
) -> List[np.ndarray]:
    """Activate dyadic cells for finest slots not yet covered by any mask."""
    masks = [m.copy() for m in masks]

    for _ in range(max_passes):
        covered = _finest_coverage(masks, num_levels, grid_shape)
        if not np.any(~covered):
            break

        for level in range(num_levels):
            stride = 2**level
            want = np.argwhere((~covered) & (owner == level))
            if len(want):
                _batch_activate_mask_cells(masks, level, want)

        masks = _carve_overlaps_from_finest(masks, num_levels)
        covered = _finest_coverage(masks, num_levels, grid_shape)
        if np.any(~covered):
            masks = _fill_coverage_gaps(masks, owner)
            masks = _carve_overlaps_from_finest(masks, num_levels)

    return masks


def _build_masks_from_owner(
    owner: np.ndarray,
    num_levels: int,
    owner_floor: np.ndarray | None = None,
) -> List[np.ndarray]:
    """
    Build a non-overlapping, fully covering, strongly-balanced partition.

    Owner is balanced and thickened for tileable transitions, tiled with block
    uniformity, then any remaining level jumps are removed by subdividing coarse
    cells (never carving, which left coverage holes).
    """
    floor = owner.copy() if owner_floor is None else owner_floor
    owner = _finalize_owner_grid(owner, floor, num_levels)
    owner = _ensure_tileable_transition_bands(owner, num_levels)
    owner = _finalize_owner_grid(owner, floor, num_levels)
    owner = _enforce_owner_block_uniformity(owner, num_levels)
    masks = _build_non_overlapping_masks_vectorized(owner, num_levels)
    masks = _repair_balance_by_subdivision(masks, num_levels, owner.shape)
    if np.any(~_finest_coverage(masks, num_levels, owner.shape)):
        masks = _fill_coverage_gaps(masks, owner)
    return masks


def _enforce_strong_balance(assigned: np.ndarray, max_passes: int = 64) -> np.ndarray:
    """
    Enforce ΔL ≤ 1 between 26-connected neighbors on the finest grid.

    When a neighbor is more than one level coarser, refine it (decrease level index).
    """
    result = assigned.copy()
    nx, ny, nz = result.shape

    for _ in range(max_passes):
        changed = False
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    level = result[i, j, k]
                    for di, dj, dk in _NEIGHBOR_OFFSETS_26:
                        ni, nj, nk = i + di, j + dj, k + dk
                        if ni < 0 or nj < 0 or nk < 0 or ni >= nx or nj >= ny or nk >= nz:
                            continue
                        neighbor_level = result[ni, nj, nk]
                        if neighbor_level > level + 1:
                            result[ni, nj, nk] = level + 1
                            changed = True
        if not changed:
            break

    return result


def _enforce_block_uniformity(assigned: np.ndarray, num_levels: int, max_passes: int = 16) -> np.ndarray:
    """
    Refine assigned levels within each coarse block to the finest level present.

    Ensures each level-l cell can represent its block without partial overlap.
    """
    result = assigned.copy()
    nx, ny, nz = result.shape

    for _ in range(max_passes):
        changed = False
        for level in range(num_levels - 1, 0, -1):
            stride = 2**level
            for i in range(0, nx, stride):
                i_end = min(i + stride, nx)
                for j in range(0, ny, stride):
                    j_end = min(j + stride, ny)
                    for k in range(0, nz, stride):
                        k_end = min(k + stride, nz)
                        block = result[i:i_end, j:j_end, k:k_end]
                        block_min = int(block.min())
                        if int(block.max()) > block_min:
                            result[i:i_end, j:j_end, k:k_end] = block_min
                            changed = True
        if not changed:
            break

    return result


def _carve_overlaps_from_finest(masks: List[np.ndarray], num_levels: int) -> List[np.ndarray]:
    """Remove coarse cells overlapped by any finer active cell."""
    for fine in range(num_levels - 1):
        for coarse in range(fine + 1, num_levels):
            scale = 2 ** (coarse - fine)
            fi, fj, fk = np.nonzero(masks[fine])
            if len(fi):
                masks[coarse][fi // scale, fj // scale, fk // scale] = False
    return masks


def _fill_coverage_gaps_from_owner(masks: List[np.ndarray], owner: np.ndarray, num_levels: int) -> List[np.ndarray]:
    """Extend coarsest cells only into finest slots not covered by any mask."""
    nx, ny, nz = owner.shape
    covered = np.zeros((nx, ny, nz), dtype=bool)
    for level, mask in enumerate(masks):
        stride = 2**level
        fi, fj, fk = np.nonzero(mask)
        if len(fi):
            for di in range(stride):
                for dj in range(stride):
                    for dk in range(stride):
                        covered[fi * stride + di, fj * stride + dj, fk * stride + dk] = True
    uncovered = np.argwhere(~covered)
    if len(uncovered) == 0:
        return masks
    coarsest = num_levels - 1
    stride = 2**coarsest
    ci, cj, ck = (
        uncovered[:, 0] // stride,
        uncovered[:, 1] // stride,
        uncovered[:, 2] // stride,
    )
    masks[coarsest][ci, cj, ck] = True
    return masks


def _fill_coverage_gaps(masks: List[np.ndarray], assigned: np.ndarray) -> List[np.ndarray]:
    """Ensure every finest-grid cell is covered, extending the coarsest level into gaps."""
    nx, ny, nz = assigned.shape
    num_levels = len(masks)
    covered = np.zeros((nx, ny, nz), dtype=bool)

    for level, mask in enumerate(masks):
        stride = 2**level
        for i, j, k in np.argwhere(mask):
            covered[
                i * stride : (i + 1) * stride,
                j * stride : (j + 1) * stride,
                k * stride : (k + 1) * stride,
            ] = True

    coarsest_level = num_levels - 1
    coarse_stride = 2**coarsest_level
    coarsest_mask = masks[coarsest_level]
    for i, j, k in np.argwhere(~covered):
        coarsest_mask[i // coarse_stride, j // coarse_stride, k // coarse_stride] = True

    return masks


def _build_non_overlapping_masks(assigned: np.ndarray, num_levels: int) -> List[np.ndarray]:
    """
    Build per-level active masks where a level-l cell is active iff every
    finest point in its block is assigned exactly level l.
    """
    nx, ny, nz = assigned.shape
    masks = []

    for level in range(num_levels):
        stride = 2**level
        nx_l, ny_l, nz_l = nx // stride, ny // stride, nz // stride
        mask = np.zeros((nx_l, ny_l, nz_l), dtype=bool)
        for i in range(nx_l):
            for j in range(ny_l):
                for k in range(nz_l):
                    block = assigned[
                        i * stride : (i + 1) * stride,
                        j * stride : (j + 1) * stride,
                        k * stride : (k + 1) * stride,
                    ]
                    mask[i, j, k] = np.all(block == level)
        masks.append(mask)

    return masks


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
        if not np.any(mask):
            raise RuntimeError(f"Level {finest_level} has no active cells after mesh construction.")

        voxel_size_level = voxel_size_finest * (2**finest_level)
        offset = mask_origins[finest_level]
        sub_origin_phys = origin_phys + offset.astype(float) * voxel_size_finest
        cuboid_build_level = num_levels - 1 - finest_level

        print(
            f"Level {finest_level}: shape {mask.shape}, active {np.count_nonzero(mask):,}, "
            f"origin offset {offset}, voxel_size {voxel_size_level}"
        )
        raw_level_data.append((mask.copy(), voxel_size_level, sub_origin_phys, cuboid_build_level))

    return list(reversed(raw_level_data))


def make_adaptive_surface_mesh(
    voxel_size: float,
    num_levels: int,
    stl_filename: str,
    domain_padding: Sequence[float] = None,
    expansion_ratio: float = 2.0,
    finest_band_cells: int = 3,
    tile_size: int = 64,
    max_dense_cells: int = 128**3,
    backend: Literal["warp", "python"] = "warp",
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
        backend: ``"warp"`` (default, GPU-accelerated) or ``"python"`` (NumPy/SciPy reference).

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

    if backend not in ("warp", "python"):
        raise ValueError(f"backend must be 'warp' or 'python', got {backend!r}")

    if n_finest > max_dense_cells:
        print("Using octree surface meshing (avoids full finest-grid allocation).", flush=True)
        if backend == "warp":
            from xlb.utils.adaptive_mesher_warp import make_masks_octree_warp

            masks, mask_origins = make_masks_octree_warp(mesh, origin_phys, grid_shape, config)
        else:
            masks, mask_origins = _make_masks_octree_refine(mesh, origin_phys, grid_shape, config)
    else:
        print(f"Using dense finest-grid meshing ({backend} backend).", flush=True)
        if backend == "warp":
            from xlb.utils.adaptive_mesher_warp import make_masks_dense_warp

            masks, mask_origins = make_masks_dense_warp(mesh, origin_phys, grid_shape, config)
        else:
            distances = _compute_distance_field(mesh, origin_phys, grid_shape, voxel_size, config)
            assigned = _assign_levels_from_distance(distances, config)
            owner_floor = assigned.copy()
            assigned = _finalize_owner_grid(assigned, owner_floor, num_levels)
            masks = _build_masks_from_owner(assigned, num_levels, owner_floor=owner_floor)
            mask_origins = [np.zeros(3, dtype=int) for _ in range(num_levels)]

    raw_level_data = _pack_level_data(masks, mask_origins, origin_phys, voxel_size, num_levels)

    return _normalize_level_data(raw_level_data, voxel_size)


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
    parser.add_argument(
        "--backend",
        choices=("warp", "python"),
        default="warp",
        help="Mesh backend: warp (default) or python reference.",
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
    print(f"  backend={args.backend}")

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
            backend=args.backend,
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
