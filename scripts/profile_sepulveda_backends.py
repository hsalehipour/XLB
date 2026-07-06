#!/usr/bin/env python3
"""Sepulveda timing: Warp adaptive mesher profiling."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, List, Tuple

import numpy as np

from xlb.utils.adaptive_mesher import (
    AdaptiveMeshConfig,
    WarpAdaptiveMesherOps,
    _build_masks_greedy_coarsest,
    _child_centers_and_keys,
    _compute_domain,
    _max_query_distance,
    _record_assignments,
    _shape_at_level,
    make_adaptive_surface_mesh,
    validate_level_data,
)


STL = "examples/cfd/stl-files/07022026_SEPULVEDA_SITE_MODEL_FORMA_NOTREES.stl"
VOXEL_SIZE = 12.0
NUM_LEVELS = 4
DOMAIN_PADDING = [0.5, 0.5, 0.5, 0.5, 0.25, 1.0]
MESH_KW = dict(
    voxel_size=VOXEL_SIZE,
    num_levels=NUM_LEVELS,
    stl_filename=STL,
    domain_padding=DOMAIN_PADDING,
    expansion_ratio=2.0,
    finest_band_cells=3,
)


@contextmanager
def timed(label: str, times: Dict[str, float]):
    t0 = time.perf_counter()
    yield
    times[label] = times.get(label, 0.0) + (time.perf_counter() - t0)


def _prepare_grid(config: AdaptiveMeshConfig) -> Tuple:
    mesh, origin, grid_shape = _compute_domain(config)
    factor = 2 ** (config.num_levels - 1)
    nx, ny, nz = grid_shape
    nx += (factor - nx % factor) % factor
    ny += (factor - ny % factor) % factor
    nz += (factor - nz % factor) % factor
    align = 2 ** config.num_levels
    grid_shape = (
        ((nx + align - 1) // align) * align,
        ((ny + align - 1) // align) * align,
        ((nz + align - 1) // align) * align,
    )
    return mesh, origin, grid_shape


def _collect_level_indices(assignments, num_levels):
    level_indices: List[np.ndarray] = []
    for target in range(num_levels):
        if assignments[target]:
            level_indices.append(np.unique(np.vstack(assignments[target]), axis=0))
        else:
            level_indices.append(np.empty((0, 3), dtype=int))
    return level_indices


def profile_octree_warp(mesh, origin, grid_shape, config, times):
    ops = WarpAdaptiveMesherOps(mesh)
    num_levels = config.num_levels
    coarsest = num_levels - 1
    assignments: List[List[np.ndarray]] = [[] for _ in range(num_levels)]

    shape_c = _shape_at_level(grid_shape, coarsest)
    voxel_c = config.voxel_size * (2**coarsest)
    with timed("coarse_distance", times):
        target_c = ops.conservative_coarse_targets(origin, shape_c, voxel_c, config)
    if len(np.argwhere(target_c == coarsest)):
        assignments[coarsest].append(np.argwhere(target_c == coarsest))

    parent_refine = np.argwhere(target_c <= coarsest - 1) if num_levels > 1 else np.empty((0, 3), dtype=int)
    parent_level = coarsest
    max_dist = _max_query_distance(origin, grid_shape, config.voxel_size)

    for level in range(coarsest - 1, -1, -1):
        if len(parent_refine) == 0:
            continue
        parent_voxel = config.voxel_size * (2**parent_level)
        child_voxel = config.voxel_size * (2**level)
        with timed(f"refine_L{level}_distance", times):
            centers, keys = _child_centers_and_keys(parent_refine, origin, parent_voxel, child_voxel)
            dists = ops.batched_distances(centers, max_dist)
            child_targets = ops.assign_levels_from_distances_1d(dists, config)
        _record_assignments(assignments, keys, child_targets, level, num_levels)
        parent_refine = keys[child_targets <= level - 1] if level > 0 else np.empty((0, 3), dtype=int)
        parent_level = level

    level_indices = _collect_level_indices(assignments, num_levels)
    nx, ny, nz = grid_shape
    with timed("owner_paint", times):
        owner = ops.paint_owner_from_assignments(
            level_indices, (nx, ny, nz), 0, 0, 0, num_levels,
        )
    owner_floor = owner.copy()
    with timed("build_masks_from_owner", times):
        masks = ops.build_masks_from_owner(
            owner, num_levels, owner_floor=owner_floor,
            origin=origin, voxel_size=config.voxel_size, config=config,
        )
    return masks


def grid_shape_finest(level_data):
    n = len(level_data)
    return tuple(int(level_data[-1][0].shape[i] * 2 ** (n - 1)) for i in range(3))


def main():
    config = AdaptiveMeshConfig(**MESH_KW)
    mesh, origin, grid_shape = _prepare_grid(config)
    print(f"Sepulveda @ voxel_size={VOXEL_SIZE} m, grid_shape={grid_shape}")

    times: Dict[str, float] = {}

    print("\n--- Warp backend (instrumented octree) ---")
    with timed("total", times):
        profile_octree_warp(mesh, origin, grid_shape, config, times)

    print("\n=== Stage breakdown (seconds) ===")
    print(f"{'Stage':<32} {'Time':>10}")
    print("-" * 44)
    total = times.pop("total", 0.0)
    for k, v in sorted(times.items(), key=lambda x: -x[1]):
        print(f"{k:<32} {v:10.2f}")
    print("-" * 44)
    print(f"{'TOTAL':<32} {total:10.2f}")

    print("\n=== End-to-end make_adaptive_surface_mesh ===")
    t0 = time.perf_counter()
    ld = make_adaptive_surface_mesh(**MESH_KW)
    elapsed = time.perf_counter() - t0
    active = sum(int(np.count_nonzero(e[0])) for e in ld)
    print(f"  warp: {elapsed:7.1f}s  total_active={active:,}")

    gs = grid_shape_finest(ld)
    stats = validate_level_data(ld, gs)
    print(f"  Validation: {stats}")


if __name__ == "__main__":
    main()
