#!/usr/bin/env python3
"""Sepulveda parity + timing: Python vs native Warp backend."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, List, Tuple

import numpy as np

from xlb.utils.adaptive_mesher import (
    AdaptiveMeshConfig,
    _assign_children_from_parents,
    _child_centers_and_keys,
    _compute_domain,
    _conservative_coarse_targets,
    _ensure_tileable_transition_bands,
    _enforce_owner_block_uniformity,
    _fill_coverage_gaps,
    _finalize_owner_grid,
    _finest_coverage,
    _build_non_overlapping_masks_vectorized,
    _record_assignments,
    _repair_balance_by_subdivision,
    _shape_at_level,
    make_adaptive_surface_mesh,
    validate_level_data,
)
from xlb.utils.adaptive_mesher_warp import WarpAdaptiveMesherOps, _max_query_distance


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


def profile_build_masks_from_owner(
    owner: np.ndarray,
    floor: np.ndarray,
    num_levels: int,
    backend: str,
    times: Dict[str, float],
    ops: WarpAdaptiveMesherOps | None = None,
) -> List[np.ndarray]:
    with timed(f"{backend}:owner_finalize_1", times):
        owner = (
            ops.finalize_owner_grid(owner, floor, num_levels)
            if backend == "warp"
            else _finalize_owner_grid(owner, floor, num_levels)
        )
    with timed(f"{backend}:tileable_bands", times):
        owner = (
            ops.ensure_tileable_transition_bands(owner, num_levels)
            if backend == "warp"
            else _ensure_tileable_transition_bands(owner, num_levels)
        )
    with timed(f"{backend}:owner_finalize_2", times):
        owner = (
            ops.finalize_owner_grid(owner, floor, num_levels)
            if backend == "warp"
            else _finalize_owner_grid(owner, floor, num_levels)
        )
    with timed(f"{backend}:block_uniformity", times):
        owner = (
            ops.enforce_owner_block_uniformity(owner, num_levels)
            if backend == "warp"
            else _enforce_owner_block_uniformity(owner, num_levels)
        )
    with timed(f"{backend}:extract_masks", times):
        masks = (
            ops.build_non_overlapping_masks_vectorized(owner, num_levels)
            if backend == "warp"
            else _build_non_overlapping_masks_vectorized(owner, num_levels)
        )
    with timed(f"{backend}:subdiv_repair", times):
        if backend == "warp":
            masks = ops.repair_balance_by_subdivision(masks, num_levels, owner.shape)
        else:
            masks = _repair_balance_by_subdivision(masks, num_levels, owner.shape)
    with timed(f"{backend}:gap_fill", times):
        if backend == "warp":
            masks = ops.fill_coverage_gaps(masks, owner.shape, num_levels)
        elif np.any(~_finest_coverage(masks, num_levels, owner.shape)):
            masks = _fill_coverage_gaps(masks, owner)
    return masks


def _collect_level_indices(assignments, num_levels):
    level_indices: List[np.ndarray] = []
    for target in range(num_levels):
        if assignments[target]:
            level_indices.append(np.unique(np.vstack(assignments[target]), axis=0))
        else:
            level_indices.append(np.empty((0, 3), dtype=int))
    return level_indices


def _embed_owner(level_indices, grid_shape, num_levels, backend, times, ops=None):
    nx, ny, nz = grid_shape
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

    with timed(f"{backend}:owner_paint", times):
        if backend == "warp":
            owner = ops.paint_owner_from_assignments(
                level_indices, (fi_max - fi_min, fj_max - fj_min, fk_max - fk_min),
                fi_min, fj_min, fk_min, num_levels,
            )
        else:
            owner = np.full((fi_max - fi_min, fj_max - fj_min, fk_max - fk_min), num_levels - 1, dtype=np.int32)
            for target in range(num_levels):
                stride = 2**target
                for i, j, k in level_indices[target]:
                    i0, i1 = i * stride - fi_min, (i + 1) * stride - fi_min
                    j0, j1 = j * stride - fj_min, (j + 1) * stride - fj_min
                    k0, k1 = k * stride - fk_min, (k + 1) * stride - fk_min
                    owner[i0:i1, j0:j1, k0:k1] = np.minimum(owner[i0:i1, j0:j1, k0:k1], target)

    full_owner = np.full((nx, ny, nz), num_levels - 1, dtype=np.int32)
    full_floor = np.full((nx, ny, nz), num_levels - 1, dtype=np.int32)
    full_owner[fi_min:fi_max, fj_min:fj_max, fk_min:fk_max] = owner
    full_floor[fi_min:fi_max, fj_min:fj_max, fk_min:fk_max] = owner.copy()
    return full_owner, full_floor


def profile_octree_python(mesh, origin, grid_shape, config, times):
    num_levels = config.num_levels
    coarsest = num_levels - 1
    assignments: List[List[np.ndarray]] = [[] for _ in range(num_levels)]

    shape_c = _shape_at_level(grid_shape, coarsest)
    voxel_c = config.voxel_size * (2**coarsest)
    with timed("python:coarse_distance", times):
        target_c = _conservative_coarse_targets(mesh, origin, shape_c, voxel_c, config)
    if len(np.argwhere(target_c == coarsest)):
        assignments[coarsest].append(np.argwhere(target_c == coarsest))

    parent_refine = np.argwhere(target_c <= coarsest - 1) if num_levels > 1 else np.empty((0, 3), dtype=int)
    parent_level = coarsest
    for level in range(coarsest - 1, -1, -1):
        if len(parent_refine) == 0:
            continue
        parent_voxel = config.voxel_size * (2**parent_level)
        child_voxel = config.voxel_size * (2**level)
        with timed(f"python:refine_L{level}_distance", times):
            child_targets, keys = _assign_children_from_parents(
                parent_refine, origin, parent_voxel, child_voxel, mesh, level, config
            )
        _record_assignments(assignments, keys, child_targets, level, num_levels)
        parent_refine = keys[child_targets <= level - 1] if level > 0 else np.empty((0, 3), dtype=int)
        parent_level = level

    level_indices = _collect_level_indices(assignments, num_levels)
    full_owner, full_floor = _embed_owner(level_indices, grid_shape, num_levels, "python", times)
    return profile_build_masks_from_owner(full_owner, full_floor, num_levels, "python", times)


def profile_octree_warp(mesh, origin, grid_shape, config, times):
    ops = WarpAdaptiveMesherOps(mesh)
    num_levels = config.num_levels
    coarsest = num_levels - 1
    assignments: List[List[np.ndarray]] = [[] for _ in range(num_levels)]

    shape_c = _shape_at_level(grid_shape, coarsest)
    voxel_c = config.voxel_size * (2**coarsest)
    with timed("warp:coarse_distance", times):
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
        with timed(f"warp:refine_L{level}_distance", times):
            centers, keys = _child_centers_and_keys(parent_refine, origin, parent_voxel, child_voxel)
            dists = ops.batched_distances(centers, max_dist)
            child_targets = ops.assign_levels_from_distances_1d(dists, config)
        _record_assignments(assignments, keys, child_targets, level, num_levels)
        parent_refine = keys[child_targets <= level - 1] if level > 0 else np.empty((0, 3), dtype=int)
        parent_level = level

    level_indices = _collect_level_indices(assignments, num_levels)
    full_owner, full_floor = _embed_owner(level_indices, grid_shape, num_levels, "warp", times, ops=ops)
    return profile_build_masks_from_owner(full_owner, full_floor, num_levels, "warp", times, ops=ops)


def bucket_times(times: Dict[str, float], prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in times.items():
        if not k.startswith(prefix):
            continue
        key = k[len(prefix) :]
        if key.startswith("refine_L"):
            out["refine_distance_all"] = out.get("refine_distance_all", 0.0) + v
        else:
            out[key] = out.get(key, 0.0) + v
    return out


def print_timing_report(py_times: Dict[str, float], wp_times: Dict[str, float]):
    py = bucket_times(py_times, "python:")
    wp = bucket_times(wp_times, "warp:")
    keys = sorted(set(py) | set(wp), key=lambda k: max(py.get(k, 0), wp.get(k, 0)), reverse=True)

    print("\n=== Stage breakdown (seconds) ===")
    print(f"{'Stage':<32} {'Python':>10} {'Warp':>10} {'Speedup':>10}")
    print("-" * 66)
    py_total = sum(py.values())
    wp_total = sum(wp.values())
    for k in keys:
        p, w = py.get(k, 0.0), wp.get(k, 0.0)
        sp = p / w if w > 1e-6 else float("inf")
        print(f"{k:<32} {p:10.2f} {w:10.2f} {sp:9.2f}x")
    print("-" * 66)
    print(f"{'TOTAL (profiled stages)':<32} {py_total:10.2f} {wp_total:10.2f} {py_total / wp_total:9.2f}x")


def grid_shape_finest(level_data):
    n = len(level_data)
    return tuple(int(level_data[-1][0].shape[i] * 2 ** (n - 1)) for i in range(3))


def check_parity(py_data, wp_data) -> bool:
    ok = True
    if len(py_data) != len(wp_data):
        print(f"PARITY FAIL: level count {len(py_data)} vs {len(wp_data)}")
        return False
    for idx, (py_entry, wp_entry) in enumerate(zip(py_data, wp_data)):
        py_mask, py_stride, py_origin, py_level = py_entry
        wp_mask, wp_stride, wp_origin, wp_level = wp_entry
        if py_level != wp_level or int(py_stride) != int(wp_stride):
            print(f"PARITY FAIL: level {idx} metadata mismatch")
            ok = False
        if not np.array_equal(py_origin, wp_origin):
            print(f"PARITY FAIL: level {idx} origin mismatch")
            ok = False
        if not np.array_equal(py_mask, wp_mask):
            diff = int(np.count_nonzero(py_mask != wp_mask))
            print(f"PARITY FAIL: level {idx} mask mismatch ({diff:,} cells differ)")
            ok = False
    return ok


def main():
    config = AdaptiveMeshConfig(**MESH_KW)
    mesh, origin, grid_shape = _prepare_grid(config)
    print(f"Sepulveda @ voxel_size={VOXEL_SIZE} m, grid_shape={grid_shape}")

    py_times: Dict[str, float] = {}
    wp_times: Dict[str, float] = {}

    print("\n--- Python backend (instrumented octree) ---")
    with timed("python:total", py_times):
        profile_octree_python(mesh, origin, grid_shape, config, py_times)

    print("\n--- Warp backend (instrumented octree) ---")
    with timed("warp:total", wp_times):
        profile_octree_warp(mesh, origin, grid_shape, config, wp_times)

    print_timing_report(py_times, wp_times)

    print("\n=== End-to-end make_adaptive_surface_mesh ===")
    py_data = wp_data = None
    for backend in ("python", "warp"):
        t0 = time.perf_counter()
        ld = make_adaptive_surface_mesh(**MESH_KW, backend=backend)
        elapsed = time.perf_counter() - t0
        active = sum(int(np.count_nonzero(e[0])) for e in ld)
        print(f"  {backend:6s}: {elapsed:7.1f}s  total_active={active:,}")
        if backend == "python":
            py_data = ld
        else:
            wp_data = ld

    gs = grid_shape_finest(py_data)
    py_stats = validate_level_data(py_data, gs)
    wp_stats = validate_level_data(wp_data, gs)

    print("\n=== Parity ===")
    parity_ok = check_parity(py_data, wp_data)
    if parity_ok:
        print("  Masks: EXACT MATCH (all levels)")
    print(f"  Python stats: {py_stats}")
    print(f"  Warp stats:   {wp_stats}")
    if py_stats == wp_stats:
        print("  Validation stats: MATCH")
    else:
        print("  Validation stats: MISMATCH")
        parity_ok = False

    if not parity_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
