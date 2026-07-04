"""Tests for surface-adaptive multi-resolution mesh generation."""

import os
import tempfile

import importlib.util

import numpy as np
import pytest
import trimesh

HAS_NEON = importlib.util.find_spec("neon") is not None

from xlb.utils.adaptive_mesher import (
    AdaptiveMeshConfig,
    _record_assignments,
    make_adaptive_surface_mesh,
    validate_level_data,
)
from xlb.utils.mesher import make_cuboid_mesh, prepare_sparsity_pattern

SPHERE_VOXEL_SIZE = 2.0


@pytest.fixture
def sphere_stl():
    """Use the checked-in sphere STL when available, else create a temp one."""
    checked_in = os.path.join(
        os.path.dirname(__file__), "..", "examples", "cfd", "stl-files", "sphere.stl"
    )
    checked_in = os.path.normpath(checked_in)
    if os.path.isfile(checked_in):
        yield checked_in
        return

    mesh = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
    mesh.apply_translation([10.0, 10.0, 10.0])
    with tempfile.TemporaryDirectory() as tmpdir:
        stl_path = os.path.join(tmpdir, "sphere.stl")
        mesh.export(stl_path)
        yield stl_path


def _grid_shape_finest(level_data):
    num_levels = len(level_data)
    return tuple(int(level_data[-1][0].shape[i] * 2 ** (num_levels - 1)) for i in range(3))


def test_record_assignments_index_mapping():
    """Child keys on a finer grid must upscale (<<), not downscale (>>), to coarser targets."""
    num_levels = 4
    assignments = [[] for _ in range(num_levels)]
    keys = np.array([[5, 3, 2], [7, 4, 1]], dtype=int)
    targets = np.array([0, 1], dtype=int)

    _record_assignments(assignments, keys, targets, source_level=1, num_levels=num_levels)

    np.testing.assert_array_equal(np.vstack(assignments[0]), [[10, 6, 4]])
    np.testing.assert_array_equal(np.vstack(assignments[1]), [[7, 4, 1]])

    assignments = [[] for _ in range(num_levels)]
    _record_assignments(
        assignments, np.array([[2, 1, 0]]), np.array([2]), source_level=1, num_levels=num_levels
    )
    np.testing.assert_array_equal(np.vstack(assignments[2]), [[1, 0, 0]])


def test_adaptive_mesh_produces_level_data_format(sphere_stl):
    level_data = make_adaptive_surface_mesh(
        voxel_size=SPHERE_VOXEL_SIZE,
        num_levels=3,
        stl_filename=sphere_stl,
        domain_padding=[2, 2, 2, 2, 2, 2],
        expansion_ratio=2.0,
        finest_band_cells=3,
    )

    assert len(level_data) == 3
    for idx, (mask, voxel_size_lattice, origin, level_id) in enumerate(level_data):
        assert mask.dtype == bool or mask.dtype == np.bool_
        assert voxel_size_lattice == 2**idx
        assert origin.shape == (3,)
        assert level_id == idx


def test_adaptive_mesh_fewer_fine_cells_than_full_grid(sphere_stl):
    adaptive_data = make_adaptive_surface_mesh(
        voxel_size=SPHERE_VOXEL_SIZE,
        num_levels=3,
        stl_filename=sphere_stl,
        domain_padding=[2, 2, 2, 2, 2, 2],
        expansion_ratio=2.0,
        finest_band_cells=3,
    )
    grid_shape = _grid_shape_finest(adaptive_data)
    finest_active = int(np.count_nonzero(adaptive_data[0][0]))
    full_grid = int(np.prod(grid_shape))

    assert finest_active < full_grid // 4


def test_adaptive_mesh_non_overlap_and_coverage(sphere_stl):
    level_data = make_adaptive_surface_mesh(
        voxel_size=SPHERE_VOXEL_SIZE,
        num_levels=3,
        stl_filename=sphere_stl,
        domain_padding=[2, 2, 2, 2, 2, 2],
        expansion_ratio=2.0,
        finest_band_cells=3,
    )
    grid_shape = _grid_shape_finest(level_data)
    stats = validate_level_data(level_data, grid_shape)

    assert stats["non_overlapping"] is True
    assert stats["fully_covering"] is True
    assert stats["strongly_balanced"] is True
    assert stats["active_counts"][1] > 0, "expected intermediate level cells at transitions"

def test_prepare_sparsity_pattern_compatible(sphere_stl):
    level_data = make_adaptive_surface_mesh(
        voxel_size=SPHERE_VOXEL_SIZE,
        num_levels=3,
        stl_filename=sphere_stl,
        domain_padding=[2, 2, 2, 2, 2, 2],
        expansion_ratio=2.0,
        finest_band_cells=3,
    )
    sparsity_pattern, level_origins = prepare_sparsity_pattern(level_data)

    assert len(sparsity_pattern) == 3
    assert len(level_origins) == 3
    for mask in sparsity_pattern:
        assert mask.dtype == np.int32
        assert mask.flags["C_CONTIGUOUS"]


def test_adaptive_mesh_config_validation():
    with pytest.raises(ValueError):
        AdaptiveMeshConfig(voxel_size=1.0, num_levels=0, stl_filename="dummy.stl")

    with pytest.raises(ValueError):
        AdaptiveMeshConfig(voxel_size=1.0, num_levels=2, expansion_ratio=1.0, stl_filename="dummy.stl")


@pytest.mark.skipif(not HAS_NEON, reason="Neon backend required")
def test_neon_grid_construction(sphere_stl):
    import neon
    import xlb
    from xlb.compute_backend import ComputeBackend
    from xlb.grid import multires_grid_factory

    compute_backend = ComputeBackend.NEON
    precision_policy = xlb.PrecisionPolicy.FP32FP32
    xlb.init(
        default_backend=compute_backend,
        default_precision_policy=precision_policy,
        velocity_set=xlb.velocity_set.D3Q27(
            precision_policy=precision_policy,
            compute_backend=compute_backend,
        ),
    )

    level_data = make_adaptive_surface_mesh(
        voxel_size=SPHERE_VOXEL_SIZE,
        num_levels=3,
        stl_filename=sphere_stl,
        domain_padding=[2, 2, 2, 2, 2, 2],
        expansion_ratio=2.0,
        finest_band_cells=3,
    )
    grid_shape = _grid_shape_finest(level_data)
    sparsity_pattern, level_origins = prepare_sparsity_pattern(level_data)

    grid = multires_grid_factory(
        grid_shape,
        sparsity_pattern_list=sparsity_pattern,
        sparsity_pattern_origins=[neon.Index_3d(*origin) for origin in level_origins],
    )
    assert grid.count_levels == 3
