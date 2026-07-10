"""Tests for surface-adaptive multi-resolution mesh generation."""

import os
import tempfile

import importlib.util

import numpy as np
import pytest
import trimesh
import warp as wp

HAS_NEON = importlib.util.find_spec("neon") is not None

from xlb.utils.adaptive_mesher import (
    AdaptiveMeshConfig,
    WarpAdaptiveMesherOps,
    _child_centers_and_keys,
    euclidean_edt_3d,
    grid_shape_finest as adaptive_grid_shape_finest,
    is_sparse_level_data,
    make_adaptive_surface_mesh,
)
from xlb.utils.mesher import make_cuboid_mesh, prepare_sparsity_pattern

SPHERE_VOXEL_SIZE = 2.0


@pytest.fixture
def sphere_stl():
    """Use the checked-in sphere STL when available, else create a temp one."""
    checked_in = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "cfd", "stl-files", "sphere.stl")
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


@pytest.fixture
def box_stl():
    with tempfile.TemporaryDirectory() as tmpdir:
        mesh = trimesh.creation.box(extents=[8.0, 6.0, 4.0])
        mesh.apply_translation([20.0, 15.0, 10.0])
        stl_path = os.path.join(tmpdir, "box.stl")
        mesh.export(stl_path)
        yield stl_path


def _grid_shape_finest(level_data):
    return adaptive_grid_shape_finest(level_data)


def _mesh_kwargs(stl_path, num_levels=3):
    return dict(
        voxel_size=2.0,
        num_levels=num_levels,
        stl_filename=stl_path,
        domain_padding=[2, 2, 2, 2, 2, 2],
        expansion_ratio=2.0,
        finest_band_cells=3,
    )


def _active_counts(level_data):
    return [int(entry[0].shape[0]) for entry in level_data]


# ---------------------------------------------------------------------------
# EDT and morphology tests (from parity file)
# ---------------------------------------------------------------------------


def test_edt_matches_scipy():
    from scipy import ndimage

    rng = np.random.default_rng(42)
    for shape in [(16, 16, 16), (12, 20, 8)]:
        mask = rng.random(shape) > 0.7
        scipy_edt = ndimage.distance_transform_edt(mask)
        warp_edt = euclidean_edt_3d(mask)
        np.testing.assert_allclose(scipy_edt, warp_edt, rtol=1e-5, atol=1e-5)


def test_morphology_filters_match_scipy():
    from scipy import ndimage

    rng = np.random.default_rng(7)
    field = rng.integers(0, 3, size=(24, 24, 24), dtype=np.int32)
    ops = WarpAdaptiveMesherOps(trimesh.creation.icosphere(subdivisions=1))
    field_wp = wp.array(field, dtype=wp.int32)
    nmin_wp = ops._min_filter_3x3(field_wp)
    nmax_wp = ops._max_filter_3x3(field_wp)
    structure = np.ones((3, 3, 3), dtype=bool)
    np.testing.assert_array_equal(nmin_wp.numpy(), ndimage.minimum_filter(field, footprint=structure, mode="nearest"))
    np.testing.assert_array_equal(nmax_wp.numpy(), ndimage.maximum_filter(field, footprint=structure, mode="nearest"))


# ---------------------------------------------------------------------------
# Octree helper tests
# ---------------------------------------------------------------------------


def test_child_centers_and_keys_offset_ordering():
    """Single parent at (1,2,3) produces 8 children with correct keys and centres."""
    parent = np.array([[1, 2, 3]])
    origin = np.array([0.0, 0.0, 0.0])
    parent_voxel = 4.0
    child_voxel = 2.0

    centers, keys = _child_centers_and_keys(parent, origin, parent_voxel, child_voxel)

    assert centers.shape == (8, 3)
    assert keys.shape == (8, 3)

    expected_keys = np.array([
        [2, 4, 6],
        [2, 4, 7],
        [2, 5, 6],
        [2, 5, 7],
        [3, 4, 6],
        [3, 4, 7],
        [3, 5, 6],
        [3, 5, 7],
    ])
    np.testing.assert_array_equal(np.sort(keys, axis=0), np.sort(expected_keys, axis=0))

    for i in range(8):
        expected_center = origin + (keys[i] + 0.5) * child_voxel
        np.testing.assert_allclose(centers[i], expected_center, atol=1e-12)


# ---------------------------------------------------------------------------
# Mesh generation tests
# ---------------------------------------------------------------------------


def test_adaptive_mesh_produces_level_data_format(sphere_stl):
    level_data = make_adaptive_surface_mesh(
        voxel_size=SPHERE_VOXEL_SIZE,
        num_levels=3,
        stl_filename=sphere_stl,
        domain_padding=[2, 2, 2, 2, 2, 2],
        expansion_ratio=2.0,
        finest_band_cells=3,
    )

    assert is_sparse_level_data(level_data)
    assert len(level_data) == 3
    for idx, (coords, voxel_size_lattice, origin, level_id) in enumerate(level_data):
        assert coords.dtype == np.int32
        assert coords.ndim == 2 and coords.shape[1] == 3
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
    finest_active = int(adaptive_data[0][0].shape[0])
    full_grid = int(np.prod(grid_shape))

    assert finest_active < full_grid // 4


def test_adaptive_mesh_sparse_levels_populated(sphere_stl):
    level_data = make_adaptive_surface_mesh(
        voxel_size=SPHERE_VOXEL_SIZE,
        num_levels=3,
        stl_filename=sphere_stl,
        domain_padding=[2, 2, 2, 2, 2, 2],
        expansion_ratio=2.0,
        finest_band_cells=3,
    )
    active_counts = _active_counts(level_data)

    assert active_counts[0] > 0
    assert active_counts[1] > 0, "expected intermediate level cells at transitions"
    assert active_counts[2] > 0
    for coords, _, _, _ in level_data:
        assert len(np.unique(coords, axis=0)) == len(coords)


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


# ---------------------------------------------------------------------------
# Mesh validity tests (from parity file)
# ---------------------------------------------------------------------------


def test_sphere_mesh_valid(sphere_stl):
    """Warp backend produces a valid sparse adaptive partition."""
    data = make_adaptive_surface_mesh(**_mesh_kwargs(sphere_stl))
    assert is_sparse_level_data(data)
    assert _active_counts(data)[0] < 8_000


def test_sphere_finest_band_near_surface(sphere_stl):
    """Finest-level cells should lie within the configured distance band of the STL."""
    from xlb.utils.mesher import _load_stl_mesh
    from trimesh.proximity import ProximityQuery

    kwargs = _mesh_kwargs(sphere_stl, num_levels=3)
    kwargs["finest_band_cells"] = 2
    data = make_adaptive_surface_mesh(**kwargs)
    mesh = _load_stl_mesh(sphere_stl)
    pq = ProximityQuery(mesh)

    voxel_size = kwargs["voxel_size"]
    d0 = kwargs["finest_band_cells"] * voxel_size
    coords = data[0][0]
    stride = int(data[0][1])
    origin = data[0][2] * stride
    centers = (coords + origin + 0.5) * voxel_size
    _, dists, _ = pq.on_surface(centers)
    assert float(np.percentile(dists, 95)) < d0 * 8.0
    assert float(dists.min()) < d0


def test_synthetic_box_valid(box_stl):
    kwargs = dict(
        voxel_size=1.5,
        num_levels=3,
        stl_filename=box_stl,
        domain_padding=[1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        expansion_ratio=2.0,
        finest_band_cells=2,
    )
    data = make_adaptive_surface_mesh(**kwargs)
    assert is_sparse_level_data(data)
    assert all(count > 0 for count in _active_counts(data))


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_adaptive_mesh_config_validation():
    with pytest.raises(ValueError):
        AdaptiveMeshConfig(voxel_size=1.0, num_levels=0, stl_filename="dummy.stl")

    with pytest.raises(ValueError):
        AdaptiveMeshConfig(voxel_size=1.0, num_levels=2, expansion_ratio=1.0, stl_filename="dummy.stl")


# ---------------------------------------------------------------------------
# Neon integration test
# ---------------------------------------------------------------------------


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
