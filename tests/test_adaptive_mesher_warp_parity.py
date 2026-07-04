"""Parity tests: Warp backend vs Python reference adaptive mesher."""

import os
import tempfile

import numpy as np
import pytest
import trimesh
import warp as wp

from xlb.utils.adaptive_mesher import make_adaptive_surface_mesh, validate_level_data
from xlb.utils.adaptive_mesher_warp import euclidean_edt_3d


@pytest.fixture
def sphere_stl():
    checked_in = os.path.join(
        os.path.dirname(__file__), "..", "examples", "cfd", "stl-files", "sphere.stl"
    )
    checked_in = os.path.normpath(checked_in)
    if os.path.isfile(checked_in):
        return checked_in

    mesh = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
    mesh.apply_translation([10.0, 10.0, 10.0])
    with tempfile.TemporaryDirectory() as tmpdir:
        stl_path = os.path.join(tmpdir, "sphere.stl")
        mesh.export(stl_path)
        return stl_path


@pytest.fixture
def box_stl():
    with tempfile.TemporaryDirectory() as tmpdir:
        mesh = trimesh.creation.box(extents=[8.0, 6.0, 4.0])
        mesh.apply_translation([20.0, 15.0, 10.0])
        stl_path = os.path.join(tmpdir, "box.stl")
        mesh.export(stl_path)
        yield stl_path


def _grid_shape_finest(level_data):
    num_levels = len(level_data)
    return tuple(int(level_data[-1][0].shape[i] * 2 ** (num_levels - 1)) for i in range(3))


def _assert_level_data_equal(python_data, warp_data):
    assert len(python_data) == len(warp_data)
    for idx, (py_entry, wp_entry) in enumerate(zip(python_data, warp_data)):
        py_mask, py_stride, py_origin, py_level = py_entry
        wp_mask, wp_stride, wp_origin, wp_level = wp_entry
        assert py_level == wp_level, f"level id mismatch at index {idx}"
        assert int(py_stride) == int(wp_stride), f"stride mismatch at level {idx}"
        assert np.array_equal(py_origin, wp_origin), f"origin mismatch at level {idx}"
        assert np.array_equal(py_mask, wp_mask), f"mask mismatch at level {idx}"


def _mesh_kwargs(stl_path, max_dense_cells=128**3, num_levels=3):
    return dict(
        voxel_size=2.0,
        num_levels=num_levels,
        stl_filename=stl_path,
        domain_padding=[2, 2, 2, 2, 2, 2],
        expansion_ratio=2.0,
        finest_band_cells=3,
        max_dense_cells=max_dense_cells,
    )


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
    from xlb.utils.adaptive_mesher_warp import WarpAdaptiveMesherOps
    import trimesh

    rng = np.random.default_rng(7)
    field = rng.integers(0, 3, size=(24, 24, 24), dtype=np.int32)
    ops = WarpAdaptiveMesherOps(trimesh.creation.icosphere(subdivisions=1))
    field_wp = wp.array(field, dtype=wp.int32)
    nmin_wp = ops._min_filter_3x3(field_wp)
    nmax_wp = ops._max_filter_3x3(field_wp)
    structure = np.ones((3, 3, 3), dtype=bool)
    np.testing.assert_array_equal(nmin_wp.numpy(), ndimage.minimum_filter(field, footprint=structure, mode="nearest"))
    np.testing.assert_array_equal(nmax_wp.numpy(), ndimage.maximum_filter(field, footprint=structure, mode="nearest"))


@pytest.mark.parametrize("max_dense_cells", [128**3, 4096])
def test_sphere_dense_and_octree_parity(sphere_stl, max_dense_cells):
    kwargs = _mesh_kwargs(sphere_stl, max_dense_cells=max_dense_cells)
    py_data = make_adaptive_surface_mesh(**kwargs, backend="python")
    wp_data = make_adaptive_surface_mesh(**kwargs, backend="warp")
    _assert_level_data_equal(py_data, wp_data)

    gs = _grid_shape_finest(py_data)
    py_stats = validate_level_data(py_data, gs)
    wp_stats = validate_level_data(wp_data, gs)
    assert py_stats["non_overlapping"] and py_stats["fully_covering"] and py_stats["strongly_balanced"]
    assert wp_stats == py_stats


def test_synthetic_box_parity(box_stl):
    kwargs = dict(
        voxel_size=1.5,
        num_levels=3,
        stl_filename=box_stl,
        domain_padding=[1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        expansion_ratio=2.0,
        finest_band_cells=2,
        max_dense_cells=128**3,
    )
    py_data = make_adaptive_surface_mesh(**kwargs, backend="python")
    wp_data = make_adaptive_surface_mesh(**kwargs, backend="warp")
    _assert_level_data_equal(py_data, wp_data)
