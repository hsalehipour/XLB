"""Tests for _filter_phantom_cells and _pad_domain_for_dyadic."""

import numpy as np
import pytest

from xlb.utils.adaptive_mesher import _filter_phantom_cells, _pad_domain_for_dyadic


class TestFilterPhantomCells:
    """Unit tests for _filter_phantom_cells."""

    def _make_level_data(self, coords_per_level, origins, voxel_sizes=None):
        """Helper to build minimal level_data tuples."""
        num_levels = len(coords_per_level)
        if voxel_sizes is None:
            voxel_sizes = [1.0 * (2**lvl) for lvl in range(num_levels)]
        entries = []
        for lvl in range(num_levels):
            pattern = np.array(coords_per_level[lvl], dtype=np.int32)
            origin = np.array(origins[lvl], dtype=np.float64)
            entries.append((pattern, voxel_sizes[lvl], origin, num_levels - 1 - lvl))
        return entries

    def test_removes_negative_virtual_coords(self):
        """Cells with negative virtual finest coords are removed."""
        coords_l0 = [[0, 0, 0], [1, 0, 0], [-1, 0, 0]]
        coords_l1 = [[0, 0, 0], [2, 2, 2]]
        level_data = self._make_level_data(
            [coords_l0, coords_l1],
            origins=[[0, 0, 0], [-1, -1, 0]],
        )

        result = _filter_phantom_cells(level_data)

        filtered_l0 = result[0][0]
        assert filtered_l0.shape[0] == 2
        assert np.all(filtered_l0 >= 0)

        filtered_l1 = result[1][0]
        origin_l1 = np.array([-1, -1, 0], dtype=np.int64)
        stride_l1 = 2
        for i in range(filtered_l1.shape[0]):
            virtual = (filtered_l1[i].astype(np.int64) + origin_l1) * stride_l1
            assert np.all(virtual >= 0), f"Row {i} has negative virtual: {virtual}"

    def test_all_positive_unchanged(self):
        """When all virtual coords are >= 0, nothing is filtered."""
        coords = [[0, 0, 0], [1, 1, 1], [2, 3, 4]]
        level_data = self._make_level_data([coords], origins=[[0, 0, 0]])

        result = _filter_phantom_cells(level_data)

        np.testing.assert_array_equal(result[0][0], np.array(coords, dtype=np.int32))

    def test_preserves_grid_shape_finest(self):
        """The grid_shape_finest attribute is preserved through filtering."""
        coords = [[0, 0, 0], [1, 1, 1]]
        level_data = self._make_level_data([coords], origins=[[0, 0, 0]])

        class LevelDataList(list):
            pass

        ld = LevelDataList(level_data)
        ld.grid_shape_finest = (16, 16, 16)

        result = _filter_phantom_cells(ld)
        assert getattr(result, "grid_shape_finest", None) == (16, 16, 16)

    def test_handles_dense_pattern_unchanged(self):
        """Dense 3D masks pass through unmodified."""
        mask = np.ones((4, 4, 4), dtype=np.int32)
        level_data = [(mask, 1.0, np.zeros(3), 0)]

        result = _filter_phantom_cells(level_data)
        np.testing.assert_array_equal(result[0][0], mask)

    def test_empty_coords_handled(self):
        """Empty coordinate arrays don't cause errors."""
        coords = np.empty((0, 3), dtype=np.int32)
        level_data = [(coords, 1.0, np.array([-5, -5, -5], dtype=np.float64), 0)]

        result = _filter_phantom_cells(level_data)
        assert result[0][0].shape == (0, 3)


class TestPadDomainForDyadic:
    """Unit tests for _pad_domain_for_dyadic."""

    def test_already_aligned_no_change(self):
        """Grid already divisible by 2^num_levels is unchanged."""
        shape = (32, 16, 64)
        origin = np.array([0.0, 0.0, 0.0])
        num_levels = 4
        padding = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        new_shape, new_origin = _pad_domain_for_dyadic(shape, origin, 1.0, num_levels, padding)

        assert new_shape == shape
        np.testing.assert_array_equal(new_origin, origin)

    def test_shape_becomes_aligned(self):
        """Result shape is divisible by 2^num_levels on all axes."""
        shape = (30, 17, 63)
        origin = np.array([0.0, 0.0, 0.0])
        num_levels = 3
        padding = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        new_shape, _ = _pad_domain_for_dyadic(shape, origin, 1.0, num_levels, padding)

        align = 2**num_levels
        assert new_shape[0] % align == 0
        assert new_shape[1] % align == 0
        assert new_shape[2] % align == 0

    def test_shape_grows_not_shrinks(self):
        """Aligned shape is always >= original on each axis."""
        shape = (30, 17, 63)
        origin = np.array([0.0, 0.0, 0.0])
        num_levels = 3
        padding = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        new_shape, _ = _pad_domain_for_dyadic(shape, origin, 1.0, num_levels, padding)

        assert new_shape[0] >= shape[0]
        assert new_shape[1] >= shape[1]
        assert new_shape[2] >= shape[2]

    def test_asymmetric_padding_preserves_offset(self):
        """Asymmetric padding puts more growth on the heavier side."""
        shape = (31, 32, 32)
        origin = np.array([0.0, 0.0, 0.0])
        num_levels = 3
        padding = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        _, new_origin = _pad_domain_for_dyadic(shape, origin, 1.0, num_levels, padding)

        assert new_origin[0] >= origin[0] - 1e-10

    def test_origin_shift_is_quantised(self):
        """Low-side growth is quantised to 2^(num_levels-1) cells."""
        shape = (29, 32, 32)
        origin = np.array([10.0, 0.0, 0.0])
        voxel_size = 2.0
        num_levels = 3
        padding = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        _, new_origin = _pad_domain_for_dyadic(shape, origin, voxel_size, num_levels, padding)

        factor = 2 ** (num_levels - 1)
        shift_cells = round((origin[0] - new_origin[0]) / voxel_size)
        assert shift_cells % factor == 0
