"""Tests for multires boundary index selection with non-zero level origins."""

import numpy as np
import pytest

from xlb.utils.mesher import _normalize_level_data


def _make_level_data_with_origin(origin_finest, coords):
    """Build minimal level_data with a negative finest origin (dyadic padding case)."""
    origin_phys = np.asarray(origin_finest, dtype=float)
    raw = [(coords, 1.0, origin_phys, 0)]
    return list(_normalize_level_data(raw, 1.0))


def test_boundary_indices_min_face_with_negative_origin():
    """Min faces must select local=-origin so finest index is 0 after masker conversion."""
    from xlb.grid.multires_grid import NeonMultiresGrid

    origin_finest = np.array([-4, -4, 0], dtype=int)
    # Domain left face at local x=4 (finest x=0); include a neighbour one cell inward.
    coords = np.array(
        [
            [4, 4, 2],  # left + front corner on domain min faces
            [5, 5, 2],  # interior neighbour
            [4, 6, 2],
        ],
        dtype=np.int32,
    )
    level_data = _make_level_data_with_origin(origin_finest, coords)

    class _GridStub:
        shape = (16, 16, 8)

        def level_to_shape(self, level):
            return tuple(x // (2**level) for x in self.shape)

    grid = NeonMultiresGrid.__new__(NeonMultiresGrid)
    grid.shape = _GridStub.shape
    grid.velocity_set = type("VS", (), {"d": 3})()
    grid.level_to_shape = _GridStub().level_to_shape

    left = grid.boundary_indices_across_levels(level_data, box_side="left", remove_edges=False)
    front = grid.boundary_indices_across_levels(level_data, box_side="front", remove_edges=False)

    assert left[0], "expected left-face voxels at finest level"
    assert front[0], "expected front-face voxels at finest level"

    left_arr = np.asarray(left[0])
    front_arr = np.asarray(front[0])

    # Local indices on the domain min faces (not lattice index 0).
    assert np.all(left_arr[0] == 4)
    assert np.all(front_arr[1] == 4)
    assert not np.any(left_arr[0] == 5), "inward neighbour must not be tagged as left BC"

    # Finest indices after masker conversion: (local + origin) << level.
    finest_left_x = (left_arr[0] + origin_finest[0]) * 1
    finest_front_y = (front_arr[1] + origin_finest[1]) * 1
    assert np.all(finest_left_x == 0)
    assert np.all(finest_front_y == 0)


def test_min_face_bound_with_shifted_domain_origin():
    """With negative origin, min face selects local=-origin (first addressable cell)."""
    from xlb.grid.multires_grid import NeonMultiresGrid

    # Origin is (-32, -32, 0).  First addressable local = -origin = 32.
    origin_finest = np.array([-32, -32, 0], dtype=int)
    coords = np.array(
        [
            [32, 32, 4],  # left + front on first addressable face
            [33, 33, 4],  # one cell inward
            [32, 33, 4],
        ],
        dtype=np.int32,
    )
    level_data = _make_level_data_with_origin(origin_finest, coords)

    grid = NeonMultiresGrid.__new__(NeonMultiresGrid)
    grid.shape = (1856, 960, 128)
    grid.velocity_set = type("VS", (), {"d": 3})()

    def level_to_shape(level):
        return tuple(x // (2**level) for x in grid.shape)

    grid.level_to_shape = level_to_shape

    left = np.asarray(grid.boundary_indices_across_levels(level_data, box_side="left", remove_edges=False)[0])
    front = np.asarray(grid.boundary_indices_across_levels(level_data, box_side="front", remove_edges=False)[0])

    assert np.all(left[0] == 32), "left face must be at local 32 = -origin"
    assert np.all(front[1] == 32), "front face must be at local 32 = -origin"

    finest_left_x = (left[0] + origin_finest[0]) * 1
    finest_front_y = (front[1] + origin_finest[1]) * 1
    assert np.all(finest_left_x == 0), "virtual coord must be 0 (first non-negative)"
    assert np.all(finest_front_y == 0)


def test_remove_edges_uses_domain_bounds_not_grid_shape():
    """remove_edges must strip face perimeter using addressable domain extent."""
    from xlb.grid.multires_grid import NeonMultiresGrid

    # Origin [0, -2, 0], stride 1.  First addressable local_y = -(-2) = 2 (virtual 0).
    # Domain_max y = 16 (from max coords).  Faces at local_x = 0.
    origin_finest = np.array([0, -2, 0], dtype=int)
    coords = []
    # locals: x=0, y in [2,3,4,5], z in [0,1,2,3]  → virtual y in [0,1,2,3]
    for y in range(2, 6):
        for z in range(4):
            coords.append([0, y, z])
    coords = np.array(coords, dtype=np.int32)
    level_data = _make_level_data_with_origin(origin_finest, coords)

    grid = NeonMultiresGrid.__new__(NeonMultiresGrid)
    grid.shape = (32, 32, 32)
    grid.velocity_set = type("VS", (), {"d": 3})()
    grid.level_to_shape = lambda l: tuple(x // (2**l) for x in grid.shape)

    left = np.asarray(grid.boundary_indices_across_levels(level_data, box_side="left", remove_edges=True)[0])
    assert left.size, "expected left-face indices after edge removal"
    o = np.asarray(level_data[0][2], dtype=np.int64).reshape(3, 1)
    finest_y = (left[1] + o[1]) * 1
    # Edges (min-y and max-y of the face) should be removed
    domain_min_y = 0  # first addressable
    domain_max_y = (5 + (-2)) * 1  # = 3
    assert np.all(finest_y > domain_min_y), f"bottom edge y not removed: {finest_y}"
    assert np.all(finest_y < domain_max_y), f"top edge y not removed: {finest_y}"


def test_virtual_finest_to_neon_global_negative_min_x():
    """neon_global == virtual (identity) for all addressable cells."""
    from xlb.grid.multires_grid import NeonMultiresGrid

    domain_min = np.array([-8, 0, 0], dtype=np.int64)
    domain_max = np.array([592, 296, 40], dtype=np.int64)
    num_levels = 4
    # Only test addressable cells (virtual >= 0)
    virtual = np.array(
        [
            [0, 4, 2],
            [8, 4, 2],
            [16, 4, 2],
            [128, 4, 2],
            [584, 4, 2],
            [592, 4, 2],
        ],
        dtype=np.int64,
    ).T
    neon = NeonMultiresGrid.virtual_finest_to_neon_global(virtual, domain_min, domain_max, level=3, num_levels=num_levels)
    # Identity: neon == virtual
    assert np.all(neon == virtual.astype(np.int32))
    roundtrip = NeonMultiresGrid.neon_global_to_virtual_finest(neon, domain_min, domain_max, level=3, num_levels=num_levels)
    assert np.all(roundtrip == virtual.astype(np.int32))


def test_virtual_finest_to_neon_l3_row_matches_embedding():
    """Coarsest-level left-face starts at local=-origin (first addressable)."""
    from xlb.grid.multires_grid import NeonMultiresGrid

    domain_min = np.array([-8, 0, 0], dtype=np.int64)
    domain_max = np.array([592, 296, 40], dtype=np.int64)
    origin_l3 = -1
    stride = 8
    # First addressable local is -origin = 1 (virtual 0)
    for local_x in range(1, 8):
        virtual_x = (local_x + origin_l3) * stride
        assert virtual_x >= 0
        virtual = np.array([[virtual_x, 0, 0]], dtype=np.int64).T
        neon = NeonMultiresGrid.virtual_finest_to_neon_global(virtual, domain_min, domain_max, level=3, num_levels=4)
        assert neon[0, 0] == virtual_x


def test_boundary_indices_cuboid_origin_zero_unchanged():
    """With zero origin, min faces remain at local index 0."""
    from xlb.grid.multires_grid import NeonMultiresGrid

    coords = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1]], dtype=np.int32)
    level_data = _make_level_data_with_origin(np.zeros(3, dtype=int), coords)

    grid = NeonMultiresGrid.__new__(NeonMultiresGrid)
    grid.shape = (8, 8, 8)
    grid.velocity_set = type("VS", (), {"d": 3})()

    def level_to_shape(level):
        return tuple(x // (2**level) for x in grid.shape)

    grid.level_to_shape = level_to_shape

    left = np.asarray(grid.boundary_indices_across_levels(level_data, box_side="left")[0])
    front = np.asarray(grid.boundary_indices_across_levels(level_data, box_side="front")[0])

    assert np.all(left[0] == 0)
    assert np.all(front[1] == 0)
