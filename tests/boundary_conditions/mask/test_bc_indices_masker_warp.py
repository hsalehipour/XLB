import pytest
import numpy as np
import xlb
from xlb.compute_backend import ComputeBackend
from xlb import DefaultConfig
from xlb.grid import grid_factory


def init_xlb_env(velocity_set):
    xlb.init(
        default_precision_policy=xlb.PrecisionPolicy.FP32FP32,
        default_backend=ComputeBackend.WARP,
        velocity_set=velocity_set(),
    )


@pytest.mark.parametrize(
    "dim,velocity_set,grid_shape",
    [
        (2, xlb.velocity_set.D2Q9, (50, 50)),
        (2, xlb.velocity_set.D2Q9, (100, 100)),
        (3, xlb.velocity_set.D3Q19, (50, 50, 50)),
        (3, xlb.velocity_set.D3Q19, (100, 100, 100)),
        (3, xlb.velocity_set.D3Q27, (50, 50, 50)),
        (3, xlb.velocity_set.D3Q27, (100, 100, 100)),
    ],
)
def test_indices_masker_warp(dim, velocity_set, grid_shape):
    init_xlb_env(velocity_set)
    my_grid = grid_factory(grid_shape)
    velocity_set = DefaultConfig.velocity_set

    missing_mask = my_grid.create_field(cardinality=velocity_set.q, dtype=xlb.Precision.BOOL)

    boundary_mask = my_grid.create_field(cardinality=1, dtype=xlb.Precision.UINT8)

    indices_boundary_masker = xlb.operator.boundary_masker.IndicesBoundaryMasker()

    # Make indices for boundary conditions (sphere)
    sphere_radius = grid_shape[0] // 4
    nr = grid_shape[0]
    x = np.arange(nr)
    y = np.arange(nr)
    z = np.arange(nr)
    if dim == 2:
        X, Y = np.meshgrid(x, y)
        indices = np.where((X - nr // 2) ** 2 + (Y - nr // 2) ** 2 < sphere_radius**2)
    else:
        X, Y, Z = np.meshgrid(x, y, z)
        indices = np.where((X - nr // 2) ** 2 + (Y - nr // 2) ** 2 + (Z - nr // 2) ** 2 < sphere_radius**2)

    indices = [tuple(indices[i]) for i in range(velocity_set.d)]

    assert len(indices) == dim
    test_bc = xlb.operator.boundary_condition.FullwayBounceBackBC(indices=indices)
    test_bc.id = 5
    boundary_mask, missing_mask = indices_boundary_masker(
        [test_bc],
        boundary_mask,
        missing_mask,
        start_index=(0, 0, 0) if dim == 3 else (0, 0),
    )
    assert missing_mask.dtype == xlb.Precision.BOOL.wp_dtype

    assert boundary_mask.dtype == xlb.Precision.UINT8.wp_dtype

    boundary_mask = boundary_mask.numpy()
    missing_mask = missing_mask.numpy()

    assert boundary_mask.shape == (1,) + grid_shape

    assert missing_mask.shape == (velocity_set.q,) + grid_shape

    if dim == 2:
        assert np.all(boundary_mask[0, indices[0], indices[1]] == test_bc.id)
        # assert that the rest of the boundary_mask is zero
        boundary_mask[0, indices[0], indices[1]] = 0
        assert np.all(boundary_mask == 0)
    if dim == 3:
        assert np.all(boundary_mask[0, indices[0], indices[1], indices[2]] == test_bc.id)
        # assert that the rest of the boundary_mask is zero
        boundary_mask[0, indices[0], indices[1], indices[2]] = 0
        assert np.all(boundary_mask == 0)


if __name__ == "__main__":
    pytest.main()