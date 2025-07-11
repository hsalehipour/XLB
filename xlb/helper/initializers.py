import warp as wp
from typing import Any
from xlb.operator import Operator
from xlb.velocity_set import VelocitySet
from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.equilibrium import MultiresQuadraticEquilibrium


def initialize_eq(f, grid, velocity_set, precision_policy, compute_backend, rho=None, u=None):
    if rho is None:
        rho = grid.create_field(cardinality=1, fill_value=1.0, dtype=precision_policy.compute_precision)
    if u is None:
        u = grid.create_field(cardinality=velocity_set.d, fill_value=0.0, dtype=precision_policy.compute_precision)
    equilibrium = QuadraticEquilibrium()

    if compute_backend == ComputeBackend.JAX:
        f = equilibrium(rho, u)
    elif compute_backend == ComputeBackend.WARP:
        f = equilibrium(rho, u, f)
    elif compute_backend == ComputeBackend.NEON:
        f = equilibrium(rho, u, f)
    else:
        raise NotImplementedError(f"Backend {compute_backend} not implemented")

    del rho, u

    return f


def initialize_multires_eq(f, grid, velocity_set, precision_policy, backend, rho, u):
    equilibrium = MultiresQuadraticEquilibrium()
    equilibrium(rho, u, f, stream=0)
    return f


# Defining an initializer for outlet only
class OutletInitializer(Operator):
    def __init__(
        self,
        wind_speed=None,
        grid_shape=None,
        velocity_set: VelocitySet = None,
        precision_policy=None,
        compute_backend=None,
    ):
        self.wind_speed = wind_speed
        self.rho = 1.0
        self.grid_shape = grid_shape
        self.equilibrium = QuadraticEquilibrium(velocity_set=velocity_set, precision_policy=precision_policy, compute_backend=compute_backend)
        super().__init__(velocity_set, precision_policy, compute_backend)

    def _construct_warp(self):
        nx, ny, nz = self.grid_shape
        _q = self.velocity_set.q
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)
        _rho = self.compute_dtype(self.rho)
        _u = _u_vec(self.wind_speed, 0.0, 0.0)
        _w = self.velocity_set.w

        # Construct the warp kernel
        @wp.kernel
        def kernel(f: wp.array4d(dtype=Any)):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # Set the velocity at the outlet (i.e. where i = nx-1)
            if index[0] == nx - 1:
                _feq = self.equilibrium.warp_functional(_rho, _u)
                for l in range(_q):
                    f[l, index[0], index[1], index[2]] = _feq[l]
            else:
                # In the rest of the domain, we assume zero velocity and equilibrium distribution.
                for l in range(_q):
                    f[l, index[0], index[1], index[2]] = _w[l]

        return None, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                f,
            ],
            dim=f.shape[1:],
        )
        return f
