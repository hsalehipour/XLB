import warp as wp
from typing import Any
from xlb.operator import Operator
from xlb.velocity_set import VelocitySet
from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.equilibrium import MultiresQuadraticEquilibrium
import neon


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
        outlet_bc_id: int = None,
        wind_vector=None,
        velocity_set: VelocitySet = None,
        precision_policy=None,
        compute_backend=None,
    ):
        assert outlet_bc_id is not None, "Outlet BC ID must be provided."
        self.outlet_bc_id = outlet_bc_id
        self.wind_vector = wind_vector
        self.rho = 1.0
        self.equilibrium = QuadraticEquilibrium(
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=ComputeBackend.WARP,
        )
        super().__init__(velocity_set, precision_policy, compute_backend)

    def _construct_warp(self):
        _q = self.velocity_set.q
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)
        _u = _u_vec(self.wind_vector[0], self.wind_vector[1], self.wind_vector[2])
        _rho = self.compute_dtype(self.rho)
        _w = self.velocity_set.w
        outlet_bc_id = self.outlet_bc_id

        @wp.func
        def functional(index: Any, bc_mask: Any, f_field: Any):
            # Check if the index corresponds to the outlet
            if self.read_field(bc_mask, index, 0) == outlet_bc_id:
                _feq = self.equilibrium.warp_functional(_rho, _u)
                for l in range(_q):
                    self.write_field(f_field, index, l, _feq[l])
            else:
                # In the rest of the domain, we assume zero velocity and equilibrium distribution.
                for l in range(_q):
                    self.write_field(f_field, index, l, _w[l])

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            bc_mask: wp.array4d(dtype=wp.uint8),
            f_field: wp.array4d(dtype=Any),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # Set the velocity at the outlet (i.e. where i = nx-1)
            functional(index, bc_mask, f_field)

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, bc_mask, f_field):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[bc_mask, f_field],
            dim=f_field.shape[1:],
        )
        return f_field

    def _construct_neon(self):
        # Use the warp functional for the NEON backend
        functional, _ = self._construct_warp()

        @neon.Container.factory(name="OutletInitializer")
        def container(
            bc_mask: Any,
            f_field: Any,
        ):
            def launcher(loader: neon.Loader):
                loader.set_grid(f_field.get_grid())
                f_field_pn = loader.get_write_handle(f_field)
                bc_mask_pn = loader.get_read_handle(bc_mask)

                @wp.func
                def kernel(index: Any):
                    # apply the functional
                    functional(index, bc_mask_pn, f_field_pn)

                loader.declare_kernel(kernel)

            return launcher

        return _, container

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, bc_mask, f_field, stream=0):
        # Launch the neon container
        c = self.neon_container(bc_mask, f_field)
        c.run(stream, container_runtime=neon.Container.ContainerRuntime.neon)
        return f_field


# Defining an initializer for outlet only
class MultiresOutletInitializer(OutletInitializer):
    def __init__(
        self,
        outlet_bc_id: int = None,
        wind_vector=None,
        velocity_set: VelocitySet = None,
        precision_policy=None,
        compute_backend=None,
    ):
        super().__init__(outlet_bc_id, wind_vector, velocity_set, precision_policy, compute_backend)

    def _construct_neon(self):
        # Use the warp functional for the NEON backend
        functional, _ = self._construct_warp()

        @neon.Container.factory(name="MultiresOutletInitializer")
        def container(
            bc_mask: Any,
            f_field: Any,
            level: Any,
        ):
            def launcher(loader: neon.Loader):
                loader.set_mres_grid(f_field.get_grid(), level)
                f_field_pn = loader.get_mres_write_handle(f_field)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask)

                @wp.func
                def kernel(index: Any):
                    # apply the functional
                    functional(index, bc_mask_pn, f_field_pn)

                loader.declare_kernel(kernel)

            return launcher

        return _, container

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, bc_mask, f_field, stream=0):
        grid = bc_mask.get_grid()
        for level in range(grid.num_levels):
            # Launch the neon container
            c = self.neon_container(bc_mask, f_field, level)
            c.run(stream, container_runtime=neon.Container.ContainerRuntime.neon)
        return f_field
