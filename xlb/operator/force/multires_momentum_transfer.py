from typing import Any

import warp as wp
import neon

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.force import MomentumTransfer


class MultiresFetchPopulations(Operator):
    """
    This operator is used to get the post-collision and post-streaming populations
    Note that for dense and single resolution simulations in XLB, the order of operations in the stepper is "stream-then-collide".
    Therefore, f_0 represents the post-collision values and post_streaming values of the current time step need to be reconstructed
    by applying the streaming and boundary conditions. These populations are readily available in XLB when using multi-resolution
    grids because the mres stepper relies on "collide-then-stream".
    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
    ):
        if compute_backend in [ComputeBackend.JAX, ComputeBackend.WARP]:
            raise NotImplementedError(f"Operator {self.__class__.__name__} not supported in {compute_backend} backend.")

        # Call the parent constructor
        super().__init__(
            velocity_set,
            precision_policy,
            compute_backend,
        )

    def _construct_neon(self):
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)

        @wp.func
        def functional(
            index: Any,
            f_0: Any,
            f_1: Any,
            _missing_mask: Any,
        ):
            # Get the distribution function
            f_post_collision = _f_vec()
            f_post_stream = _f_vec()
            for l in range(self.velocity_set.q):
                f_post_stream[l] = self.compute_dtype(self.read_field(f_0, index, l))
                f_post_collision[l] = self.compute_dtype(self.read_field(f_1, index, l))
            return f_post_collision, f_post_stream

        return functional, None


class MultiresMomentumTransfer(MomentumTransfer):
    """
    Multiresolution Momentum Transfer operator for computing the force on a multiresolution grid.
    This operator computes uses the same approach as its parent class for computing the forces.
    """

    def __init__(
        self,
        no_slip_bc_instance,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
    ):
        if compute_backend in [ComputeBackend.JAX, ComputeBackend.WARP]:
            raise NotImplementedError(f"Operator {self.__class__.__name__} not supported in {compute_backend} backend.")

        # Call super
        super().__init__(no_slip_bc_instance, velocity_set, precision_policy, compute_backend)

        # Define the **minimal** stepper operator needed for the multi-res momentum transfer
        self.fetcher = MultiresFetchPopulations(
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend,
        )

    def _construct_neon(self):
        # Use the warp functional for the NEON backend
        functional, _ = self._construct_warp()

        @neon.Container.factory(name="MomentumTransfer")
        def container(
            f_0: Any,
            f_1: Any,
            bc_mask: Any,
            missing_mask: Any,
            force: Any,
            level: Any,
        ):
            def container_launcher(loader: neon.Loader):
                loader.set_mres_grid(bc_mask.get_grid(), level)
                bc_mask_pn = loader.get_mres_write_handle(bc_mask)
                missing_mask_pn = loader.get_mres_write_handle(missing_mask)
                f_0_pn = loader.get_mres_write_handle(f_0)
                f_1_pn = loader.get_mres_write_handle(f_1)

                @wp.func
                def container_kernel(index: Any):
                    # apply the functional
                    functional(
                        index,
                        f_0_pn,
                        f_1_pn,
                        bc_mask_pn,
                        missing_mask_pn,
                        force,
                    )

                loader.declare_kernel(container_kernel)

            return container_launcher

        return functional, container

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(
        self,
        f_0,
        f_1,
        bc_mask,
        missing_mask,
        stream=0,
    ):
        # Ensure the force is initialized to zero
        self.force *= self.compute_dtype(0.0)

        # Define the neon functionals needed for this operation
        self.fetcher_functional = self.fetcher.neon_functional

        grid = bc_mask.get_grid()
        for level in range(grid.num_levels):
            # Launch the neon container
            c = self.neon_container(f_0, f_1, bc_mask, missing_mask, self.force, level)
            c.run(stream, container_runtime=neon.Container.ContainerRuntime.neon)
        return self.force.numpy()[0]
