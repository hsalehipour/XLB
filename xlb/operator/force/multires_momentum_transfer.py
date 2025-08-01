from typing import Any

import warp as wp
import neon

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.force import MomentumTransfer


class MultiresMomentumTransfer(MomentumTransfer):
    """
    Multiresolution Momentum Transfer operator for computing the force on a multiresolution grid.
    This operator computes uses the same approach as its parent class for computing the forces.
    """

    def __init__(
        self,
        no_slip_bc_instance,
        collision_type="BGK",
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
    ):
        # Call super
        super().__init__(no_slip_bc_instance, velocity_set, precision_policy, compute_backend)
        if self.compute_backend in [ComputeBackend.JAX, ComputeBackend.WARP]:
            raise NotImplementedError(f"Operator {self.__class__.__name__} not supported in {self.compute_backend} backend.")

        # TODO! The current implementation does not support encoding and decoding of mesh distance in f_1!
        assert not self.no_slip_bc_instance.needs_mesh_distance, "Mesh distance is not supported for Force Calculation!"

        # Print a warning to the user about the boundary voxels
        print(
            "WARNING! make sure boundary voxels are all at the same level and not among the transition regions from one level to another. "
            "Otherwise, the results of force calculation are not correct!\n"
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

                # Important: Note the swap to the order of f_0 and f_1 in the functional call.
                # This is because the multiresolution simulation first performs collision and then streaming and hence
                # f_0 refers to the post-streaming distribution function and f_1 refers to the post-collision distribution function.
                # This is in contrast to our dense implementations (all backends) where streaming occurs first and is followed by
                # collision which makes f_0 post-collision and f_1 post-streaming.
                # So as a workaround, we can simply swap f_0 and f_1 in the functional call.

                @wp.func
                def container_kernel(index: Any):
                    # apply the functional
                    functional(
                        index,
                        f_1_pn,
                        f_0_pn,
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
        self.stepper_functional = self.stepper.neon_functional

        grid = bc_mask.get_grid()
        for level in range(grid.num_levels):
            # Launch the neon container
            c = self.neon_container(f_0, f_1, bc_mask, missing_mask, self.force, level)
            c.run(stream, container_runtime=neon.Container.ContainerRuntime.neon)
        return self.force.numpy()[0]
