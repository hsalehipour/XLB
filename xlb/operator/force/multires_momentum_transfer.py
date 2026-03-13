"""
Multi-resolution momentum-transfer force operator for the Neon backend.
"""

from typing import Any

import warp as wp
import neon

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.force import MomentumTransfer
from xlb.mres_perf_optimization_type import MresPerfOptimizationType


class MultiresMomentumTransfer(MomentumTransfer):
    """Momentum-transfer force computation on a multi-resolution grid.

    Extends :class:`MomentumTransfer` with Neon-specific container code that
    iterates over all grid levels.  The LBM operation sequence (collide-then-
    stream vs. stream-then-collide) is inferred from the performance
    optimization type.

    Parameters
    ----------
    no_slip_bc_instance : BoundaryCondition
        The no-slip BC whose tagged voxels define the force integration
        surface.
    mres_perf_opt : MresPerfOptimizationType
        Multi-resolution performance strategy.
    velocity_set : VelocitySet, optional
    precision_policy : PrecisionPolicy, optional
    compute_backend : ComputeBackend, optional
    """

    def __init__(
        self,
        no_slip_bc_instance,
        mres_perf_opt=MresPerfOptimizationType.NAIVE_COLLIDE_STREAM,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
    ):
        from xlb.operator.force.momentum_transfer import LBMOperationSequence

        if compute_backend in [ComputeBackend.JAX, ComputeBackend.WARP]:
            raise NotImplementedError(f"Operator {self.__class__.__name__} not supported in {compute_backend} backend.")

        # Set the sequence of operations based on the performance optimization type
        if mres_perf_opt == MresPerfOptimizationType.NAIVE_COLLIDE_STREAM:
            operation_sequence = LBMOperationSequence.COLLIDE_THEN_STREAM
        elif mres_perf_opt in (
            MresPerfOptimizationType.FUSION_AT_FINEST,
            MresPerfOptimizationType.FUSION_AT_FINEST_SFV,
            MresPerfOptimizationType.FUSION_AT_FINEST_SFV_ALL,
        ):
            operation_sequence = LBMOperationSequence.STREAM_THEN_COLLIDE
        else:
            raise ValueError(f"Unknown performance optimization type: {mres_perf_opt}")

        # Check if the performance optimization type is compatible with the use of mesh distance
        if operation_sequence != LBMOperationSequence.STREAM_THEN_COLLIDE:
            assert not no_slip_bc_instance.needs_mesh_distance, (
                "Mesh distance is only supported in the MultiresMomentumTransfer operator when the LBM operation sequence is STREAM_THEN_COLLIDE."
            )

        # Print a warning to the user about the boundary voxels
        print(
            "WARNING! make sure boundary voxels are all at the same level and not among the transition regions from one level to another. "
            "Otherwise, the results of force calculation are not correct!\n"
        )

        # Call super
        super().__init__(no_slip_bc_instance, operation_sequence, velocity_set, precision_policy, compute_backend)

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
