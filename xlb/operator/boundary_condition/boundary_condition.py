"""
Base class for boundary conditions in a LBM simulation.
"""

from enum import Enum, auto
import warp as wp
from typing import Any
from jax import jit
from functools import partial
import numpy as np

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb import DefaultConfig
from xlb.operator.boundary_condition.boundary_condition_registry import boundary_condition_registry
from xlb.operator.boundary_condition import HelperFunctionsBC
from xlb.operator.boundary_masker.mesh_voxelization_method import MeshVoxelizationMethod
import neon


# Enum for implementation step
class ImplementationStep(Enum):
    COLLISION = auto()
    STREAMING = auto()


class BoundaryCondition(Operator):
    """
    Base class for boundary conditions in a LBM simulation.
    """

    def __init__(
        self,
        implementation_step: ImplementationStep,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        indices=None,
        mesh_vertices=None,
        voxelization_method: MeshVoxelizationMethod = None,
    ):
        self.id = boundary_condition_registry.register_boundary_condition(self.__class__.__name__ + "_" + str(hash(self)))
        velocity_set = velocity_set or DefaultConfig.velocity_set
        precision_policy = precision_policy or DefaultConfig.default_precision_policy
        compute_backend = compute_backend or DefaultConfig.default_backend

        super().__init__(velocity_set, precision_policy, compute_backend)

        # Set the BC indices
        self.indices = indices
        self.mesh_vertices = mesh_vertices

        # Set the implementation step
        self.implementation_step = implementation_step

        # A flag to indicate whether bc indices need to be padded in both normal directions to identify missing directions
        # when inside/outside of the geoemtry is not known
        self.needs_padding = False

        # A flag for BCs that need normalized distance between the grid and a mesh (to be set to True if applicable inside each BC)
        self.needs_mesh_distance = False

        # A flag for BCs that need auxiliary data initialization before stepper
        self.needs_aux_init = False

        # A flag to track if the BC is initialized with auxiliary data
        self.is_initialized_with_aux_data = False

        # Number of auxiliary data needed for the BC (for prescribed values)
        self.num_of_aux_data = 0

        # A flag for BCs that need auxiliary data recovery after streaming
        self.needs_aux_recovery = False

        # Voxelization method. For BC's specified on a mesh, the user can specify the voxelization scheme.
        # Currently we support three methods based on (a) aabb method (b) ray casting and (c) winding number.
        self.voxelization_method = voxelization_method

        # Construct a default warp functional for assembling auxiliary data if needed
        if self.compute_backend in [ComputeBackend.WARP, ComputeBackend.NEON]:

            @wp.func
            def assemble_auxiliary_data(
                index: Any,
                timestep: Any,
                missing_mask: Any,
                f_0: Any,
                f_1: Any,
                f_pre: Any,
                f_post: Any,
                level: Any = 0,
            ):
                return f_post

            self.assemble_auxiliary_data = assemble_auxiliary_data

    def pad_indices(self):
        """
        This method pads the indices to ensure that the boundary condition can be applied correctly.
        It is used to find missing directions in indices_boundary_masker when the BC is imposed on a
        geometry that is in the domain interior.
        """
        _d = self.velocity_set.d
        bc_indices = np.array(self.indices)
        lattice_velocity_np = self.velocity_set._c
        if self.needs_padding:
            bc_indices_padded = bc_indices[:, :, None] + lattice_velocity_np[:, None, :]
            return np.unique(bc_indices_padded.reshape(_d, -1), axis=1)
        else:
            return bc_indices

    @partial(jit, static_argnums=(0,), inline=True)
    def assemble_auxiliary_data(self, f_pre, f_post, bc_mask, missing_mask):
        """
        A placeholder function for prepare the auxiliary distribution functions for the boundary condition.
        currently being called after collision only.
        """
        return f_post

    def _construct_kernel(self, functional):
        """
        Constructs the warp kernel for the boundary condition.
        The functional is specific to each boundary condition and should be passed as an argument.
        """
        bc_helper = HelperFunctionsBC(velocity_set=self.velocity_set, precision_policy=self.precision_policy, compute_backend=self.compute_backend)
        _id = wp.uint8(self.id)

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            f_pre: wp.array4d(dtype=Any),
            f_post: wp.array4d(dtype=Any),
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.uint8),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # read tid data
            _f_pre, _f_post, _boundary_id, _missing_mask = bc_helper.get_bc_thread_data(f_pre, f_post, bc_mask, missing_mask, index)

            # Apply the boundary condition
            if _boundary_id == _id:
                timestep = 0
                _f = functional(index, timestep, _missing_mask, f_pre, f_post, _f_pre, _f_post)
            else:
                _f = _f_post

            # Write the result
            for l in range(self.velocity_set.q):
                f_post[l, index[0], index[1], index[2]] = self.store_dtype(_f[l])

        return kernel
