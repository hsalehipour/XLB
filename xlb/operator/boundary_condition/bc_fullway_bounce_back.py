"""
Base class for boundary conditions in a LBM simulation.
"""

import jax.numpy as jnp
from jax import jit
import jax.lax as lax
from functools import partial
import warp as wp
from typing import Any

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator import Operator
from xlb.operator.boundary_condition.boundary_condition import (
    BoundaryCondition,
    ImplementationStep,
)
from xlb.operator.boundary_condition.boundary_condition_registry import (
    boundary_condition_registry,
)


class FullwayBounceBackBC(BoundaryCondition):
    """
    Full Bounce-back boundary condition for a lattice Boltzmann method simulation.
    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        indices=None,
        mesh_vertices=None,
    ):
        super().__init__(
            ImplementationStep.COLLISION,
            velocity_set,
            precision_policy,
            compute_backend,
            indices,
            mesh_vertices,
        )

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f_pre, f_post, bc_mask, missing_mask):
        boundary = bc_mask == self.id
        new_shape = (self.velocity_set.q,) + boundary.shape[1:]
        boundary = lax.broadcast_in_dim(boundary, new_shape, tuple(range(self.velocity_set.d + 1)))
        return jnp.where(boundary, f_pre[self.velocity_set.opp_indices, ...], f_post)

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _opp_indices = self.velocity_set.opp_indices
        _q = wp.constant(self.velocity_set.q)
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)

        # Construct the functional for this BC
        @wp.func
        def functional(
            index: Any,
            timestep: Any,
            missing_mask: Any,
            f_0: Any,
            f_1: Any,
            f_pre: Any,
            f_post: Any,
        ):
            fliped_f = _f_vec()
            for l in range(_q):
                fliped_f[l] = f_pre[_opp_indices[l]]
            return fliped_f

        @wp.kernel
        def kernel2d(
            f_pre: wp.array3d(dtype=Any),
            f_post: wp.array3d(dtype=Any),
            bc_mask: wp.array3d(dtype=wp.uint8),
            missing_mask: wp.array3d(dtype=wp.bool),
        ):  # Get the global index
            i, j = wp.tid()
            index = wp.vec2i(i, j)

            # read tid data
            _f_pre, _f_post, _boundary_id, _missing_mask = self._get_thread_data_2d(f_pre, f_post, bc_mask, missing_mask, index)

            # Check if the boundary is active
            if _boundary_id == wp.uint8(FullwayBounceBackBC.id):
                timestep = 0
                _f = functional(index, timestep, _missing_mask, f_pre, f_post, _f_pre, _f_post)
            else:
                _f = _f_post

            # Write the result to the output
            for l in range(self.velocity_set.q):
                f_post[l, index[0], index[1]] = self.store_dtype(_f[l])

        # Construct the warp kernel
        @wp.kernel
        def kernel3d(
            f_pre: wp.array4d(dtype=Any),
            f_post: wp.array4d(dtype=Any),
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.bool),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # read tid data
            _f_pre, _f_post, _boundary_id, _missing_mask = self._get_thread_data_3d(f_pre, f_post, bc_mask, missing_mask, index)

            # Check if the boundary is active
            if _boundary_id == wp.uint8(FullwayBounceBackBC.id):
                timestep = 0
                _f = functional(index, timestep, _missing_mask, f_pre, f_post, _f_pre, _f_post)
            else:
                _f = _f_post

            # Write the result to the output
            for l in range(self.velocity_set.q):
                f_post[l, index[0], index[1], index[2]] = self.store_dtype(_f[l])

        kernel = kernel3d if self.velocity_set.d == 3 else kernel2d

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_pre, f_post, bc_mask, missing_mask):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[f_pre, f_post, bc_mask, missing_mask],
            dim=f_pre.shape[1:],
        )
        return f_post
