"""
Base class for boundary conditions in a LBM simulation.
"""

import jax.numpy as jnp
from jax import jit
import jax.lax as lax
from functools import partial
import warp as wp
from typing import Any
from collections import Counter
import numpy as np

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.macroscopic.zero_moment import ZeroMoment
from xlb.operator.macroscopic.second_moment import SecondMoment as MomentumFlux
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.boundary_condition.boundary_condition import (
    ImplementationStep,
    BoundaryCondition,
)
from xlb.operator.boundary_condition.boundary_condition_registry import (
    boundary_condition_registry,
)


class HybridBC(BoundaryCondition):
    """
    The hybrid BC methods in this boundary condition have been originally developed by H. Salehipour and are inspired from
    various previous publications, in particular [1]. The reformulations are aimed to provide local formulations that are
    computationally efficient and numerically stable at exessively large Reynolds numbers.

    [1] Dorschner, B., Chikatamarla, S. S., Bösch, F., & Karlin, I. V. (2015). Grad's approximation for moving and
        stationary walls in entropic lattice Boltzmann simulations. Journal of Computational Physics, 295, 340-354.
    """

    def __init__(
        self,
        bc_method,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        indices=None,
        mesh_vertices=None,
    ):
        assert bc_method in [
            "bounceback_regularized",
            "bounceback_grads",
            "dorschner_localized",
        ], f"type = {bc_method} not supported! Use 'bounceback_regularized', 'bounceback_grads' or 'dorschner_localized'."
        self.bc_method = bc_method

        # TODO: the input velocity must be suitably stored elesewhere when mesh is moving.
        self.u = (0, 0, 0)

        # Call the parent constructor
        super().__init__(
            ImplementationStep.STREAMING,
            velocity_set,
            precision_policy,
            compute_backend,
            indices,
            mesh_vertices,
        )

        # Instantiate the operator for computing macroscopic values
        self.macroscopic = Macroscopic()
        self.zero_moment = ZeroMoment()
        self.equilibrium = QuadraticEquilibrium()
        self.momentum_flux = MomentumFlux()

        if self.bc_method == "dorschner_localized":
            # This BC needs implicit distance to the mesh
            self.needs_mesh_distance = True

            # This BC needs auxilary data recovery after streaming
            self.needs_aux_recovery = True

        # If this BC is defined using indices, it would need padding in order to find missing directions
        # when imposed on a geometry that is in the domain interior
        if self.mesh_vertices is None:
            assert self.indices is not None
            self.needs_padding = True

        # Raise error if used for 2d examples:
        if self.velocity_set.d == 2:
            raise NotImplementedError("This BC is not implemented in 2D!")

        # if indices is not None:
        #     # this BC would be limited to stationary boundaries
        #     # assert mesh_vertices is None
        # if mesh_vertices is not None:
        #     # this BC would be applicable for stationary and moving boundaries
        #     assert indices is None
        #     if mesh_velocity_function is not None:
        #         # mesh is moving and/or deforming

        assert self.compute_backend == ComputeBackend.WARP, "This BC is currently only implemented with the Warp backend!"

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f_pre, f_post, bc_mask, missing_mask):
        # TODO
        raise NotImplementedError(f"Operation {self.__class__.__name} not implemented in JAX!")
        return

    def _construct_warp(self):
        # load helper functions
        from xlb.helper.bc_warp_functions import regularize_fpop, grads_approximate_fpop, interpolated_bounceback, moving_wall_fpop_correction

        # Set local variables and constants
        _c = self.velocity_set.c
        _q = self.velocity_set.q
        _d = self.velocity_set.d
        _w = self.velocity_set.w
        _opp_indices = self.velocity_set.opp_indices
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)
        _u_wall = _u_vec(self.u[0], self.u[1], self.u[2]) if _d == 3 else _u_vec(self.u[0], self.u[1])

        # Construct the functionals for this BC
        @wp.func
        def hybrid_bounceback_regularized(
            index: Any,
            timestep: Any,
            missing_mask: Any,
            f_0: Any,
            f_1: Any,
            f_pre: Any,
            f_post: Any,
        ):
            # Using regularization technique [1] to represent fpop using macroscopic values derived from interpolated bounceback scheme of [2].
            # missing data in lattice Boltzmann.
            # [1] Latt, J., Chopard, B., Malaspinas, O., Deville, M., Michler, A., 2008. Straight velocity
            #     boundaries in the lattice Boltzmann method. Physical Review E 77, 056703.
            # [2] Yu, D., Mei, R., Shyy, W., 2003. A uniﬁed boundary treatment in lattice boltzmann method,
            #     in: 41st aerospace sciences meeting and exhibit, p. 953.

            # Apply interpolated bounceback first to find missing populations at the boundary
            f_post = interpolated_bounceback(missing_mask, f_0, f_1, f_pre, f_post)

            # Compute density, velocity using all f_post-streaming values
            rho, u = self.macroscopic.warp_functional(f_post)

            # Regularize the resulting populations
            feq = self.equilibrium.warp_functional(rho, u)
            f_post = regularize_fpop(f_post, feq)
            return f_post

        @wp.func
        def hybrid_bounceback_grads(
            index: Any,
            timestep: Any,
            missing_mask: Any,
            f_0: Any,
            f_1: Any,
            f_pre: Any,
            f_post: Any,
        ):
            # Using Grad's approximation [1] to represent fpop using macroscopic values derived from interpolated bounceback scheme of [2].
            # missing data in lattice Boltzmann.
            # [1] Dorschner, B., Chikatamarla, S. S., Bösch, F., & Karlin, I. V. (2015). Grad's approximation for moving and
            #    stationary walls in entropic lattice Boltzmann simulations. Journal of Computational Physics, 295, 340-354.
            # [2] Yu, D., Mei, R., Shyy, W., 2003. A uniﬁed boundary treatment in lattice boltzmann method,
            #     in: 41st aerospace sciences meeting and exhibit, p. 953.

            # Apply interpolated bounceback first to find missing populations at the boundary
            f_post = interpolated_bounceback(missing_mask, f_0, f_1, f_pre, f_post)

            # Compute density, velocity using all f_post-streaming values
            rho, u = self.macroscopic.warp_functional(f_post)

            # Compute Grad's appriximation using full equation as in Eq (10) of Dorschner et al.
            f_post = grads_approximate_fpop(rho, u, f_post)
            return f_post

        # Construct the functionals for this BC
        @wp.func
        def dorschner_localized(
            index: Any,
            timestep: Any,
            missing_mask: Any,
            f_0: Any,
            f_1: Any,
            f_pre: Any,
            f_post: Any,
        ):
            # A localized reformulation of [1] derived by H. Salehipour.
            # [1] Dorschner, B., Chikatamarla, S. S., Bösch, F., & Karlin, I. V. (2015). Grad's approximation for moving and
            #     stationary walls in entropic lattice Boltzmann simulations. Journal of Computational Physics, 295, 340-354.
            # NOTE: this BC has been reformulated to become entirely local and so has differences compared to the original paper.
            #       Here we use the current time-step populations (f_pre = f_post_collision and f_post = f_post_streaming).
            #
            # here I need to compute all terms in Eq (10)
            # Strategy:
            # 1) "weights" should have been stored somewhere to be used here.
            # 2) Given "weights", "u_w" (input to the BC) and "u_f" (computed from f_aux), compute "u_target" as per Eq (14)
            #    NOTE: in the original paper "u_target" is associated with the previous time step not current time.
            # 3) Given "weights" use differentiable interpolated BB to find f_missing as I had before:
            # fmissing = ((1. - weights) * f_poststreaming_iknown + weights * (f_postcollision_imissing + f_postcollision_iknown)) / (1.0 + weights)
            # 4) Add contribution due to u_w to f_missing as is usual in regular Bouzidi BC (ie. -6.0 * self.lattice.w * jnp.dot(self.vel, c)
            # 5) Compute rho_target = \sum(f_ibb) based on these values
            # 6) Compute feq using feq = self.equilibrium(rho_target, u_target)
            # 7) Compute Pi_neq and Pi_eq using all f_post-streaming values as per:
            #       Pi_neq = self.momentum_flux(fneq) and Pi_eq = self.momentum_flux(feq)
            # 8) Compute Grad's appriximation using full equation as in Eq (10)
            #    NOTE: this is very similar to the regularization procedure.

            _f_nbr = _f_vec()
            zero = self.compute_dtype(0.0)
            one = self.compute_dtype(1.0)
            u_target = _u_vec(zero, zero, zero)
            num_missing = self.compute_dtype(0.0)
            for l in range(_q):
                # If the mask is missing then take the opposite index
                if missing_mask[l] == wp.uint8(1):
                    # Get index associated with the fluid neighbours
                    fluid_nbr_index = type(index)()
                    for d in range(_d):
                        fluid_nbr_index[d] = index[d] + _c[d, l]
                    # Find the neighbour and its velocity value
                    for ll in range(_q):
                        # f_0 is the post-collision values of the current time-step
                        # The following is the post-collision values of the fluid neighbor cell
                        _f_nbr[ll] = self.compute_dtype(f_0[ll, fluid_nbr_index[0], fluid_nbr_index[1], fluid_nbr_index[2]])

                    # Compute the velocity vector at the fluid neighbouring cells
                    _, u_f = self.macroscopic.warp_functional(_f_nbr)

                    # The implicit distance to the boundary or "weights" have been stored in known directions of f_1
                    weight = f_1[_opp_indices[l], index[0], index[1], index[2]]

                    # Given "weights", "u_w" (input to the BC) and "u_f" (computed from f_aux), compute "u_target" as per Eq (14)
                    for d in range(_d):
                        u_target[d] += (weight * u_f[d] + _u_wall[d]) / (one + weight)

                    # Use differentiable interpolated BB to find f_missing:
                    f_post[l] = ((one - weight) * f_post[_opp_indices[l]] + weight * (f_pre[l] + f_pre[_opp_indices[l]])) / (one + weight)

                    # Add contribution due to moving_wall to f_missing as is usual in regular Bouzidi BC
                    # f_post = moving_wall_fpop_correction(_u_wall, l, f_post)

                    # Record the number of missing directions
                    num_missing += one

            # Compute rho_target = \sum(f_ibb) based on these values
            rho_target = self.zero_moment.warp_functional(f_post)
            for d in range(_d):
                u_target[d] /= num_missing

            # Compute Grad's appriximation using full equation as in Eq (10)
            f_post = grads_approximate_fpop(rho_target, u_target, f_post)
            return f_post

        if self.bc_method == "bounceback_regularized":
            functional = hybrid_bounceback_regularized
        elif self.bc_method == "bounceback_grads":
            functional = hybrid_bounceback_grads
        elif self.bc_method == "dorschner_localized":
            functional = dorschner_localized

        kernel = self._construct_kernel(functional)

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
