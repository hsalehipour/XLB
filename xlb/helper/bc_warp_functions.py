from xlb import DefaultConfig
from xlb.compute_backend import ComputeBackend
from xlb.operator.macroscopic.second_moment import SecondMoment as MomentumFlux
import warp as wp
from typing import Any

# Set the compute and Store dtypes
if DefaultConfig.default_backend == ComputeBackend.JAX:
    compute_dtype = DefaultConfig.default_precision_policy.compute_precision.jax_dtype
    store_dtype = DefaultConfig.default_precision_policy.store_precision.jax_dtype
elif DefaultConfig.default_backend == ComputeBackend.WARP:
    compute_dtype = DefaultConfig.default_precision_policy.compute_precision.wp_dtype
    compute_dtype = DefaultConfig.default_precision_policy.store_precision.wp_dtype

# Set local constants
_d = DefaultConfig.velocity_set.d
_q = DefaultConfig.velocity_set.q
_u_vec = wp.vec(_d, dtype=compute_dtype)
_opp_indices = DefaultConfig.velocity_set.opp_indices
_w = DefaultConfig.velocity_set.w
_c = DefaultConfig.velocity_set.c
_c_float = DefaultConfig.velocity_set.c_float
_qi = DefaultConfig.velocity_set.qi


# Define the operator needed for computing the momentum flux
momentum_flux = MomentumFlux()


@wp.func
def get_bc_fsum(
    fpop: Any,
    missing_mask: Any,
):
    fsum_known = compute_dtype(0.0)
    fsum_middle = compute_dtype(0.0)
    for l in range(_q):
        if missing_mask[_opp_indices[l]] == wp.uint8(1):
            fsum_known += compute_dtype(2.0) * fpop[l]
        elif missing_mask[l] != wp.uint8(1):
            fsum_middle += fpop[l]
    return fsum_known + fsum_middle


@wp.func
def get_normal_vectors(
    missing_mask: Any,
):
    if wp.static(_d == 3):
        for l in range(_q):
            if missing_mask[l] == wp.uint8(1) and wp.abs(_c[0, l]) + wp.abs(_c[1, l]) + wp.abs(_c[2, l]) == 1:
                return -_u_vec(_c_float[0, l], _c_float[1, l], _c_float[2, l])
    else:
        for l in range(_q):
            if missing_mask[l] == wp.uint8(1) and wp.abs(_c[0, l]) + wp.abs(_c[1, l]) == 1:
                return -_u_vec(_c_float[0, l], _c_float[1, l])


@wp.func
def bounceback_nonequilibrium(
    fpop: Any,
    feq: Any,
    missing_mask: Any,
):
    for l in range(_q):
        if missing_mask[l] == wp.uint8(1):
            fpop[l] = fpop[_opp_indices[l]] + feq[l] - feq[_opp_indices[l]]
    return fpop


@wp.func
def regularize_fpop(
    fpop: Any,
    feq: Any,
):
    """
    Regularizes the distribution functions by adding non-equilibrium contributions based on second moments of fpop.
    """
    # Compute momentum flux of off-equilibrium populations for regularization: Pi^1 = Pi^{neq}
    f_neq = fpop - feq
    PiNeq = momentum_flux.warp_functional(f_neq)

    # Compute double dot product Qi:Pi1 (where Pi1 = PiNeq)
    nt = _d * (_d + 1) // 2
    for l in range(_q):
        QiPi1 = compute_dtype(0.0)
        for t in range(nt):
            QiPi1 += _qi[l, t] * PiNeq[t]

        # assign all populations based on eq 45 of Latt et al (2008)
        # fneq ~ f^1
        fpop1 = compute_dtype(4.5) * _w[l] * QiPi1
        fpop[l] = feq[l] + fpop1
    return fpop


@wp.func
def grads_approximate_fpop(
    rho: Any,
    u: Any,
    f_post: Any,
):
    # Purpose: Using Grad's approximation to represent fpop based on macroscopic inputs used for outflow [1] and
    # Dirichlet BCs [2]
    # [1] S. Chikatax`marla, S. Ansumali, and I. Karlin, "Grad's approximation for missing data in lattice Boltzmann
    #   simulations", Europhys. Lett. 74, 215 (2006).
    # [2] Dorschner, B., Chikatamarla, S. S., Bösch, F., & Karlin, I. V. (2015). Grad's approximation for moving and
    #    stationary walls in entropic lattice Boltzmann simulations. Journal of Computational Physics, 295, 340-354.

    # Note: See also self.regularize_fpop function which is somewhat similar.

    # Compute pressure tensor Pi using all f_post-streaming values
    Pi = momentum_flux.warp_functional(f_post)

    # Compute double dot product Qi:Pi1 (where Pi1 = PiNeq)
    nt = _d * (_d + 1) // 2
    for l in range(_q):
        # compute dot product of qi and Pi
        QiPi = compute_dtype(0.0)
        for t in range(nt):
            if t == 0 or t == 3 or t == 5:
                QiPi += _qi[l, t] * (Pi[t] - rho / compute_dtype(3.0))
            else:
                QiPi += _qi[l, t] * Pi[t]

        # Compute c.u
        cu = compute_dtype(0.0)
        for d in range(_d):
            if _c[d, l] == 1:
                cu += u[d]
            elif _c[d, l] == -1:
                cu -= u[d]
        cu *= compute_dtype(3.0)

        # change f_post using the Grad's approximation
        f_post[l] = rho * _w[l] * (compute_dtype(1.0) + cu) + _w[l] * compute_dtype(4.5) * QiPi

    return f_post


@wp.func
def moving_wall_fpop_correction(
    u_w: Any,
    lattice_direction: Any,
    f_post: Any,
):
    # Add forcing term necessary to account for the local density changes caused by the mass displacement as the object moves with velocity u_w.
    # [1] L.-S. Luo, Unified theory of lattice Boltzmann models for nonideal gases, Phys. Rev. Lett. 81 (1998) 1618-1621.
    # [2] L.-S. Luo, Theory of the lattice Boltzmann method: Lattice Boltzmann models for nonideal gases, Phys. Rev. E 62 (2000) 4982-4996.
    #
    # Note: this function must be called within a for-loop over all lattice directions and the populations to be modified must
    # be only those in the missing direction (the check for missing direction must be outside of this function).
    cu = compute_dtype(0.0)
    l = lattice_direction
    for d in range(_d):
        if _c[d, l] == 1:
            cu += u_w[d]
        elif _c[d, l] == -1:
            cu -= u_w[d]
    cu *= compute_dtype(-6.0) * _w[l]
    f_post[l] += cu
    return f_post


@wp.func
def interpolated_bounceback(
    missing_mask: Any,
    f_0: Any,
    f_1: Any,
    f_pre: Any,
    f_post: Any,
):
    # A local single-node version of the interpolated bounce-back boundary condition due to Bouzidi for a lattice
    # Boltzmann method simulation.
    # Ref:
    # [1] Yu, D., Mei, R., Shyy, W., 2003. A uniﬁed boundary treatment in lattice boltzmann method,
    # in: 41st aerospace sciences meeting and exhibit, p. 953.

    one = compute_dtype(1.0)
    for l in range(_q):
        # If the mask is missing then take the opposite index
        if missing_mask[l] == wp.uint8(1):
            # The implicit distance to the boundary or "weights" have been stored in known directions of f_1
            # weight = f_1[_opp_indices[l], index[0], index[1], index[2]]
            # TODO: use weights associated with curved boundaries that are properly stored in f_1. There needs to be an input flag for this!
            weight = compute_dtype(0.5)

            # Use differentiable interpolated BB to find f_missing:
            f_post[l] = ((one - weight) * f_post[_opp_indices[l]] + weight * (f_pre[l] + f_pre[_opp_indices[l]])) / (one + weight)

            # TODO: Add u_wall associated with moving boundaries that are properly stored in f_1 or f_0. There needs to be an input flag for this!
            # Add contribution due to moving_wall to f_missing as is usual in regular Bouzidi BC
            # f_post = moving_wall_fpop_correction(_u_wall, l, f_post)
    return f_post
