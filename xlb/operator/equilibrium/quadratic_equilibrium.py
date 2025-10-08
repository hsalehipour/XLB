from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp
import os

import neon
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium import Equilibrium
from xlb.operator import Operator


class QuadraticEquilibrium(Equilibrium):
    """
    Quadratic equilibrium of Boltzmann equation using hermite polynomials.
    Standard equilibrium model for LBM.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, rho, u):
        cu = 3.0 * jnp.tensordot(self.velocity_set.c, u, axes=(0, 0))
        usqr = 1.5 * jnp.sum(jnp.square(u), axis=0, keepdims=True)
        w = self.velocity_set.w.reshape((-1,) + (1,) * (len(rho.shape) - 1))
        feq = rho * w * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)
        return feq

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _c = self.velocity_set.c
        _d = self.velocity_set.d
        _q = self.velocity_set.q
        _w = self.velocity_set.w
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)

        # Construct the equilibrium functional
        # D2Q9 Kernel (2D, 9 directions)        
        @wp.func
        def functional_d2q9(
            rho: Any,
            u: Any,
        ):
            # Precompute constants
            zero     = self.compute_dtype(0.0)
            half     = self.compute_dtype(0.5)
            one      = self.compute_dtype(1.0)
            one_half = self.compute_dtype(1.5)
            three    = self.compute_dtype(3.0)

            # Allocate the equilibrium distribution array
            feq = _f_vec()

            # Compute usqr (velocity magnitude term)
            usqr = one_half * (u[0] * u[0] + u[1] * u[1])

            # Rest particle: l=0 (0,0), w[0] = 4/9
            cu = zero
            base = one + half * cu * cu - usqr
            feq[0] = rho * _w[0] * base  # Simplifies to rho * w[0] * (1 - usqr)

            # Pair 1: l=1 (0,1) and l=2 (0,-1), w[1] = w[2] = 1/9
            cu_l = three * u[1]
            base = one + half * cu_l * cu_l - usqr
            feq[1] = rho * _w[1] * (base + cu_l)
            feq[2] = rho * _w[2] * (base - cu_l)

            # Pair 2: l=3 (1,0) and l=4 (-1,0), w[3] = w[4] = 1/9
            cu_l = three * u[0]
            base = one + half * cu_l * cu_l - usqr
            feq[3] = rho * _w[3] * (base + cu_l)
            feq[4] = rho * _w[4] * (base - cu_l)

            # Pair 3: l=5 (1,1) and l=8 (-1,-1), w[5] = w[8] = 1/36
            cu_l = three * (u[0] + u[1])
            base = one + half * cu_l * cu_l - usqr
            feq[5] = rho * _w[5] * (base + cu_l)
            feq[8] = rho * _w[8] * (base - cu_l)

            # Pair 4: l=6 (-1,1) and l=7 (1,-1), w[6] = w[7] = 1/36
            cu_l = three * (-u[0] + u[1])
            base = one + half * cu_l * cu_l - usqr
            feq[6] = rho * _w[6] * (base + cu_l)
            feq[7] = rho * _w[7] * (base - cu_l)

            return feq
        # D3Q19 Kernel (3D, 19 directions)
        @wp.func
        def functional_d3q19(
            rho: Any,
            u: Any,
        ):
            # Precompute constants
            zero     = self.compute_dtype(0.0)
            half     = self.compute_dtype(0.5)
            one      = self.compute_dtype(1.0)
            one_half = self.compute_dtype(1.5)
            three    = self.compute_dtype(3.0)

            # Allocate the equilibrium distribution array
            feq = _f_vec()

            # Compute usqr (velocity magnitude term)
            usqr = one_half * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2])

            # Rest particle: l=0 (0,0,0), w[0] = 1/3
            cu = zero
            base = one + half * cu * cu - usqr
            feq[0] = rho * _w[0] * base

            # Pair 1: l=1 (0,0,-1) and l=2 (0,0,1), w[1] = w[2] = 1/18
            cu_l = three * (-u[2])
            base = one + half * cu_l * cu_l - usqr
            feq[1] = rho * _w[1] * (base + cu_l)
            feq[2] = rho * _w[2] * (base - cu_l)

            # Pair 2: l=3 (0,-1,0) and l=6 (0,1,0), w[3] = w[6] = 1/18
            cu_l = three * (-u[1])
            base = one + half * cu_l * cu_l - usqr
            feq[3] = rho * _w[3] * (base + cu_l)
            feq[6] = rho * _w[6] * (base - cu_l)

            # Pair 3: l=4 (0,-1,-1) and l=8 (0,1,1), w[4] = w[8] = 1/36
            cu_l = three * (-u[1] - u[2])
            base = one + half * cu_l * cu_l - usqr
            feq[4] = rho * _w[4] * (base + cu_l)
            feq[8] = rho * _w[8] * (base - cu_l)

            # Pair 4: l=5 (0,-1,1) and l=7 (0,1,-1), w[5] = w[7] = 1/36
            cu_l = three * (-u[1] + u[2])
            base = one + half * cu_l * cu_l - usqr
            feq[5] = rho * _w[5] * (base + cu_l)
            feq[7] = rho * _w[7] * (base - cu_l)

            # Pair 5: l=9 (-1,0,0) and l=14 (1,0,0), w[9] = w[14] = 1/18
            cu_l = three * (-u[0])
            base = one + half * cu_l * cu_l - usqr
            feq[9] = rho * _w[9] * (base + cu_l)
            feq[14] = rho * _w[14] * (base - cu_l)

            # Pair 6: l=10 (-1,0,-1) and l=16 (1,0,1), w[10] = w[16] = 1/36
            cu_l = three * (-u[0] - u[2])
            base = one + half * cu_l * cu_l - usqr
            feq[10] = rho * _w[10] * (base + cu_l)
            feq[16] = rho * _w[16] * (base - cu_l)

            # Pair 7: l=11 (-1,0,1) and l=15 (1,0,-1), w[11] = w[15] = 1/36
            cu_l = three * (-u[0] + u[2])
            base = one + half * cu_l * cu_l - usqr
            feq[11] = rho * _w[11] * (base + cu_l)
            feq[15] = rho * _w[15] * (base - cu_l)

            # Pair 8: l=12 (-1,-1,0) and l=18 (1,1,0), w[12] = w[18] = 1/36
            cu_l = three * (-u[0] - u[1])
            base = one + half * cu_l * cu_l - usqr
            feq[12] = rho * _w[12] * (base + cu_l)
            feq[18] = rho * _w[18] * (base - cu_l)

            # Pair 9: l=13 (-1,1,0) and l=17 (1,-1,0), w[13] = w[17] = 1/36
            cu_l = three * (-u[0] + u[1])
            base = one + half * cu_l * cu_l - usqr
            feq[13] = rho * _w[13] * (base + cu_l)
            feq[17] = rho * _w[17] * (base - cu_l)

            return feq
        # D3Q27 Kernel (3D, 27 directions)
        @wp.func
        def functional_d3q27(rho: Any,
                         u: Any,
                        ):
                   
            # Precompute constants
            zero     = self.compute_dtype(0.0)
            half     = self.compute_dtype(0.5)
            one      = self.compute_dtype(1.0)
            one_half = self.compute_dtype(1.5)
            three    = self.compute_dtype(3.0)

            # Allocate the equilibrium distribution array
            feq = _f_vec()

            # Compute usqr once (velocity magnitude term)
            usqr = one_half * wp.dot(u, u)

            # Rest particle: l=0 (0,0,0) - No opposite
            cu = zero
            base = one + half * cu * cu - usqr
            feq[0] = rho * _w[0] * base  # cu = 0, so feq[0] = rho * w[0] * (1 - usqr)

            # Pair 1: l=1 (0,0,-1) and l=2 (0,0,1)
            cu_l = -three * u[2]
            base = one + half * cu_l * cu_l - usqr
            feq[1] = rho * _w[1] * (base + cu_l)
            feq[2] = rho * _w[2] * (base - cu_l)

            # Pair 2: l=3 (0,-1,0) and l=6 (0,1,0)
            cu_l = -three * u[1]
            base = one + half * cu_l * cu_l - usqr
            feq[3] = rho * _w[3] * (base + cu_l)
            feq[6] = rho * _w[6] * (base - cu_l)

            # Pair 3: l=4 (0,-1,-1) and l=8 (0,1,1)
            cu_l = three * (-u[1] - u[2])
            base = one + half * cu_l * cu_l - usqr
            feq[4] = rho * _w[4] * (base + cu_l)
            feq[8] = rho * _w[8] * (base - cu_l)

            # Pair 4: l=5 (0,-1,1) and l=7 (0,1,-1)
            cu_l = three * (-u[1] + u[2])
            base = one + half * cu_l * cu_l - usqr
            feq[5] = rho * _w[5] * (base + cu_l)
            feq[7] = rho * _w[7] * (base - cu_l)

            # Pair 5: l=9 (-1,0,0) and l=18 (1,0,0)
            cu_l = -three * u[0]
            base = one + half * cu_l * cu_l - usqr
            feq[9] = rho * _w[9] * (base + cu_l)
            feq[18] = rho * _w[18] * (base - cu_l)

            # Pair 6: l=10 (-1,0,-1) and l=20 (1,0,1)
            cu_l = three * (-u[0] - u[2])
            base = one + half * cu_l * cu_l - usqr
            feq[10] = rho * _w[10] * (base + cu_l)
            feq[20] = rho * _w[20] * (base - cu_l)

            # Pair 7: l=11 (-1,0,1) and l=19 (1,0,-1)
            cu_l = three * (-u[0] + u[2])
            base = one + half * cu_l * cu_l - usqr
            feq[11] = rho * _w[11] * (base + cu_l)
            feq[19] = rho * _w[19] * (base - cu_l)

            # Pair 8: l=12 (-1,-1,0) and l=24 (1,1,0)
            cu_l = three * (-u[0] - u[1])
            base = one + half * cu_l * cu_l - usqr
            feq[12] = rho * _w[12] * (base + cu_l)
            feq[24] = rho * _w[24] * (base - cu_l)

            # Pair 9: l=13 (-1,-1,-1) and l=26 (1,1,1)
            cu_l = three * (-u[0] - u[1] - u[2])
            base = one + half * cu_l * cu_l - usqr
            feq[13] = rho * _w[13] * (base + cu_l)
            feq[26] = rho * _w[26] * (base - cu_l)

            # Pair 10: l=14 (-1,-1,1) and l=25 (1,1,-1)
            cu_l = three * (-u[0] - u[1] + u[2])
            base = one + half * cu_l * cu_l - usqr
            feq[14] = rho * _w[14] * (base + cu_l)
            feq[25] = rho * _w[25] * (base - cu_l)

            # Pair 11: l=15 (-1,1,0) and l=21 (1,-1,0)
            cu_l = three * (-u[0] + u[1])
            base = one + half * cu_l * cu_l - usqr
            feq[15] = rho * _w[15] * (base + cu_l)
            feq[21] = rho * _w[21] * (base - cu_l)

            # Pair 12: l=16 (-1,1,-1) and l=23 (1,-1,1)
            cu_l = three * (-u[0] + u[1] - u[2])
            base = one + half * cu_l * cu_l - usqr
            feq[16] = rho * _w[16] * (base + cu_l)
            feq[23] = rho * _w[23] * (base - cu_l)

            # Pair 13: l=17 (-1,1,1) and l=22 (1,-1,-1)
            cu_l = three * (-u[0] + u[1] + u[2])
            base = one + half * cu_l * cu_l - usqr
            feq[17] = rho * _w[17] * (base + cu_l)
            feq[22] = rho * _w[22] * (base - cu_l)

            return feq

        @wp.func
        def functional_loop(
            rho: Any,
            u: Any,
        ):
            # Allocate the equilibrium
            feq = _f_vec()

            # Compute the equilibrium
            for l in range(self.velocity_set.q):
                # Compute cu
                cu = self.compute_dtype(0.0)
                for d in range(self.velocity_set.d):
                    if _c[d, l] == 1:
                        cu += u[d]
                    elif _c[d, l] == -1:
                        cu -= u[d]
                cu *= self.compute_dtype(3.0)

                # Compute usqr
                usqr = self.compute_dtype(1.5) * wp.dot(u, u)

                # Compute feq
                feq[l] = rho * _w[l] * (self.compute_dtype(1.0) + cu * (self.compute_dtype(1.0) + self.compute_dtype(0.5) * cu) - usqr)

            return feq
        
        # Determine the lattice type and return the appropriate kernel
        if _d == 2 and _q == 9:
            functional = functional_d2q9
        elif _d == 3 and _q == 19:
            functional = functional_d3q19
        elif _d == 3 and _q == 27:
            functional = functional_d3q27
        else:
            functional = functional_loop

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            rho: wp.array4d(dtype=Any),
            u: wp.array4d(dtype=Any),
            f: wp.array4d(dtype=Any),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # Get the equilibrium
            _u = _u_vec()
            for d in range(self.velocity_set.d):
                _u[d] = u[d, index[0], index[1], index[2]]
            _rho = rho[0, index[0], index[1], index[2]]
            feq = functional(_rho, _u)

            # Set the output
            for l in range(self.velocity_set.q):
                f[l, index[0], index[1], index[2]] = self.store_dtype(feq[l])

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, rho, u, f):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                rho,
                u,
                f,
            ],
            dim=rho.shape[1:],
        )
        return f

    def _construct_neon(self):
        import neon, typing

        # Use the warp functional for the NEON backend
        functional, _ = self._construct_warp()

        # Set local constants TODO: This is a hack and should be fixed with warp update
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)

        @neon.Container.factory(name="QuadraticEquilibrium")
        def container(
            rho: Any,
            u: Any,
            f: Any,
        ):
            def quadratic_equilibrium_ll(loader: neon.Loader):
                loader.set_grid(rho.get_grid())
                rho_pn = loader.get_read_handle(rho)
                u_pn = loader.get_read_handle(u)
                f_pn = loader.get_write_handle(f)

                @wp.func
                def quadratic_equilibrium_cl(index: typing.Any):
                    _u = _u_vec()
                    for d in range(self.velocity_set.d):
                        _u[d] = wp.neon_read(u_pn, index, d)
                    _rho = wp.neon_read(rho_pn, index, 0)
                    feq = functional(_rho, _u)

                    # Set the output
                    for l in range(self.velocity_set.q):
                        # wp.neon_write(f_pn, index, l, self.store_dtype(feq[l]))
                        wp.neon_write(f_pn, index, l, feq[l])

                loader.declare_kernel(quadratic_equilibrium_cl)

            return quadratic_equilibrium_ll

        return functional, container

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, rho, u, f):
        c = self.neon_container(rho, u, f)
        c.run(0, container_runtime=neon.Container.ContainerRuntime.neon)
        return f
