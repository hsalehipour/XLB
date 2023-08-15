import time
import numpy as np
import jax.numpy as jnp
from jax import vjp, jit
from functools import partial
from src.adjoint import LBMBaseDifferentiable
import os
from src.lattice import LatticeD2Q9, LatticeD3Q19



class UnitTest(LBMBaseDifferentiable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, f):
        """
        BGK collision step for lattice.

        The collision step is where the main physics of the LBM is applied. In the BGK approximation,
        the distribution function is relaxed towards the equilibrium distribution function.
        """
        f = self.precisionPolicy.cast_to_compute(f)
        rho, u = self.update_macroscopic(f)
        feq = self.equilibrium(rho, u, castOutput=False)
        fneq = f - feq
        fout = f - self.omega * fneq
        if self.force is not None:
            fout = self.apply_force(fout, feq, rho, u)
        return self.precisionPolicy.cast_to_output(fout)

    def collision_adj_math(self, f, fhat):
        rho, vel = test.update_macroscopic(f)
        feq = test.equilibrium(rho, vel)
        omega = 1.0
        feq_adj = self.equilibrium_adj_math(fhat, feq, rho, vel)
        fneq_adj = fhat - feq_adj
        fhat = fhat - omega * fneq_adj
        return fhat


    def test_adjoint(self, fhat, dfunc_ref, func_name, func, *args):
        # Construct the adjoints for the forward model.
        _, dfunc_AD = vjp(func, *args)
        start_time = time.time()
        dfunc_AD = dfunc_AD(fhat)[0]
        print(f'AD time is: {time.time() - start_time}')

        if np.allclose(dfunc_ref, dfunc_AD, tol, tol):
            print(f'**** PASSED **** unit test for {func_name} up to tol={tol}')
        else:
            print(f'!!!! FAILED !!!! unit test for {func_name} equilibrium up to tol={tol}')

# def streaming(f, c):
#     def streaming_i(f, c):
#         if dim == 2:
#             return jnp.roll(f, (c[0], c[1]), axis=(0, 1))
#         elif dim == 3:
#             return jnp.roll(f, (c[0], c[1], c[2]), axis=(0, 1, 2))
#     return vmap(streaming_i, in_axes=(-1, 0), out_axes=-1)(f, c.T)
#
# def streaming_adj_math(fhat, c):
#     def streaming_i(fhat, c):
#         if dim == 2:
#             return jnp.roll(fhat, (-c[0], -c[1]), axis=(0, 1))
#         elif dim == 3:
#             return jnp.roll(fhat, (-c[0], -c[1], -c[2]), axis=(0, 1, 2))
#     return vmap(streaming_i, in_axes=(-1, 0), out_axes=-1)(fhat, c.T)




# def collision_and_stream(f):
#     f = collision(f)
#     f = streaming(f, c)
#     return f
#
# def collision_and_stream_adj_math(fhat, feq, rho, vel):
#     fhat = streaming_adj_math(fhat, c)
#     fhat = collision_adj_math(fhat, feq, rho, vel)
#     return fhat
#
# def apply_bc1(f):
#     return 5.0*f
#
# def apply_bc2(f):
#     return f + 2.0
#
# def apply_bc1_adj_math(fhat):
#     return 5.0*fhat
#
# def apply_bc2_adj_math(fhat):
#     return fhat
#
# def step(f):
#     f = collision(f)
#     f = apply_bc1(f)
#     f = streaming(f, c)
#     f = apply_bc2(f)
#     return f

# def step_adj_math(fhat, feq, rho, vel):
#     fhat = streaming_adj_math(fhat, c)
#     fhat = apply_bc2_adj_math(fhat)
#     fhat = collision_adj_math(fhat, feq, rho, vel)
#     fhat = apply_bc1_adj_math(fhat)
#     return fhat


if __name__ == "__main__":
    precision = "f32/f32"
    lattice = LatticeD3Q19(precision)

    # Input test parameters
    nx, ny, nz = 5, 7, 3
    tol = 1e-6

    os.system("rm -rf ./*.vtk && rm -rf ./*.png")
    kwargs = {
        'optimize': True,
        'lattice': lattice,
        'omega': 1.0,
        'nx': nx,
        'ny': ny,
        'nz': nz,
        'precision': precision,
        'io_rate': 500,
        'print_info_rate': 500
    }

    test = UnitTest(**kwargs)
    f = test.assign_fields_sharded()
    timestep = 0
    fhat = jnp.array(np.random.random(f.shape), dtype=test.precisionPolicy.compute_dtype)
    start_time = time.time()
    fhat_postcollision = test.collision_adj_math(f, fhat)
    print(f'Ref time is: {time.time() - start_time}')
    test.test_adjoint(fhat, fhat_postcollision, '"BGK Collision"', test.collision, f)

    # Streaming
    f = test.assign_fields_sharded()
    timestep = 0
    fhat_poststreaming = test.streaming_adj(fhat)
    print(f'Ref time is: {time.time() - start_time}')
    test.test_adjoint(fhat, fhat_poststreaming, '"Streaming"', test.streaming, f)



# # Equilibrium
# feq = equilibrium(fpop)
# _, equilibrium_adj = vjp(equilibrium, fpop)
# rho, vel = update_macroscopic(fpop)
# fhat_eq_math = equilibrium_adj_math(fhat, feq, rho, vel)
# fhat_eq_AD = equilibrium_adj(fhat)[0]
# if np.allclose(fhat_eq_math, fhat_eq_AD, tol, tol):
#     print(f'PASSED unit test for adjoint equilibrium up to tol={tol}')
# else:
#     print(f'FAILED unit test for adjoint equilibrium up to tol={tol}')
#
# # Streaming
# _, streaming_adj = vjp(streaming, fpop, c)
# fhat_poststreaming_math = streaming_adj_math(fhat, c)
# fhat_poststreaming_AD = streaming_adj(fhat)[0]
# if np.allclose(fhat_poststreaming_math, fhat_poststreaming_AD, tol, tol):
#     print(f'PASSED unit test for adjoint streaming up to tol={tol}')
# else:
#     print(f'FAILED unit test for adjoint streaming up to tol={tol}')
#
# # Collide and Streaming sequence of operations
# _, collision_adj = vjp(collision, fpop)
# fhat_postcollision_math = collision_adj_math(fhat, feq, rho, vel)
# fhat_postcollision_AD = collision_adj(fhat)[0]
# if np.allclose(fhat_postcollision_math, fhat_postcollision_AD, tol, tol):
#     print(f'PASSED unit test for adjoint collision up to tol={tol}')
# else:
#     print(f'FAILED unit test for adjoint collision up to tol={tol}')
#
#
#
# # Collide and Streaming sequence of operations
# _, collision_and_stream_adj = vjp(collision_and_stream, fpop)
# fhat_math = collision_and_stream_adj_math(fhat, feq, rho, vel)
# fhat_AD = collision_and_stream_adj(fhat)[0]
# if np.allclose(fhat_math, fhat_AD, tol, tol):
#     print(f'PASSED unit test for adjoint collide-stream up to tol={tol}')
# else:
#     print(f'FAILED unit test for adjoint collide-stream up to tol={tol}')
#
#
# # Complete LBM time step with BCs
# _, step_adj = vjp(step, fpop)
# fhat_next_math = step_adj_math(fhat, feq, rho, vel)
# fhat_next_AD = step_adj(fhat)[0]
# if np.allclose(fhat_next_math, fhat_next_AD, tol, tol):
#     print(f'PASSED unit test for an LBM time-step up to tol={tol}')
# else:
#     print(f'FAILED unit test for an LBM time-step up to tol={tol}')


