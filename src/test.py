import time
import numpy as np
import jax.numpy as jnp
from jax import vjp, jit
from functools import partial
from src.adjoint import LBMBaseDifferentiable
import os
from src.lattice import LatticeD2Q9, LatticeD3Q19
from src.boundary_conditions import *


class UnitTest(LBMBaseDifferentiable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_boundary_conditions(self):

        # concatenate the indices of the left, right, and bottom walls
        # walls = self.boundingBoxIndices["bottom"]
        walls = np.concatenate((self.boundingBoxIndices['bottom'],
                                self.boundingBoxIndices['top'],
                                self.boundingBoxIndices['left'],
                                self.boundingBoxIndices['right']))
        walls = np.unique(walls, axis=0)
        # apply bounce back boundary condition to the walls
        self.BCs.append(BounceBackHalfway(tuple(walls.T), self.gridInfo, self.precisionPolicy))
        self.BCs[0].needsExtraConfiguration = False
        self.BCs[0].isSolid = False


    @partial(jit, static_argnums=(0,))
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

    def collision_adj(self, f, fhat):
        rho, vel = test.update_macroscopic(f)
        feq = test.equilibrium(rho, vel)
        omega = self.omega
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
            print(f'!!!! FAILED !!!! unit test for {func_name} up to tol={tol}')

    def apply_bounceback_halfway_adj(self, fhat, fhat_poststreaming):
        # only at BC
        bc = self.BCs[0]
        nbd = len(bc.indices[0])
        bindex = np.arange(nbd)[:, None]
        fbd = fhat_poststreaming[bc.indices]
        fbd = fbd.at[bindex, bc.iknown].set(fhat[bc.indices][bindex, bc.imissing])
        fhat_poststreaming = fhat_poststreaming.at[bc.indices].set(fbd)
        return fhat_poststreaming

    @partial(jit, static_argnums=(0,))
    def step_adjoint_test(self, f, fhat):
        """
        Adjoint of halfway bounceback boundary condition.
        """
        # all voxels
        fhat_poststreaming = self.streaming_adj(fhat)
        fhat_poststreaming = self.apply_bounceback_halfway_adj(fhat, fhat_poststreaming)
        fhat_postcollision = self.collision_adj(f, fhat_poststreaming)
        return fhat_postcollision

if __name__ == "__main__":
    precision = "f32/f32"
    lattice = LatticeD2Q9(precision)

    # Input test parameters
    nx, ny, nz = 3, 4, 0
    tol = 1e-6
    timestep = 0

    os.system("rm -rf ./*.vtk && rm -rf ./*.png")
    kwargs = {
        'optimize': True,
        'lattice': lattice,
        'omega': 1.5,
        'nx': nx,
        'ny': ny,
        'nz': nz,
        'precision': precision,
        'io_rate': 500,
        'print_info_rate': 500
    }

    # Define the test class
    test = UnitTest(**kwargs)

    # TEST 1:
    # Collision
    f = test.assign_fields_sharded()
    f = jnp.array(np.random.random(f.shape), dtype=test.precisionPolicy.compute_dtype)
    fhat = jnp.array(np.random.random(f.shape), dtype=test.precisionPolicy.compute_dtype)
    start_time = time.time()
    fhat_postcollision = test.collision_adj(f, fhat)
    print(f'Ref time is: {time.time() - start_time}')
    test.test_adjoint(fhat, fhat_postcollision, '"BGK Collision"', test.collision, f)

    # TEST 2:
    # Streaming
    f = jnp.array(np.random.random(f.shape), dtype=test.precisionPolicy.compute_dtype)
    start_time = time.time()
    fhat_poststreaming = test.streaming_adj(fhat)
    print(f'Ref time is: {time.time() - start_time}')
    test.test_adjoint(fhat, fhat_poststreaming, '"Streaming"', test.streaming, f)


    # TEST 3:
    # Collide-then-Stream
    f = jnp.array(np.random.random(f.shape), dtype=test.precisionPolicy.compute_dtype)
    start_time = time.time()
    fhat_poststreaming = test.collision_adj(f, test.streaming_adj(fhat))
    print(f'Ref time is: {time.time() - start_time}')
    def lbm_step(f):
        return test.streaming(test.collision(f))
    test.test_adjoint(fhat, fhat_poststreaming, '"Collide-then-Stream"', lbm_step, f)

    # TEST 4:
    # Apply post-streaming BC after stream
    f = jnp.array(np.random.random(f.shape), dtype=test.precisionPolicy.compute_dtype)
    start_time = time.time()
    fhat_poststreaming = test.streaming_adj(fhat)
    fhat_poststreaming = test.apply_bounceback_halfway_adj(fhat, fhat_poststreaming)
    print(f'Ref time is: {time.time() - start_time}')
    def lbm_step_bc(f):
        f_poststreaming = test.streaming(f)
        f_poststreaming = test.apply_bc(f_poststreaming, f, timestep, "PostStreaming")
        return f_poststreaming

    ffunc, dfunc_AD = vjp(lbm_step_bc, f)
    dfunc_AD = dfunc_AD(fhat)[0]
    test.test_adjoint(fhat, fhat_poststreaming, '"Stream with halfway BB"', lbm_step_bc, f)


    # TEST 5:
    # Apply post-streaming boudary condition after BGK Collision & Streaming
    f = jnp.array(np.random.random(f.shape), dtype=test.precisionPolicy.compute_dtype)
    start_time = time.time()
    fhat_poststreaming = test.step_adjoint_test(f, fhat)
    print(f'Ref time is: {time.time() - start_time}')
    def lbm_step_full(f):
        f_postcollision = test.collision(f)
        f_poststreaming = test.streaming(f_postcollision)
        f_poststreaming = test.apply_bc(f_poststreaming, f_postcollision, timestep, "PostStreaming")
        return f_poststreaming

    ffunc, dfunc_AD = vjp(lbm_step_full, f)
    dfunc_AD = dfunc_AD(fhat)[0]
    test.test_adjoint(fhat, fhat_poststreaming, '"BGK collide-stream with halfway BB"', lbm_step_full, f)
