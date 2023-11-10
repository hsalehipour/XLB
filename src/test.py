import time
from jax import vjp
import os
import copy

from src.adjoint import LBMBaseDifferentiable
from src.lattice import LatticeD2Q9, LatticeD3Q19
from src.boundary_conditions import *

np.random.seed(0)

class UnitTest(LBMBaseDifferentiable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_boundary_conditions(self):

        # concatenate the indices of the left, right, and bottom walls
        walls = [self.boundingBoxIndices['bottom'], self.boundingBoxIndices['top'],
                 self.boundingBoxIndices['left'], self.boundingBoxIndices['right']]

        inlet = self.boundingBoxIndices['left'][1:-1]
        outlet = self.boundingBoxIndices['right'][1:-1]

        # apply bounce back boundary condition to the walls
        for wall in walls:
            self.BCs.append(BounceBackHalfway(tuple(wall.T), self.gridInfo, self.precisionPolicy))
            self.BCs[-1].needsExtraConfiguration = False
            self.BCs[-1].isSolid = False

        vel_inlet = np.random.random(inlet.shape)
        self.BCs.append(ZouHe(tuple(inlet.T), self.gridInfo, self.precisionPolicy, 'velocity', vel_inlet))

        rho_outlet = np.random.random((outlet.shape[0], 1))
        self.BCs.append(ZouHe(tuple(outlet.T), self.gridInfo, self.precisionPolicy, 'pressure', rho_outlet))


    @partial(jit, static_argnums=(0,))
    def collision(self, f):
        """
        BGK collision step for lattice.

        The collision step is where the main physics of the LBM is applied. In the BGK approximation,
        the distribution function is relaxed towards the equilibrium distribution function.
        """
        f = self.precisionPolicy.cast_to_compute(f)
        rho, u = self.update_macroscopic(f)
        feq = self.equilibrium(rho, u, cast_output=False)
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

    def apply_bounceback_halfway_adj(self, bc, fhat_poststreaming, fhat, implementationStep):
        # only at BC
        if implementationStep == 'PostStreaming':
            nbd = len(bc.indices[0])
            bindex = np.arange(nbd)[:, None]
            fbd = fhat_poststreaming[bc.indices]
            fbd = fbd.at[bindex, bc.iknown].set(fhat[bc.indices][bindex, bc.imissing])
            fhat_poststreaming = fhat_poststreaming.at[bc.indices].set(fbd)
        return fhat_poststreaming

    def apply_zouhe_adj(self, bc, fhat_poststreaming, fhat, fpop_fwd, implementationStep):
        """
        # The ZouHe Adjoint BC:
        # since in the forward computation, the boundary condition is imposed on post-streaming populations and has no
        # dependence on pre-streaming fpop, the adjoint ZouHe would be identically be zero for both types of pressure
        # and velocity BC. This is confirmed by AD unit test.
        """
        nbd = len(bc.indices[0])
        bindex = np.arange(nbd)[:, None]

        if implementationStep == 'PostCollision':

            if bc.type == 'velocity':
                vel = bc.prescribed
                unormal = jnp.sum(bc.normals * vel, keepdims=True, axis=-1)*jnp.ones((1, self.q))
                coeff = 1.0 / (1.0 + unormal)
            elif bc.type == 'pressure':
                rho = bc.prescribed
                vel = bc.calculate_vel(fpop_fwd, rho)
                coeff = 1.0 / rho
            else:
                raise ValueError(f"type = {bc.type} not supported! Use \'pressure\' or \'velocity\'.")

            # compute dBC/df
            fbd = fhat[bc.indices]
            c = jnp.array(self.lattice.c, dtype=self.precisionPolicy.compute_dtype)
            cu = 3.0 * jnp.dot(vel, c)
            fsum = 2.0 * jnp.sum(self.w * cu * fbd * bc.imissingBitmask, keepdims=True, axis=-1)
            ddf = coeff * fsum

            # Note: the zero'th index needs to be corrected before --->
            fbd = fbd.at[bindex, bc.iknown].add(fbd[bindex, bc.imissing])
            fbd = fbd.at[bindex, 0].set(fhat[bc.indices][bindex, 0])

            # ---> this line. In other words, the above two lines must be executed before the following lines.
            fbd = fbd.at[bc.imiddleBitmask].add(ddf[bc.imiddleBitmask])
            fbd = fbd.at[bc.iknownBitmask].add(2.*ddf[bc.iknownBitmask])
            fhat_poststreaming = fhat_poststreaming.at[bc.indices].set(fbd)

        elif implementationStep == 'PostStreaming':
            fbd = fhat_poststreaming[bc.indices]
            fbd = fbd.at[bc.iknownBitmask].set(0.0)
            fhat_poststreaming = fhat_poststreaming.at[bc.indices].set(fbd)
        else:
            raise ValueError(f"Failed to impose adjoint Zou-He BC.")

        return fhat_poststreaming

    @partial(jit, static_argnums=(0, 4))
    def apply_bc_adj(self, fhat_poststreaming, fhat, fpop_fwd, implementationStep):
        for bc in self.BCs:
            if bc.name == "BounceBackHalfway":
                fhat_poststreaming = self.apply_bounceback_halfway_adj(bc, fhat_poststreaming, fhat, implementationStep)
            if bc.name == "ZouHe":
                fhat_poststreaming = self.apply_zouhe_adj(bc, fhat_poststreaming, fhat, fpop_fwd, implementationStep)
        return fhat_poststreaming

    @partial(jit, static_argnums=(0,))
    def step_adjoint_test(self, f, fhat):
        """
        Adjoint of halfway bounceback boundary condition.
        """
        # all voxels
        fhat_poststreaming = self.streaming_adj(fhat)
        fhat_poststreaming = self.apply_bc_adj(fhat_poststreaming, fhat, f, "PostStreaming")
        fhat_postcollision = self.collision_adj(f, fhat_poststreaming)
        return fhat_postcollision

    @partial(jit, static_argnums=(0,))
    def step_adjoint_complete(self, f, fhat):
        """
        Adjoint of LBM step
        """
        fhat_postcollision = self.apply_bc_adj(fhat, fhat, f, "PostCollision")
        fhat_poststreaming = self.streaming_adj(fhat_postcollision)
        fhat_poststreaming = self.apply_bc_adj(fhat_poststreaming, fhat_postcollision, f, "PostStreaming")
        # return fhat_poststreaming
        fhat_postcollision = self.collision_adj(f, fhat_poststreaming)
        return fhat_postcollision

if __name__ == "__main__":
    precision = "f32/f32"
    lattice = LatticeD2Q9(precision)

    # Input test parameters
    nx, ny, nz = 5, 8, 0
    tol = 1e-5
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
    BCCopy = copy.deepcopy(test.BCs)
    bottomWall, topWall, leftWall, rightWall, leftInlet, rightOutlet = BCCopy

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

    # # TEST 4:
    # # Apply post-streaming BC after stream
    # test.BCs = [bottomWall, topWall, leftWall, rightWall]
    # f = jnp.array(np.random.random(f.shape), dtype=test.precisionPolicy.compute_dtype)
    # start_time = time.time()
    # fhat_poststreaming = test.streaming_adj(fhat)
    # fhat_poststreaming = test.apply_bc_adj(fhat_poststreaming, fhat, f, "PostStreaming")
    # print(f'Ref time is: {time.time() - start_time}')
    # def lbm_step_bc(f):
    #     f_poststreaming = test.streaming(f)
    #     f_poststreaming = test.apply_bc(f_poststreaming, f, timestep, "PostStreaming")
    #     return f_poststreaming
    #
    # ffunc, dfunc_AD = vjp(lbm_step_bc, f)
    # dfunc_AD = dfunc_AD(fhat)[0]
    # test.test_adjoint(fhat, fhat_poststreaming, '"Stream with halfway BB"', lbm_step_bc, f)
    #
    #
    # # TEST 5:
    # # Apply post-streaming boudary condition after BGK Collision & Streaming
    # test.BCs = [bottomWall, topWall, leftWall, rightWall]
    # f = jnp.array(np.random.random(f.shape), dtype=test.precisionPolicy.compute_dtype)
    # start_time = time.time()
    # fhat_poststreaming = test.step_adjoint_test(f, fhat)
    # print(f'Ref time is: {time.time() - start_time}')
    # def lbm_step_full(f):
    #     f_postcollision = test.collision(f)
    #     f_poststreaming = test.streaming(f_postcollision)
    #     f_poststreaming = test.apply_bc(f_poststreaming, f_postcollision, timestep, "PostStreaming")
    #     return f_poststreaming
    #
    # ffunc, dfunc_AD = vjp(lbm_step_full, f)
    # dfunc_AD = dfunc_AD(fhat)[0]
    # test.test_adjoint(fhat, fhat_poststreaming, '"BGK collide-stream with halfway BB"', lbm_step_full, f)


    # TEST 6:
    # Apply post-streaming BC after stream
    test.BCs = [bottomWall, topWall, rightWall, leftInlet]
    f = jnp.array(np.random.random(f.shape), dtype=test.precisionPolicy.compute_dtype)
    start_time = time.time()
    fhat_postcollision = fhat
    fhat_poststreaming = fhat
    fhat_postcollision = test.apply_bc_adj(fhat_postcollision, fhat_poststreaming, f, "PostCollision")
    fhat_poststreaming = test.streaming_adj(fhat_postcollision)
    fhat_poststreaming = test.apply_bc_adj(fhat_poststreaming, fhat_postcollision, f, "PostStreaming")
    print(f'Ref time is: {time.time() - start_time}')
    def lbm_step_bc(f):
        f_poststreaming = test.streaming(f)
        f_poststreaming = test.apply_bc(f_poststreaming, f, timestep, "PostStreaming")
        return f_poststreaming

    ffunc, dfunc_AD = vjp(lbm_step_bc, f)
    dfunc_AD = dfunc_AD(fhat)[0]
    test.test_adjoint(fhat, fhat_poststreaming, '"Stream with halfway BB and ZouHe Velocity BC"', lbm_step_bc, f)


    # TEST 7:
    # Apply post-streaming BC after stream
    test.BCs = [bottomWall, topWall, rightOutlet, leftInlet]
    f = jnp.array(np.random.random(f.shape), dtype=test.precisionPolicy.compute_dtype)
    start_time = time.time()
    fhat_postcollision = fhat
    fhat_poststreaming = fhat
    fhat_postcollision = test.apply_bc_adj(fhat_postcollision, fhat_poststreaming, f, "PostCollision")
    fhat_poststreaming = test.streaming_adj(fhat_postcollision)
    fhat_poststreaming = test.apply_bc_adj(fhat_poststreaming, fhat_postcollision, f, "PostStreaming")
    print(f'Ref time is: {time.time() - start_time}')
    def lbm_step_bc(f):
        f_poststreaming = test.streaming(f)
        f_poststreaming = test.apply_bc(f_poststreaming, f, timestep, "PostStreaming")
        return f_poststreaming

    ffunc, dfunc_AD = vjp(lbm_step_bc, f)
    dfunc_AD = dfunc_AD(fhat)[0]
    test.test_adjoint(fhat, fhat_poststreaming, '"Stream with halfway BB and ZouHe Velocity and Pressure BC"', lbm_step_bc, f)


    # TEST 8:
    test.BCs = [bottomWall, topWall, rightOutlet, leftInlet]
    f = jnp.array(np.random.random(f.shape), dtype=test.precisionPolicy.compute_dtype)
    start_time = time.time()
    fhat_poststreaming = test.step_adjoint_complete(f, fhat)
    print(f'Ref time is: {time.time() - start_time}')
    def lbm_step_complete(f):
        f_postcollision = test.collision(f)
        # f_postcollision = f
        f_poststreaming = test.streaming(f_postcollision)
        f_poststreaming = test.apply_bc(f_poststreaming, f_postcollision, timestep, "PostStreaming")
        return f_poststreaming

    ffunc, dfunc_AD = vjp(lbm_step_complete, f)
    dfunc_AD = dfunc_AD(fhat)[0]
    test.test_adjoint(fhat, fhat_poststreaming, '"BGK collide-stream with halfway BB and ZouHe pressure and Velocity"',
                      lbm_step_complete, f)