import time
from jax import vjp
import jax
import os

from src.adjoint import LBMBaseDifferentiable
from src.lattice import LatticeD2Q9, LatticeD3Q19
from src.boundary_conditions import *

jax.config.update('jax_enable_x64', True)
np.random.seed(0)

def discard_solids(bclist, fld1, fld2):
    solid_halo_list = [np.array(bc.solid_indices).T for bc in bclist if bc.isSolid]
    solid_halo_voxels = np.unique(np.vstack(solid_halo_list), axis=0) if solid_halo_list else None
    if solid_halo_voxels is not None:
        solid_voxels = tuple(solid_halo_voxels.T)
        fld1 = fld1.at[solid_voxels].set(fld2[solid_voxels])
    return fld1, fld2

class UnitTest(LBMBaseDifferentiable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_boundary_conditions(self):

        # Define the cylinder surface
        diam = self.ny/10
        coord = np.array([(i, j) for i in range(self.nx) for j in range(self.ny)])
        xx, yy = coord[:, 0], coord[:, 1]
        cx, cy = self.nx/2, self.ny/2
        cylinder = (xx - cx)**2 + (yy-cy)**2 <= (diam/2.)**2
        cylinder = coord[cylinder]
        implicit_distance = np.reshape((xx - cx)**2 + (yy-cy)**2 - (diam/2.)**2, (self.nx, self.ny))
        self.BCs.append(InterpolatedBounceBackDifferentiable(tuple(cylinder.T),
                                                             implicit_distance, self.gridInfo, self.precisionPolicy))
        self.BCs.append(BounceBackHalfway(tuple(cylinder.T), self.gridInfo, self.precisionPolicy))

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
        self.BCs[-1].needsExtraConfiguration = False

        rho_outlet = np.random.random((outlet.shape[0], 1))
        self.BCs.append(ZouHe(tuple(outlet.T), self.gridInfo, self.precisionPolicy, 'pressure', rho_outlet))
        self.BCs[-1].needsExtraConfiguration = False

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
        rho, vel = self.update_macroscopic(f)
        feq = self.equilibrium(rho, vel)
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

        # Discard the dissimilarity at the solid voxels
        dfunc_ref, dfunc_AD = discard_solids(self.BCs, dfunc_ref, dfunc_AD)

        flag = np.allclose(dfunc_ref, dfunc_AD, tol, tol)
        print('=' * 50)
        if flag:
            print(f'**** PASSED **** unit test for {func_name} up to tol={tol}')
        else:
            print(f'!!!! FAILED !!!! unit test for {func_name} up to tol={tol}')
        print('=' * 50)
        return flag

    @partial(jit, static_argnums=(0, 1, 4))
    def apply_bounceback_halfway_adj(self, bc, fhat_poststreaming, fhat, implementationStep):
        # only at BC
        if implementationStep == 'PostStreaming':
            nbd = len(bc.indices[0])
            bindex = np.arange(nbd)[:, None]
            fbd = fhat_poststreaming[bc.indices]
            fbd = fbd.at[bindex, bc.iknown].set(fhat[bc.indices][bindex, bc.imissing])
            fhat_poststreaming = fhat_poststreaming.at[bc.indices].set(fbd)

        return fhat_poststreaming

    @partial(jit, static_argnums=(0, 1, 4))
    def apply_zouhe_adj(self, bc, fhat_poststreaming, fhat, implementationStep):
        """
        The ZouHe Adjoint BC:
        Ihe forward computations, the boundary condition is imposed on post-streaming populations and has no
        dependence on pre-streaming fpop. As such, the adjoint ZouHe would set the post-streaming iknown values to zero
        for both types of pressure and velocity BC and all the non-zero contributions are applied before the adjoint
        streaming step. This is confirmed by AD unit test.
        """
        nbd = len(bc.indices[0])
        bindex = np.arange(nbd)[:, None]

        if implementationStep == 'PostCollision':

            if bc.type == 'velocity':
                vel = bc.prescribed
                unormal = jnp.sum(bc.normals * vel, keepdims=True, axis=-1)
                coeff = 1.0 / (1.0 + unormal)
            elif bc.type == 'pressure':
                vel = bc.normals
                coeff = 1.0
            else:
                raise ValueError(f"type = {bc.type} not supported! Use \'pressure\' or \'velocity\'.")

            # compute dBC/df
            fbd = fhat[bc.indices]
            c = jnp.array(self.lattice.c, dtype=self.precisionPolicy.compute_dtype)
            cu = 3.0 * jnp.dot(vel, c)
            fsum = 2.0 * jnp.sum(self.w * cu * fbd * bc.imissingMask, keepdims=True, axis=-1)
            ddf = coeff * fsum * jnp.ones((1, self.q))

            # Note: the zero'th index needs to be corrected before --->
            fbd = fbd.at[bindex, bc.iknown].add(fbd[bindex, bc.imissing])
            fbd = fbd.at[bindex, 0].set(fhat[bc.indices][bindex, 0])

            # ---> this line. In other words, the above two lines must be executed before the following lines.
            fbd = fbd.at[bc.imiddleMask].add(ddf[bc.imiddleMask])
            fbd = fbd.at[bc.iknownMask].add(2.*ddf[bc.iknownMask])
            fhat_poststreaming = fhat_poststreaming.at[bc.indices].set(fbd)

        elif implementationStep == 'PostStreaming':
            fbd = fhat_poststreaming[bc.indices]
            fbd = fbd.at[bc.iknownMask].set(0.0)
            fhat_poststreaming = fhat_poststreaming.at[bc.indices].set(fbd)
        else:
            raise ValueError(f"Failed to impose adjoint Zou-He BC.")

        return fhat_poststreaming

    @partial(jit, static_argnums=(0, 1, 4))
    def apply_InterpolatedBounceBackDifferentiable_adj(self, bc, fhat_poststreaming, fhat, implementationStep):
        """
        The Adjoint BC of InterpolatedBounceBackDifferentiable:
        """
        nbd = len(bc.indices[0])
        bindex = np.arange(nbd)[:, None]

        if bc.weights is None:
            bc.set_proximity_ratio()

        if implementationStep == 'PostCollision':
            # compute dBC/df
            # Note: The zero'th index needs to be corrected due to the addition above
            fbd = fhat[bc.indices]
            ddf = fbd * (1. - bc.weights) / (1. + bc.weights)
            fbd = fbd.at[bindex, bc.iknown].add(ddf[bindex, bc.imissing])
            fbd = fbd.at[bindex, 0].set(fhat[bc.indices][bindex, 0])
            fhat_poststreaming = fhat_poststreaming.at[bc.indices].set(fbd)

        elif implementationStep == 'PostStreaming':
            # compute dBC/df
            # Note: The zero'th index needs to be corrected due to the addition above
            fbd = fhat_poststreaming[bc.indices]
            ddf = fhat[bc.indices] * bc.weights / (1. + bc.weights)
            fbd = fbd.at[bc.imissingMask].add(ddf[bc.imissingMask])
            fbd = fbd.at[bindex, bc.iknown].set(ddf[bindex, bc.imissing])
            fbd = fbd.at[bindex, 0].set(fhat_poststreaming[bc.indices][bindex, 0])
            fhat_poststreaming = fhat_poststreaming.at[bc.indices].set(fbd)
        else:
            raise ValueError(f"Failed to impose adjoint InterpolatedBounceback BC.")

        return fhat_poststreaming


    @partial(jit, static_argnums=(0, 3))
    def apply_bc_adj(self, fhat_poststreaming, fhat, implementationStep):
        for bc in self.BCs:
            if bc.name == "BounceBackHalfway":
                fhat_poststreaming = self.apply_bounceback_halfway_adj(bc, fhat_poststreaming, fhat, implementationStep)
            if bc.name == "ZouHe":
                fhat_poststreaming = self.apply_zouhe_adj(bc, fhat_poststreaming, fhat, implementationStep)
            if bc.name == "InterpolatedBounceBackDifferentiable":
                fhat_poststreaming = self.apply_InterpolatedBounceBackDifferentiable_adj(bc,
                                                                                         fhat_poststreaming,
                                                                                         fhat, implementationStep)
        return fhat_poststreaming

    @partial(jit, static_argnums=(0,))
    def step_adjoint_noCollision(self, f, fhat):
        """
        Adjoint of halfway bounceback boundary condition.
        """
        # all voxels
        fhat_poststreaming = self.streaming_adj(fhat)
        fhat_poststreaming = self.apply_bc_adj(fhat_poststreaming, fhat, "PostStreaming")
        fhat_postcollision = self.collision_adj(f, fhat_poststreaming)
        return fhat_postcollision

    @partial(jit, static_argnums=(0,))
    def step_adjoint_complete(self, f, fhat):
        """
        Adjoint of LBM step
        """
        fhat_postcollision = self.apply_bc_adj(fhat, fhat, "PostCollision")
        fhat_poststreaming = self.streaming_adj(fhat_postcollision)
        fhat_poststreaming = self.apply_bc_adj(fhat_poststreaming, fhat_postcollision, "PostStreaming")
        fhat_postcollision = self.collision_adj(f, fhat_poststreaming)
        return fhat_postcollision

    def lbm_step_complete(self, f):
        f_postcollision = self.collision(f)
        f_poststreaming = self.streaming(f_postcollision)
        f_poststreaming = self.apply_bc(f_poststreaming, f_postcollision, timestep, "PostStreaming")
        return f_poststreaming

def init_unit_test(**kwargs):
    test = UnitTest(**kwargs)
    f = test.assign_fields_sharded()
    f = jnp.array(np.random.random(f.shape), dtype=test.precisionPolicy.compute_dtype)
    fhat = jnp.array(np.random.random(f.shape), dtype=test.precisionPolicy.compute_dtype)
    return test, f, fhat

def unit_test1(**kwargs):
    # TEST 1: Collision
    test, f, fhat = init_unit_test(**kwargs)
    start_time = time.time()
    fhat_postcollision = test.collision_adj(f, fhat)
    print(f'Ref time is: {time.time() - start_time}')
    test_result = test.test_adjoint(fhat, fhat_postcollision, '"BGK Collision"', test.collision, f)
    return test_result

def unit_test2(**kwargs):
    # TEST 2: Streaming
    test, f, fhat = init_unit_test(**kwargs)
    start_time = time.time()
    fhat_poststreaming = test.streaming_adj(fhat)
    print(f'Ref time is: {time.time() - start_time}')
    test_result = test.test_adjoint(fhat, fhat_poststreaming, '"Streaming"', test.streaming, f)
    return test_result

def unit_test3(**kwargs):
    # TEST 3: Collide-then-Stream
    test, f, fhat = init_unit_test(**kwargs)
    start_time = time.time()
    fhat_poststreaming = test.collision_adj(f, test.streaming_adj(fhat))
    print(f'Ref time is: {time.time() - start_time}')
    def lbm_step(f):
        return test.streaming(test.collision(f))
    test_result = test.test_adjoint(fhat, fhat_poststreaming, '"Collide-then-Stream"', lbm_step, f)
    return test_result

def unit_test4(**kwargs):
    # TEST 4:apply post-streaming BC after stream
    test, f, fhat = init_unit_test(**kwargs)
    cylinder, cylinder_BB, bottomWall, topWall, leftWall, rightWall, leftInlet, rightOutlet = test.BCs
    test.BCs = [bottomWall, topWall, leftWall, rightWall]
    start_time = time.time()
    fhat_poststreaming = test.streaming_adj(fhat)
    fhat_poststreaming = test.apply_bc_adj(fhat_poststreaming, fhat, "PostStreaming")
    print(f'Ref time is: {time.time() - start_time}')
    def lbm_step_bc(f):
        f_poststreaming = test.streaming(f)
        f_poststreaming = test.apply_bc(f_poststreaming, f, timestep, "PostStreaming")
        return f_poststreaming
    test_result = test.test_adjoint(fhat, fhat_poststreaming, '"Stream with halfway BB"', lbm_step_bc, f)
    return test_result

def unit_test5(**kwargs):
    # TEST 5: Apply post-streaming boudary condition after BGK Collision & Streaming
    test, f, fhat = init_unit_test(**kwargs)
    cylinder, cylinder_BB, bottomWall, topWall, leftWall, rightWall, leftInlet, rightOutlet = test.BCs
    test.BCs = [bottomWall, topWall, leftWall, rightWall]
    start_time = time.time()
    fhat_poststreaming = test.step_adjoint_noCollision(f, fhat)
    print(f'Ref time is: {time.time() - start_time}')
    test_result = test.test_adjoint(fhat, fhat_poststreaming, '"BGK collide-stream with halfway BB"',
                                    test.lbm_step_complete, f)
    return test_result

def unit_test6(**kwargs):
    # TEST 6: Apply post-streaming BC after stream
    test, f, fhat = init_unit_test(**kwargs)
    cylinder, cylinder_BB, bottomWall, topWall, leftWall, rightWall, leftInlet, rightOutlet = test.BCs
    test.BCs = [bottomWall, topWall, rightWall, leftInlet]
    start_time = time.time()
    fhat_postcollision = fhat
    fhat_poststreaming = fhat
    fhat_postcollision = test.apply_bc_adj(fhat_postcollision, fhat_poststreaming, "PostCollision")
    fhat_poststreaming = test.streaming_adj(fhat_postcollision)
    fhat_poststreaming = test.apply_bc_adj(fhat_poststreaming, fhat_postcollision, "PostStreaming")
    print(f'Ref time is: {time.time() - start_time}')
    def lbm_step_bc(f):
        f_poststreaming = test.streaming(f)
        f_poststreaming = test.apply_bc(f_poststreaming, f, timestep, "PostStreaming")
        return f_poststreaming
    test_result = test.test_adjoint(fhat, fhat_poststreaming,
                                    '"Stream with halfway BB and ZouHe Velocity BC"', lbm_step_bc, f)
    return test_result

def unit_test7(**kwargs):
    # TEST 7: Apply post-streaming BC after stream
    test, f, fhat = init_unit_test(**kwargs)
    cylinder, cylinder_BB, bottomWall, topWall, leftWall, rightWall, leftInlet, rightOutlet = test.BCs
    test.BCs = [bottomWall, topWall, rightOutlet, leftInlet]
    start_time = time.time()
    fhat_postcollision = fhat
    fhat_poststreaming = fhat
    fhat_postcollision = test.apply_bc_adj(fhat_postcollision, fhat_poststreaming, "PostCollision")
    fhat_poststreaming = test.streaming_adj(fhat_postcollision)
    fhat_poststreaming = test.apply_bc_adj(fhat_poststreaming, fhat_postcollision, "PostStreaming")
    print(f'Ref time is: {time.time() - start_time}')
    def lbm_step_bc(f):
        f_poststreaming = test.streaming(f)
        f_poststreaming = test.apply_bc(f_poststreaming, f, timestep, "PostStreaming")
        return f_poststreaming
    test_result = test.test_adjoint(fhat, fhat_poststreaming,
                                    '"Stream with halfway BB and ZouHe Velocity and Pressure BC"', lbm_step_bc, f)
    return test_result

def unit_test8(**kwargs):
    # TEST 8:
    test, f, fhat = init_unit_test(**kwargs)
    cylinder, cylinder_BB, bottomWall, topWall, leftWall, rightWall, leftInlet, rightOutlet = test.BCs
    test.BCs = [bottomWall, topWall, rightOutlet, leftInlet]
    start_time = time.time()
    fhat_poststreaming = test.step_adjoint_complete(f, fhat)
    print(f'Ref time is: {time.time() - start_time}')
    test_result = test.test_adjoint(fhat, fhat_poststreaming,
                                    '"BGK collide-stream with halfway BB and ZouHe pressure and Velocity"',
                                    test.lbm_step_complete, f)
    return test_result

def unit_test9(**kwargs):
    # TEST 9:
    test, f, fhat = init_unit_test(**kwargs)
    cylinder, cylinder_BB, bottomWall, topWall, leftWall, rightWall, leftInlet, rightOutlet = test.BCs
    test.BCs = [bottomWall, topWall, leftWall, rightWall, cylinder_BB]

    # This is necessary to ensure fhat streamed from outside of fluid domain does not contribute to adjoint BC
    # which AD assumes incorrectly and by default when those solid voxels are inside the array and not at the matrix
    # boundaries
    fhat, _ = discard_solids(test.BCs, fhat, fhat*0.0)

    start_time = time.time()
    fhat_poststreaming = test.step_adjoint_complete(f, fhat)
    print(f'Ref time is: {time.time() - start_time}')
    test_result = test.test_adjoint(fhat, fhat_poststreaming,
                                    '"LBM full step with half-way BB also in the interior of the domain!"',
                                    test.lbm_step_complete, f)
    return test_result

def unit_test10(**kwargs):
    # TEST 10:
    test, f, fhat = init_unit_test(**kwargs)
    cylinder, cylinder_BB, bottomWall, topWall, leftWall, rightWall, leftInlet, rightOutlet = test.BCs
    test.BCs = [bottomWall, topWall, leftWall, rightWall, cylinder]

    # This is necessary to ensure fhat streamed from outside of fluid domain does not contribute to adjoint BC
    # which AD assumes incorrectly and by default when those solid voxels are inside the array and not at the matrix
    # boundaries
    fhat, _ = discard_solids(test.BCs, fhat, fhat*0.0)

    start_time = time.time()
    fhat_poststreaming = test.step_adjoint_complete(f, fhat)
    print(f'Ref time is: {time.time() - start_time}')
    test_result = test.test_adjoint(fhat, fhat_poststreaming,
                                    '"LBM full step with interp-BB also in the interior of the domain!"',
                                    test.lbm_step_complete, f)
    return test_result


def unit_test11(**kwargs):
    # TEST 11:
    test, f, fhat = init_unit_test(**kwargs)
    cylinder, cylinder_BB, bottomWall, topWall, leftWall, rightWall, leftInlet, rightOutlet = test.BCs
    test.BCs = [bottomWall, topWall, leftInlet, rightOutlet, cylinder]
    start_time = time.time()

    # backward
    fhat = vjp(test.apply_bc, f, f, timestep, None)[1](fhat)[1]
    fhat_poststreaming = vjp(test.streaming, f)[1](fhat)[0]
    fhat_poststreaming = vjp(test.apply_bc, f, f, timestep, None)[1](fhat_poststreaming)[0]
    fhat_poststreaming = vjp(test.collision, f)[1](fhat_poststreaming)[0]

    print(f'Ref time is: {time.time() - start_time}')
    test_result = test.test_adjoint(fhat, fhat_poststreaming,
                                    'Testing construction of individual vjp calls to forward functions',
                                    test.lbm_step_complete, f)
    return test_result


def unit_test12(**kwargs):
    # TEST 12: Testing addition of objective function
    test, f, fhat = init_unit_test(**kwargs)
    cylinder, cylinder_BB, bottomWall, topWall, leftWall, rightWall, leftInlet, rightOutlet = test.BCs
    test.BCs = [bottomWall, topWall, leftInlet, rightOutlet]

    def objective_func(sdf, fpop):
        # rho, _ = test.update_macroscopic(fpop)
        inlet = test.BCs[2]
        outlet = test.BCs[3]
        pout = np.mean(outlet.prescribed) / 3.0
        pin = np.mean(inlet.calculate_rho(fpop, inlet.prescribed)) / 3.0
        prd = pin - pout
        return prd

    # backward
    start_time = time.time()
    fhat_ref = test.step_adjoint_complete(f, fhat)
    bc = test.BCs[2]
    unormal = jnp.sum(bc.normals * bc.prescribed, keepdims=True, axis=-1)
    grad = 2.0 / 3.0 / (1.0 + unormal) * jnp.ones((1, test.q))
    grad = grad.at[bc.imissingMask].set(0.0)
    grad = grad.at[bc.imiddleMask].divide(2.0)
    grad /= unormal.shape[0]
    fhat_ref = fhat_ref.at[bc.indices].add(-grad)
    print(f'Ref time is: {time.time() - start_time}')

    # Construct the adjoints for the forward model.
    obj_vjp = vjp(objective_func, 0.0, f)[1]
    step_vjp = vjp(test.lbm_step_complete, f)[1]
    fhat_AD = step_vjp(fhat)[0]
    fhat_AD = fhat_AD - obj_vjp(1.0)[1]

    # Discard the dissimilarity at the solid voxels
    fhat_ref, fhat_AD = discard_solids(test.BCs, fhat_ref, fhat_AD)

    flag = np.allclose(fhat_ref, fhat_AD, tol, tol)
    print('=' * 50)
    if flag:
        print(f'**** PASSED **** unit test for {func_name} up to tol={tol}')
    else:
        print(f'!!!! FAILED !!!! unit test for {func_name} up to tol={tol}')
    print('=' * 50)

    return


if __name__ == "__main__":
    precision = "f64/f64"
    lattice = LatticeD2Q9(precision)

    # Input test parameters
    nx, ny, nz = 80, 50, 0
    tol = 1e-9
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

    unit_test_list = {
        'collision': unit_test1,
        'streaming': unit_test2,
        'C + S'    : unit_test3,
        'S + BB'   : unit_test4,
        'C + S + BB': unit_test5,
        'S + BB + ZouHe_vel': unit_test6,
        'S + BB + ZouHe_vel + ZouHePress': unit_test7,
        'C + S + BB + ZouHe_vel + ZouHePress': unit_test8,
        'C + S + BB with BC configuration=True': unit_test9,
        'C + S + IBB': unit_test10,
        'Individual vjp calls': unit_test11,
        'pressure drop obj func': unit_test12,
    }

    for test_name, func_name in unit_test_list.items():
        test_result = func_name(**kwargs)
