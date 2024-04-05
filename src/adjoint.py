import jax.numpy as jnp
from termcolor import colored
from jax import jit, vjp
from jax.sharding import PartitionSpec
from jax.experimental.shard_map import shard_map
from functools import partial
from src.base import LBMBase
from src.models import BGKSim, KBCSim
from src.utils import save_fields_vtk
import numpy as np


class LBMBaseDifferentiable(KBCSim):
    """
    Same as LBMBase class but with added adjoint capabilities either through manual computation or leveraging AD.
    # Currently we have 2 methods of introducing level-set field into the TO pipeleine:
    #   v1: through collision operator using tanh function
    #   v2: through a differentiable iinterpolation bc 
    %TODO: need to fix this: BGK with D3Q19 cannot be handled because LBMBaseDifferentiable inherits from KBCSim
    """

    def __init__(self, sdf, **kwargs):
        # set the SDFGrid objetc
        self.sdf = sdf

        # get the TO method
        self.TO_method = kwargs.setdefault('TO_method', 'v1')

        # call the parent class
        super().__init__(**kwargs)

        # create keepin mask
        keepin_mask = jnp.full(shape=(self.nx, self.ny, self.nz), dtype = jnp.bool_, fill_value=False)
        for keepin in kwargs.get('keepins'):
            keepin_mask = keepin_mask.at[keepin.array[..., 0] <= 0.0].set(True)
        self.keepin_mask = keepin_mask
        
        # get the collision method:
        self.collision_model = kwargs.setdefault('collision_model', 'kbc')
    
        if self.dim == 2:
            inout_specs = PartitionSpec("x", None, None)
        elif self.dim == 3:
            inout_specs = PartitionSpec("x", None, None, None)

        self.streaming_adj = jit(shard_map(self.streaming_adj_m, mesh=self.mesh,
                                           in_specs=inout_specs, out_specs=inout_specs, check_rep=False))

    @partial(jit, static_argnums=(0,))
    def add_design_variable_effect(self, u, sdf_array):
        eta = 0.5 - 0.5*jnp.tanh(sdf_array/ (0.5*self.sdf.spacing))
        eta = eta.at[self.keepin_mask].set(1.0)
        return u*eta[..., None]    
        
    def get_solid_voxels(self):
        # Accumulate the indices of all BCs to create the grid mask with FALSE along directions that
        # stream into a boundary voxel.
        solid_list = [np.array(bc.indices).T for bc in self.BCs if bc.isSolid]
        solid_voxels = np.unique(np.vstack(solid_list), axis=0) if solid_list else None

        # add external solid walls to the wall indices
        # Note: the SDF data structure is that of SDFGrid class in doppler
        if hasattr(self, "sdf"):
            voxel_coordinates = self.sdf.voxel_grid_coordinates()
            solid_wall = self.sdf.array > 0.0
            solid_wall_cord = voxel_coordinates[solid_wall[..., 0], :]
            solid_wall_indices = self.sdf.index_from_coord(solid_wall_cord)
            if solid_voxels is None:
                return solid_wall_indices
            else:
                return np.unique(np.vstack([solid_voxels, solid_wall_indices]), axis=0)

    def objective(self, sdf, fpop):
        """
        Define the objective function

        Parameters
        ----------
        sdf : SDFGrid to evaluate the objective on
        fpop: state variable in LBM

        Returns
        -------
        Objective function value
        """
        raise NotImplementedError

    def streaming_adj_m(self, f):
        """
        This function performs the adjoint streaming step in the Lattice Boltzmann Method and propagates
        the distribution functions in the opposite of lattice directions.

        To enable multi-GPU/TPU functionality, it extracts the left and right boundary slices of the
        distribution functions that need to be communicated to the neighboring processes.

        The function then sends the left boundary slice to the left neighboring process and the right
        boundary slice to the right neighboring process. The received data is then set to the
        corresponding indices in the receiving domain.

        Parameters
        ----------
        f: jax.numpy.ndarray
            The array holding the adjoint distribution functions for the simulation.

        Returns
        -------
        jax.numpy.ndarray
            The adjoint distribution functions after the adjoint streaming operation.
        """
        f = self.streaming_p(f, -self.c)
        left_comm, right_comm = f[:1, ..., self.lattice.left_indices], f[-1:, ..., self.lattice.right_indices]

        left_comm, right_comm = self.send_right(left_comm, 'x'), self.send_left(right_comm, 'x')
        f = f.at[:1, ..., self.lattice.left_indices].set(left_comm)
        f = f.at[-1:, ..., self.lattice.right_indices].set(right_comm)
        return f

    def density_adj(self, fhat, feq, rho):
        """ adjoint density """
        return jnp.sum(feq * fhat, axis=-1, keepdims=True) / rho

    @partial(jit, static_argnums=(0,))
    def equilibrium_adj_math(self, fhat, feq, rho, vel):
        """
        Adjoint Equilibrium distribution function.
        """
        dim = self.dim
        q = self.q
        c = self.c
        w = self.w

        # adjoint density
        rho_adj = self.density_adj(fhat, feq, rho)

        # adjoint momentum
        mhat = jnp.zeros_like(vel)
        umhat = jnp.zeros_like(rho)
        cu = jnp.dot(vel, c)

        for d in range(dim):
            for i in range(q):
                val = fhat[..., i] * w[i] * (c[d, i] + 3.0 * (c[d, i] * cu[..., i] - vel[..., d] / 3.0))
                mhat = mhat.at[..., d].add(val)
            umhat += jnp.expand_dims(vel[..., d] * mhat[..., d], -1)

        cmhat = jnp.dot(mhat, c)
        feq_adj = rho_adj + 3.0 * (cmhat - umhat)
        return feq_adj

    def construct_adjoints(self, sdf_array, fpop):
        """
        Construct the adjoints for the forward model.
        ================================
        Note:
        ================================
        fhat = vjp(self.step, fpop, 0., False)[1](fhat)[0]

        is equivalent to:

        def step_vjp(fhat)
            fhat = vjp(test.apply_bc, f, f, timestep, None)[1](fhat)[1]
            fhat_poststreaming = vjp(test.streaming, f)[1](fhat)[0]
            fhat_poststreaming = vjp(test.apply_bc, f, f, timestep, None)[1](fhat_poststreaming)[0]
            fhat_poststreaming = vjp(test.collision, f)[1](fhat_poststreaming)[0]
            return fhat_poststreaming
        ================================
        """
        _, step_vjp = vjp(self.step, fpop, 0., sdf_array, False)
        _, objective_vjp = vjp(self.objective, sdf_array, fpop)
        return step_vjp, objective_vjp

    @partial(jit, static_argnums=(0,))
    def step_adjoint(self, fhat, step_vjp, objective_vjp):
        """
        This function performs a single step of the adjoint LBM simulation.

        It first performs applies the respective boundary conditions to the post-collision
        distribution functions. Then it performs adjoint streaming step, which is the inverse propagation of the
        distribution functions in the lattice. Again it applies the respective boundary
        conditions to the post-streaming distribution functions. Finally it performs adjoint collision step, which
        is similar to the relaxation of the forward distribution functions towards and adjoint equilibrium state.
        Parameters
        ----------
        fhat: jax.numpy.ndarray
            The adjoint distribution functions.

        Returns
        -------
        fhat: jax.numpy.ndarray
            The adjoint distribution functions after the adjoint simulation step.
        """
        fhat = step_vjp((fhat, None))[0]
        fhat = fhat + objective_vjp(1.0)[1]
        return fhat

    def run_adjoint(self, step_vjp, objective_vjp, t_max):
        """
        This function runs the adjoint LBM simulation for a specified number of time steps.

        It first initializes the adjoint distribution functions and then enters a loop where it performs the
        adjoint simulation steps (collision, streaming, and boundary conditions) for each time step.

        The function can also print the progress of the simulation, save the simulation data, and
        compute the performance of the simulation in million lattice updates per second (MLUPS).

        Parameters
        ----------
        t_max: int
            The total number of time steps to run the simulation.
        Returns
        -------
        fhat: jax.numpy.ndarray
            The distribution functions after the simulation.
        """
        if self.dim == 2:
            shape = (self.nx, self.ny, self.lattice.q)
        elif self.dim == 3:
            shape = (self.nx, self.ny, self.nz, self.lattice.q)
        else:
            raise NotImplemented
        fhat = self.distributed_array_init(shape, self.precisionPolicy.output_dtype, init_val=0.0)

        # Loop over all time steps
        start_step = 0
        for timestep in range(start_step, t_max + 1):
            io_flag = self.ioRate > 0 and (timestep % self.ioRate == 0 or timestep == t_max)
            print_iter_flag = self.printInfoRate > 0 and timestep % self.printInfoRate == 0

            # Perform one time-step (collision, streaming, and boundary conditions)
            fhat = self.step_adjoint(fhat, step_vjp, objective_vjp)

            # Print the progress of the simulation
            if print_iter_flag:
                print(
                    colored("Timestep ", 'blue') + colored(f"{timestep}", 'green') + colored(" of ", 'blue') + colored(
                        f"{t_max}", 'green') + colored(" completed", 'blue'))

            # if io_flag:
            #     # Save the simulation data
            #     print(f"Saving data at timestep {timestep}/{t_max}")
            #     rho_adj = jnp.sum(fhat, axis=-1, keepdims=True)
            #     c = jnp.array(self.c, dtype=self.precisionPolicy.compute_dtype).T
            #     u = jnp.dot(fhat, c)
            #     lbm_shapeDerivative = step_vjp((fhat, None))[2]
            #     fields = {"rho": rho_adj[..., 0],
            #               "u_x": u[..., 0], "u_y": u[..., 1], "u_z": u[..., 2], "umag": np.sqrt(u[..., 0]**2+u[..., 1]**2+u[..., 2]**2),
            #               "shape_derivative": lbm_shapeDerivative}
            #     save_fields_vtk(timestep, fields, prefix='adjfields')

        return fhat
    

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, f, sdf_array):
        if self.collision_model == 'bgk':
            return self.collision_bgk(f, sdf_array)
        else:
            return self.collision_kbc(f, sdf_array)

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision_bgk(self, f, sdf_array):
        """
        BGK collision step for lattice.

        The collision step is where the main physics of the LBM is applied. In the BGK approximation, 
        the distribution function is relaxed towards the equilibrium distribution function.
        """
        f = self.precisionPolicy.cast_to_compute(f)
        rho, u = self.update_macroscopic(f)
        if self.TO_method == 'v1':
            u = self.add_design_variable_effect(u, sdf_array)
        feq = self.equilibrium(rho, u, cast_output=False)
        fneq = f - feq
        fout = f - self.omega * fneq
        if self.force is not None:
            fout = self.apply_force(fout, feq, rho, u)
        return self.precisionPolicy.cast_to_output(fout)

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision_kbc(self, f, sdf_array):
        """
        Alternative KBC collision step for lattice.
        Note: 
        At low Reynolds number the orignal KBC collision above produces inaccurate results because
        it does not check for the entropy increase/decrease. The KBC stabalizations should only be 
        applied in principle to cells whose entropy decrease after a regular BGK collision. This is 
        the case in most cells at higher Reynolds numbers and hence a check may not be needed. 
        Overall the following alternative collision is more reliable and may replace the original 
        implementation. The issue at the moment is that it is about 60-80% slower than the above method.
        """
        f = self.precisionPolicy.cast_to_compute(f)
        tiny = 1e-32
        beta = self.omega * 0.5
        rho, u = self.update_macroscopic(f)
        if self.TO_method == 'v1':
            u = self.add_design_variable_effect(u, sdf_array)
        feq = self.equilibrium(rho, u, cast_output=False)

        # Alternative KBC: only stabalizes for voxels whose entropy decreases after BGK collision.
        f_bgk = f - self.omega * (f - feq)
        H_fin = jnp.sum(f * jnp.log(f / self.w), axis=-1, keepdims=True)
        H_fout = jnp.sum(f_bgk * jnp.log(f_bgk / self.w), axis=-1, keepdims=True)

        # the rest is identical to collision_deprecated
        fneq = f - feq
        if self.dim == 2:
            deltaS = self.fdecompose_shear_d2q9(fneq) * rho / 4.0
        else:
            deltaS = self.fdecompose_shear_d3q27(fneq) * rho
        deltaH = fneq - deltaS
        invBeta = 1.0 / beta
        gamma = invBeta - (2.0 - invBeta) * self.entropic_scalar_product(deltaS, deltaH, feq) / (tiny + self.entropic_scalar_product(deltaH, deltaH, feq))

        f_kbc = f - beta * (2.0 * deltaS + gamma * deltaH)
        fout = jnp.where(H_fout > H_fin, f_kbc, f_bgk)

        # add external force
        if self.force is not None:
            fout = self.apply_force(fout, feq, rho, u)
        return self.precisionPolicy.cast_to_output(fout)