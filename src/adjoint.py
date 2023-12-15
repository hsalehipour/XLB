import jax.numpy as jnp
from termcolor import colored
from jax import jit, vjp
from jax.sharding import PartitionSpec
from jax.experimental.shard_map import shard_map
from functools import partial
from src.base import LBMBase


class LBMBaseDifferentiable(LBMBase):
    """
    Same as LBMBase class but with added adjoint capabilities either through manual computation or leveraging AD.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.dim == 2:
            inout_specs = PartitionSpec("x", None, None)
        elif self.dim == 3:
            inout_specs = PartitionSpec("x", None, None, None)

        self.streaming_adj = jit(shard_map(self.streaming_adj_m, mesh=self.mesh,
                                           in_specs=inout_specs, out_specs=inout_specs, check_rep=False))

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
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
        return jnp.sum(feq*fhat, axis=-1, keepdims=True)/rho

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

    def construct_adjoints(self, sdf, fpop):
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
        _, self.step_vjp = vjp(self.step, fpop, 0., False)
        _, self.objective_vjp = vjp(self.objective, sdf, fpop)
        return

    @partial(jit, static_argnums=(0,))
    def step_adjoint(self, fhat):
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
        fhat = self.step_vjp((fhat, None))[0]
        fhat = fhat - self.objective_vjp(1.0)[1]
        return fhat

    def run_adjoint(self, fpop, t_max):
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

        # construct gradients of needed function for performing adjoint computations
        sdf = self.sdf.array
        self.construct_adjoints(sdf, fpop)

        # Loop over all time steps
        start_step = 0
        for timestep in range(start_step, t_max + 1):
            print_iter_flag = self.printInfoRate > 0 and timestep % self.printInfoRate == 0

            # Perform one time-step (collision, streaming, and boundary conditions)
            fhat = self.step_adjoint(fhat)

            # Print the progress of the simulation
            if print_iter_flag:
                print(
                    colored("Timestep ", 'blue') + colored(f"{timestep}", 'green') + colored(" of ", 'blue') + colored(
                        f"{t_max}", 'green') + colored(" completed", 'blue'))


        return fhat