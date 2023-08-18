import jax.numpy as jnp
import numpy as np
from jax import jit, vjp
from jax.sharding import PartitionSpec
from jax.experimental.shard_map import shard_map
from functools import partial
from src.base import LBMBase
"""
Collision operators are defined in this file for different models.
"""

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

    def construct_adjoints(self, fin, fout, timestep):
        """
        Construct the adjoints for the forward model.
        """
        # _, self.collision_adj = vjp(self.collision, fin)
        # _, self.apply_bc_adj = vjp(self.apply_bc, fout, fin, timestep, 'PostCollision')
        # _, compute_J_adj = vjp(self.compute_J, fout, phi, self.Js)
        return

    @partial(jit, static_argnums=(0,))
    def step_adjoint(self, fhat):
        """
        This function performs a single step of the adjoint LBM simulation.

        It first performs the adjoint streaming step, which is the inverse propagation of the distribution
        functions in the lattice. It then applies the respective boundary conditions to the post-streaming
        distribution functions.

        The function then performs adjoint collision step, which is similar to the relaxation of the forward
        distribution functions towards and adjoint equilibrium state. It then applies the respective boundary
        conditions to the post-collision distribution functions.

        Parameters
        ----------
        fhat: jax.numpy.ndarray
            The adjoint distribution functions.

        Returns
        -------
        fhat: jax.numpy.ndarray
            The adjoint distribution functions after the adjoint simulation step.
        """
        fhat = self.streaming_adj(fhat)
        fhat = self.apply_bc_adj(fhat)[1]
        fhat = self.collision_adj(fhat)[0]
        fhat = self.apply_bc_adj(fhat)[1]
        return fhat