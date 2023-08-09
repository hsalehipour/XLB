import numpy as np
import jax.numpy as jnp
from jax import vjp

dim = 3
q = 19
c = [(x, y, z) for x in [0, -1, 1] for y in [0, -1, 1] for z in [0, -1, 1]]
c = np.array([ci for ci in c if np.linalg.norm(ci) < 1.5]).T

# Initialize the weights to be 1/36
w = 1.0 / 36.0 * np.ones(q)
w[np.linalg.norm(c, axis=0) < 1.1] = 2.0 / 36.0
w[0] = 1.0 / 3.0


def update_macroscopic(f):
    rho = jnp.sum(f, axis=-1)
    u = jnp.dot(f, c.T) / rho[..., None]
    return rho, u

def equilibrium(f):
    # Cast c to compute precision so that XLA call FXX matmul,
    # which is faster (it is faster in some older versions of JAX, newer versions are smart enough to do this automatically)
    rho, u = update_macroscopic(f)
    cu = 3.0 * jnp.dot(u, c)
    usqr = 1.5 * jnp.sum(jnp.square(u), axis=-1, keepdims=True)
    feq = rho[..., None] * w * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)
    return feq

def density_adj(fhat, feq, rho):
    """ adjoint density """
    return jnp.sum(feq*fhat, axis=-1)/rho

def equilibrium_adj_math(fhat, feq, rho, vel):
    """
    Adjoint Equilibrium distribution function.
    """
    ntot = fhat.shape[0]

    # adjoint density
    rho_adj = density_adj(fhat, feq, rho)

    # adjoint momentum
    mhat = np.zeros((ntot, dim))
    umhat = np.zeros(ntot)
    feq_adj = np.zeros((ntot, q))
    cu = jnp.dot(vel, c)

    for d in range(dim):
        for i in range(q):
            mhat[:, d] += fhat[:, i] * w[i] * \
                          (c[d, i] + 3.0 * (c[d, i] * cu[:, i] - vel[:, d] / 3.0))
        umhat += vel[:, d] * mhat[:, d]

    cmhat = np.dot(mhat, c)
    for i in range(q):
        feq_adj[:, i] = rho_adj + 3.0 * (cmhat[:, i] - umhat)

    return feq_adj


ncell = 10
f = jnp.array(np.random.random((ncell, q)))
fhat = jnp.array(np.random.random((ncell, q)))
feq = equilibrium(f)
_, equilibrium_adj = vjp(equilibrium, f)
rho, vel = update_macroscopic(f)
fhat_eq_math = equilibrium_adj_math(fhat, feq, rho, vel)
fhat_eq_AD = equilibrium_adj(fhat)[0]
print(np.allclose(fhat_eq_math, fhat_eq_AD, 1e-6, 1e-6))

