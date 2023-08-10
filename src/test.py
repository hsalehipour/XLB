import numpy as np
import jax.numpy as jnp
from jax import vjp, vmap

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
    nx, ny, nz = fhat.shape[:-1]

    # adjoint density
    rho_adj = density_adj(fhat, feq, rho)

    # adjoint momentum
    mhat = np.zeros((nx, ny, nz, dim))
    umhat = np.zeros((nx, ny, nz))
    feq_adj = np.zeros((nx, ny, nz, q))
    cu = jnp.dot(vel, c)

    for d in range(dim):
        for i in range(q):
            mhat[..., d] += fhat[..., i] * w[i] * \
                            (c[d, i] + 3.0 * (c[d, i] * cu[..., i] - vel[..., d] / 3.0))
        umhat += vel[..., d] * mhat[..., d]

    cmhat = np.dot(mhat, c)
    for i in range(q):
        feq_adj[..., i] = rho_adj + 3.0 * (cmhat[..., i] - umhat)

    return feq_adj


def streaming(f, c):
    def streaming_i(f, c):
        if dim == 2:
            return jnp.roll(f, (c[0], c[1]), axis=(0, 1))
        elif dim == 3:
            return jnp.roll(f, (c[0], c[1], c[2]), axis=(0, 1, 2))
    return vmap(streaming_i, in_axes=(-1, 0), out_axes=-1)(f, c.T)

def streaming_adj_math(f, c):
    fhat_rolled = np.zeros_like(f)
    for i in range(q):
        fhat_rolled[..., i] = np.roll(f[..., i], (-c[0, i], -c[1, i], -c[2, i]), axis=(0, 1, 2))
    return fhat_rolled

nx, ny, nz = 5, 7, 3
f = jnp.array(np.random.random((nx, ny, nz, q)))
fhat = jnp.array(np.random.random((nx, ny, nz, q)))
tol = 1e-6


# Equilibrium
feq = equilibrium(f)
_, equilibrium_adj = vjp(equilibrium, f)
rho, vel = update_macroscopic(f)
fhat_eq_math = equilibrium_adj_math(fhat, feq, rho, vel)
fhat_eq_AD = equilibrium_adj(fhat)[0]
if np.allclose(fhat_eq_math, fhat_eq_AD, tol, tol):
    print(f'PASSED unit test for adjoint equilibrium up to tol={tol}')
else:
    print(f'FAILED unit test for adjoint equilibrium up to tol={tol}')

# Streaming
_, streaming_adj = vjp(streaming, f, c)
fhat_streamed_math = streaming_adj_math(fhat, c)
fhat_streamed_AD = streaming_adj(fhat)[0]
if np.allclose(fhat_streamed_math, fhat_streamed_AD, tol, tol):
    print(f'PASSED unit test for adjoint streaming up to tol={tol}')
else:
    print(f'FAILED unit test for adjoint streaming up to tol={tol}')



