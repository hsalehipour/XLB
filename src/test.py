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

def streaming_adj_math(fhat, c):
    def streaming_i(fhat, c):
        if dim == 2:
            return jnp.roll(fhat, (-c[0], -c[1]), axis=(0, 1))
        elif dim == 3:
            return jnp.roll(fhat, (-c[0], -c[1], -c[2]), axis=(0, 1, 2))
    return vmap(streaming_i, in_axes=(-1, 0), out_axes=-1)(fhat, c.T)

def collision(f):
    omega = 1.0
    feq = equilibrium(f)
    fneq = f - feq
    fout = f - omega * fneq
    return fout

def collision_adj_math(fhat, feq, rho, vel):
    omega = 1.0
    feq_adj = equilibrium_adj_math(fhat, feq, rho, vel)
    fneq_adj = fhat - feq_adj
    fhat = fhat - omega * fneq_adj
    return fhat

def collision_and_stream(f):
    f = collision(f)
    f = streaming(f, c)
    return f

def collision_and_stream_adj_math(fhat, feq, rho, vel):
    fhat = streaming_adj_math(fhat, c)
    fhat = collision_adj_math(fhat, feq, rho, vel)
    return fhat



# Input test parameters
nx, ny, nz = 5, 7, 3
fpop = jnp.array(np.random.random((nx, ny, nz, q)))
fhat = jnp.array(np.random.random((nx, ny, nz, q)))
tol = 1e-6


# Equilibrium
feq = equilibrium(fpop)
_, equilibrium_adj = vjp(equilibrium, fpop)
rho, vel = update_macroscopic(fpop)
fhat_eq_math = equilibrium_adj_math(fhat, feq, rho, vel)
fhat_eq_AD = equilibrium_adj(fhat)[0]
if np.allclose(fhat_eq_math, fhat_eq_AD, tol, tol):
    print(f'PASSED unit test for adjoint equilibrium up to tol={tol}')
else:
    print(f'FAILED unit test for adjoint equilibrium up to tol={tol}')

# Streaming
_, streaming_adj = vjp(streaming, fpop, c)
fhat_poststreaming_math = streaming_adj_math(fhat, c)
fhat_poststreaming_AD = streaming_adj(fhat)[0]
if np.allclose(fhat_poststreaming_math, fhat_poststreaming_AD, tol, tol):
    print(f'PASSED unit test for adjoint streaming up to tol={tol}')
else:
    print(f'FAILED unit test for adjoint streaming up to tol={tol}')

# Collide and Streaming sequence of operations
_, collision_adj = vjp(collision, fpop)
fhat_postcollision_math = collision_adj_math(fhat, feq, rho, vel)
fhat_postcollision_AD = collision_adj(fhat)[0]
if np.allclose(fhat_postcollision_math, fhat_postcollision_AD, tol, tol):
    print(f'PASSED unit test for adjoint collision up to tol={tol}')
else:
    print(f'FAILED unit test for adjoint collision up to tol={tol}')
    


# Collide and Streaming sequence of operations
_, collision_and_stream_adj = vjp(collision_and_stream, fpop)
fhat_math = collision_and_stream_adj_math(fhat, feq, rho, vel)
fhat_AD = collision_and_stream_adj(fhat)[0]
if np.allclose(fhat_math, fhat_AD, tol, tol):
    print(f'PASSED unit test for adjoint collide-stream up to tol={tol}')
else:
    print(f'FAILED unit test for adjoint collide-stream up to tol={tol}')



