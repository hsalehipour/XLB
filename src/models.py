import jax.numpy as jnp
from jax import jit
from functools import partial
from src.base import LBMBase
"""
Collision operators are defined in this file for different models.
"""

class BGKSim(LBMBase):
    """
    BGK simulation class.

    This class implements the Bhatnagar-Gross-Krook (BGK) approximation for the collision step in the Lattice Boltzmann Method.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, f, sdf=None):
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

class KBCSim(LBMBase):
    """
    KBC simulation class.

    This class implements the Karlin-Bösch-Chikatamarla (KBC) model for the collision step in the Lattice Boltzmann Method.
    """
    def __init__(self, **kwargs):
        if kwargs.get('lattice').name != 'D3Q27' and kwargs.get('nz') > 0:
            raise ValueError("KBC collision operator in 3D must only be used with D3Q27 lattice.")
        super().__init__(**kwargs)

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, f, sdf=None):
        """
        KBC collision step for lattice.
        """
        f = self.precisionPolicy.cast_to_compute(f)
        tiny = 1e-32
        beta = self.omega * 0.5
        rho, u = self.update_macroscopic(f)
        feq = self.equilibrium(rho, u, cast_output=False)
        fneq = f - feq
        if self.dim == 2:
            deltaS = self.fdecompose_shear_d2q9(fneq) * rho / 4.0
        else:
            deltaS = self.fdecompose_shear_d3q27(fneq) * rho
        deltaH = fneq - deltaS
        invBeta = 1.0 / beta
        gamma = invBeta - (2.0 - invBeta) * self.entropic_scalar_product(deltaS, deltaH, feq) / (tiny + self.entropic_scalar_product(deltaH, deltaH, feq))

        fout = f - beta * (2.0 * deltaS + gamma * deltaH)

        # add external force
        if self.force is not None:
            fout = self.apply_force(fout, feq, rho, u)
        return self.precisionPolicy.cast_to_output(fout)
    
    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision_modified(self, f):
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


class AdvectionDiffusionBGK(LBMBase):
    """
    Advection Diffusion Model based on the BGK model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vel = kwargs.get("vel", None)
        if self.vel is None:
            raise ValueError("Velocity must be specified for AdvectionDiffusionBGK.")

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, f):
        """
        BGK collision step for lattice.
        """
        f = self.precisionPolicy.cast_to_compute(f)
        rho =jnp.sum(f, axis=-1, keepdims=True)
        feq = self.equilibrium(rho, self.vel, cast_output=False)
        fneq = f - feq
        fout = f - self.omega * fneq
        return self.precisionPolicy.cast_to_output(fout)