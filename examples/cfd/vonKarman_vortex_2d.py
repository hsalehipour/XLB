import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.distribute import distribute
from xlb.operator.boundary_condition import (
    FullwayBounceBackBC,
    HalfwayBounceBackBC,
    RegularizedBC,
    ExtrapolationOutflowBC,
)
from xlb.operator.macroscopic import Macroscopic
from xlb.utils import save_image
import warp as wp
import numpy as np
import jax.numpy as jnp
import time


class vonKarmanVortex:
    def __init__(self, omega, inlet_velocity_mean, cylinder_diameter, grid_shape, velocity_set, backend, precision_policy):
        # initialize backend
        xlb.init(
            velocity_set=velocity_set,
            default_backend=backend,
            default_precision_policy=precision_policy,
        )

        self.grid_shape = grid_shape
        self.velocity_set = velocity_set
        self.backend = backend
        self.precision_policy = precision_policy
        self.omega = omega
        self.boundary_conditions = []
        self.u_max = 1.5 * inlet_velocity_mean
        self.cylinder_diameter = cylinder_diameter

        # Create grid using factory
        self.grid = grid_factory(grid_shape, compute_backend=backend)

        # Setup the simulation BC and stepper
        self._setup()

    def _setup(self):
        self.setup_boundary_conditions()
        self.setup_stepper()

    def define_boundary_indices(self):
        box = self.grid.bounding_box_indices()
        box_no_edge = self.grid.bounding_box_indices(remove_edges=True)
        inlet = box_no_edge["left"]
        outlet = box_no_edge["right"]
        walls = [box["bottom"][i] + box["top"][i] for i in range(self.velocity_set.d)]
        walls = np.unique(np.array(walls), axis=-1).tolist()

        cyliner_radius = self.cylinder_diameter // 2
        x = np.arange(self.grid_shape[0])
        y = np.arange(self.grid_shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")
        indices = np.where((X - 2.0 * self.cylinder_diameter) ** 2 + (Y - 2.0 * self.cylinder_diameter) ** 2 < cyliner_radius**2)
        cyliner = [tuple(indices[i]) for i in range(self.velocity_set.d)]

        return inlet, outlet, walls, cyliner

    def setup_boundary_conditions(self):
        inlet, outlet, walls, cyliner = self.define_boundary_indices()
        bc_left = RegularizedBC("velocity", profile=self.bc_profile(), indices=inlet)
        bc_walls = HalfwayBounceBackBC(indices=walls)
        bc_outlet = ExtrapolationOutflowBC(indices=outlet)
        bc_cyliner = HalfwayBounceBackBC(indices=cyliner)
        self.boundary_conditions = [bc_walls, bc_left, bc_outlet, bc_cyliner]

    def setup_stepper(self):
        stepper = IncompressibleNavierStokesStepper(
            omega=self.omega,
            grid=self.grid,
            boundary_conditions=self.boundary_conditions,
            collision_type="BGK",
        )
        # Distribute the stepper
        self.stepper = distribute(
            stepper,
            self.grid,
            self.velocity_set,
        )
        self.f_0, self.f_1, self.bc_mask, self.missing_mask = self.stepper.prepare_fields()

    def bc_profile(self):
        u_max = self.u_max  # u_max = 0.04
        # Get the grid dimensions for the y direction
        H_y = float(self.grid_shape[1] - 1)  # Height in y direction

        @wp.func
        def bc_profile_warp(index: wp.vec3i):
            # Poiseuille flow profile: parabolic velocity distribution
            y = self.precision_policy.store_precision.wp_dtype(index[1])

            # Calculate normalized distance from center
            y_center = y - (H_y / 2.0)
            r_squared = (2.0 * y_center / H_y) ** 2.0

            # Parabolic profile: u = u_max * (1 - r²)
            return wp.vec(u_max * wp.max(0.0, 1.0 - r_squared), length=1)

        def bc_profile_jax():
            y = jnp.arange(self.grid_shape[1])

            # Calculate normalized distance from center
            y_center = y - (H_y / 2.0)
            r_squared = (2.0 * y_center / H_y) ** 2.0

            # Parabolic profile for x velocity, zero for y and z
            u_x = u_max * jnp.maximum(0.0, 1.0 - r_squared)
            u_y = jnp.zeros_like(u_x)

            return jnp.stack([u_x, u_y])

        if self.backend == ComputeBackend.JAX:
            return bc_profile_jax
        elif self.backend == ComputeBackend.WARP:
            return bc_profile_warp

    def run(self, num_steps, post_process_interval=100):
        start_time = time.time()
        for i in range(num_steps):
            self.f_0, self.f_1 = self.stepper(self.f_0, self.f_1, self.bc_mask, self.missing_mask, i)
            self.f_0, self.f_1 = self.f_1, self.f_0

            if i % post_process_interval == 0 or i == num_steps - 1:
                self.post_process(i)
                end_time = time.time()
                print(f"Completing {i} iterations. Time elapsed for 1000 LBM steps in {end_time - start_time:.6f} seconds.")
                start_time = time.time()

    def post_process(self, i):
        # Write the results. We'll use JAX backend for the post-processing
        if not isinstance(self.f_0, jnp.ndarray):
            f_0 = wp.to_jax(self.f_0)
        else:
            f_0 = self.f_0

        macro = Macroscopic(
            compute_backend=ComputeBackend.JAX,
            precision_policy=self.precision_policy,
            velocity_set=xlb.velocity_set.D2Q9(precision_policy=self.precision_policy, backend=ComputeBackend.JAX),
        )
        rho, u = macro(f_0)
        u = u[:, :-1, :]
        rho = rho[:, :-1, :]
        u_magnitude = (u[0] ** 2 + u[1] ** 2) ** 0.5

        fields = {"u_magnitude": u_magnitude, "u_x": u[0], "u_y": u[1], "rho": rho[0]}

        # save_fields_vtk(fields, timestep=i)
        save_image(fields["u_magnitude"], timestep=i)


if __name__ == "__main__":
    # Running the simulation
    cylinder_diameter = 20
    scaling_factor = 80/cylinder_diameter
    nx = int(22 * cylinder_diameter)
    ny = int(4.1 * cylinder_diameter)
    grid_shape = (nx, ny)

    backend = ComputeBackend.JAX
    precision_policy = PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, backend=backend)

    Re = 100.0
    inlet_velocity_mean = 0.003 * scaling_factor
    visc = inlet_velocity_mean * cylinder_diameter / Re
    omega = 1.0 / (3.0 * visc + 0.5)

    simulation = vonKarmanVortex(omega, inlet_velocity_mean, cylinder_diameter, grid_shape, velocity_set, backend, precision_policy)
    simulation.run(num_steps=100000, post_process_interval=500)
