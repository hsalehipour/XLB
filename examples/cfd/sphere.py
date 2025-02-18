import xlb
import trimesh
import time
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import (
    HalfwayBounceBackBC,
    FullwayBounceBackBC,
    RegularizedBC,
    DoNothingBC,
    HybridBC,
)
from xlb.operator.force.momentum_transfer import MomentumTransfer
from xlb.operator.macroscopic import Macroscopic
from xlb.utils import save_fields_vtk, save_image
import warp as wp
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator import Operator
from typing import Any
from xlb.velocity_set.velocity_set import VelocitySet


class OutletInitializer(Operator):
    def __init__(
        self,
        wind_speed=None,
        grid_shape=None,
        velocity_set: VelocitySet = None,
        precision_policy=None,
        compute_backend=None,
    ):
        self.wind_speed = wind_speed
        self.rho = 1.0
        self.grid_shape = grid_shape
        self.equilibrium = QuadraticEquilibrium(velocity_set=velocity_set, precision_policy=precision_policy, compute_backend=compute_backend)
        super().__init__(velocity_set, precision_policy, compute_backend)

    def _construct_warp(self):
        nx, ny, nz = self.grid_shape
        _q = self.velocity_set.q
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)
        _rho = self.compute_dtype(self.rho)
        _u = _u_vec(self.wind_speed, 0.0, 0.0)
        _w = self.velocity_set.w

        # Construct the warp kernel
        @wp.kernel
        def kernel(f: wp.array4d(dtype=Any)):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # Set the velocity at the outlet (i.e. where i = nx-1)
            if index[0] == nx - 1:
                _feq = self.equilibrium.warp_functional(_rho, _u)
                for l in range(_q):
                    f[l, index[0], index[1], index[2]] = _feq[l]
            else:
                # In the rest of the domain, we assume zero velocity and equilibrium distribution.
                for l in range(_q):
                    f[l, index[0], index[1], index[2]] = _w[l]

        return None, kernel

    @Operator.register_backend(xlb.ComputeBackend.WARP)
    def warp_implementation(self, f):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                f,
            ],
            dim=f.shape[1:],
        )
        return f


class WindTunnel3D:
    def __init__(self, omega, wind_speed, grid_shape, velocity_set, backend, precision_policy):
        # initialize backend
        wp.config.max_unroll = 27
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
        self.wind_speed = wind_speed

        # Create grid using factory
        self.grid = grid_factory(grid_shape, compute_backend=backend)

        # Setup the simulation BC and stepper
        self._setup()

        # Make list to store drag coefficients
        self.time_steps = []
        self.force_x = []
        self.force_y = []
        self.force_z = []

    def _setup(self):
        self.setup_boundary_conditions()
        self.setup_stepper()

        # Make initializer operator
        initializer = OutletInitializer(
            wind_speed=self.wind_speed,
            grid_shape=self.grid_shape,
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.backend,
        )

        # Initialize fields using the stepper
        self.f_0, self.f_1, self.bc_mask, self.missing_mask = self.stepper.prepare_fields(initializer=initializer)

    def define_boundary_indices(self):
        box = self.grid.bounding_box_indices()
        box_no_edge = self.grid.bounding_box_indices(remove_edges=True)
        inlet = box_no_edge["left"]
        outlet = box["right"]
        walls = [box["bottom"][i] + box["top"][i] + box["front"][i] + box["back"][i] for i in range(self.velocity_set.d)]
        walls = np.unique(np.array(walls), axis=-1).tolist()

        # Load the mesh (replace with your own mesh)
        stl_filename = "examples/cfd/stl-files/sphere.stl"
        mesh = trimesh.load_mesh(stl_filename, process=False)
        mesh_vertices = mesh.vertices

        # Transform the mesh points to be located in the right position in the wind tunnel
        mesh_vertices -= mesh_vertices.min(axis=0)
        mesh_extents = mesh_vertices.max(axis=0)
        length_phys_unit = mesh_extents.max()
        length_lbm_unit = self.grid_shape[1] / 7
        dx = length_phys_unit / length_lbm_unit
        mesh_vertices = mesh_vertices / dx
        shift = np.array([self.grid_shape[0] / 3, (self.grid_shape[1] - mesh_extents[1] / dx) / 2, (self.grid_shape[2] - mesh_extents[2] / dx) / 2])
        sphere = mesh_vertices + shift.astype(int)
        diam = np.max(sphere.max(axis=0) - sphere.min(axis=0))
        self.sphere_cross_section = np.pi * diam**2 / 4.0

        return inlet, outlet, walls, sphere

    def setup_boundary_conditions(self):
        inlet, outlet, walls, sphere = self.define_boundary_indices()
        bc_left = RegularizedBC("velocity", prescribed_value=(self.wind_speed, 0.0, 0.0), indices=inlet)
        bc_do_nothing = DoNothingBC(indices=outlet)
        bc_sphere = HybridBC(bc_method="dorschner_localized", mesh_vertices=sphere, voxelization_method="winding")
        # bc_sphere = HybridBC(bc_method="bounceback_grads", mesh_vertices=sphere, use_mesh_distance=True)
        # bc_sphere = HalfwayBounceBackBC(mesh_vertices=sphere)
        # Not assining BC for walls makes them periodic.
        self.boundary_conditions = [bc_left, bc_do_nothing, bc_sphere]

    def setup_stepper(self):
        self.stepper = IncompressibleNavierStokesStepper(
            omega=self.omega,
            grid=self.grid,
            boundary_conditions=self.boundary_conditions,
            collision_type="KBC",
        )

    def run(self, num_steps, print_interval, post_process_interval=100):
        # Setup the operator for computing surface forces at the interface of the specified BC
        bc_sphere = self.boundary_conditions[-1]
        self.momentum_transfer = MomentumTransfer(bc_sphere)

        start_time = time.time()
        for i in range(num_steps):
            self.f_0, self.f_1 = self.stepper(self.f_0, self.f_1, self.bc_mask, self.missing_mask, i)
            self.f_0, self.f_1 = self.f_1, self.f_0

            if i % post_process_interval == 0 or i == num_steps - 1:
                self.post_process(i)

            if (i + 1) % print_interval == 0:
                elapsed_time = time.time() - start_time
                print(f"Iteration: {i + 1}/{num_steps} | Time elapsed: {elapsed_time:.2f}s")

    def post_process(self, i):
        # Write the results. We'll use JAX backend for the post-processing
        if not isinstance(self.f_0, jnp.ndarray):
            f_0 = wp.to_jax(self.f_0)
            bmask = wp.to_jax(self.bc_mask)
        else:
            f_0 = self.f_0

        macro = Macroscopic(
            compute_backend=ComputeBackend.JAX,
            precision_policy=self.precision_policy,
            velocity_set=xlb.velocity_set.D3Q27(precision_policy=self.precision_policy, backend=ComputeBackend.JAX),
        )

        rho, u = macro(f_0)

        # remove boundary cells
        # u = u[:, 1:-1, 1:-1, 1:-1]
        u_magnitude = (u[0] ** 2 + u[1] ** 2 + u[2] ** 2) ** 0.5

        fields = {"u_magnitude": u_magnitude, "bmask": bmask[0]}

        save_fields_vtk(fields, timestep=i)
        save_image(fields["u_magnitude"][:, self.grid_shape[1] // 2, :], timestep=i)

        # Compute lift and drag
        boundary_force = self.momentum_transfer(self.f_0, self.f_1, self.bc_mask, self.missing_mask)
        drag = boundary_force[0]
        drag2 = boundary_force[1]
        lift = boundary_force[2]
        c_d = 2.0 * drag / (self.wind_speed**2 * self.sphere_cross_section)
        c_d2 = 2.0 * drag2 / (self.wind_speed**2 * self.sphere_cross_section)
        c_l = 2.0 * lift / (self.wind_speed**2 * self.sphere_cross_section)
        self.force_x.append(c_d)
        self.force_y.append(c_d2)
        self.force_z.append(c_l)
        # self.lift_coefficients.append(c_l)
        self.time_steps.append(i)

        # # Save monitor plot
        self.plot_drag_coefficient(self.force_x, prefix="x_drag")
        self.plot_drag_coefficient(self.force_y, prefix="y_drag")
        self.plot_drag_coefficient(self.force_z, prefix="lift")

    def plot_drag_coefficient(self, drag_coefficients, prefix="drag"):
        # Compute moving average of drag coefficient, 100, 1000, 10000
        # drag_coefficients = np.array(self.drag_coefficients)
        drag_coefficients_ma_10 = np.convolve(drag_coefficients, np.ones(10) / 10, mode="valid")
        drag_coefficients_ma_100 = np.convolve(drag_coefficients, np.ones(100) / 100, mode="valid")
        drag_coefficients_ma_1000 = np.convolve(drag_coefficients, np.ones(1000) / 1000, mode="valid")
        drag_coefficients_ma_10000 = np.convolve(drag_coefficients, np.ones(10000) / 10000, mode="valid")
        drag_coefficients_ma_100000 = np.convolve(drag_coefficients, np.ones(100000) / 100000, mode="valid")

        # Plot drag coefficient
        plt.plot(self.time_steps, drag_coefficients, label="Raw")
        if len(self.time_steps) > 10:
            plt.plot(self.time_steps[9:], drag_coefficients_ma_10, label="MA 10")
        if len(self.time_steps) > 100:
            plt.plot(self.time_steps[99:], drag_coefficients_ma_100, label="MA 100")
        if len(self.time_steps) > 1000:
            plt.plot(self.time_steps[999:], drag_coefficients_ma_1000, label="MA 1,000")
        if len(self.time_steps) > 10000:
            plt.plot(self.time_steps[9999:], drag_coefficients_ma_10000, label="MA 10,000")
        if len(self.time_steps) > 100000:
            plt.plot(self.time_steps[99999:], drag_coefficients_ma_100000, label="MA 100,000")

        plt.ylim(-1.0, 1.0)
        plt.legend()
        plt.xlabel("Time step")
        plt.ylabel("Drag coefficient")
        plt.savefig(prefix + ".png")
        plt.close()


if __name__ == "__main__":
    # Grid parameters
    wp.clear_kernel_cache()
    res = 32
    grid_size_x, grid_size_y, grid_size_z = 10 * res, 7 * res, 7 * res
    grid_shape = (grid_size_x, grid_size_y, grid_size_z)

    # Configuration
    backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32

    velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, backend=backend)
    wind_speed = 0.02
    num_steps = 100000
    print_interval = 1000

    # Set up Reynolds number and deduce relaxation time (omega)
    # Re = 500000000.0
    Re = 100000.0
    clength = grid_size_y / 7.0 - 1
    visc = wind_speed * clength / Re
    omega = 1.0 / (3.0 * visc + 0.5)

    # Print simulation info
    print("\n" + "=" * 50 + "\n")
    print("Simulation Configuration:")
    print(f"Grid size: {grid_size_x} x {grid_size_y} x {grid_size_z}")
    print(f"Backend: {backend}")
    print(f"Velocity set: {velocity_set}")
    print(f"Precision policy: {precision_policy}")
    print(f"Prescribed velocity: {wind_speed}")
    print(f"Reynolds number: {Re}")
    print(f"Max iterations: {num_steps}")
    print("\n" + "=" * 50 + "\n")

    simulation = WindTunnel3D(omega, wind_speed, grid_shape, velocity_set, backend, precision_policy)
    simulation.run(num_steps, print_interval, post_process_interval=1000)
