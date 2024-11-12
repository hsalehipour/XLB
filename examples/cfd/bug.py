import trimesh
import time, os, sys
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.helper import create_nse_fields, initialize_eq, check_bc_overlaps
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import (
    FullwayBounceBackBC,
    EquilibriumBC,
    DoNothingBC,
    RegularizedBC,
    HalfwayBounceBackBC,
    ExtrapolationOutflowBC,
    GradsApproximationBC,
    ZouHeBC,
)
from xlb.operator.force.momentum_transfer import MomentumTransfer
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.boundary_masker import IndicesBoundaryMasker, MeshBoundaryMasker
from xlb.utils import save_fields_vtk, save_image
import warp as wp
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt


def get_physical_timestep(physical_discretization_step, physical_velocity, ulb):
    dx = physical_discretization_step  # meters
    dt = dx * ulb / abs(physical_velocity)
    return dt


class Ahmed:
    def __init__(self, **kwargs):
        # initialize backend
        xlb.init(
            velocity_set=kwargs["velocity_set"],
            default_backend=kwargs["backend"],
            default_precision_policy=kwargs["precision_policy"],
        )

        self.grid_shape = kwargs["grid_shape"]
        self.velocity_set = kwargs["velocity_set"]
        self.backend = kwargs["backend"]
        self.precision_policy = kwargs["precision_policy"]
        self.grid, self.f_0, self.f_1, self.missing_mask, self.bc_mask = create_nse_fields(grid_shape)
        self.stepper = None
        self.boundary_conditions = []
        self.wind_speed = kwargs["wind_speed"]
        self.omega = kwargs["omega"]
        self.voxel_size = kwargs["voxel_size"]
        self.dt = kwargs["physical_timestep"]

        # Make list to store drag coefficients
        self._setup()

    def _setup(self):
        self.initialize_fields()
        self.setup_boundary_conditions()
        self.setup_boundary_masker()
        self.setup_stepper()

    def define_boundary_indices(self):
        boundingBoxIndices = self.grid.bounding_box_indices()
        boundingBoxIndices_noEdge = self.grid.bounding_box_indices(remove_edges=True)

        inlet = boundingBoxIndices_noEdge["left"]
        outlet = boundingBoxIndices_noEdge["right"]
        walls = [
            boundingBoxIndices["bottom"][i] + boundingBoxIndices["top"][i] + boundingBoxIndices["front"][i] + boundingBoxIndices["back"][i]
            for i in range(self.velocity_set.d)
        ]
        walls = np.unique(np.array(walls), axis=-1).tolist()
        return inlet, outlet, walls

    def setup_boundary_conditions(self):
        inlet, outlet, walls = self.define_boundary_indices()
        bc_walls = GradsApproximationBC(indices=walls)
        # bc_inlet = RegularizedBC("velocity", (self.wind_speed, 0.0, 0.0), indices=inlet)
        bc_inlet = EquilibriumBC(rho=1.0, u=(self.wind_speed, 0.0, 0.0), indices=inlet)
        bc_outlet = DoNothingBC(indices=outlet)
        self.boundary_conditions = [bc_inlet, bc_outlet, bc_walls]

    def setup_boundary_masker(self):
        check_bc_overlaps(self.boundary_conditions, self.velocity_set.d, self.backend)
        indices_boundary_masker = IndicesBoundaryMasker(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.backend,
        )
        self.bc_mask, self.missing_mask = indices_boundary_masker(self.boundary_conditions, self.bc_mask, self.missing_mask)

    def initialize_fields(self):
        self.f_0 = initialize_eq(self.f_0, self.grid, self.velocity_set, self.precision_policy, self.backend)
        self.f_1 = initialize_eq(self.f_1, self.grid, self.velocity_set, self.precision_policy, self.backend)

    def setup_stepper(self):
        self.stepper = IncompressibleNavierStokesStepper(self.omega, boundary_conditions=self.boundary_conditions, collision_type="KBC")

    def run(self, num_steps, print_interval):
        for i in range(num_steps):
            self.f_0, self.f_1 = self.stepper(self.f_0, self.f_1, self.bc_mask, self.missing_mask, i)
            self.f_0, self.f_1 = self.f_1, self.f_0
            wp.synchronize()

            # Setup output fields
            if i % print_interval == 0:
                self.output_data(i)

    def getMacro(self):
        start = time.time()

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
        u = u[:, :, :, :] * self.voxel_size / self.dt
        rho = rho[:, :, :, :]

        fields = {"umag": (u[0] ** 2 + u[1] ** 2 + u[2] ** 2) ** 0.5, "u_x": u[0], "u_y": u[1], "u_z": u[2], "rho": rho[0], "bmap": bmask[0]}
        print(f"Time to compute Macro {time.time() - start} sec")
        return fields

    def output_data(self, i):
        # Get macro
        fields = self.getMacro()
        # Output vtk and image
        fields = {key: value.astype(np.float32) for key, value in fields.items()}
        save_fields_vtk(fields, timestep=i)
        save_image(fields["umag"][:, self.grid_shape[1] // 2, :], timestep=i)
        if np.isnan(np.average(fields["u_x"])):
            print("NaN in Velocity")
            sys.exit()


if __name__ == "__main__":
    # clear kernel cash
    wp.clear_kernel_cache()

    # Assume meters
    duct_width = 6
    # Hydraulic Diameter
    dh = (4 * duct_width**2) / (2 * (duct_width + duct_width))
    # Number of diameters for length
    duct_length = 9 * dh

    voxel_size = 0.15  # 0.06
    # How many vehicle lengths for the domain
    grid_size_x = int(duct_length / voxel_size)
    grid_size_y = int(duct_width / voxel_size)
    grid_size_z = int(duct_width / voxel_size)
    grid_shape = (grid_size_x, grid_size_y, grid_size_z)

    # Assume meters / second (current Exp Data is 40m/s )
    velocity = 300
    ulb = 0.05
    air_kin_visc = 1.508e-5

    # Flow Passes
    flow_passes = 10

    print_interval = 500

    dt = get_physical_timestep(voxel_size, velocity, ulb)
    visc_lbm = air_kin_visc * (dt / voxel_size**2)
    omega = 1.0 / (3.0 * visc_lbm + 0.5)

    # Configuration
    backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, backend=backend)

    Re = ulb * (dh / voxel_size**2) / visc_lbm

    # Print simulation info
    print("\n" + "=" * 50 + "\n")
    print("Simulation Configuration:")
    print(f"Grid size: {grid_size_x} x {grid_size_y} x {grid_size_z}")
    print(f"Voxel Count: {grid_size_x * grid_size_y * grid_size_z:,}")
    print(f"Backend: {backend}")
    print(f"Velocity set: {velocity_set}")
    print(f"Precision policy: {precision_policy}")
    print(f"Prescribed velocity: {ulb}")
    print(f"Reynolds Dh: {Re:,}")
    print("\n" + "=" * 50 + "\n")

    kwargs = {
        "backend": backend,
        "velocity_set": velocity_set,
        "omega": omega,
        "grid_shape": grid_shape,
        "wind_speed": ulb,
        "precision_policy": precision_policy,
        "voxel_size": voxel_size,
        "physical_timestep": dt,
    }

    simulation = Ahmed(**kwargs)
    simulation.run(1001, 1000)
