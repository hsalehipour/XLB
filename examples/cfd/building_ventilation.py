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
    ExtrapolationOutflowBC,
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
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Building simulation")
    parser.add_argument("input_mesh", type=str, help="Input mesh file name")
    return parser.parse_args()

# ----------------- Input Mesh and Output directory --------------------
args = parse_arguments()
output_name = args.input_mesh
stl_filename = "/home/hesam/REPOs/XLB_refactored/XLB/examples/cfd/stl-files/" + output_name + ".stl"
current_dir = os.path.join(os.path.dirname(__file__))
output_dir = os.path.join(current_dir, output_name)

# Clean out old results if they exist
if os.path.exists(output_dir):
    os.system("rm -r " + output_dir)
# Start new folder for results
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# -------------------------- Simulation Setup --------------------------

# Grid parameters
grid_size_x, grid_size_y, grid_size_z = 640, 320, 160
grid_shape = (grid_size_x, grid_size_y, grid_size_z)

# Simulation Configuration
compute_backend = ComputeBackend.WARP
precision_policy = PrecisionPolicy.FP32FP32

velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)
wind_speed = 0.02
num_steps = 100000
print_interval = 1000
post_process_interval = 1000

# Physical Parameters
Re = 50000.0
clength = grid_size_x - 1
visc = wind_speed * clength / Re
omega = 1.0 / (3.0 * visc + 0.5)

# Print simulation info
print("\n" + "=" * 50 + "\n")
print("Simulation Configuration:")
print(f"Grid size: {grid_size_x} x {grid_size_y} x {grid_size_z}")
print(f"Backend: {compute_backend}")
print(f"Velocity set: {velocity_set}")
print(f"Precision policy: {precision_policy}")
print(f"Prescribed velocity: {wind_speed}")
print(f"Reynolds number: {Re}")
print(f"Max iterations: {num_steps}")
print("\n" + "=" * 50 + "\n")

# Initialize XLB
xlb.init(
    velocity_set=velocity_set,
    default_backend=compute_backend,
    default_precision_policy=precision_policy,
)

# Create Grid
grid = grid_factory(grid_shape, compute_backend=compute_backend)

# Bounding box indices
box = grid.bounding_box_indices()
box_no_edge = grid.bounding_box_indices(remove_edges=True)
inlet = box_no_edge["right"]
outlet = box_no_edge["left"]
walls = [box["bottom"][i] + box["top"][i] + box["front"][i] + box["back"][i] for i in range(velocity_set.d)]
walls = np.unique(np.array(walls), axis=-1).tolist()

# Load the mesh (replace with your own mesh)
voxelization_method = "aabb"
mesh = trimesh.load_mesh(stl_filename, process=False)
mesh_vertices = mesh.vertices

# Get the minimum vertex coordinates along each axis (x, y, z)
min_values = mesh_vertices.min(axis=0)

# Transform the mesh points to align with the grid
mesh_vertices -= mesh_vertices.min(axis=0)
mesh_extents = mesh_vertices.max(axis=0)
length_phys_unit = mesh_extents.max()
length_lbm_unit = grid_shape[0] / 3
dx = length_phys_unit / length_lbm_unit
mesh_vertices = mesh_vertices / dx

# Depending on the voxelization method, shift_z ensures the bottom ground does not intersect with the voxelized mesh
# Any smaller shift value would lead to large lift computations due to the initial equilibrium distributions. Bigger
# values would be fine but leave a gap between surfaces that are supposed to touch.
if voxelization_method in ["ray", "winding"]:
    shift_z = 2
elif voxelization_method in ["aabb", "aabb_fill_in"]:
    shift_z = 3
shift = np.array([0.5 * grid_shape[0], (grid_shape[1] - mesh_extents[1] / dx) / 2, shift_z])
building_vertices = mesh_vertices + shift


bc_left = RegularizedBC("velocity", prescribed_value=(wind_speed, 0.0, 0.0), indices=inlet)
bc_walls = FullwayBounceBackBC(indices=walls)
bc_do_nothing = ExtrapolationOutflowBC(indices=outlet)
bc_building = HybridBC(bc_method="nonequilibrium_regularized", mesh_vertices=building_vertices, voxelization_method=voxelization_method)
boundary_conditions = [bc_walls, bc_left, bc_do_nothing, bc_building]


# Setup Stepper
stepper = IncompressibleNavierStokesStepper(
    grid=grid,
    boundary_conditions=boundary_conditions,
    collision_type="KBC",
)

# Prepare Fields
f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()


# -------------------------- Helper Functions --------------------------


def post_process(
    step,
    f_0,
    f_1,
    grid_shape,
    macro,
    missing_mask,
    bc_mask,
    wind_speed,
):
    """
    Post-process simulation data: save fields, compute forces, and plot drag coefficient.

    Args:
        step (int): Current time step.
        f_current: Current distribution function.
        grid_shape (tuple): Shape of the grid.
        macro: Macroscopic operator object.
        momentum_transfer: MomentumTransfer operator object.
        missing_mask: Missing mask from stepper.
        bc_mask: Boundary condition mask from stepper.
        wind_speed (float): Prescribed wind speed.
        car_cross_section (float): Cross-sectional area of the car.
        drag_coefficients (list): List to store drag coefficients.
        lift_coefficients (list): List to store lift coefficients.
        time_steps (list): List to store time steps.
    """
    # Convert to JAX array if necessary
    if not isinstance(f_0, jnp.ndarray):
        f_0_jax = wp.to_jax(f_0)
    else:
        f_0_jax = f_0

    # Compute macroscopic quantities
    rho, u = macro(f_0_jax)

    # Remove boundary cells
    u = u[:, 1:-1, 1:-1, 1:-1]
    fields = {"umag": jnp.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2), "ux": u[0], "uy": u[1], "uz": u[2]}

    # Save fields in VTK format
    voxel_size = dx
    org_X = min_values[0] - (shift[0]) * voxel_size
    org_Y = min_values[1] - (shift[1]) * voxel_size
    org_Z = min_values[2] - (shift[2]) * voxel_size
    origin = np.array([org_X, org_Y, org_Z])
    save_fields_vtk(fields, timestep=step, output_dir=output_dir, shift_coords=(origin[0], origin[1], origin[2]), scale=voxel_size)

    # Save the u_magnitude slice at the mid y-plane
    mid_y = grid_shape[1] // 2
    save_image(fields["umag"][:, mid_y, :], timestep=step, prefix=output_dir + "/" + output_name)


# Define Macroscopic Calculation
macro = Macroscopic(
    compute_backend=ComputeBackend.JAX,
    precision_policy=precision_policy,
    velocity_set=xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=ComputeBackend.JAX),
)


# -------------------------- Simulation Loop --------------------------

start_time = time.time()
for step in range(num_steps):
    # Perform simulation step
    f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, step)
    f_0, f_1 = f_1, f_0  # Swap the buffers

    # Print progress at intervals
    if step % print_interval == 0:
        if compute_backend == ComputeBackend.WARP:
            wp.synchronize()
        elapsed_time = time.time() - start_time
        print(f"Iteration: {step}/{num_steps} | Time elapsed: {elapsed_time:.2f}s")
        start_time = time.time()

    # Post-process at intervals and final step
    if (step % post_process_interval == 0) or (step == num_steps - 1):
        post_process(
            step,
            f_0,
            f_1,
            grid_shape,
            macro,
            missing_mask,
            bc_mask,
            wind_speed,
        )

print("Simulation completed successfully.")
