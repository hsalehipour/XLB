import neon
import warp as wp
import numpy as np
import time
import os  # Added for directory and file handling
import re
import matplotlib.pyplot as plt  # Added for plotting
import trimesh


import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import multires_grid_factory
from xlb.operator.boundary_condition import (
    FullwayBounceBackBC,
    HalfwayBounceBackBC,
    RegularizedBC,
    ExtrapolationOutflowBC,
    DoNothingBC,
    ZouHeBC,
    HybridBC,
)
from xlb.operator.boundary_masker import MeshVoxelizationMethod
from xlb.utils.mesher import MultiresIO
from xlb.utils.makemesh import generate_mesh
from xlb.operator.force import MultiresMomentumTransfer
from xlb.helper.initializers import MultiresOutletInitializer


def generate_makemesh_mesh(stl_filename, num_finest_voxels_across_part):
    """
    Generate a cuboid mesh based on the provided voxel size, domain multipliers, and padding values.
    """
    # Number of refinement levels
    num_levels = 6

    # Domain multipliers for the full domain
    domainMultiplier = {
        "-x": 5,
        "x": 12,
        "-y": 12,
        "y": 12,
        "-z": 12,
        "z": 12,
    }

    # Padding values to control voxel growth
    padding_values = {
        0: (8, 8, 8, 8, 8, 8),
        1: (8, 8, 8, 8, 8, 8),
        2: (8, 36, 8, 8, 8, 8),
        3: (8, 24, 8, 8, 8, 8),
        4: (8, 12, 8, 8, 8, 8),
        5: (8, 8, 8, 8, 8, 8),
        6: (4, 4, 4, 4, 4, 4),
        7: (4, 4, 4, 4, 4, 4),
        8: (4, 4, 4, 4, 4, 4),
        9: (4, 4, 4, 4, 4, 4),
        10: (4, 4, 4, 4, 4, 4),
        11: (4, 4, 4, 4, 4, 4),
        12: (4, 4, 4, 4, 4, 4),
        13: (4, 4, 4, 4, 4, 4),
        14: (4, 4, 4, 4, 4, 4),
        15: (4, 4, 4, 4, 4, 4),
    }

    # Load the mesh
    mesh = trimesh.load_mesh(stl_filename, process=False)
    assert not mesh.is_empty, ValueError("Loaded mesh is empty or invalid.")

    # Compute original bounds
    min_bound = mesh.vertices.min(axis=0)
    max_bound = mesh.vertices.max(axis=0)
    partSize = max_bound - min_bound

    # Smallest voxel size
    voxel_size = min(partSize) / num_finest_voxels_across_part

    # Compute translation to put mesh into first octant of the domain
    shift = np.array(
        [
            domainMultiplier["-x"] * partSize[0] - min_bound[0],
            domainMultiplier["-y"] * partSize[1] - min_bound[1],
            domainMultiplier["-z"] * partSize[2] - min_bound[2],
        ],
        dtype=float,
    )

    # Apply translation and save out temp STL
    mesh.apply_translation(shift)
    _ = mesh.vertex_normals
    mesh_vertices = np.asarray(mesh.vertices) / voxel_size
    mesh.export("temp.stl")

    # Generate mesh using generate_mesh
    level_data, _ = generate_mesh(
        num_levels,
        "temp.stl",
        voxel_size,
        padding_values,
        domainMultiplier,
    )
    grid_shape_finest = tuple([i * 2 ** (len(level_data) - 1) for i in level_data[-1][0].shape])
    print(f"Full shape based on finest voxels size is {grid_shape_finest}")
    os.remove("temp.stl")
    return level_data, mesh_vertices, tuple([int(a) for a in grid_shape_finest])

def prepare_sparsity_pattern(level_data):
    """
    Prepare the sparsity pattern for the multiresolution grid based on the level data.
    """
    num_levels = len(level_data)
    sparsity_pattern = []
    level_origins = []
    for lvl in range(num_levels):
        # Get the level mask from the level data
        level_mask = level_data[lvl][0]
        # Ensure level mask is contiguous int32
        level_mask = np.ascontiguousarray(level_mask, dtype=np.int32)
        sparsity_pattern.append(level_mask)
        # Get the origin for this level
        level_origins.append(level_data[lvl][2])
    return sparsity_pattern, level_origins

# -------------------------- Simulation Setup --------------------------

# Define resolution of the voxelized grid
num_finest_voxels_across_part = 100
sphere_radius = num_finest_voxels_across_part / 2

# Other setup parameters
Re = 450000
compute_backend = ComputeBackend.NEON
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)
ulb = 0.05
flow_passes = 10
print_interval_percentage = 1.0  # Print every 1% of iterations
file_output_crossover_percentage = 80.0  # Crossover at 80% of iterations
num_file_outputs_pre_crossover = 8  # 8 outputs before crossover (e.g., every 10% up to 80%)
num_file_outputs_post_crossover = 40  # 20 outputs in the last 20%

# Initialize XLB
# wp.config.max_unroll = 27
xlb.init(
    velocity_set=velocity_set,
    default_backend=compute_backend,
    default_precision_policy=precision_policy,
)

# Generate the cuboid mesh and sphere vertices
stl_filename = "examples/cfd/stl-files/sphere.stl"
level_data, sphere, grid_shape_zip = generate_makemesh_mesh(stl_filename, num_finest_voxels_across_part)

# Create output directory based on script name
script_name = os.path.splitext(os.path.basename(__file__))[0]  # Get script name without extension
output_dir = os.path.join("examples/cfd/grid_refinement", script_name)  # e.g., examples/cfd/grid_refinement/sphere_mres

# Clear and recreate the output directory
import shutil
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)  # Remove the directory and all its contents
os.makedirs(output_dir)  # Recreate the empty directory

# Define an exporter for the multiresolution data
field_name_cardinality_dict = {"velocity": 3, "density": 1}
h5exporter = MultiresIO(field_name_cardinality_dict, level_data)

# Define a separate exporter for the initial bc_mask output
bc_mask_exporter = MultiresIO({"bc_mask": 1}, level_data)


def fix_xmf_paths(xmf_filename, hdf5_basename):
    """
    Modify the XMF file to use relative HDF5 paths.
    
    Args:
        xmf_filename (str): Path to the XMF file (e.g., 'examples/cfd/grid_refinement/<script_name>/<script_name>_0000.xmf')
        hdf5_basename (str): Desired HDF5 filename (e.g., '<script_name>_0000.h5')
    """
    with open(xmf_filename, 'r') as f:
        content = f.read()
    
    # Replace paths like 'examples/cfd/grid_refinement/<script_name>/<script_name>_XXXX.h5'
    # with just '<script_name>_XXXX.h5'
    pattern = rf'examples/cfd/grid_refinement/[^/]+/{re.escape(hdf5_basename)}'
    fixed_content = re.sub(pattern, hdf5_basename, content)
    
    with open(xmf_filename, 'w') as f:
        f.write(fixed_content)

# Prepare the sparsity pattern and origins from the level data
sparsity_pattern, level_origins = prepare_sparsity_pattern(level_data)

# Compute active voxels per level
active_voxels = [np.count_nonzero(mask) for mask in sparsity_pattern]
print("Active voxels per level:", active_voxels)

# Get the number of levels
num_levels = len(level_data)

# Compute total lattice updates per global step
total_lattice_updates_per_step = sum(active_voxels[lvl] * (2 ** (num_levels - 1 - lvl)) for lvl in range(num_levels))
print("Total lattice updates per global step:", total_lattice_updates_per_step)

# Create the multires grid
grid = multires_grid_factory(
    grid_shape_zip,
    velocity_set=velocity_set,
    sparsity_pattern_list=sparsity_pattern,
    sparsity_pattern_origins=[neon.Index_3d(*box_origin) for box_origin in level_origins],
)

# Calculate iterations based on flow passes
coarsest_level = grid.count_levels - 1
grid_shape_x_coarsest = grid.level_to_shape(coarsest_level)[0]
num_steps = int(flow_passes * (grid_shape_x_coarsest / ulb))

# Calculate print and file output intervals
print_interval = max(1, int(num_steps * (print_interval_percentage / 100.0)))
crossover_step = int(num_steps * (file_output_crossover_percentage / 100.0))
file_output_interval_pre_crossover = max(1, int(crossover_step / num_file_outputs_pre_crossover)) if num_file_outputs_pre_crossover > 0 else num_steps + 1
file_output_interval_post_crossover = max(1, int((num_steps - crossover_step) / num_file_outputs_post_crossover)) if num_file_outputs_post_crossover > 0 else num_steps + 1

# Define Boundary Indices
coarsest_level = grid.count_levels - 1
box = grid.bounding_box_indices(shape=grid.level_to_shape(coarsest_level))
box_no_edge = grid.bounding_box_indices(shape=grid.level_to_shape(coarsest_level), remove_edges=True)
inlet = box_no_edge["left"]
outlet = box_no_edge["right"]
walls = [box["bottom"][i] + box["top"][i] + box["front"][i] + box["back"][i] for i in range(velocity_set.d)]
walls = np.unique(np.array(walls), axis=-1).tolist()

# Define Boundary Conditions
def bc_profile():
    assert compute_backend == ComputeBackend.NEON
    nx, ny, nz = grid_shape_zip
    H_y = float(ny // 2 ** (num_levels - 1) - 1)  # Height in y direction
    H_z = float(nz // 2 ** (num_levels - 1) - 1)  # Height in z direction

    @wp.func
    def bc_profile_warp(index: wp.vec3i):
        y = wp.float32(index[1])
        z = wp.float32(index[2])
        y_center = y - (H_y / 2.0)
        z_center = z - (H_z / 2.0)
        r_squared = (2.0 * y_center / H_y) ** 2.0 + (2.0 * z_center / H_z) ** 2.0
        return wp.vec(ulb * wp.max(0.0, 1.0 - r_squared), length=1)

    return bc_profile_warp

# Convert bc indices to a list of lists
inlet = [[] for _ in range(num_levels - 1)] + [inlet]
outlet = [[] for _ in range(num_levels - 1)] + [outlet]
walls = [[] for _ in range(num_levels - 1)] + [walls]

# Initialize Boundary Conditions
bc_left = RegularizedBC("velocity", prescribed_value=(ulb, 0.0, 0.0), indices=inlet)
bc_walls = FullwayBounceBackBC(indices=walls)
# bc_walls = RegularizedBC("velocity", prescribed_value=(ulb, 0.0, 0.0), indices=walls)
bc_outlet = DoNothingBC(indices=outlet)
bc_sphere = HybridBC(bc_method="nonequilibrium_regularized", mesh_vertices=sphere, voxelization_method=MeshVoxelizationMethod.AABB, use_mesh_distance=False)
# bc_sphere = HalfwayBounceBackBC(mesh_vertices=sphere, voxelization_method=MeshVoxelizationMethod.AABB)

boundary_conditions = [bc_walls, bc_left, bc_outlet, bc_sphere]

# Make initializer operator
from xlb.helper.initializers import MultiresOutletInitializer

initializer = MultiresOutletInitializer(
    outlet_bc_id=bc_outlet.id,
    wind_vector=(ulb, 0.0, 0.0),
    velocity_set=velocity_set,
    precision_policy=precision_policy,
    compute_backend=compute_backend,
)

# Configure the simulation relaxation time
visc = ulb * num_finest_voxels_across_part / Re
omega = 1.0 / (3.0 * visc + 0.5)

# Define a multi-resolution simulation manager
sim = xlb.helper.MultiresSimulationManager(
    omega=omega,
    grid=grid,
    boundary_conditions=boundary_conditions,
    collision_type="KBC",
    initializer=initializer,
)

# Save bc_mask at initialization (step 0)
sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
filename = os.path.join(output_dir, f"{script_name}_initial_bc_mask")
try:
    bc_mask_exporter.to_hdf5(filename, {"bc_mask": sim.bc_mask}, compression="gzip", compression_opts=2)
    xmf_filename = f"{filename}.xmf"
    hdf5_basename = f"{script_name}_initial_bc_mask.h5"
    fix_xmf_paths(xmf_filename, hdf5_basename)
except Exception as e:
    print(f"Error during initial bc_mask output: {e}")
wp.synchronize()

# Setup Momentum Transfer for Force Calculation
bc_sphere = boundary_conditions[-1]
momentum_transfer = MultiresMomentumTransfer(bc_sphere, compute_backend=compute_backend)

def print_lift_drag(sim):
    boundary_force = momentum_transfer(sim.f_0, sim.f_1, sim.bc_mask, sim.missing_mask)
    drag = boundary_force[0]  # x-direction
    lift = boundary_force[2]
    sphere_cross_section = np.pi * sphere_radius**2
    u_avg = ulb
    cd = 2.0 * drag / (u_avg**2 * sphere_cross_section)
    cl = 2.0 * lift / (u_avg**2 * sphere_cross_section)
    # Check for NaN values
    if np.isnan(cd) or np.isnan(cl):
        raise ValueError(f"NaN detected in coefficients at step {step}: Cd={cd}, Cl={cl}")
    drag_values.append([cd, cl])
    print(f"CD={cd}, CL={cl}")

def plot_drag_lift(drag_values, output_dir, print_interval, percentile_range=(15, 85), use_log_scale=False):
    """
    Plot CD and CL over time and save the plot to the output directory with better min/max control.
    
    Parameters:
    - drag_values: List or array of [Cd, Cl] values.
    - output_dir: Directory to save the plot.
    - post_process_interval: Interval between simulation steps.
    - percentile_range: Tuple of (lower, upper) percentiles for y-axis limits (default: (5, 95)).
    - use_log_scale: If True, use logarithmic y-axis (default: False).
    """
    drag_values_array = np.array(drag_values)
    steps = np.arange(0, len(drag_values) * print_interval, print_interval)
    
    # Calculate percentile-based y-axis limits
    cd_values = drag_values_array[:, 0]
    cl_values = drag_values_array[:, 1]
    y_min = min(np.percentile(cd_values, percentile_range[0]), np.percentile(cl_values, percentile_range[0]))
    y_max = max(np.percentile(cd_values, percentile_range[1]), np.percentile(cl_values, percentile_range[1]))
    
    # Add some padding to the limits for better visualization
    padding = (y_max - y_min) * 0.1
    y_min, y_max = y_min - padding, y_max + padding
    
    # Ensure positive range for log scale if used
    if use_log_scale:
        y_min = max(y_min, 1e-6)  # Avoid zero or negative values for log scale

    plt.figure(figsize=(10, 6))
    plt.plot(steps, cd_values, label='Drag Coefficient (Cd)', color='blue')
    plt.plot(steps, cl_values, label='Lift Coefficient (Cl)', color='red')
    
    plt.xlabel('Simulation Step')
    plt.ylabel('Coefficient')
    plt.title('Drag and Lift Coefficients Over Time')
    plt.legend()
    plt.grid(True)
    
    # Apply y-axis limits
    plt.ylim(y_min, y_max)
    
    # Use log scale if specified
    if use_log_scale:
        plt.yscale('log')
    
    # Save and close the plot
    plt.savefig(os.path.join(output_dir, 'drag_lift_plot.png'))
    plt.close()

# -------------------------- Simulation Loop --------------------------

wp.synchronize()
start_time = time.time()
compute_time = 0.0
steps_since_last_print = 0
drag_values = []

# Print simulation info
print("\n" + "=" * 50 + "\n")
print("Simulation Configuration:")
print(f"Grid shape at finest level: {grid_shape_zip}")
print(f"Grid shape at coarsest level: {grid.level_to_shape(coarsest_level)}")
print(f"Number of flow passes: {flow_passes}")
print(f"Calculated iterations: {num_steps:,}")
print(f"Output directory: {output_dir}")
print(f"Print interval: {print_interval} steps (every {print_interval_percentage}% of iterations)")
print(f"File output interval pre-crossover (0-{file_output_crossover_percentage}%): {file_output_interval_pre_crossover} steps")
print(f"File output interval post-crossover ({file_output_crossover_percentage}-100%): {file_output_interval_post_crossover} steps")
print("\n" + "=" * 50 + "\n")

for step in range(num_steps):
    step_start = time.time()
    sim.step()
    wp.synchronize()
    compute_time += time.time() - step_start
    steps_since_last_print += 1
    # Handle printing and drag/lift calculation
    if step % print_interval == 0 or step == num_steps - 1:
        sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
        print_lift_drag(sim)
        wp.synchronize()
        end_time = time.time()
        elapsed = end_time - start_time
        total_lattice_updates = total_lattice_updates_per_step * steps_since_last_print
        MLUPS = total_lattice_updates / compute_time / 1e6 if compute_time > 0 else 0.0
        current_flow_passes = step * ulb / grid_shape_x_coarsest
        remaining_steps = num_steps - step - 1
        # Estimate time remaining
        time_remaining = 0.0 if MLUPS == 0 else (total_lattice_updates_per_step * remaining_steps) / (MLUPS * 1e6)
        # Format time remaining as hours, minutes, seconds
        hours, rem = divmod(time_remaining, 3600)
        minutes, seconds = divmod(rem, 60)
        time_remaining_str = f"{int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s"
        print(f"Completed step {step}/{num_steps} ({remaining_steps} remaining). "
              f"Flow Passes: {current_flow_passes:.2f}. "
              f"Time elapsed for last {steps_since_last_print} steps: {elapsed:.6f} seconds. "
              f"Compute time: {compute_time:.6f} seconds. "
              f"MLUPS: {MLUPS:.2f}. "
              f"Estimated time remaining: {time_remaining_str}")
        start_time = time.time()
        compute_time = 0.0
        steps_since_last_print = 0
    # Handle file output
    file_output_interval = file_output_interval_pre_crossover if step < crossover_step else file_output_interval_post_crossover
    if step % file_output_interval == 0 or step == num_steps - 1:
        sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
        filename = os.path.join(output_dir, f"{script_name}_{step:04d}")
        try:
            h5exporter.to_hdf5(filename, {"velocity": sim.u, "density": sim.rho}, compression="gzip", compression_opts=2)
            xmf_filename = f"{filename}.xmf"
            hdf5_basename = f"{script_name}_{step:04d}.h5"
            fix_xmf_paths(xmf_filename, hdf5_basename)
        except Exception as e:
            print(f"Error during file output at step {step}: {e}")
        wp.synchronize()

# Save drag and lift data to CSV
if len(drag_values) > 0:
    with open(os.path.join(output_dir, "drag_lift.csv"), 'w') as fd:
        fd.write("Step,Cd,Cl\n")
        for i, (cd, cl) in enumerate(drag_values):
            fd.write(f"{i * print_interval},{cd},{cl}\n")

    # Plot drag and lift coefficients
    plot_drag_lift(drag_values, output_dir, print_interval)

# Convert drag_values to a NumPy array
drag_values_array = np.array(drag_values)

# Check if there are enough values
if len(drag_values) > 0:
    # Calculate the index to start from for the last 50%
    start_index = len(drag_values) // 2
    
    # Select the last 50% of the values
    last_half = drag_values_array[start_index:, :]
    
    # Calculate the average Cd and Cl for the last 50%
    avg_cd = np.mean(last_half[:, 0])  # Average of Cd (first column)
    avg_cl = np.mean(last_half[:, 1])  # Average of Cl (second column)
    
    # Print the average values
    print(f"Average Drag Coefficient (Cd) for last 50%: {avg_cd:.6f}")
    print(f"Average Lift Coefficient (Cl) for last 50%: {avg_cl:.6f}")
else:
    print("No drag or lift data collected.")