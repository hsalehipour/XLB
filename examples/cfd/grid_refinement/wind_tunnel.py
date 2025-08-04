import neon
import warp as wp
import numpy as np
import time
import os
import re
import matplotlib.pyplot as plt
import trimesh
import shapely
import networkx

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
from xlb.utils.mesher import make_cuboid_mesh, MultiresIO
from xlb.utils.makemesh import generate_mesh
from xlb.operator.force import MultiresMomentumTransfer
from xlb.helper.initializers import MultiresOutletInitializer
from xlb import MresPerfOptimizationType

wp.clear_kernel_cache()

def generate_makemesh_mesh(stl_filename, voxel_size, ground_refinement_level=2, ground_voxel_height=4):
    """
    Generate a makemesh mesh based on the provided voxel size in meters, domain multipliers, and padding values.
    """
    # Number of requested refinement levels
    num_levels = 4

    # Domain multipliers for the full domain
    domainMultiplier = {
        "-x": 1.5,
        "x": 2,
        "-y": 3,
        "y": 3,
        "-z": 0.173611,
        "z": 4,
    }

    # Padding values to control voxel growth
    padding_values = {
        0: (12, 12, 12, 12, 12, 12),
        1: (8, 20, 8, 8, 8, 8),
        2: (8, 40, 8, 8, 8, 8),
        3: (8, 20, 8, 8, 8, 8),
        4: (8, 20, 8, 8, 8, 8),
        5: (4, 4, 4, 4, 4, 4),
        6: (4, 4, 4, 4, 4, 4),
        7: (4, 4, 4, 4, 4, 4),
        8: (4, 4, 4, 4, 4, 4),
    }

    # Load the mesh
    mesh = trimesh.load_mesh(stl_filename, process=False)
    if mesh.is_empty:
        raise ValueError("Loaded mesh is empty or invalid.")

    # Compute original bounds
    min_bound = mesh.vertices.min(axis=0)
    max_bound = mesh.vertices.max(axis=0)
    partSize = max_bound - min_bound

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

    # Generate mesh using generate_mesh with ground refinement
    level_data, _ = generate_mesh(
        num_levels,
        "temp.stl",
        voxel_size,
        padding_values,
        domainMultiplier,
        ground_refinement_level=ground_refinement_level,
        ground_voxel_height=ground_voxel_height,
    )
    actual_num_levels = len(level_data)
    grid_shape_finest = tuple([int(i * 2 ** (actual_num_levels - 1)) for i in level_data[-1][0].shape])
    print(f"Requested levels: {num_levels}, Actual levels: {actual_num_levels}")
    print(f"Full shape based on finest voxel size is {grid_shape_finest}")
    os.remove("temp.stl")
    return level_data, mesh_vertices, tuple([int(a) for a in grid_shape_finest]), partSize, actual_num_levels

def generate_cuboid_mesh(stl_filename, voxel_size):
    """
    Alternative cuboid mesh generation based on Apolo's method with domain multipliers per level.
    """
    # Domain multipliers for each refinement level
    domainMultiplier = [
        [5, 8, 5, 5, 5, 5],
        [2, 4, 2, 2, 2, 2],
        [.6, 2, 0.6, 0.6, 0.6, 0.6],
        [0.4, 0.9, 0.4, 0.4, 0.4, 0.4],
    ]

    # Load the mesh
    mesh = trimesh.load_mesh(stl_filename, process=False)
    if mesh.is_empty:
        raise ValueError("Loaded mesh is empty or invalid.")

    # Compute original bounds
    min_bound = mesh.vertices.min(axis=0)
    max_bound = mesh.vertices.max(axis=0)
    partSize = max_bound - min_bound

    # Compute translation to put mesh into first octant of the domain
    shift = np.array(
        [
            domainMultiplier[0][0] * partSize[0] - min_bound[0],
            domainMultiplier[0][2] * partSize[1] - min_bound[1],
            domainMultiplier[0][4] * partSize[2] - min_bound[2],
        ],
        dtype=float,
    )

    # Apply translation and save out temp STL
    mesh.apply_translation(shift)
    _ = mesh.vertex_normals
    mesh_vertices = np.asarray(mesh.vertices) / voxel_size
    mesh.export("temp.stl")

    # Generate mesh using make_cuboid_mesh
    level_data = make_cuboid_mesh(
        voxel_size,
        domainMultiplier,
        "temp.stl",
    )
    actual_num_levels = len(level_data)
    grid_shape_finest = tuple([int(i * 2 ** (actual_num_levels - 1)) for i in level_data[-1][0].shape])
    print(f"Requested levels: {len(domainMultiplier)}, Actual levels: {actual_num_levels}")
    print(f"Full shape based on finest voxel size is {grid_shape_finest}")
    os.remove("temp.stl")
    return level_data, mesh_vertices, tuple([int(a) for a in grid_shape_finest]), partSize, actual_num_levels

def prepare_sparsity_pattern(level_data):
    """
    Prepare the sparsity pattern for the multiresolution grid based on the level data.
    """
    sparsity_pattern = []
    level_origins = []
    for lvl in range(len(level_data)):
        level_mask = level_data[lvl][0]
        level_mask = np.ascontiguousarray(level_mask, dtype=np.int32)
        sparsity_pattern.append(level_mask)
        level_origins.append(level_data[lvl][2])
    return sparsity_pattern, level_origins

# -------------------------- Simulation Setup --------------------------

# Define physical and simulation parameters
voxel_size = 0.008  # Finest voxel size in meters
u_physical = 10  # Physical inlet velocity in m/s
ulb = 0.05  # Lattice velocity
flow_passes = 10 # Domain flow passes
kinematic_viscosity = 1.508e-5  # Kinematic viscosity of air in m^2/s

# Generate the mesh and body vertices
stl_filename = "examples/cfd/stl-files/Ahmed_25_NoLegs.stl"
script_name = "Ahmed 8mm"

level_data, body_vertices, grid_shape_zip, partSize, actual_num_levels = generate_makemesh_mesh(
    stl_filename,
    voxel_size,
)

# I/O settings
print_interval_percentage = 1  # Print every 1% of iterations
file_output_crossover_percentage = 90  # Crossover at 80% of iterations
num_file_outputs_pre_crossover = 9  # 8 outputs before crossover (e.g., every 10% up to 80%)
num_file_outputs_post_crossover = 10  # 20 outputs in the last 20%

# Other setup parameters
compute_backend = ComputeBackend.NEON
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)

# Initialize XLB
xlb.init(
    velocity_set=velocity_set,
    default_backend=compute_backend,
    default_precision_policy=precision_policy,
)

# Calculate Reynolds number
L = partSize[0]  # Characteristic length. Use x-dimension (length) of the body
Re = u_physical * L / kinematic_viscosity

# Calculate lattice viscosity using coarsest voxel size
delta_x_coarse = voxel_size * 2 ** (actual_num_levels - 1)
delta_t = delta_x_coarse * ulb / u_physical
nu_lattice = kinematic_viscosity * delta_t / (delta_x_coarse ** 2)
omega = 1.0 / (3.0 * nu_lattice + 0.5)

# Create output directory based on script name
output_dir = os.path.join("examples/cfd/grid_refinement", script_name)

# Clear and recreate the output directory
import shutil
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# Define exporter object for HDF5 output
field_name_cardinality_dict = {"velocity": 3, "density": 1}
h5exporter = MultiresIO(field_name_cardinality_dict, level_data)

# Define a separate exporter for the initial bc_mask output
bc_mask_exporter = MultiresIO({"bc_mask": 1}, level_data)

# Prepare the sparsity pattern and origins
sparsity_pattern, level_origins = prepare_sparsity_pattern(level_data)

# get the number of levels
num_levels = len(level_data)

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
# Get boundary indices for all sides across levels
box = grid.bounding_box_indices(shape=grid.level_to_shape(coarsest_level))
left_indices = grid.boundary_indices_across_levels(level_data, box_side="left", remove_edges=True)
right_indices = grid.boundary_indices_across_levels(level_data, box_side="right", remove_edges=True)
top_indices = grid.boundary_indices_across_levels(level_data, box_side="top", remove_edges=False)
bottom_indices = grid.boundary_indices_across_levels(level_data, box_side="bottom", remove_edges=False)
front_indices = grid.boundary_indices_across_levels(level_data, box_side="front", remove_edges=False)
back_indices = grid.boundary_indices_across_levels(level_data, box_side="back", remove_edges=False)

# Filter front and back indices to remove overlaps with top and bottom at each level
filtered_front_indices = []
filtered_back_indices = []
for level in range(num_levels):
    # Convert indices to sets of tuples (x, y, z) for set operations
    top_set = set(zip(*top_indices[level])) if top_indices[level] else set()
    bottom_set = set(zip(*bottom_indices[level])) if bottom_indices[level] else set()
    front_set = set(zip(*front_indices[level])) if front_indices[level] else set()
    back_set = set(zip(*back_indices[level])) if back_indices[level] else set()
    
    # Remove top and bottom indices from front and back
    filtered_front_set = front_set - (top_set | bottom_set)
    filtered_back_set = back_set - (top_set | bottom_set)
    
    # Convert back to list of lists format: [[x_coords], [y_coords], [z_coords]]
    filtered_front_indices.append(
        [list(coords) for coords in zip(*filtered_front_set)] if filtered_front_set else []
    )
    filtered_back_indices.append(
        [list(coords) for coords in zip(*filtered_back_set)] if filtered_back_set else []
    )

# Turbulent Flow Profile
def bc_profile(taper_fraction=0.05):
    assert compute_backend == ComputeBackend.NEON

    # Note nx, ny, nz are the dimensions of the grid at the finest level while the inlet is defined at the coarsest level
    _, ny, nz = grid_shape_zip
    dtype = precision_policy.compute_precision.wp_dtype
    H_y = dtype(ny // 2 ** (num_levels - 1) - 1)  # Height in y direction
    H_z = dtype(nz // 2 ** (num_levels - 1) - 1)  # Height in z direction
    two = dtype(2.0)
    ulb_wp = dtype(ulb)
    taper_frac = dtype(taper_fraction)  # Fraction of distance from center where tapering begins
    core_frac = dtype(1.0 - 2.0 * taper_fraction)  # Fraction of core region with max velocity

    @wp.func
    def bc_profile_warp(index: wp.vec3i):
        # Turbulent flow profile: constant velocity in core, linear taper near walls
        y = dtype(index[1])
        z = dtype(index[2])

        # Calculate normalized distance from center
        y_center = wp.abs(y - (H_y / two))  # Distance from center in y
        z_center = wp.abs(z - (H_z / two))  # Distance from center in z

        # Calculate normalized distances (0 at center, 1 at wall)
        y_norm = two * y_center / H_y
        z_norm = two * z_center / H_z

        # Find the maximum normalized distance to determine if we're in the taper region
        max_norm = wp.max(y_norm, z_norm)

        # If within core region (within core_frac from center), use max velocity
        # Otherwise, linearly taper to zero at the wall
        velocity = ulb_wp
        if max_norm > core_frac:
            # Linear taper in the outer taper_fraction region
            velocity = ulb_wp * (dtype(1.0) - (max_norm - core_frac) / taper_frac)

        # Ensure velocity doesn't go negative
        velocity = wp.max(dtype(0.0), velocity)

        return wp.vec(velocity, length=1)

    return bc_profile_warp

# Initialize boundary conditions
bc_inlet = RegularizedBC("velocity", prescribed_value=(ulb, 0.0, 0.0), indices=left_indices)
# bc_inlet = RegularizedBC("velocity", profile=bc_profile(), indices=left_indices)
bc_outlet = DoNothingBC(indices=right_indices)
bc_top = FullwayBounceBackBC(indices=top_indices)
bc_bottom = FullwayBounceBackBC(indices=bottom_indices)
# bc_bottom = HybridBC(bc_method="nonequilibrium_regularized", indices=bottom_indices, prescribed_value= (ulb,0.0,0.0))
bc_front = FullwayBounceBackBC(indices=filtered_front_indices)
bc_back = FullwayBounceBackBC(indices=filtered_back_indices)
bc_body = HybridBC(
    bc_method="nonequilibrium_regularized",
    mesh_vertices=body_vertices,
    voxelization_method=MeshVoxelizationMethod.AABB,
    use_mesh_distance=False
)

# Combine all boundary conditions
boundary_conditions = [bc_top, bc_front, bc_back, bc_inlet, bc_outlet, bc_bottom, bc_body]

# Make initializer operator
initializer = MultiresOutletInitializer(
    outlet_bc_id=bc_outlet.id,
    wind_vector=(ulb, 0.0, 0.0),
    velocity_set=velocity_set,
    precision_policy=precision_policy,
    compute_backend=compute_backend,
)

# Define a multi-resolution simulation manager
sim = xlb.helper.MultiresSimulationManager(
    omega=omega,
    grid=grid,
    boundary_conditions=boundary_conditions,
    collision_type="KBC",
    initializer=initializer,
    mres_perf_opt=xlb.MresPerfOptimizationType.FUSION_AT_FINEST,
)

# Compute active voxels per level and solid voxels (bc_mask == 255) per level
active_voxels = [np.count_nonzero(mask) for mask in sparsity_pattern]
sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)  # Ensure bc_mask is populated
fields_data = bc_mask_exporter.get_fields_data({"bc_mask": sim.bc_mask})
bc_mask_data = fields_data["bc_mask_0"]  # Concatenated bc_mask values for all active voxels
level_id_field = bc_mask_exporter.level_id_field  # Level ID for each voxel
solid_voxels = []
for lvl in range(actual_num_levels):
    # In XLB, finest level has ID 0 (maps to level_data[0]), coarsest has ID actual_num_levels-1 (maps to level_data[actual_num_levels-1])
    level_mask = level_id_field == lvl  # Map level_id_field directly to level_data index
    solid_voxels.append(np.sum(bc_mask_data[level_mask] == 255))
# Adjust active voxels by subtracting solid voxels
active_voxels = [max(0, active_voxels[lvl] - solid_voxels[lvl]) for lvl in range(actual_num_levels)]
total_voxels = sum(active_voxels)
total_lattice_updates_per_step = sum(active_voxels[lvl] * (2 ** (actual_num_levels - 1 - lvl)) for lvl in range(actual_num_levels))

# Save bc_mask at initialization (step 0)
sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
filename = os.path.join(output_dir, f"{script_name}_initial_bc_mask")
try:
    bc_mask_exporter.to_hdf5(filename, {"bc_mask": sim.bc_mask}, compression="gzip", compression_opts=0)
    xmf_filename = f"{filename}.xmf"
    hdf5_basename = f"{script_name}_initial_bc_mask.h5"
except Exception as e:
    print(f"Error during initial bc_mask output: {e}")
wp.synchronize()

# Setup Momentum Transfer for Force Calculation
momentum_transfer = MultiresMomentumTransfer(bc_body, compute_backend=compute_backend)

# Compute reference area using existing bc_mask processing
fields_data = bc_mask_exporter.get_fields_data({"bc_mask": sim.bc_mask})
bc_mask_data = fields_data["bc_mask_0"]  # Concatenated bc_mask values for all active voxels
level_id_field = bc_mask_exporter.level_id_field  # Level ID for each voxel
finest_level = 0  # Finest level has ID 0 in xlb framework
mask_finest = level_id_field == finest_level  # Boolean mask for finest level voxels
bc_mask_finest = bc_mask_data[mask_finest]  # bc_mask values at finest level

# Get 3D indices of active voxels at finest level from the level_data mask
active_indices_finest = np.argwhere(level_data[0][0])  # level_data[0][0] is the finest level mask

# Filter for solid voxels (assuming bc_body.id identifies the solid body)
solid_voxels_indices = active_indices_finest[bc_mask_finest == bc_body.id]

# Compute projected area as number of unique (y, z) pairs (j, k indices)
# Assuming i=x, j=y, k=z in the grid
unique_jk = np.unique(solid_voxels_indices[:, 1:3], axis=0)
reference_area = unique_jk.shape[0]  # Area in lattice units (number of lattice sites)
reference_area_physical = reference_area * (voxel_size ** 2)

# Hard-coded reference area
# reference_area = 0.0225 / (voxel_size**2)

def print_lift_drag(sim):
    boundary_force = momentum_transfer(sim.f_0, sim.f_1, sim.bc_mask, sim.missing_mask)
    drag = boundary_force[0]
    lift = boundary_force[2]
    cd = 2.0 * drag / (ulb**2 * reference_area)
    cl = 2.0 * lift / (ulb**2 * reference_area)
    # Check for NaN values
    if np.isnan(cd) or np.isnan(cl):
        raise ValueError(f"NaN detected in coefficients at step {step}: Cd={cd}, Cl={cl}")
    drag_values.append([cd, cl])
    print(f"CD={cd:.3f}, CL={cl:.3f}, Drag Force (lattice units)={drag:.6f}")

def plot_drag_lift(drag_values, output_dir, print_interval, percentile_range=(15, 85), use_log_scale=False):
    """
    Plot CD and CL over time and save the plot to the output directory.
    """
    drag_values_array = np.array(drag_values)
    steps = np.arange(0, len(drag_values) * print_interval, print_interval)
    cd_values = drag_values_array[:, 0]
    cl_values = drag_values_array[:, 1]
    y_min = min(np.percentile(cd_values, percentile_range[0]), np.percentile(cl_values, percentile_range[0]))
    y_max = max(np.percentile(cd_values, percentile_range[1]), np.percentile(cl_values, percentile_range[1]))
    padding = (y_max - y_min) * 0.1
    y_min, y_max = y_min - padding, y_max + padding
    if use_log_scale:
        y_min = max(y_min, 1e-6)
    plt.figure(figsize=(10, 6))
    plt.plot(steps, cd_values, label='Drag Coefficient (Cd)', color='blue')
    plt.plot(steps, cl_values, label='Lift Coefficient (Cl)', color='red')
    plt.xlabel('Simulation Step')
    plt.ylabel('Coefficient')
    plt.title('Drag and Lift Coefficients Over Time')
    plt.legend()
    plt.grid(True)
    plt.ylim(y_min, y_max)
    if use_log_scale:
        plt.yscale('log')
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
print(f"Finest voxel size: {voxel_size} meters")
print(f"Coarsest voxel size: {delta_x_coarse} meters")
print(f"Total voxels: {sum(np.count_nonzero(mask) for mask in sparsity_pattern):,}")
print(f"Total active voxels: {total_voxels:,}")
print(f"Active voxels per level: {active_voxels}")
print(f"Solid voxels per level: {solid_voxels}")
print(f"Total lattice updates per global step: {total_lattice_updates_per_step:,}")
print(f"Actual number of refinement levels: {actual_num_levels}")
print(f"Physical inlet velocity: {u_physical} m/s")
print(f"Lattice velocity (ulb): {ulb}")
print(f"Characteristic length: {L: .4f} meters")
print(f"Kinematic viscosity: {kinematic_viscosity} m^2/s")
print(f"Computed reference area (bc_mask): {reference_area} lattice units")
print(f"Physical reference area (bc_mask): {reference_area_physical:.6f} m^2")
print(f"Reynolds number: {Re:,.2f}")
print(f"Lattice viscosity: {nu_lattice}")
print(f"Relaxation parameter (omega): {omega: .5f}")
print("\n" + "=" * 50 + "\n")

for step in range(num_steps):
    step_start = time.time()
    sim.step()
    wp.synchronize()
    compute_time += time.time() - step_start
    steps_since_last_print += 1
    if step % print_interval == 0 or step == num_steps - 1:
        sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
        wp.synchronize()
        print_lift_drag(sim)
        end_time = time.time()
        elapsed = end_time - start_time
        total_lattice_updates = total_lattice_updates_per_step * steps_since_last_print
        MLUPS = total_lattice_updates / compute_time / 1e6 if compute_time > 0 else 0.0
        current_flow_passes = step * ulb / grid_shape_x_coarsest
        remaining_steps = num_steps - step - 1
        time_remaining = 0.0 if MLUPS == 0 else (total_lattice_updates_per_step * remaining_steps) / (MLUPS * 1e6)
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
    file_output_interval = file_output_interval_pre_crossover if step < crossover_step else file_output_interval_post_crossover
    if step % file_output_interval == 0 or step == num_steps - 1:
        sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
        filename = os.path.join(output_dir, f"{script_name}_{step:04d}")
        try:
            h5exporter.to_hdf5(filename, {"velocity": sim.u, "density": sim.rho}, compression="gzip", compression_opts=0)
            xmf_filename = f"{filename}.xmf"
            hdf5_basename = f"{script_name}_{step:04d}.h5"
        except Exception as e:
            print(f"Error during file output at step {step}: {e}")
        wp.synchronize()

# Save drag and lift data to CSV
if len(drag_values) > 0:
    with open(os.path.join(output_dir, "drag_lift.csv"), 'w') as fd:
        fd.write("Step,Cd,Cl\n")
        for i, (cd, cl) in enumerate(drag_values):
            fd.write(f"{i * print_interval},{cd},{cl}\n")
    plot_drag_lift(drag_values, output_dir, print_interval)

# Calculate and print average Cd and Cl for the last 50%
drag_values_array = np.array(drag_values)
if len(drag_values) > 0:
    start_index = len(drag_values) // 2
    last_half = drag_values_array[start_index:, :]
    avg_cd = np.mean(last_half[:, 0])
    avg_cl = np.mean(last_half[:, 1])
    print(f"Average Drag Coefficient (Cd) for last 50%: {avg_cd:.6f}")
    print(f"Average Lift Coefficient (Cl) for last 50%: {avg_cl:.6f}")
else:
    print("No drag or lift data collected.")