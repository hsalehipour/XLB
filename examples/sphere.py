import neon
import warp as wp
import numpy as np
import time
import os
import re
import matplotlib.pyplot as plt
import trimesh
import shutil

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
from xlb.helper.initializers import CustomMultiresInitializer
from xlb import MresPerfOptimizationType

wp.clear_kernel_cache()
wp.config.quiet = True

# User Configuration
# =================
# Physical and simulation parameters
voxel_size = 0.15/300.0 # Finest voxel size in meters
ulb = 0.05         # Lattice velocity
flow_passes = 2    # Domain flow passes
kinematic_viscosity = 1.508e-5  # Kinematic viscosity of air in m^2/s

# STL filename
stl_filename = "examples/stl/sphere.stl"
base_script_name = "Sphere 300D"

# List of Reynolds numbers to simulate
#reynolds_numbers = [30, 50, 100, 300, 500, 1000, 3000, 10000, 30000, 100000, 200000, 500000, 1000000]

reynolds_numbers = [ 10000, 1000000]


# I/O settings
print_interval_percentage = 5   # Print every 1% of iterations
file_output_crossover_percentage = 50  # Crossover at 50% of iterations
num_file_outputs_pre_crossover = 1    # Outputs before crossover
num_file_outputs_post_crossover = 1   # Outputs after crossover

# Other setup parameters
compute_backend = ComputeBackend.NEON
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)

# Choose mesher type
mesher_type = "makemesh"  # Options: "makemesh" or "cuboid"

# Mesh Generation Functions
# =========================
def generate_makemesh_mesh(stl_filename, voxel_size, ground_refinement_level=None, ground_voxel_height=4):
    """
    Generate a makemesh mesh based on the provided voxel size in meters, domain multipliers, and padding values.
    """
    # Number of requested refinement levels
    num_levels = 8

    # Domain multipliers for the full domain
    domainMultiplier = {
        "-x": 3,
        "x": 6,
        "-y": 4,
        "y": 4,
        "-z": 4,
        "z": 4,
    }

    padding_values = [
        [6, 6, 6, 6, 6, 6],
        [6, 6, 6, 6, 6, 6],
        [6, 6, 6, 6, 6, 6],
        [6, 6, 6, 6, 6, 6],
        [6, 6, 6, 6, 6, 6],
        [6, 6, 6, 6, 6, 6],
        [6, 6, 6, 6, 6, 6],
        [6, 6, 6, 6, 6, 6],
        [6, 6, 6, 6, 6, 6],
        
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
    level_data, _, sparsity_pattern, level_origins = generate_mesh(
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

    return level_data, mesh_vertices, tuple([int(a) for a in grid_shape_finest]), partSize, actual_num_levels, shift, sparsity_pattern, level_origins

def generate_cuboid_mesh(stl_filename, voxel_size):
    """
    Alternative cuboid mesh generation based on Apolo's method with domain multipliers per level.
    """
    # Domain multipliers for each refinement level
    domain_multiplier = [
        [3, 6, 4, 4, 4, 4],  # -x, x, -y, y, -z, z
        [1.5, 3, 1.5, 1.5, 1.5, 1.5],  # -x, x, -y, y, -z, z
        [1, 2, 1, 1, 1, 1],
        [0.25, 0.5, 0.25, 0.25, 0.25, 0.25],
        # [1, 2, 1, 1, 1, 1],
        # [0.4, 1, 0.4, 0.4, 0.4, 0.4],
        # [0.2, 0.4, 0.2, 0.2, 0.2, 0.2],
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
            domain_multiplier[0][0] * partSize[0] - min_bound[0],
            domain_multiplier[0][2] * partSize[1] - min_bound[1],
            domain_multiplier[0][4] * partSize[2] - min_bound[2],
        ],
        dtype=float,
    )

    # Apply translation and save out temp STL
    mesh.apply_translation(shift)
    _ = mesh.vertex_normals
    mesh_vertices = np.asarray(mesh.vertices) / voxel_size
    mesh.export("temp.stl")

    # Generate mesh using make_cuboid_mesh
    level_data, sparsity_pattern, level_origins = make_cuboid_mesh(
        voxel_size,
        domain_multiplier,
        "temp.stl",
    )
    actual_num_levels = len(level_data)
    grid_shape_finest = tuple([int(i * 2 ** (actual_num_levels - 1)) for i in level_data[-1][0].shape])
    print(f"Requested levels: {len(domain_multiplier)}, Actual levels: {actual_num_levels}")
    print(f"Full shape based on finest voxel size is {grid_shape_finest}")
    os.remove("temp.stl")

    return level_data, mesh_vertices, tuple([int(a) for a in grid_shape_finest]), partSize, actual_num_levels, shift, sparsity_pattern, level_origins

# Boundary Conditions Setup
# =========================
def setup_boundary_conditions(grid, level_data, body_vertices, ulb, nu_lattice, compute_backend=ComputeBackend.NEON):
    """
    Set up boundary conditions for the simulation.
    """
    num_levels = len(level_data)
    coarsest_level = num_levels - 1
    box = grid.bounding_box_indices(shape=grid.level_to_shape(coarsest_level))
    left_indices = grid.boundary_indices_across_levels(level_data, box_side="left", remove_edges=True)
    right_indices = grid.boundary_indices_across_levels(level_data, box_side="right", remove_edges=True)
    top_indices = grid.boundary_indices_across_levels(level_data, box_side="top", remove_edges=False)
    bottom_indices = grid.boundary_indices_across_levels(level_data, box_side="bottom", remove_edges=False)
    front_indices = grid.boundary_indices_across_levels(level_data, box_side="front", remove_edges=False)
    back_indices = grid.boundary_indices_across_levels(level_data, box_side="back", remove_edges=False)

    # box_no_edge = grid.bounding_box_indices(shape=grid.level_to_shape(coarsest_level), remove_edges=True)
    # inlet = box_no_edge["left"]
    # outlet = box_no_edge["right"]
    # walls = [box["bottom"][i] + box["top"][i] + box["front"][i] + box["back"][i] for i in range(velocity_set.d)]
    # walls = np.unique(np.array(walls), axis=-1).tolist()

    # inlet = [[] for _ in range(num_levels - 1)] + [inlet]
    # outlet = [[] for _ in range(num_levels - 1)] + [outlet]
    # walls = [[] for _ in range(num_levels - 1)] + [walls]

    # Filter front and back indices to remove overlaps with top and bottom at each level
    filtered_front_indices = []
    filtered_back_indices = []
    filtered_top_indices = []
    filtered_bottom_indices = []
    for level in range(num_levels):
        left_set = set(zip(*left_indices[level])) if left_indices[level] else set()
        right_set = set(zip(*right_indices[level])) if right_indices[level] else set()
        top_set = set(zip(*top_indices[level])) if top_indices[level] else set()
        bottom_set = set(zip(*bottom_indices[level])) if bottom_indices[level] else set()
        front_set = set(zip(*front_indices[level])) if front_indices[level] else set()
        back_set = set(zip(*back_indices[level])) if back_indices[level] else set()
        filtered_front_set = front_set - (top_set | bottom_set )
        filtered_back_set = back_set - (top_set | bottom_set )
        filtered_top_set = top_set - (left_set | right_set)
        filtered_bottom_set = bottom_set - (left_set | right_set)
        filtered_front_indices.append(
            [list(coords) for coords in zip(*filtered_front_set)] if filtered_front_set else []
        )
        filtered_back_indices.append(
            [list(coords) for coords in zip(*filtered_back_set)] if filtered_back_set else []
        )
        filtered_top_indices.append(
            [list(coords) for coords in zip(*filtered_top_set)] if filtered_top_set else []
        )
        filtered_bottom_indices.append(
            [list(coords) for coords in zip(*filtered_bottom_set)] if filtered_bottom_set else []
        )

    # Turbulent Flow Profile
    def bc_profile(taper_fraction=0.05):
        assert compute_backend == ComputeBackend.NEON
        _, ny, nz = grid_shape_zip
        dtype = precision_policy.compute_precision.wp_dtype
        H_y = dtype(ny // 2 ** (num_levels - 1) - 1)
        H_z = dtype(nz // 2 ** (num_levels - 1) - 1)
        two = dtype(2.0)
        ulb_wp = dtype(ulb)
        taper_frac = dtype(taper_fraction)
        core_frac = dtype(1.0 - 2.0 * taper_fraction)

        @wp.func
        def bc_profile_warp(index: wp.vec3i):
            y = dtype(index[1])
            z = dtype(index[2])
            y_center = wp.abs(y - (H_y / two))
            z_center = wp.abs(z - (H_z / two))
            y_norm = two * y_center / H_y
            z_norm = two * z_center / H_z
            max_norm = wp.max(y_norm, z_norm)
            velocity = ulb_wp
            if max_norm > core_frac:
                velocity = ulb_wp * (dtype(1.0) - (max_norm - core_frac) / taper_frac)
            velocity = wp.max(dtype(0.0), velocity)
            return wp.vec(velocity, length=1)

        return bc_profile_warp

    bc_inlet = RegularizedBC(
        "velocity",
        #profile=bc_profile_taper(),
        prescribed_value=(ulb, 0.0, 0.0),
       indices=left_indices,
    )

    bc_outlet = DoNothingBC(indices=right_indices)

    #bc_top = FullwayBounceBackBC(indices=top_indices)
    bc_top = HybridBC(bc_method="nonequilibrium_regularized",prescribed_value=(ulb, 0.0, 0.0),indices=top_indices)
    
    bc_bottom = HybridBC(bc_method="nonequilibrium_regularized",prescribed_value=(ulb, 0.0, 0.0),indices=bottom_indices)
    #bc_bottom = FullwayBounceBackBC(indices=bottom_indices)
    #bc_front = FullwayBounceBackBC(indices=filtered_front_indices)
    bc_front = HybridBC(bc_method="nonequilibrium_regularized",prescribed_value=(ulb, 0.0, 0.0),indices=filtered_front_indices)
    #bc_back = FullwayBounceBackBC(indices=filtered_back_indices)
    bc_back = HybridBC(bc_method="nonequilibrium_regularized",prescribed_value=(ulb, 0.0, 0.0),indices=filtered_back_indices)

    bc_body = HybridBC(
        bc_method="bounceback_grads",
        mesh_vertices=body_vertices,
        voxelization_method=MeshVoxelizationMethod("AABB_CLOSE", close_voxels=3),
        use_mesh_distance=True,
    )

    return [bc_top, bc_bottom, bc_front, bc_back, bc_inlet, bc_outlet, bc_body] # Body must be last. Outlet must be second to last
    # return [bc_walls, bc_inlet, bc_outlet, bc_body]


# Simulation Initialization
# =========================
def initialize_simulation(grid, boundary_conditions, omega, initializer, collision_type="KBC", mres_perf_opt=xlb.MresPerfOptimizationType.FUSION_AT_FINEST):
    """
    Initialize the multiresolution simulation manager.
    """
    sim = xlb.helper.MultiresSimulationManager(
        omega=omega,
        grid=grid,
        boundary_conditions=boundary_conditions,
        collision_type=collision_type,
        initializer=initializer,
        mres_perf_opt=mres_perf_opt,
    )
    return sim

# Utility Functions
# =================
def print_lift_drag(sim, step, momentum_transfer, ulb, reference_area, voxel_size):
    """
    Calculate and print lift and drag coefficients.
    """
    boundary_force = momentum_transfer(sim.f_0, sim.f_1, sim.bc_mask, sim.missing_mask)
    drag = boundary_force[0]
    lift = boundary_force[2]
    cd = 2.0 * drag / (ulb**2 * reference_area)
    cl = 2.0 * lift / (ulb**2 * reference_area)
    if np.isnan(cd) or np.isnan(cl):
        raise ValueError(f"NaN detected in coefficients at step {step}: Cd={cd}, Cl={cl}")
    drag_values.append([cd, cl])
    print(f"CD={cd:.3f}, CL={cl:.3f}, Drag Force (lattice units)={drag:.6f}")

def plot_drag_lift(drag_values, output_dir, print_interval, script_name, percentile_range=(15, 85), use_log_scale=False):
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
    plt.title(f'{script_name}: Drag and Lift Coefficients Over Time')
    plt.legend()
    plt.grid(True)
    plt.ylim(y_min, y_max)
    if use_log_scale:
        plt.yscale('log')
    plt.savefig(os.path.join(output_dir, 'drag_lift_plot.png'))
    plt.close()

def compute_voxel_statistics_and_reference_area(sim, bc_mask_exporter, level_data, actual_num_levels, sparsity_pattern, boundary_conditions, voxel_size):
    """
    Compute active/solid voxels, totals, lattice updates, and reference area based on simulation data.
    """
    # Compute macro fields
    sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
    fields_data = bc_mask_exporter.get_fields_data({"bc_mask": sim.bc_mask})
    bc_mask_data = fields_data["bc_mask_0"]
    level_id_field = bc_mask_exporter.level_id_field

    # Compute solid voxels per level (assuming 255 is the solid marker)
    solid_voxels = []
    for lvl in range(actual_num_levels):
        level_mask = level_id_field == lvl
        solid_voxels.append(np.sum(bc_mask_data[level_mask] == 255))

    # Compute active voxels (total non-zero in sparsity minus solids)
    active_voxels = [np.count_nonzero(mask) for mask in sparsity_pattern]
    active_voxels = [max(0, active_voxels[lvl] - solid_voxels[lvl]) for lvl in range(actual_num_levels)]

    # Totals
    total_voxels = sum(active_voxels)
    total_lattice_updates_per_step = sum(active_voxels[lvl] * (2 ** (actual_num_levels - 1 - lvl)) for lvl in range(actual_num_levels))

    # Compute reference area (projected on YZ plane at finest level)
    finest_level = 0
    mask_finest = level_id_field == finest_level
    bc_mask_finest = bc_mask_data[mask_finest]
    active_indices_finest = np.argwhere(level_data[0][0])
    bc_body_id = boundary_conditions[-1].id  # Assuming last BC is bc_body
    solid_voxels_indices = active_indices_finest[bc_mask_finest == bc_body_id]
    unique_jk = np.unique(solid_voxels_indices[:, 1:3], axis=0)
    reference_area = unique_jk.shape[0]
    reference_area_physical = reference_area * (voxel_size ** 2)

    return {
        "active_voxels": active_voxels,
        "solid_voxels": solid_voxels,
        "total_voxels": total_voxels,
        "total_lattice_updates_per_step": total_lattice_updates_per_step,
        "reference_area": reference_area,
        "reference_area_physical": reference_area_physical
    }

# Experimental data
experimental_re = [
    19.42, 32.29, 58.43, 88.00, 144.28, 272.54, 422.44, 664.10, 1153.20, 1788.30,
    3019.50, 5172.00, 8615.00, 13949.00, 20987.00, 32653.00, 52119.00, 8.32e4,
    1.37e5, 1.95e5, 2.36e5, 2.65e5, 2.84e5, 3.00e5, 3.17e5, 3.45e5, 3.86e5,
    4.54e5, 5.12e5, 6.25e5, 7.74e5, 9.05e5, 9.86e5
]
experimental_cd = [
    2.6658, 1.9542, 1.4126, 1.1271, 0.8867, 0.6878, 0.5889, 0.50424, 0.45045,
    0.41389, 0.39122, 0.38037, 0.39687, 0.41409, 0.43947, 0.45079, 0.45724,
    0.45088, 0.43223, 0.41434, 0.39601, 0.36492, 0.2528, 0.15643, 0.11629,
    0.09957, 0.09278, 0.08992, 0.09279, 0.10536, 0.12135, 0.13208, 0.13976
]

# Dictionary to store simulated average Cd for each Re
simulated_cds = {}

# Main Script
# ===========
# Loop over each Reynolds number
for Re in reynolds_numbers:
    # Initialize XLB
    wp.clear_kernel_cache()
    xlb.init(
        velocity_set=velocity_set,
        default_backend=compute_backend,
        default_precision_policy=precision_policy,
    )

    # Generate mesh (done once, as it's the same for all Re)
    if mesher_type == "makemesh":
        level_data, body_vertices, grid_shape_zip, partSize, actual_num_levels, shift, sparsity_pattern, level_origins = generate_makemesh_mesh(
            stl_filename, voxel_size
        )
    elif mesher_type == "cuboid":
        level_data, body_vertices, grid_shape_zip, partSize, actual_num_levels, shift, sparsity_pattern, level_origins = generate_cuboid_mesh(
            stl_filename, voxel_size
        )
    else:
        raise ValueError(f"Invalid mesher_type: {mesher_type}. Must be 'makemesh' or 'cuboid'.")

    # Characteristic length
    L = partSize[0]
    L = float(L)  # Cast to built-in float to avoid NumPy type propagation issues with Warp
    
    # Compute u_physical for this Re
    u_physical = Re * kinematic_viscosity / L

    # Set script name based on Re
    if Re >= 1000000:
        re_str = f"Re{int(Re / 1000000)}M"
    elif Re >= 1000:
        re_str = f"Re{int(Re / 1000)}K"
    else:
        re_str = f"Re{Re}"
    script_name = f"{base_script_name} {re_str}"

    # Calculate lattice parameters
    delta_x_coarse = voxel_size * 2 ** (actual_num_levels - 1)
    delta_t = voxel_size * ulb / u_physical
    nu_lattice = kinematic_viscosity * delta_t / (voxel_size ** 2)
    
    omega = 1.0 / (3.0 * nu_lattice + 0.5)

    # Create output directory
    current_dir = os.path.join(os.path.dirname(__file__))
    output_dir = os.path.join(current_dir, script_name)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Define exporter objects
    field_name_cardinality_dict = {"velocity": 3, "density": 1}
    h5exporter = MultiresIO(
        field_name_cardinality_dict,
        level_data,
        scale=voxel_size,
        offset=-shift,
        timestep_size=delta_t,
    )
    bc_mask_exporter = MultiresIO({"bc_mask": 1}, level_data)

    # Create grid
    grid = multires_grid_factory(
        grid_shape_zip,
        velocity_set=velocity_set,
        sparsity_pattern_list=sparsity_pattern,
        sparsity_pattern_origins=[neon.Index_3d(*box_origin) for box_origin in level_origins],
    )

    # Calculate num_steps
    coarsest_level = grid.count_levels - 1
    grid_shape_x_coarsest = grid.level_to_shape(coarsest_level)[0]
    num_steps = int(flow_passes * (grid_shape_x_coarsest / ulb))

    # Calculate print and file output intervals
    print_interval = max(1, int(num_steps * (print_interval_percentage / 100.0)))
    crossover_step = int(num_steps * (file_output_crossover_percentage / 100.0))
    file_output_interval_pre_crossover = max(1, int(crossover_step / num_file_outputs_pre_crossover)) if num_file_outputs_pre_crossover > 0 else num_steps + 1
    file_output_interval_post_crossover = max(1, int((num_steps - crossover_step) / num_file_outputs_post_crossover)) if num_file_outputs_post_crossover > 0 else num_steps + 1

    # Setup boundary conditions
    boundary_conditions = setup_boundary_conditions(grid, level_data, body_vertices, ulb, nu_lattice, compute_backend)

    # Create initializer
    initializer = CustomMultiresInitializer(
   # bc_id=boundary_conditions[-2].id,  # bc_outlet
    constant_velocity_vector=(ulb, 0.0, 0.0),
    velocity_set=velocity_set,
    precision_policy=precision_policy,
    compute_backend=compute_backend,
    )


    # Initialize simulation
    sim = initialize_simulation(grid, boundary_conditions, omega, initializer)

    # Compute voxel statistics and reference area
    stats = compute_voxel_statistics_and_reference_area(sim, bc_mask_exporter, level_data, actual_num_levels, sparsity_pattern, boundary_conditions, voxel_size)
    active_voxels = stats["active_voxels"]
    solid_voxels = stats["solid_voxels"]
    total_voxels = stats["total_voxels"]
    total_lattice_updates_per_step = stats["total_lattice_updates_per_step"]
    reference_area = stats["reference_area"]
    reference_area_physical = stats["reference_area_physical"]

    # Save initial bc_mask
    #bc_mask_exporter.to_hdf5(filename, {"bc_mask": sim.bc_mask}, compression="gzip", compression_opts=0)
       
    wp.synchronize()   
    momentum_transfer = MultiresMomentumTransfer(
        boundary_conditions[-1],
        mres_perf_opt=xlb.MresPerfOptimizationType.FUSION_AT_FINEST,
        compute_backend=compute_backend,
    )

    # Print simulation info
    print("\n" + "=" * 50 + "\n")
    print(f"Simulation Configuration for Re = {Re}:")
    # print(f"Grid shape at finest level: {grid_shape_zip}")
    # print(f"Grid shape at coarsest level: {grid.level_to_shape(coarsest_level)}")
    print(f"Number of flow passes: {flow_passes}")
    print(f"Calculated iterations: {num_steps:,}")
    # print(f"Output directory: {output_dir}")
    # print(f"Print interval: {print_interval} steps (every {print_interval_percentage}% of iterations)")
    # print(f"File output interval pre-crossover (0-{file_output_crossover_percentage}%): {file_output_interval_pre_crossover} steps")
    # print(f"File output interval post-crossover ({file_output_crossover_percentage}-100%): {file_output_interval_post_crossover} steps")
    print(f"Finest voxel size: {voxel_size} meters")
    print(f"Coarsest voxel size: {delta_x_coarse} meters")
    print(f"Total voxels: {sum(np.count_nonzero(mask) for mask in sparsity_pattern):,}")
    print(f"Total active voxels: {total_voxels:,}")
    print(f"Active voxels per level: {active_voxels}")
    print(f"Solid voxels per level: {solid_voxels}")
    print(f"Total lattice updates per global step: {total_lattice_updates_per_step:,}")
    print(f"Actual number of refinement levels: {actual_num_levels}")
    print(f"Physical inlet velocity: {u_physical:.4f} m/s")
    print(f"Lattice velocity (ulb): {ulb}")
    print(f"Characteristic length: {L: .4f} meters")
    # print(f"Kinematic viscosity: {kinematic_viscosity} m^2/s")
    print(f"Computed reference area (bc_mask): {reference_area} lattice units")
    print(f"Physical reference area (bc_mask): {reference_area_physical:.6f} m^2")
    print(f"Reynolds number: {Re:,.2f}")
    # print(f"Lattice viscosity: {nu_lattice:.5f}")
    print(f"Relaxation parameter (omega): {omega:.5f}")
    print("\n" + "=" * 50 + "\n")

    # -------------------------- Simulation Loop --------------------------
    wp.synchronize()
    start_time = time.time()
    compute_time = 0.0
    steps_since_last_print = 0
    drag_values = []

    for step in range(num_steps):
        step_start = time.time()
        sim.step()
        wp.synchronize()
        compute_time += time.time() - step_start
        steps_since_last_print += 1
        if step % print_interval == 0 or step == num_steps - 1:
            sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
            wp.synchronize()
            print_lift_drag(sim, step, momentum_transfer, ulb, reference_area, voxel_size)
            filename = os.path.join(output_dir, f"{script_name}_{step:04d}")
            h5exporter.to_slice_image(
                filename,
                {"velocity": sim.u},
                plane_point=(1, 0, 0),
                plane_normal=(0, 1, 0),
                grid_res=2500,
                bounds=(0.1, .9, 0.1, .9),
                show_axes=False,
                show_colorbar=False,
                slice_thickness=delta_x_coarse, #needed when using model units
                normalize = u_physical*1.5, #eventually we could have the 1.5 read from json as we did before
            )
            end_time = time.time()
            elapsed = end_time - start_time
            total_lattice_updates = total_lattice_updates_per_step * steps_since_last_print
            MLUPS = total_lattice_updates / compute_time / 1e6 if compute_time > 0 else 0.0
            current_flow_passes = step * ulb / grid_shape_x_coarsest
            remaining_steps = num_steps - step - 1
            time_remaining = 0.0 if MLUPS == 0 else (total_lattice_updates_per_step * remaining_steps) / (MLUPS * 1e6)
            hours, rem = divmod(time_remaining, 3600)
            minutes, seconds = divmod(rem, 60)
            time_remaining_str = f"{int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s\n"
            print(f" \n"
                  f"Completed step {step}/{num_steps} ({remaining_steps} remaining). \n"
                  f"Flow Passes: {current_flow_passes:.2f}. \n"
                  f"Time elapsed for last {steps_since_last_print} steps: {elapsed:.6f} seconds. \n"
                  f"Compute time: {compute_time:.6f} seconds. \n"
                  f"MLUPS: {MLUPS:.2f}. \n"
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
        plot_drag_lift(drag_values, output_dir, print_interval, script_name)

    # Calculate and print average Cd and Cl for the last 50%
    drag_values_array = np.array(drag_values)
    if len(drag_values) > 0:
        start_index = len(drag_values) // 2
        last_half = drag_values_array[start_index:, :]
        avg_cd = np.mean(last_half[:, 0])
        avg_cl = np.mean(last_half[:, 1])
        print(f"Average Drag Coefficient (Cd) for last 50%: {avg_cd:.6f}")
        print(f"Average Lift Coefficient (Cl) for last 50%: {avg_cl:.6f}")
        # Store the average Cd for this Re
        simulated_cds[Re] = avg_cd
    else:
        print("No drag or lift data collected.")

    # Create or update the comparison plot
    plt.figure(figsize=(10, 6))
    plt.semilogx(experimental_re, experimental_cd, label='Experimental', color='blue')
    sim_re = sorted(simulated_cds.keys())
    sim_cd = [simulated_cds[r] for r in sim_re]
    plt.semilogx(sim_re, sim_cd, label='Simulated', marker='x', linestyle='--', color='red')
    plt.xlabel('Reynolds Number (Re)')
    plt.ylabel('Drag Coefficient (Cd)')
    plt.title(f"{base_script_name} Re vs Cd: Experimental vs Simulated")
    plt.legend()
    plt.grid(True)
    comparison_plot_path = os.path.join(current_dir, f"{base_script_name.replace(' ', '_')}_comparison_plot.png")
    plt.savefig(comparison_plot_path)
    plt.close()
    print(f"Updated comparison plot saved to {comparison_plot_path}")