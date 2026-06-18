import neon
import warp as wp
import numpy as np
import time
import os
import trimesh
import shutil

import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import multires_grid_factory
from xlb.operator.boundary_condition import (
    RegularizedBC,
    DoNothingBC,
    HybridBC,
)
from xlb.operator.boundary_masker import MeshVoxelizationMethod

from xlb.utils.mesher import MultiresIO
from xlb.utils import UnitConvertor
from xlb.utils.mesher import make_cuboid_mesh, prepare_sparsity_pattern
from xlb.helper.initializers import CustomMultiresInitializer


wp.clear_kernel_cache()
wp.config.quiet = True

# User Configuration
# =================
# Physical and simulation parameters
wind_speed_lbm = 0.05  # Lattice velocity
wind_speed_mps = 1.0  # Physical inlet velocity in m/s (user input)
flow_passes = 4  # Domain flow passes
kinematic_viscosity = 1.508e-5  # Kinematic viscosity of air in m^2/s 1.508e-5
voxel_size = 1  # Finest voxel size in meters (user input)

# STL filename
stl_filename = "examples/cfd/stl-files/university_ave_buildings.obj"
script_name = "test_buildings"

# I/O settings
print_interval_percentage = 1  # Print every 1% of iterations
file_output_crossover_percentage = 1  # Crossover at 50% of iterations
num_file_outputs_pre_crossover = 3  # Outputs before crossover
num_file_outputs_post_crossover = 3  # Outputs after crossover

# Other setup parameters
compute_backend = ComputeBackend.NEON
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)


# Utility Functions
# =================
def compute_voxel_statistics(
    bc_mask,
    bc_mask_exporter,
    sparsity_pattern,
):
    """
    Compute active/solid voxels, totals, lattice updates per step
    """
    num_levels = len(sparsity_pattern)
    fields_data = bc_mask_exporter.get_fields_data({"bc_mask": bc_mask})
    bc_mask_data = fields_data["bc_mask_0"]
    level_id_field = bc_mask_exporter.level_id_field

    # Compute solid voxels per level (assuming 255 is the solid marker)
    solid_voxels = []
    for lvl in range(num_levels):
        level_mask = level_id_field == lvl
        solid_voxels.append(np.sum(bc_mask_data[level_mask] == 255))

    # Compute active voxels (total non-zero in sparsity minus solids)
    active_voxels = [np.count_nonzero(mask) for mask in sparsity_pattern]
    active_voxels = [max(0, active_voxels[lvl] - solid_voxels[lvl]) for lvl in range(num_levels)]

    # Totals
    total_voxels = sum(active_voxels)
    total_lattice_updates_per_step = sum(active_voxels[lvl] * (2 ** (num_levels - 1 - lvl)) for lvl in range(num_levels))

    return {
        "active_voxels": active_voxels,
        "solid_voxels": solid_voxels,
        "total_voxels": total_voxels,
        "total_lattice_updates_per_step": total_lattice_updates_per_step,
    }


# Mesh Generation Functions
# =========================
def generate_cuboid_mesh(stl_filename, voxel_size):
    """
    Generate a makemesh mesh based on the provided voxel size in meters, domain multipliers, and padding values.
    """

    # Domain multipliers for each refinement level
    # First entry should be full domain size
    # Domain multipliers
    domainMultiplier = [
        [1.5, 3, 0, 6, 1.5, 1.5],  # -x, x, -y, y, -z, z
        [1, 2, 0, 4.5, 1, 1],  # -x, x, -y, y, -z, z
        [0.5, 1, 0, 3, 0.5, 0.5],
        [0.25, 0.5, 0, 2, 0.25, 0.25],
        # [1, 2, 1, 1, 1, 1],
        # [0.4, 1, 0.4, 0.4, 0.4, 0.4],
        # [0.2, 0.4, 0.2, 0.2, 0.2, 0.2],
    ]

    # Number of requested refinement levels
    num_levels = len(domainMultiplier)

    # Load the mesh (OBJ files may load as a Scene with multiple geometries)
    loaded = trimesh.load(stl_filename, process=False)
    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            raise ValueError("Loaded mesh is empty or invalid.")
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    else:
        mesh = loaded
    if mesh.is_empty:
        raise ValueError("Loaded mesh is empty or invalid.")

    # Rotate the mesh if needed
    # angle = -90*(np.pi/180)    # 90 degrees in radians
    # axis = [0, 1, 0]     # Rotate around the Y-axis
    # center = [0, 0, 0]   # Rotate around the origin
    # rot_matrix = trimesh.transformations.rotation_matrix(angle, axis, center)
    # mesh.apply_transform(rot_matrix)

    # Compute original bounds
    min_bound = mesh.vertices.min(axis=0)
    max_bound = mesh.vertices.max(axis=0)
    partSize = max_bound - min_bound

    # Compute translation to put mesh into first octant of the domain
    stl_shift = np.array(
        [
            domainMultiplier[0][0] * partSize[0] - min_bound[0],
            domainMultiplier[0][2] * partSize[1] - min_bound[1],
            domainMultiplier[0][4] * partSize[2] - min_bound[2],
        ],
        dtype=float,
    )

    # Apply translation and save out temp STL
    mesh.apply_translation(stl_shift)
    _ = mesh.vertex_normals
    # XLB expects triangle soup: each row triplet is one face (not indexed vertices)
    mesh_vertices = np.asarray(mesh.vertices[mesh.faces].reshape(-1, 3))
    mesh.export("temp.stl")

    # Generate mesh using generate_mesh with ground refinement
    level_data = make_cuboid_mesh(voxel_size, domainMultiplier, "temp.stl")

    # Print some info
    grid_shape_finest = tuple([int(i * 2 ** (num_levels - 1)) for i in level_data[-1][0].shape])
    print(f"Requested levels: {num_levels}, Actual levels: {num_levels}")
    print(f"Full shape based on finest voxel size is {grid_shape_finest}")
    os.remove("temp.stl")

    return (
        level_data,
        mesh_vertices,
        tuple([int(a) for a in grid_shape_finest]),
        stl_shift,
    )


# Boundary Conditions Setup
# =========================
def setup_boundary_conditions(grid, level_data, building_vertices, wind_speed_mps):
    """
    Set up boundary conditions for the simulation.
    """
    # Convert wind speed to lattice units
    wind_speed_lbm = unit_convertor.velocity_to_lbm(wind_speed_mps)

    num_levels = len(level_data)
    coarsest_level = num_levels - 1
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
    filtered_top_indices = []
    filtered_bottom_indices = []
    for level in range(num_levels):
        left_set = set(zip(*left_indices[level])) if left_indices[level] else set()
        right_set = set(zip(*right_indices[level])) if right_indices[level] else set()
        top_set = set(zip(*top_indices[level])) if top_indices[level] else set()
        bottom_set = set(zip(*bottom_indices[level])) if bottom_indices[level] else set()
        front_set = set(zip(*front_indices[level])) if front_indices[level] else set()
        back_set = set(zip(*back_indices[level])) if back_indices[level] else set()
        filtered_front_set = front_set - (top_set | bottom_set)
        filtered_back_set = back_set - (top_set | bottom_set)
        filtered_top_set = top_set - (left_set | right_set)
        filtered_bottom_set = bottom_set - (left_set | right_set)
        filtered_front_indices.append([list(coords) for coords in zip(*filtered_front_set)] if filtered_front_set else [])
        filtered_back_indices.append([list(coords) for coords in zip(*filtered_back_set)] if filtered_back_set else [])
        filtered_top_indices.append([list(coords) for coords in zip(*filtered_top_set)] if filtered_top_set else [])
        filtered_bottom_indices.append([list(coords) for coords in zip(*filtered_bottom_set)] if filtered_bottom_set else [])

    # Turbulent Flow Profile
    def bc_profile_taper(taper_fraction=0.07):
        assert compute_backend == ComputeBackend.NEON
        _, ny, nz = grid_shape_finest
        dtype = precision_policy.compute_precision.wp_dtype
        H_y = dtype(ny)
        H_z = dtype(nz)
        two = dtype(2.0)
        wind_speed_lbm_wp = dtype(wind_speed_lbm)
        taper_frac = dtype(taper_fraction)
        core_frac = dtype(1.0 - 2.0 * taper_fraction)
        _u_vec = wp.vec(velocity_set.d, dtype=dtype)

        @wp.func
        def bc_profile_warp(index: wp.vec3i):
            y = dtype(index[1])
            z = dtype(index[2])
            y_center = wp.abs(y - (H_y / two))
            z_center = wp.abs(z - (H_z / two))
            y_norm = two * y_center / H_y
            z_norm = two * z_center / H_z
            max_norm = wp.max(y_norm, z_norm)
            velocity = wind_speed_lbm_wp
            if max_norm > core_frac:
                velocity = wind_speed_lbm_wp * (dtype(1.0) - (max_norm - core_frac) / taper_frac)
            velocity = wp.max(dtype(0.0), velocity)
            return wp.vec(velocity, length=1)

        return bc_profile_warp

    # Initialize boundary conditions

    bc_inlet = RegularizedBC(
        "velocity",
        # profile=bc_profile_taper(),
        prescribed_value=(wind_speed_lbm, 0.0, 0.0),
        indices=left_indices,
    )

    bc_outlet = DoNothingBC(indices=right_indices)
    bc_side1 = HybridBC(bc_method="nonequilibrium_regularized", prescribed_value=(wind_speed_lbm, 0.0, 0.0), indices=top_indices)
    bc_side2 = HybridBC(bc_method="nonequilibrium_regularized", prescribed_value=(wind_speed_lbm, 0.0, 0.0), indices=bottom_indices)
    bc_ground = HybridBC(bc_method="nonequilibrium_regularized", prescribed_value=(0.0, 0.0, 0.0), indices=filtered_front_indices)
    bc_top = HybridBC(bc_method="nonequilibrium_regularized", prescribed_value=(wind_speed_lbm, 0.0, 0.0), indices=filtered_back_indices)

    bc_body = HybridBC(
        bc_method="nonequilibrium_regularized",
        mesh_vertices=unit_convertor.length_to_lbm(building_vertices),
        voxelization_method=MeshVoxelizationMethod("AABB"),
        use_mesh_distance=False,
    )

    return [bc_side1, bc_side2, bc_ground, bc_top, bc_inlet, bc_outlet, bc_body]  # Body must be last. Outlet must be second to last

# Main Script
# ===========
# Initialize XLB
xlb.init(
    velocity_set=velocity_set,
    default_backend=compute_backend,
    default_precision_policy=precision_policy,
)

# Generate mesh
level_data, building_vertices, grid_shape_finest, stl_shift = generate_cuboid_mesh(stl_filename, voxel_size)

# Prepare the sparsity pattern and origins from the level data
sparsity_pattern, level_origins = prepare_sparsity_pattern(level_data)

# Define a unit convertor
unit_convertor = UnitConvertor(
    velocity_lbm_unit=wind_speed_lbm,
    velocity_physical_unit=wind_speed_mps,
    voxel_size_physical_unit=voxel_size,
)


# Calculate lattice parameters
num_levels = len(level_data)
delta_x_coarse = voxel_size * 2 ** (num_levels - 1)
nu_lattice = unit_convertor.viscosity_to_lbm(kinematic_viscosity)
omega_finest = 1.0 / (3.0 * nu_lattice + 0.5)

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
    offset=-stl_shift,
    unit_convertor=unit_convertor,
)
bc_mask_exporter = MultiresIO(
    {"bc_mask": 1},
    level_data,
    offset=-stl_shift,
    unit_convertor=unit_convertor,
)


# Create grid
grid = multires_grid_factory(
    grid_shape_finest,
    velocity_set=velocity_set,
    sparsity_pattern_list=sparsity_pattern,
    sparsity_pattern_origins=[neon.Index_3d(*box_origin) for box_origin in level_origins],
)

# Calculate num_steps
coarsest_level = grid.count_levels - 1
grid_shape_x_coarsest = grid.level_to_shape(coarsest_level)[0]
num_steps = int(flow_passes * (grid_shape_x_coarsest / wind_speed_lbm))

# Calculate print and file output intervals
print_interval = max(1, int(num_steps * (print_interval_percentage / 100.0)))
crossover_step = int(num_steps * (file_output_crossover_percentage / 100.0))
file_output_interval_pre_crossover = (
    max(1, int(crossover_step / num_file_outputs_pre_crossover)) if num_file_outputs_pre_crossover > 0 else num_steps + 1
)
file_output_interval_post_crossover = (
    max(1, int((num_steps - crossover_step) / num_file_outputs_post_crossover)) if num_file_outputs_post_crossover > 0 else num_steps + 1
)
final_print_interval = max(1, int((num_steps - crossover_step) * (print_interval_percentage / 100.0)))

# Setup boundary conditions
boundary_conditions = setup_boundary_conditions(grid, level_data, building_vertices, wind_speed_mps)

# Create initializer
wind_speed_lbm = unit_convertor.velocity_to_lbm(wind_speed_mps)
initializer = CustomMultiresInitializer(
    bc_id=boundary_conditions[-2].id,  # bc_outlet
    constant_velocity_vector=(wind_speed_lbm, 0.0, 0.0),
    velocity_set=velocity_set,
    precision_policy=precision_policy,
    compute_backend=compute_backend,
)

# Initialize simulation
sim = xlb.helper.MultiresSimulationManager(
    omega_finest=omega_finest,
    grid=grid,
    boundary_conditions=boundary_conditions,
    collision_type="KBC",
    initializer=initializer,
    mres_perf_opt=xlb.mres_perf_optimization_type.MresPerfOptimizationType.FUSION_AT_FINEST,
)

# Compute voxel statistics
stats = compute_voxel_statistics(sim.bc_mask, bc_mask_exporter, sparsity_pattern)
active_voxels = stats["active_voxels"]
solid_voxels = stats["solid_voxels"]
total_voxels = stats["total_voxels"]
total_lattice_updates_per_step = stats["total_lattice_updates_per_step"]

# Save initial bc_mask
filename = os.path.join(output_dir, f"{script_name}_initial_bc_mask")
try:
    bc_mask_exporter.to_hdf5(filename, {"bc_mask": sim.bc_mask}, compression="gzip", compression_opts=0)
except Exception as e:
    print(f"Error during initial bc_mask output: {e}")
wp.synchronize()


# Print simulation info
print("\n" + "=" * 50 + "\n")
print(f"Number of flow passes: {flow_passes}")
print(f"Calculated iterations: {num_steps:,}")
print(f"Finest voxel size: {voxel_size} meters")
print(f"Coarsest voxel size: {delta_x_coarse} meters")
print(f"Total voxels: {sum(np.count_nonzero(mask) for mask in sparsity_pattern):,}")
print(f"Total active voxels: {total_voxels:,}")
print(f"Active voxels per level: {active_voxels}")
print(f"Solid voxels per level: {solid_voxels}")
print(f"Total lattice updates per global step: {total_lattice_updates_per_step:,}")
print(f"Actual number of refinement levels: {num_levels}")
print(f"Physical inlet velocity: {wind_speed_mps:.4f} m/s")
print(f"Lattice velocity (wind_speed_lbm): {wind_speed_lbm}")
print(f"Relaxation parameter (omega): {omega_finest:.5f}")
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
        end_time = time.time()
        elapsed = end_time - start_time
        total_lattice_updates = total_lattice_updates_per_step * steps_since_last_print
        MLUPS = total_lattice_updates / compute_time / 1e6 if compute_time > 0 else 0.0
        current_flow_passes = step * wind_speed_lbm / grid_shape_x_coarsest
        remaining_steps = num_steps - step - 1
        time_remaining = 0.0 if MLUPS == 0 else (total_lattice_updates_per_step * remaining_steps) / (MLUPS * 1e6)
        hours, rem = divmod(time_remaining, 3600)
        minutes, seconds = divmod(rem, 60)
        time_remaining_str = f"{int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s"
        percent_complete = (step + 1) / num_steps * 100
        print(f"Completed step {step}/{num_steps} ({percent_complete:.2f}% complete)")
        print(f"  Flow Passes: {current_flow_passes:.2f}")
        print(f"  Time elapsed: {elapsed:.1f}s, Compute time: {compute_time:.1f}s, ETA: {time_remaining_str}")
        print(f"  MLUPS: {MLUPS:.1f}")
        start_time = time.time()
        compute_time = 0.0
        steps_since_last_print = 0
    file_output_interval = file_output_interval_pre_crossover if step < crossover_step else file_output_interval_post_crossover
    if step % file_output_interval == 0 or step == num_steps - 1:
        sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
        filename = os.path.join(output_dir, f"{script_name}_{step:04d}")
        try:
            h5exporter.to_hdf5(filename, {"velocity": sim.u, "density": sim.rho}, compression="gzip", compression_opts=1)
        except Exception as e:
            print(f"Error during file output at step {step}: {e}")
        wp.synchronize()
    if step >= crossover_step and step % final_print_interval == 0:
        sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
        wp.synchronize()
        filename = os.path.join(output_dir, f"{script_name}_{step:04d}")
        # h5exporter.to_slice_image(
        #     filename,
        #     {"velocity": sim.u},
        #     plane_point=(1, 0, 0),
        #     plane_normal=(0, 1, 0),
        #     grid_res=2000,
        #     bounds=(0, 1, 0, 1),
        #     show_axes=False,
        #     show_colorbar=False,
        #     slice_thickness=delta_x_coarse,  # needed when using model units
        #     normalize=wind_speed_mps * 1.5,  # eventually we could have the 1.5 read from json as we did before
        # )
        h5exporter.to_hdf5(filename, {"velocity": sim.u, "density": sim.rho}, compression="gzip", compression_opts=1)
