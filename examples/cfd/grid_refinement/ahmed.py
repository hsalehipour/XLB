import neon
import warp as wp
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import trimesh
import shutil

import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import multires_grid_factory
from xlb.operator.boundary_condition import (
    DoNothingBC,
    HybridBC,
    RegularizedBC,
)
from xlb.operator.boundary_masker import MeshVoxelizationMethod
from xlb.utils.mesher import prepare_sparsity_pattern, make_cuboid_mesh, MultiresIO
from xlb.utils import UnitConvertor
from xlb.operator.force import MultiresMomentumTransfer
from xlb.helper.initializers import CustomMultiresInitializer

wp.clear_kernel_cache()
wp.config.quiet = True

# User Configuration
# =================
# Physical and simulation parameters
wind_speed_lbm = 0.05  # Lattice velocity
wind_speed_mps = 38.0  # Physical inlet velocity in m/s (user input)
flow_passes = 2  # Domain flow passes
kinematic_viscosity = 1.508e-5  # Kinematic viscosity of air in m^2/s 1.508e-5
voxel_size = 0.005  # Finest voxel size in meters

# STL filename
stl_filename = "examples/cfd/stl-files/Ahmed_25_NoLegs.stl"
script_name = "Ahmed"

# I/O settings
print_interval_percentage = 1  # Print every 1% of iterations
file_output_crossover_percentage = 10  # Crossover at 50% of iterations
num_file_outputs_pre_crossover = 20  # Outputs before crossover
num_file_outputs_post_crossover = 5  # Outputs after crossover

# Other setup parameters
compute_backend = ComputeBackend.NEON
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)


def generate_cuboid_mesh(stl_filename, voxel_size):
    """
    Alternative cuboid mesh generation based on Apolo's method with domain multipliers per level.
    """
    # Domain multipliers for each refinement level
    domain_multiplier = [
        [3.0, 4.0, 2.5, 2.5, 0.0, 4.0],     # -x, x, -y, y, -z, z
        [1.2, 1.25, 1.75, 1.75, 0.0, 1.5],
        [0.8, 1.0, 1.25, 1.25, 0.0, 1.2],
        [0.5, 0.65, 0.6, 0.60, 0.0, 0.6],
        [0.25, 0.25, 0.25, 0.25, 0.0, 0.25],
    ]

    # Load the mesh
    mesh = trimesh.load_mesh(stl_filename, process=False)
    if mesh.is_empty:
        raise ValueError("Loaded mesh is empty or invalid.")

    # Compute original bounds
    min_bound = mesh.vertices.min(axis=0)
    max_bound = mesh.vertices.max(axis=0)
    partSize = max_bound - min_bound
    x0 = max_bound[0]  # End of car for Ahmed

    # Compute translation to put mesh into first octant of the domain
    stl_shift = np.array(
        [
            domain_multiplier[0][0] * partSize[0] - min_bound[0],
            domain_multiplier[0][2] * partSize[1] - min_bound[1],
            domain_multiplier[0][4] * partSize[2] - min_bound[2],
        ],
        dtype=float,
    )

    # Apply translation and save out temp STL
    mesh.apply_translation(stl_shift)
    _ = mesh.vertex_normals
    mesh_vertices = np.asarray(mesh.vertices)
    mesh.export("temp.stl")

    # Generate mesh using make_cuboid_mesh
    level_data = make_cuboid_mesh(
        voxel_size,
        domain_multiplier,
        "temp.stl",
    )

    num_levels = len(level_data)
    grid_shape_finest = tuple([int(i * 2 ** (num_levels - 1)) for i in level_data[-1][0].shape])
    print(f"Full shape based on finest voxel size is {grid_shape_finest}")
    os.remove("temp.stl")

    return (
        level_data,
        mesh_vertices,
        tuple([int(a) for a in grid_shape_finest]),
        stl_shift,
        x0,
    )


# Boundary Conditions Setup
# =========================
def setup_boundary_conditions(grid, level_data, body_vertices, wind_speed_mps):
    """
    Set up boundary conditions for the simulation.
    """
    # Convert wind speed to lattice units
    wind_speed_lbm = unit_convertor.velocity_to_lbm(wind_speed_mps)

    left_indices = grid.boundary_indices_across_levels(level_data, box_side="left", remove_edges=True)
    right_indices = grid.boundary_indices_across_levels(level_data, box_side="right", remove_edges=True)
    top_indices = grid.boundary_indices_across_levels(level_data, box_side="top", remove_edges=False)
    bottom_indices = grid.boundary_indices_across_levels(level_data, box_side="bottom", remove_edges=False)
    front_indices = grid.boundary_indices_across_levels(level_data, box_side="front", remove_edges=False)
    back_indices = grid.boundary_indices_across_levels(level_data, box_side="back", remove_edges=False)

    # Initialize boundary conditions
    bc_inlet = RegularizedBC("velocity", prescribed_value=(wind_speed_lbm, 0.0, 0.0), indices=left_indices)
    bc_outlet = DoNothingBC(indices=right_indices)
    bc_top = HybridBC(bc_method="nonequilibrium_regularized", indices=top_indices)
    bc_bottom = HybridBC(bc_method="nonequilibrium_regularized", indices=bottom_indices)
    bc_front = HybridBC(bc_method="nonequilibrium_regularized", indices=front_indices)
    bc_back = HybridBC(bc_method="nonequilibrium_regularized", indices=back_indices)
    bc_body = HybridBC(
        bc_method="nonequilibrium_regularized",
        mesh_vertices=unit_convertor.length_to_lbm(body_vertices),
        voxelization_method=MeshVoxelizationMethod("AABB_CLOSE", close_voxels=4),
        use_mesh_distance=True,
    )

    return [bc_top, bc_bottom, bc_front, bc_back, bc_inlet, bc_outlet, bc_body]


# Simulation Initialization
# =========================
def initialize_simulation(
    grid, boundary_conditions, omega_finest, initializer, collision_type="KBC", mres_perf_opt=xlb.MresPerfOptimizationType.FUSION_AT_FINEST
):
    """
    Initialize the multiresolution simulation manager.
    """
    sim = xlb.helper.MultiresSimulationManager(
        omega_finest=omega_finest,
        grid=grid,
        boundary_conditions=boundary_conditions,
        collision_type=collision_type,
        initializer=initializer,
        mres_perf_opt=mres_perf_opt,
    )
    return sim


# Utility Functions
# =================
def print_lift_drag(sim, step, momentum_transfer, wind_speed_lbm, reference_area):
    """
    Calculate and print lift and drag coefficients.
    """
    boundary_force = momentum_transfer(sim.f_0, sim.f_1, sim.bc_mask, sim.missing_mask)
    drag = boundary_force[0]
    lift = boundary_force[2]
    cd = 2.0 * drag / (wind_speed_lbm**2 * reference_area)
    cl = 2.0 * lift / (wind_speed_lbm**2 * reference_area)
    if np.isnan(cd) or np.isnan(cl):
        print(f"NaN detected in coefficients at step {step}")
        raise ValueError(f"NaN detected in coefficients at step {step}: Cd={cd}, Cl={cl}")
    drag_values.append([cd, cl])
    return cd, cl, drag


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
    plt.plot(steps, cd_values, label="Drag Coefficient (Cd)", color="blue")
    plt.plot(steps, cl_values, label="Lift Coefficient (Cl)", color="red")
    plt.xlabel("Simulation Step")
    plt.ylabel("Coefficient")
    plt.title(f"{script_name}: Drag and Lift Coefficients Over Time")
    plt.legend()
    plt.grid(True)
    plt.ylim(y_min, y_max)
    if use_log_scale:
        plt.yscale("log")
    plt.savefig(os.path.join(output_dir, "drag_lift_plot.png"))
    plt.close()


def compute_voxel_statistics(sim, bc_mask_exporter, sparsity_pattern, boundary_conditions, unit_convertor):
    """
    Compute active/solid voxels, totals, lattice updates, and reference area based on simulation data.
    """
    fields_data = bc_mask_exporter.get_fields_data({"bc_mask": sim.bc_mask})
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

    # Compute reference area (projected on YZ plane at finest level)
    finest_level = 0
    mask_finest = level_id_field == finest_level
    bc_mask_finest = bc_mask_data[mask_finest]
    active_indices_finest = np.argwhere(sparsity_pattern[0])
    bc_body_id = boundary_conditions[-1].id  # Assuming last BC is bc_body
    solid_voxels_indices = active_indices_finest[bc_mask_finest == bc_body_id]
    unique_jk = np.unique(solid_voxels_indices[:, 1:3], axis=0)
    reference_area = unique_jk.shape[0]
    reference_area_physical = reference_area * unit_convertor.reference_length**2

    return {
        "active_voxels": active_voxels,
        "solid_voxels": solid_voxels,
        "total_voxels": total_voxels,
        "total_lattice_updates_per_step": total_lattice_updates_per_step,
        "reference_area": reference_area,
        "reference_area_physical": reference_area_physical,
    }


def plot_data(x0, output_dir, delta_x_coarse, sim, IOexporter, prefix="Ahmed"):
    """
    Ahmed Car Model, slant - angle = 25 degree
    Profiles on symmetry plane (y=0) covering entire field
    Origin of coordinate system:
         x=0: end of the car, y=0: symmetry plane, z=0: ground plane

    S.Becker/H. Lienhart/C.Stoots
    Insitute of Fluid Mechanics
    University Erlangen-Nuremberg
    Erlangen, Germany
    Coordaintes in meters need to convert to voxels
    Velocity data in m/s
    """

    def _load_sim_line(csv_path):
        """
        Read a CSV exported by IOexporter.to_line without pandas.
        Returns (z, Ux).
        """
        # Read with header as column names
        data = np.genfromtxt(
            csv_path,
            delimiter=",",
            names=True,
            autostrip=True,
            dtype=None,
            encoding="utf-8",
        )
        if data.size == 0:
            raise ValueError(f"No data in {csv_path}")

        z = np.asarray(data["z"], dtype=float)
        ux = np.asarray(data["value"], dtype=float)
        return z, ux

    # Load reference data
    import json

    ref_data_path = "examples/cfd/data/ahmed.json"
    with open(ref_data_path, "r") as file:
        data = json.load(file)

    for x_str in data["data"].keys():
        # Extract reference horizontal velocity in m/s and its corresponding height in m
        refX = np.array(data["data"][x_str]["x-velocity"])
        refY = np.array(data["data"][x_str]["height"])

        # From reference x0 (rear of body) find x1 for plot
        x_pos = float(x_str)
        x1 = x0 + x_pos

        print(f" x1 is {x1}")
        sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
        filename = os.path.join(output_dir, f"{prefix}_{x_str}")
        wp.synchronize()
        IOexporter.to_line(
            filename,
            {"velocity": sim.u},
            start_point=(x1, 0, 0),
            end_point=(x1, 0, 0.8),
            resolution=250,
            component=0,
            radius=delta_x_coarse,  # needed with model units
        )
        # read the CSV written by the exporter
        csv_path = filename + "_velocity_0.csv"
        print(f"CSV path is {csv_path}")

        try:
            sim_z, sim_ux = _load_sim_line(csv_path)
        except Exception as e:
            print(f"Failed to read {csv_path}: {e}")
            continue

        # plot reference vs simulation
        plt.figure(figsize=(4.5, 6))
        plt.plot(refX, refY, "o", mfc="none", label="Experimental)")
        plt.plot(sim_ux, sim_z, "-", lw=2, label="Simulation")
        plt.xlim(np.min(refX) * 0.9, np.max(refX) * 1.1)
        plt.ylim(np.min(refY), np.max(refY))
        plt.xlabel("Ux [m/s]")
        plt.ylabel("z [m]")
        plt.title(f"Velocity Plot at {x_pos:+.3f}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename + ".png", dpi=150)
        plt.close()


# Main Script
# ===========
# Initialize XLB

xlb.init(
    velocity_set=velocity_set,
    default_backend=compute_backend,
    default_precision_policy=precision_policy,
)

# Generate mesh
level_data, body_vertices, grid_shape_zip, stl_shift, x0 = generate_cuboid_mesh(stl_filename, voxel_size)

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
    grid_shape_zip,
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

# Setup boundary conditions
boundary_conditions = setup_boundary_conditions(grid, level_data, body_vertices, wind_speed_mps)

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
sim = initialize_simulation(grid, boundary_conditions, omega_finest, initializer)

# Compute voxel statistics and reference area
stats = compute_voxel_statistics(sim, bc_mask_exporter, sparsity_pattern, boundary_conditions, unit_convertor)
active_voxels = stats["active_voxels"]
solid_voxels = stats["solid_voxels"]
total_voxels = stats["total_voxels"]
total_lattice_updates_per_step = stats["total_lattice_updates_per_step"]
reference_area = stats["reference_area"]
reference_area_physical = stats["reference_area_physical"]

# Save initial bc_mask
filename = os.path.join(output_dir, f"{script_name}_initial_bc_mask")
try:
    bc_mask_exporter.to_hdf5(filename, {"bc_mask": sim.bc_mask}, compression="gzip", compression_opts=0)
    xmf_filename = f"{filename}.xmf"
    hdf5_basename = f"{script_name}_initial_bc_mask.h5"
except Exception as e:
    print(f"Error during initial bc_mask output: {e}")
wp.synchronize()


# Setup momentum transfer
momentum_transfer = MultiresMomentumTransfer(
    boundary_conditions[-1],
    mres_perf_opt=xlb.MresPerfOptimizationType.FUSION_AT_FINEST,
    compute_backend=compute_backend,
)

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
print(f"Number of refinement levels: {num_levels}")
print(f"Physical inlet velocity: {wind_speed_mps:.4f} m/s")
print(f"Lattice velocity (ulb): {wind_speed_lbm}")
print(f"Computed reference area (bc_mask): {reference_area} lattice units")
print(f"Physical reference area (bc_mask): {reference_area_physical:.6f} m^2")
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
        cd, cl, drag = print_lift_drag(sim, step, momentum_transfer, wind_speed_lbm, reference_area)
        filename = os.path.join(output_dir, f"{script_name}_{step:04d}")
        h5exporter.to_hdf5(filename, {"velocity": sim.u, "density": sim.rho}, compression="gzip", compression_opts=0)
        h5exporter.to_slice_image(
            filename,
            {"velocity": sim.u},
            plane_point=(1, 0, 0),
            plane_normal=(0, 1, 0),
            grid_res=2000,
            bounds=(0.25, 0.75, 0, 0.5),
            show_axes=False,
            show_colorbar=False,
            slice_thickness=delta_x_coarse,  # needed when using model units
        )
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
        print(f"  Cd={cd:.3f}, Cl={cl:.3f}, Drag Force (lattice units)={drag:.3f}")
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
    if step == num_steps - 1:
        plot_data(x0, output_dir, delta_x_coarse, sim, h5exporter, prefix="Ahmed")

# Save drag and lift data to CSV
if len(drag_values) > 0:
    with open(os.path.join(output_dir, "drag_lift.csv"), "w") as fd:
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
    print(f"Experimental Drag Coefficient (Cd): {0.3088}")
    print(f"Error Drag Coefficient (Cd): {((avg_cd - 0.3088) / 0.3088) * 100:.2f}%")

else:
    print("No drag or lift data collected.")
