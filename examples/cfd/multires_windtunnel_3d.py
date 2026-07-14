"""
Ahmed body aerodynamics with multi-resolution LBM.

Simulates turbulent flow around the Ahmed body (25-degree slant angle)
using the XLB multi-resolution Neon backend.  Computes drag and lift
coefficients via momentum transfer and exports HDF5/XDMF data for
post-processing.

Coordinate convention (mesh-domain frame after STL shift):
    +x : streamwise (inlet at left, outlet at right)
    +y : spanwise  (front / back lateral faces)
    +z : vertical  (ground at bottom, open sky at top)
"""

import json
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
from xlb.operator.boundary_condition import DoNothingBC, HybridBC, RegularizedBC
from xlb.operator.boundary_masker import MeshVoxelizationMethod
from xlb.utils.mesher import MultiresIO, is_sparse_level_data, make_cuboid_mesh, prepare_sparsity_pattern
from xlb.utils.adaptive_mesher import grid_shape_finest, load_and_shift_stl, make_adaptive_surface_mesh
from xlb.utils import UnitConvertor
from xlb.operator.force import MultiresMomentumTransfer
from xlb.helper.initializers import CustomMultiresInitializer

wp.clear_kernel_cache()
wp.config.quiet = True

# User Configuration
# =================
# Physical and simulation parameters
wind_speed_lbm = 0.05  # Reference lattice velocity used for unit conversion
wind_speed_mps = 38.0  # Physical inlet velocity [m/s]
flow_passes = 2  # Number of domain-length transits before stopping
kinematic_viscosity = 1.508e-5  # Air kinematic viscosity [m^2/s]
voxel_size = 0.005  # Finest lattice cell size [m]

# Mesh generation: "cuboid" or "adaptive"
mesher_type = "adaptive"

# Cuboid mesher: nested domain multipliers per level [-x, +x, -y, +y, -z, +z]
domain_multiplier = [
    [3.0, 4.0, 2.5, 2.5, 0.0, 4.0],
    [1.2, 1.25, 1.75, 1.75, 0.0, 1.5],
    [0.8, 1.0, 1.25, 1.25, 0.0, 1.2],
    [0.5, 0.65, 0.6, 0.60, 0.0, 0.6],
    [0.25, 0.25, 0.25, 0.25, 0.0, 0.25],
]

# Adaptive mesher (see xlb.utils.adaptive_mesher)
num_levels = len(domain_multiplier)
domain_padding = domain_multiplier[0]
expansion_ratio = 2.0
finest_band_cells = 8

# Geometry and output
stl_filename = "examples/cfd/stl-files/Ahmed_25_NoLegs.stl"
script_name = "Ahmed"

# Progress reporting and HDF5 output scheduling
print_interval_percentage = 1
file_output_crossover_percentage = 10
num_file_outputs_pre_crossover = 20
num_file_outputs_post_crossover = 5

# Backend
compute_backend = ComputeBackend.NEON
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)


def _active_voxel_count(mask):
    """Count active voxels in a dense boolean mask or sparse (N, 3) coord array."""
    return mask.shape[0] if mask.ndim == 2 else int(np.count_nonzero(mask))


def _active_voxel_indices(mask):
    """Return active voxel indices as (N, 3) for dense or sparse level patterns."""
    return mask if mask.ndim == 2 else np.argwhere(mask)


def generate_cuboid_mesh(stl_filename, voxel_size):
    """Build a nested cuboid multires mesh around the Ahmed body."""
    mesh = trimesh.load_mesh(stl_filename, process=False)
    if mesh.is_empty:
        raise ValueError("Loaded mesh is empty or invalid.")

    min_bound = mesh.vertices.min(axis=0)
    max_bound = mesh.vertices.max(axis=0)
    part_size = max_bound - min_bound
    x0 = max_bound[0]

    stl_shift = np.array(
        [
            domain_multiplier[0][0] * part_size[0] - min_bound[0],
            domain_multiplier[0][2] * part_size[1] - min_bound[1],
            domain_multiplier[0][4] * part_size[2] - min_bound[2],
        ],
        dtype=float,
    )

    mesh.apply_translation(stl_shift)
    _ = mesh.vertex_normals
    mesh_vertices = np.asarray(mesh.vertices)
    mesh.export("temp.stl")

    level_data = make_cuboid_mesh(voxel_size, domain_multiplier, "temp.stl")
    finest_shape = tuple(int(i * 2 ** (len(level_data) - 1)) for i in level_data[-1][0].shape)
    print(f"[cuboid] levels={len(level_data)}, grid_shape_finest={finest_shape}")
    os.remove("temp.stl")

    return level_data, mesh_vertices, finest_shape, stl_shift, x0


def generate_adaptive_mesh(stl_filename, voxel_size):
    """Build a surface-adaptive multires mesh around the Ahmed body."""
    mesh = trimesh.load_mesh(stl_filename, process=False)
    if mesh.is_empty:
        raise ValueError("Loaded mesh is empty or invalid.")
    x0 = mesh.vertices.max(axis=0)[0]

    temp_stl = None
    try:
        temp_stl, stl_shift = load_and_shift_stl(stl_filename, domain_padding)
        level_data = make_adaptive_surface_mesh(
            voxel_size=voxel_size,
            num_levels=num_levels,
            stl_filename=temp_stl,
            domain_padding=domain_padding,
            expansion_ratio=expansion_ratio,
            finest_band_cells=finest_band_cells,
        )
        finest_shape = grid_shape_finest(level_data)
        shifted_mesh = trimesh.load_mesh(temp_stl, process=False)
        mesh_vertices = np.asarray(shifted_mesh.vertices)

        print(f"[adaptive] levels={len(level_data)}, sparse={is_sparse_level_data(level_data)}")
        print(f"[adaptive] grid_shape_finest={finest_shape}")

        return level_data, mesh_vertices, tuple(int(a) for a in finest_shape), stl_shift, x0
    finally:
        if temp_stl is not None and os.path.isfile(temp_stl):
            os.remove(temp_stl)


def generate_mesh(stl_filename, voxel_size):
    """Dispatch to cuboid or adaptive mesher based on ``mesher_type``."""
    if mesher_type == "cuboid":
        return generate_cuboid_mesh(stl_filename, voxel_size)
    if mesher_type == "adaptive":
        return generate_adaptive_mesh(stl_filename, voxel_size)
    raise ValueError(f"Unknown mesher_type: {mesher_type!r}. Use 'cuboid' or 'adaptive'.")


def setup_boundary_conditions(grid, level_data, body_vertices, unit_convertor, wind_speed_mps):
    """
    Configure domain-face and body-surface boundary conditions.

    The body mesh BC must remain last; the outlet must be second-to-last
    (required by ``CustomMultiresInitializer`` and force operators).
    """
    wind_speed_lbm_local = unit_convertor.velocity_to_lbm(wind_speed_mps)
    sides = ("left", "right", "top", "bottom", "front", "back")
    indices = {
        side: grid.boundary_indices_across_levels(
            level_data,
            box_side=side,
            remove_edges=(side in ("left", "right")),
        )
        for side in sides
    }

    return [
        HybridBC(bc_method="nonequilibrium_regularized", indices=indices["top"]),
        HybridBC(bc_method="nonequilibrium_regularized", indices=indices["bottom"]),
        HybridBC(bc_method="nonequilibrium_regularized", indices=indices["front"]),
        HybridBC(bc_method="nonequilibrium_regularized", indices=indices["back"]),
        RegularizedBC("velocity", prescribed_value=(wind_speed_lbm_local, 0.0, 0.0), indices=indices["left"]),
        DoNothingBC(indices=indices["right"]),
        HybridBC(
            bc_method="nonequilibrium_regularized",
            mesh_vertices=unit_convertor.length_to_lbm(body_vertices),
            voxelization_method=MeshVoxelizationMethod("AABB_CLOSE", close_voxels=4),
            use_mesh_distance=True,
        ),
    ]


def compute_voxel_statistics(bc_mask, bc_mask_exporter, sparsity_pattern, boundary_conditions, unit_convertor):
    """
    Summarize sparsity, solid counts, lattice updates, and projected body area.
    """
    num_levels_local = len(sparsity_pattern)
    fields_data = bc_mask_exporter.get_fields_data({"bc_mask": bc_mask})
    bc_mask_data = fields_data["bc_mask_0"]
    level_id_field = bc_mask_exporter.level_id_field

    solid_voxels = [
        np.sum(bc_mask_data[level_id_field == lvl] == 255) for lvl in range(num_levels_local)
    ]
    active_voxels = [_active_voxel_count(mask) for mask in sparsity_pattern]
    active_voxels = [max(0, active_voxels[lvl] - solid_voxels[lvl]) for lvl in range(num_levels_local)]

    finest_mask = level_id_field == 0
    bc_mask_finest = bc_mask_data[finest_mask]
    active_indices_finest = _active_voxel_indices(sparsity_pattern[0])
    body_indices = active_indices_finest[bc_mask_finest == boundary_conditions[-1].id]
    reference_area = np.unique(body_indices[:, 1:3], axis=0).shape[0]

    return {
        "active_voxels": active_voxels,
        "solid_voxels": solid_voxels,
        "total_voxels": sum(active_voxels),
        "total_lattice_updates_per_step": sum(
            active_voxels[lvl] * (2 ** (num_levels_local - 1 - lvl)) for lvl in range(num_levels_local)
        ),
        "reference_area": reference_area,
        "reference_area_physical": reference_area * unit_convertor.reference_length**2,
    }


def check_fields_finite(sim, step, h5exporter):
    """Raise ``ValueError`` if velocity or density contains NaN."""
    fields = h5exporter.get_fields_data({"velocity": sim.u, "density": sim.rho})
    for name, data in fields.items():
        if np.isnan(data).any():
            raise ValueError(f"NaN detected in {name} at step {step}")


def save_fields(h5exporter, sim, output_dir, step):
    """Write velocity and density fields to HDF5/XDMF at the given step."""
    filename = os.path.join(output_dir, f"{script_name}_{step:04d}")
    h5exporter.to_hdf5(filename, {"velocity": sim.u, "density": sim.rho}, compression="gzip", compression_opts=0)
    wp.synchronize()


def print_progress(
    step,
    num_steps,
    grid_shape_x_coarsest,
    total_lattice_updates_per_step,
    steps_since_last_print,
    start_time,
    compute_time,
    cd=None,
    cl=None,
    drag=None,
):
    """Print flow-pass progress, wall time, ETA, MLUPS, and optional force coefficients."""
    elapsed = time.time() - start_time
    total_lattice_updates = total_lattice_updates_per_step * steps_since_last_print
    mlups = total_lattice_updates / compute_time / 1e6 if compute_time > 0 else 0.0
    remaining_steps = num_steps - step - 1
    time_remaining = 0.0 if mlups == 0 else (total_lattice_updates_per_step * remaining_steps) / (mlups * 1e6)
    hours, rem = divmod(time_remaining, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"Completed step {step}/{num_steps} ({(step + 1) / num_steps * 100:.2f}% complete)")
    print(f"  Flow Passes: {step * wind_speed_lbm / grid_shape_x_coarsest:.2f}")
    print(f"  Time elapsed: {elapsed:.1f}s, Compute time: {compute_time:.1f}s, ETA: {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s")
    print(f"  MLUPS: {mlups:.1f}")
    if cd is not None:
        print(f"  Cd={cd:.3f}, Cl={cl:.3f}, Drag Force (lattice units)={drag:.3f}")


def plot_force_coefficients(drag_values, output_dir, print_interval, percentile_range=(15, 85)):
    """Plot Cd and Cl over time and save to the output directory."""
    drag_values_array = np.array(drag_values)
    steps = np.arange(0, len(drag_values) * print_interval, print_interval)
    cd_values = drag_values_array[:, 0]
    cl_values = drag_values_array[:, 1]
    y_min = min(np.percentile(cd_values, percentile_range[0]), np.percentile(cl_values, percentile_range[0]))
    y_max = max(np.percentile(cd_values, percentile_range[1]), np.percentile(cl_values, percentile_range[1]))
    padding = (y_max - y_min) * 0.1
    y_min, y_max = y_min - padding, y_max + padding

    plt.figure(figsize=(10, 6))
    plt.plot(steps, cd_values, label="Drag Coefficient (Cd)", color="blue")
    plt.plot(steps, cl_values, label="Lift Coefficient (Cl)", color="red")
    plt.xlabel("Simulation Step")
    plt.ylabel("Coefficient")
    plt.title(f"{script_name}: Drag and Lift Coefficients Over Time")
    plt.legend()
    plt.grid(True)
    plt.ylim(y_min, y_max)
    plt.savefig(os.path.join(output_dir, "drag_lift_plot.png"))
    plt.close()


def plot_velocity_profiles(x0, output_dir, delta_x_coarse, sim, io_exporter, prefix="Ahmed"):
    """
    Compare streamwise velocity profiles on the symmetry plane to Ahmed reference data.
    """
    with open("examples/cfd/data/ahmed.json", "r") as file:
        ref_data = json.load(file)

    for x_str, profile in ref_data["data"].items():
        ref_ux = np.array(profile["x-velocity"])
        ref_z = np.array(profile["height"])
        x_pos = float(x_str)
        x1 = x0 + x_pos

        sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
        filename = os.path.join(output_dir, f"{prefix}_{x_str}")
        wp.synchronize()
        io_exporter.to_line(
            filename,
            {"velocity": sim.u},
            start_point=(x1, 0, 0),
            end_point=(x1, 0, 0.8),
            resolution=250,
            component=0,
            radius=delta_x_coarse,
        )

        csv_path = filename + "_velocity_0.csv"
        try:
            data = np.genfromtxt(csv_path, delimiter=",", names=True, autostrip=True, dtype=None, encoding="utf-8")
            if data.size == 0:
                raise ValueError(f"No data in {csv_path}")
            sim_z = np.asarray(data["z"], dtype=float)
            sim_ux = np.asarray(data["value"], dtype=float)
        except Exception as e:
            print(f"Failed to read {csv_path}: {e}")
            continue

        plt.figure(figsize=(4.5, 6))
        plt.plot(ref_ux, ref_z, "o", mfc="none", label="Experimental")
        plt.plot(sim_ux, sim_z, "-", lw=2, label="Simulation")
        plt.xlim(np.min(ref_ux) * 0.9, np.max(ref_ux) * 1.1)
        plt.ylim(np.min(ref_z), np.max(ref_z))
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
xlb.init(
    velocity_set=velocity_set,
    default_backend=compute_backend,
    default_precision_policy=precision_policy,
)

# --- Mesh and unit conversion ---
level_data, body_vertices, grid_shape_finest, stl_shift, x0 = generate_mesh(stl_filename, voxel_size)
sparsity_pattern, level_origins = prepare_sparsity_pattern(level_data)

unit_convertor = UnitConvertor(
    velocity_lbm_unit=wind_speed_lbm,
    velocity_physical_unit=wind_speed_mps,
    voxel_size_physical_unit=voxel_size,
)

num_levels = len(level_data)
delta_x_coarse = voxel_size * 2 ** (num_levels - 1)
omega_finest = 1.0 / (3.0 * unit_convertor.viscosity_to_lbm(kinematic_viscosity) + 0.5)

# --- Output directory and field exporters ---
output_dir = os.path.join(os.path.dirname(__file__), script_name)
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

h5exporter = MultiresIO({"velocity": 3, "density": 1}, level_data, offset=-stl_shift, unit_convertor=unit_convertor)
bc_mask_exporter = MultiresIO({"bc_mask": 1}, level_data, offset=-stl_shift, unit_convertor=unit_convertor)

# --- Grid and time-stepping schedule ---
grid = multires_grid_factory(
    grid_shape_finest,
    velocity_set=velocity_set,
    sparsity_pattern_list=sparsity_pattern,
    sparsity_pattern_origins=[neon.Index_3d(*origin) for origin in level_origins],
)

coarsest_level = grid.count_levels - 1
grid_shape_x_coarsest = grid.level_to_shape(coarsest_level)[0]
num_steps = int(flow_passes * (grid_shape_x_coarsest / wind_speed_lbm))

print_interval = max(1, int(num_steps * (print_interval_percentage / 100.0)))
crossover_step = int(num_steps * (file_output_crossover_percentage / 100.0))
file_output_interval_pre = (
    max(1, int(crossover_step / num_file_outputs_pre_crossover)) if num_file_outputs_pre_crossover else num_steps + 1
)
file_output_interval_post = (
    max(1, int((num_steps - crossover_step) / num_file_outputs_post_crossover))
    if num_file_outputs_post_crossover
    else num_steps + 1
)

# --- Boundary conditions, initializer, and simulation manager ---
boundary_conditions = setup_boundary_conditions(grid, level_data, body_vertices, unit_convertor, wind_speed_mps)
initializer = CustomMultiresInitializer(
    bc_id=boundary_conditions[-2].id,
    constant_velocity_vector=(wind_speed_lbm, 0.0, 0.0),
    velocity_set=velocity_set,
    precision_policy=precision_policy,
    compute_backend=compute_backend,
)

sim = xlb.helper.MultiresSimulationManager(
    omega_finest=omega_finest,
    grid=grid,
    boundary_conditions=boundary_conditions,
    collision_type="KBC",
    initializer=initializer,
    mres_perf_opt=xlb.MresPerfOptimizationType.FUSION_AT_FINEST,
)

momentum_transfer = MultiresMomentumTransfer(
    boundary_conditions[-1],
    mres_perf_opt=xlb.MresPerfOptimizationType.FUSION_AT_FINEST,
    compute_backend=compute_backend,
)

# --- Initial diagnostics and bc_mask export ---
stats = compute_voxel_statistics(sim.bc_mask, bc_mask_exporter, sparsity_pattern, boundary_conditions, unit_convertor)
reference_area = stats["reference_area"]

bc_mask_exporter.to_hdf5(
    os.path.join(output_dir, f"{script_name}_initial_bc_mask"),
    {"bc_mask": sim.bc_mask},
    compression="gzip",
    compression_opts=0,
)
wp.synchronize()

print("\n" + "=" * 50)
print(f"Iterations: {num_steps:,}  |  Levels: {num_levels}  |  Mesher: {mesher_type}")
print(f"Voxel: {voxel_size} m (coarsest {delta_x_coarse} m)")
print(f"Active voxels: {stats['total_voxels']:,}  |  Lattice updates/step: {stats['total_lattice_updates_per_step']:,}")
print(f"Inlet: {wind_speed_mps} m/s  |  omega: {omega_finest:.5f}")
print(f"Reference area: {reference_area} lattice units ({stats['reference_area_physical']:.6f} m^2)")
print("=" * 50 + "\n")

# --- Time integration ---
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
        check_fields_finite(sim, step, h5exporter)

        boundary_force = momentum_transfer(sim.f_0, sim.f_1, sim.bc_mask, sim.missing_mask)
        drag = boundary_force[0]
        lift = boundary_force[2]
        cd = 2.0 * drag / (wind_speed_lbm**2 * reference_area)
        cl = 2.0 * lift / (wind_speed_lbm**2 * reference_area)
        drag_values.append([cd, cl])

        save_fields(h5exporter, sim, output_dir, step)
        h5exporter.to_slice_image(
            os.path.join(output_dir, f"{script_name}_{step:04d}"),
            {"velocity": sim.u},
            plane_point=(1, 0, 0),
            plane_normal=(0, 1, 0),
            grid_res=2000,
            bounds=(0.25, 0.75, 0, 0.5),
            show_axes=False,
            show_colorbar=False,
            slice_thickness=delta_x_coarse,
        )

        print_progress(
            step,
            num_steps,
            grid_shape_x_coarsest,
            stats["total_lattice_updates_per_step"],
            steps_since_last_print,
            start_time,
            compute_time,
            cd=cd,
            cl=cl,
            drag=drag,
        )
        start_time = time.time()
        compute_time = 0.0
        steps_since_last_print = 0

    file_output_interval = file_output_interval_pre if step < crossover_step else file_output_interval_post
    if step % file_output_interval == 0 or step == num_steps - 1:
        sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
        check_fields_finite(sim, step, h5exporter)
        save_fields(h5exporter, sim, output_dir, step)

    if step == num_steps - 1:
        plot_velocity_profiles(x0, output_dir, delta_x_coarse, sim, h5exporter)

if drag_values:
    with open(os.path.join(output_dir, "drag_lift.csv"), "w") as fd:
        fd.write("Step,Cd,Cl\n")
        for i, (cd, cl) in enumerate(drag_values):
            fd.write(f"{i * print_interval},{cd},{cl}\n")
    plot_force_coefficients(drag_values, output_dir, print_interval)

    last_half = np.array(drag_values)[len(drag_values) // 2 :]
    avg_cd = np.mean(last_half[:, 0])
    avg_cl = np.mean(last_half[:, 1])
    print(f"Average Drag Coefficient (Cd) for last 50%: {avg_cd:.6f}")
    print(f"Average Lift Coefficient (Cl) for last 50%: {avg_cl:.6f}")
    print(f"Experimental Drag Coefficient (Cd): {0.3088}")
    print(f"Error Drag Coefficient (Cd): {((avg_cd - 0.3088) / 0.3088) * 100:.2f}%")
else:
    print("No drag or lift data collected.")
