"""
Urban canopy wind simulation with multi-resolution LBM.

Simulates atmospheric flow over a building site using the XLB multi-resolution
Neon backend.  The domain is built with the adaptive surface mesher, which
refines voxels near STL geometry and coarsens away from buildings.  Exports
velocity and density fields to HDF5/XDMF for ParaView post-processing.

Coordinate convention (mesh-domain frame after STL shift):
    +x : streamwise (inlet at left, outlet at right)
    +y : spanwise  (front / back lateral faces)
    +z : vertical  (ground at bottom, open sky at top)
"""

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
from xlb.operator.boundary_condition import RegularizedBC, DoNothingBC, HybridBC
from xlb.operator.boundary_masker import MeshVoxelizationMethod
from xlb.utils.mesher import MultiresIO, is_sparse_level_data, prepare_sparsity_pattern
from xlb.utils import UnitConvertor
from xlb.utils.adaptive_mesher import grid_shape_finest, load_and_shift_stl, make_adaptive_surface_mesh
from xlb.helper.initializers import CustomMultiresInitializer

wp.clear_kernel_cache()
wp.config.quiet = True

# User Configuration
# =================
# Physical parameters
wind_speed_lbm = 0.05  # Reference lattice velocity used for unit conversion
wind_speed_mps = 1.0  # Physical inlet velocity [m/s]
flow_passes = 4  # Number of domain-length transits before stopping
kinematic_viscosity = 1.508e-5  # Air kinematic viscosity [m^2/s]
voxel_size = 8  # Finest lattice cell size [m]

# Adaptive mesh parameters (see xlb.utils.adaptive_mesher)
num_levels = 4
domain_padding = [1.5, 3.0, 1.5, 1.5, 0.0, 4.0]  # [-x, +x, -y, +y, -z, +z] x geometry extent
expansion_ratio = 2.0  # Geometric ratio between consecutive refinement shells
finest_band_cells = 6  # Thickness of finest-level band near surfaces [cells]

# Geometry and output
stl_filename = "examples/cfd/stl-files/07022026_SEPULVEDA_SITE_MODEL_FORMA_NOTREES.stl"
script_name = "sepulveda_site_notrees"

# Progress reporting and HDF5 output scheduling
print_interval_percentage = 1  # Console progress every N % of iterations
file_output_crossover_percentage = 1  # Switch to post-crossover output rate at N % of run
num_file_outputs_pre_crossover = 3
num_file_outputs_post_crossover = 3

# Backend
compute_backend = ComputeBackend.NEON
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)


def _active_voxel_count(mask):
    """Count active voxels in a dense boolean mask or sparse (N, 3) coord array."""
    return mask.shape[0] if mask.ndim == 2 else int(np.count_nonzero(mask))


def _coords_to_set(indices):
    """Convert per-level BC indices ``[[x...], [y...], [z...]]`` to a set of (x, y, z) tuples."""
    return set(zip(*indices)) if indices else set()


def _set_to_indices(coord_set):
    """Convert a set of (x, y, z) tuples back to ``[[x...], [y...], [z...]]`` BC index lists."""
    return [list(coords) for coords in zip(*coord_set)] if coord_set else []


def compute_voxel_statistics(bc_mask, bc_mask_exporter, sparsity_pattern):
    """
    Summarize sparsity and solid-voxel counts across refinement levels.

    Returns per-level active/solid counts, total fluid voxels, and equivalent
    finest-level lattice updates per global time step.
    """
    num_levels = len(sparsity_pattern)
    fields_data = bc_mask_exporter.get_fields_data({"bc_mask": bc_mask})
    bc_mask_data = fields_data["bc_mask_0"]
    level_id_field = bc_mask_exporter.level_id_field

    solid_voxels = [
        np.sum(bc_mask_data[level_id_field == lvl] == 255) for lvl in range(num_levels)
    ]
    active_voxels = [_active_voxel_count(mask) for mask in sparsity_pattern]
    active_voxels = [max(0, active_voxels[lvl] - solid_voxels[lvl]) for lvl in range(num_levels)]

    return {
        "active_voxels": active_voxels,
        "solid_voxels": solid_voxels,
        "total_voxels": sum(active_voxels),
        "total_lattice_updates_per_step": sum(
            active_voxels[lvl] * (2 ** (num_levels - 1 - lvl)) for lvl in range(num_levels)
        ),
    }


def generate_adaptive_mesh(stl_filename, voxel_size):
    """
    Build a surface-adaptive multires mesh and return geometry for the simulation.

    Shifts the STL into the mesh-domain frame, runs ``make_adaptive_surface_mesh``,
    and extracts triangle-soup vertices for building voxelization.

    Returns:
        level_data: sparse per-level coords, compatible with ``prepare_sparsity_pattern``
        mesh_vertices: (N, 3) triangle soup in physical coordinates
        grid_shape_finest: (nx, ny, nz) finest lattice dimensions
        stl_shift: translation applied to the STL (used to offset HDF5 exports)
    """
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

        loaded = trimesh.load(temp_stl, process=False)
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values())) if isinstance(loaded, trimesh.Scene) else loaded
        if mesh.is_empty:
            raise ValueError("Loaded mesh is empty or invalid.")
        mesh_vertices = np.asarray(mesh.vertices[mesh.faces].reshape(-1, 3))

        print(f"Requested levels: {num_levels}, Actual levels: {len(level_data)}")
        print(f"Mesh representation: {'sparse coords' if is_sparse_level_data(level_data) else 'dense masks'}")
        print(f"Full shape based on finest voxel size is {finest_shape}")

        return level_data, mesh_vertices, tuple(int(a) for a in finest_shape), stl_shift
    finally:
        if temp_stl is not None and os.path.isfile(temp_stl):
            os.remove(temp_stl)


def setup_boundary_conditions(grid, level_data, building_vertices, wind_speed_mps):
    """
    Configure domain-face and building-surface boundary conditions.

    Face layout:
        left (+x inlet)  : velocity inlet
        right (-x outlet): do-nothing outlet
        front/back (±y)  : slip walls
        bottom/top (±z)  : ground / lid; corner cells shared with inlet/outlet
                           are removed to avoid duplicate BC tagging

    The building mesh BC must remain last; the outlet must be second-to-last
    (required by ``CustomMultiresInitializer`` and force operators).
    """
    wind_speed_lbm_local = unit_convertor.velocity_to_lbm(wind_speed_mps)
    sides = ("left", "right", "top", "bottom", "front", "back")
    indices = {
        side: grid.boundary_indices_across_levels(
            level_data,
            box_side=side,
            remove_edges=(side == "left" or side == "right"),
        )
        for side in sides
    }

    filtered_top = []
    filtered_bottom = []
    for lvl in range(len(level_data)):
        lr_exclude = _coords_to_set(indices["left"][lvl]) | _coords_to_set(indices["right"][lvl])
        filtered_top.append(_set_to_indices(_coords_to_set(indices["top"][lvl]) - lr_exclude))
        filtered_bottom.append(_set_to_indices(_coords_to_set(indices["bottom"][lvl]) - lr_exclude))

    return [
        HybridBC(bc_method="nonequilibrium_regularized", prescribed_value=(wind_speed_lbm_local, 0.0, 0.0), indices=indices["front"]),
        HybridBC(bc_method="nonequilibrium_regularized", prescribed_value=(wind_speed_lbm_local, 0.0, 0.0), indices=indices["back"]),
        HybridBC(bc_method="nonequilibrium_regularized", prescribed_value=(0.0, 0.0, 0.0), indices=filtered_bottom),
        HybridBC(bc_method="nonequilibrium_regularized", prescribed_value=(wind_speed_lbm_local, 0.0, 0.0), indices=filtered_top),
        RegularizedBC("velocity", prescribed_value=(wind_speed_lbm_local, 0.0, 0.0), indices=indices["left"]),
        DoNothingBC(indices=indices["right"]),
        # HybridBC(
        #     bc_method="nonequilibrium_regularized",
        #     mesh_vertices=unit_convertor.length_to_lbm(building_vertices),
        #     voxelization_method=MeshVoxelizationMethod("AABB"),
        #     use_mesh_distance=False,
        # ),
    ]


def check_fields_finite(sim, step, h5exporter):
    """
    Validate macroscopic fields before writing HDF5 output.

    Raises:
        ValueError: if velocity or density contains NaN at the given step.
    """
    fields = h5exporter.get_fields_data({"velocity": sim.u, "density": sim.rho})
    for name, data in fields.items():
        if np.isnan(data).any():
            raise ValueError(f"NaN detected in {name} at step {step}")


def save_fields(h5exporter, sim, output_dir, step):
    """Write velocity and density fields to HDF5/XDMF at the given step."""
    filename = os.path.join(output_dir, f"{script_name}_{step:04d}")
    h5exporter.to_hdf5(filename, {"velocity": sim.u, "density": sim.rho}, compression="gzip", compression_opts=1)
    wp.synchronize()


def print_progress(step, num_steps, grid_shape_x_coarsest, total_lattice_updates_per_step, steps_since_last_print, start_time, compute_time):
    """Print flow-pass progress, wall time, ETA, and MLUPS since the last report."""
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


# Main Script
# ===========
xlb.init(
    velocity_set=velocity_set,
    default_backend=compute_backend,
    default_precision_policy=precision_policy,
)

# --- Mesh and unit conversion ---
level_data, building_vertices, grid_shape_finest, stl_shift = generate_adaptive_mesh(stl_filename, voxel_size)
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
file_output_interval_pre = max(1, int(crossover_step / num_file_outputs_pre_crossover)) if num_file_outputs_pre_crossover else num_steps + 1
file_output_interval_post = (
    max(1, int((num_steps - crossover_step) / num_file_outputs_post_crossover))
    if num_file_outputs_post_crossover
    else num_steps + 1
)

# --- Boundary conditions, initializer, and simulation manager ---
boundary_conditions = setup_boundary_conditions(grid, level_data, building_vertices, wind_speed_mps)
initializer = CustomMultiresInitializer(
    bc_id=boundary_conditions[-1].id,
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
    mres_perf_opt=xlb.mres_perf_optimization_type.MresPerfOptimizationType.FUSION_AT_FINEST,
)

# --- Initial diagnostics and bc_mask export ---
stats = compute_voxel_statistics(sim.bc_mask, bc_mask_exporter, sparsity_pattern)

bc_mask_exporter.to_hdf5(
    os.path.join(output_dir, f"{script_name}_initial_bc_mask"),
    {"bc_mask": sim.bc_mask},
    compression="gzip",
    compression_opts=0,
)
wp.synchronize()

print("\n" + "=" * 50)
print(f"Iterations: {num_steps:,}  |  Levels: {num_levels}  |  Voxel: {voxel_size} m (coarsest {delta_x_coarse} m)")
print(f"Active voxels: {stats['total_voxels']:,}  |  Lattice updates/step: {stats['total_lattice_updates_per_step']:,}")
print(f"Inlet: {wind_speed_mps} m/s  |  omega: {omega_finest:.5f}")
print("=" * 50 + "\n")

# --- Time integration ---
start_time = time.time()
compute_time = 0.0
steps_since_last_print = 0

for step in range(num_steps):
    step_start = time.time()
    sim.step()
    wp.synchronize()
    compute_time += time.time() - step_start
    steps_since_last_print += 1

    if step % print_interval == 0 or step == num_steps - 1:
        sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
        print_progress(
            step, num_steps, grid_shape_x_coarsest, stats["total_lattice_updates_per_step"],
            steps_since_last_print, start_time, compute_time,
        )
        start_time = time.time()
        compute_time = 0.0
        steps_since_last_print = 0

    file_output_interval = file_output_interval_pre if step < crossover_step else file_output_interval_post
    if step % file_output_interval == 0 or step == num_steps - 1:
        sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
        check_fields_finite(sim, step, h5exporter)
        save_fields(h5exporter, sim, output_dir, step)
