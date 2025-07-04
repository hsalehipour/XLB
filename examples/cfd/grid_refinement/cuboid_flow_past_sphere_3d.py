import neon
import warp as wp
import numpy as np
import time

import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import multires_grid_factory
from xlb.operator.stepper import MultiresIncompressibleNavierStokesStepper
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
from xlb.utils.mesher import make_cuboid_mesh
from xlb.operator.force import MultiresMomentumTransfer


def generate_cuboid_mesh(stl_filename, num_finest_voxels_across_part):
    """
    Generate a cuboid mesh based on the provided voxel size and domain multipliers.
    """
    import open3d as o3d
    import os

    # Domain multipliers for each refinement level
    # First entry should be full domain size
    # Domain multipliers
    domainMultiplier = [
        [15, 15, 7, 7, 7, 7],  # -x, x, -y, y, -z, z
        [6, 8, 5, 5, 5, 5],  # -x, x, -y, y, -z, z
        [4, 6, 4, 4, 4, 4],
        [2, 4, 2, 2, 2, 2],
        # [1, 2, 1, 1, 1, 1],
        # [0.4, 1, 0.4, 0.4, 0.4, 0.4],
        # [0.2, 0.4, 0.2, 0.2, 0.2, 0.2],
    ]

    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(stl_filename)
    if mesh.is_empty():
        raise ValueError("Loaded mesh is empty or invalid.")

    # Compute original bounds
    aabb = mesh.get_axis_aligned_bounding_box()
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()
    partSize = max_bound - min_bound

    # smallest voxel size
    voxel_size = min(partSize) / num_finest_voxels_across_part

    # Compute translation to put mesh into first octant of that domain—
    shift = np.array(
        [
            domainMultiplier[0][0] * partSize[0] - min_bound[0],
            domainMultiplier[0][2] * partSize[1] - min_bound[1],
            domainMultiplier[0][4] * partSize[2] - min_bound[2],
        ],
        dtype=float,
    )

    # Apply translation and save out temp stl
    mesh.translate(shift)
    mesh.compute_vertex_normals()
    mesh_vertices = np.asarray(mesh.vertices) / voxel_size
    o3d.io.write_triangle_mesh("temp.stl", mesh)

    # Mesh based on temp stl
    level_data = make_cuboid_mesh(
        voxel_size,
        domainMultiplier,
        "temp.stl",
    )
    grid_shape_finest = tuple([i * 2 ** (len(level_data) - 1) for i in level_data[-1][0].shape])
    print(f"Full shape based on finest voxels size is {grid_shape_finest}")
    os.remove("temp.stl")
    return level_data, mesh_vertices, tuple([int(a) for a in grid_shape_finest])


def prepare_sparsity_pattern(level_data):
    """
    Prepare the sparsity pattern for the multiresolution grid based on the level data. "level_data" is expected to be formatted as in
    the output of "make_cuboid_mesh".
    """
    num_levels = len(level_data)
    sparsity_pattern = []
    level_origins = []
    sparsity_pattern = []
    for lvl in range(num_levels):
        # Get the level mask from the level data
        level_mask = level_data[lvl][0]

        # Ensure level_0 is contiguous int32
        level_mask = np.ascontiguousarray(level_mask, dtype=np.int32)

        # Append the padded level mask to the sparsity pattern
        sparsity_pattern.append(level_mask)

        # Get the origin for this level
        level_origins.append(level_data[lvl][2])

    return sparsity_pattern, level_origins


# -------------------------- Simulation Setup --------------------------

# The following parameters define the resolution of the voxelized grid
sphere_radius = 5
num_finest_voxels_across_part = 2 * sphere_radius

# Other setup parameters
Re = 5000.0
compute_backend = ComputeBackend.NEON
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)
u_max = 0.04
num_steps = 10000
post_process_interval = 1000

# Initialize XLB
xlb.init(
    velocity_set=velocity_set,
    default_backend=compute_backend,
    default_precision_policy=precision_policy,
)

# Generate the cuboid mesh and sphere vertices
stl_filename = "examples/cfd/stl-files/sphere.stl"
level_data, sphere, grid_shape_finest = generate_cuboid_mesh(stl_filename, num_finest_voxels_across_part)

# Prepare the sparsity pattern and origins from the level data
sparsity_pattern, level_origins = prepare_sparsity_pattern(level_data)

# get the number of levels
num_levels = len(level_data)

# Create the multires grid
grid = multires_grid_factory(
    grid_shape_finest,
    velocity_set=velocity_set,
    sparsity_pattern_list=sparsity_pattern,
    sparsity_pattern_origins=[neon.Index_3d(*box_origin) for box_origin in level_origins],
)

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

    # Note nx, ny, nz are the dimensions of the grid at the finest level while the inlet is defined at the coarsest level
    nx, ny, nz = grid_shape_finest
    H_y = float(ny // 2 ** (num_levels - 1) - 1)  # Height in y direction
    H_z = float(nz // 2 ** (num_levels - 1) - 1)  # Height in z direction

    @wp.func
    def bc_profile_warp(index: wp.vec3i):
        # Poiseuille flow profile: parabolic velocity distribution
        y = wp.float32(index[1])
        z = wp.float32(index[2])

        # Calculate normalized distance from center
        y_center = y - (H_y / 2.0)
        z_center = z - (H_z / 2.0)
        r_squared = (2.0 * y_center / H_y) ** 2.0 + (2.0 * z_center / H_z) ** 2.0

        # Parabolic profile: u = u_max * (1 - r²)
        return wp.vec(u_max * wp.max(0.0, 1.0 - r_squared), length=1)

    return bc_profile_warp


# Convert bc indices to a list of list (first entry corresponds to the finest level)
inlet = [[] for _ in range(num_levels - 1)] + [inlet]
outlet = [[] for _ in range(num_levels - 1)] + [outlet]
walls = [[] for _ in range(num_levels - 1)] + [walls]

# Initialize Boundary Conditions
bc_left = RegularizedBC("velocity", profile=bc_profile(), indices=inlet)
# Alternatively, use a prescribed velocity profile
# bc_left = RegularizedBC("velocity", prescribed_value=(u_max, 0.0, 0.0), indices=inlet)
bc_walls = FullwayBounceBackBC(indices=walls)  # TODO: issues with halfway bounce back only here!
# bc_outlet = ExtrapolationOutflowBC(indices=outlet)
bc_outlet = DoNothingBC(indices=outlet)
bc_sphere = HalfwayBounceBackBC(mesh_vertices=sphere, voxelization_method=MeshVoxelizationMethod.AABB)
# bc_sphere = HybridBC(
#     bc_method="nonequilibrium_regularized", mesh_vertices=sphere, voxelization_method=MeshVoxelizationMethod.AABB, use_mesh_distance=False
# )

boundary_conditions = [bc_walls, bc_left, bc_outlet, bc_sphere]

# Configure the simulation relaxation time
visc = u_max * num_finest_voxels_across_part / Re
omega = 1.0 / (3.0 * visc + 0.5)

# Define a multi-resolution simulation manager
sim = xlb.helper.MultiresSimulationManager(
    omega=omega,
    grid=grid,
    boundary_conditions=boundary_conditions,
    collision_type="KBC",
)

# Setup Momentum Transfer for Force Calculation
bc_sphre = boundary_conditions[-1]
momentum_transfer = MultiresMomentumTransfer(bc_sphere, compute_backend=compute_backend)


def print_lift_drag(sim):
    # Compute lift and drag
    wp.synchronize()
    boundary_force = momentum_transfer(sim.f_0, sim.f_1, sim.bc_mask, sim.missing_mask)
    wp.synchronize()
    drag = boundary_force[0]  # x-direction
    lift = boundary_force[2]
    sphere_cross_section = np.pi * sphere_radius**2
    u_avg = 0.5 * u_max
    cd = 2.0 * drag / (u_avg**2 * sphere_cross_section)
    cl = 2.0 * lift / (u_avg**2 * sphere_cross_section)
    print(f"CD={cd}, CL={cl}")


# -------------------------- Simulation Loop --------------------------

wp.synchronize()
start_time = time.time()
for step in range(num_steps):
    sim.step()

    if step % post_process_interval == 0 or step == num_steps - 1:
        # TODO: Issues in the vtk output for rectangular cuboids (as if a duboid grid with the largest side is assumed)
        sim.export_macroscopic("multires_flow_over_sphere_3d_")
        print_lift_drag(sim)
        wp.synchronize()
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Completed step {step}. Time elapsed for {post_process_interval} steps: {elapsed:.6f} seconds.")
        start_time = time.time()
