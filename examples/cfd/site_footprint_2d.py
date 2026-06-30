"""
2D flow past an urban site footprint (D2Q9).

This example takes a 3D site mesh (the same ``.obj`` used by the 3D
``urban_canopy_3d.py`` example) and runs a **2D** lattice-Boltzmann simulation
on the *horizontal footprint* of the buildings. The building heights are
ignored: every (streamwise, crosswind) grid column that is covered by any part
of a building at any height becomes a solid obstacle. This is the 2D analogue
of ``flow_past_cylinder_2d.py`` where the single cylinder is replaced by the
projected building footprints.

Geometry / axes
---------------
The Rhino-exported mesh uses ``Y`` as the vertical axis. The horizontal plane is
therefore ``(X, Z)``. By default the wind blows along ``+X`` (streamwise) and
``Z`` is the crosswind direction. The mesh is projected onto the ``XZ`` plane,
rasterised onto the lattice, and used as a no-slip obstacle.

Physical units
--------------
The flow is set in real-world units (wind speed in m/s, kinematic viscosity of
air) and converted to lattice units with ``UnitConvertor`` exactly as in
``urban_canopy_3d.py``. The lattice viscosity follows from the physical
viscosity and the chosen resolution, so the relaxation rate ``omega`` lands very
close to 2.0 at realistic Reynolds numbers; ``KBC`` collision is used to keep the
simulation stable in that regime.

Boundary conditions
--------------------
* **Inlet (left):**  ``RegularizedBC`` with a uniform free-stream velocity.
* **Outlet (right):** ``DoNothingBC``.
* **Lateral (top/bottom):** ``FullwayBounceBackBC`` (no-slip channel walls).
* **Buildings:** ``FullwayBounceBackBC`` (no-slip).

Compute backends
----------------
* **WARP (default):** recommended for the large 2D grids urban sites produce.
* **JAX:** set ``compute_backend = ComputeBackend.JAX``.

Post-processing (always on JAX) saves velocity-magnitude and vorticity PNGs with
prefix ``site_footprint_2d`` plus a one-off ``*_footprint.png`` preview of the
rasterised obstacle mask.

Usage
-----
From the repository root (with ``PYTHONPATH`` pointing at this repo)::

    python3 examples/cfd/site_footprint_2d.py
"""

import os
import time

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

import jax

jax.config.update("jax_default_matmul_precision", "highest")

import jax.numpy as jnp
import numpy as np
import warp as wp
from scipy import ndimage
from tqdm import tqdm

import xlb
from xlb.compute_backend import ComputeBackend
from xlb.grid import grid_factory
from xlb.operator.boundary_condition import (
    DoNothingBC,
    HalfwayBounceBackBC,
    FullwayBounceBackBC,
    RegularizedBC,
)
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.precision_policy import PrecisionPolicy
from xlb.utils import UnitConvertor, save_image, warp_array_to_jax

wp.clear_kernel_cache()
wp.config.quiet = True

# -------------------------- User configuration --------------------------

# Geometry input. The mesh vertical axis is Y; the horizontal plane is (X, Z).
stl_filename = "examples/cfd/stl-files/20260629_Site_mesh.obj"

# Index of the mesh axis used as the streamwise (wind) direction in the
# horizontal plane: 0 -> X, 2 -> Z. The remaining horizontal axis is crosswind.
streamwise_axis = 0  # wind blows along +X

# Resolution: physical size (in mesh units, here metres) of one lattice cell.
# Smaller -> finer geometry and a larger grid. Start coarse for quick tests.
voxel_size = 8.0

# The mesh contains a flat ground/terrain plane at the base (vertical axis = Y).
# Only mesh faces that rise above this height (in mesh units) are treated as
# buildings; the ground plane is discarded so footprints are per-building rather
# than one solid blob covering the whole site.
ground_height_threshold = 0.5

# Domain padding as a fraction of the site footprint extent, so the obstacle is
# embedded in a larger channel with room for an inflow region and a wake.
pad_upstream = 0.5  # fraction of streamwise site length, in front of the site
pad_downstream = 1.0  # fraction of streamwise site length, behind the site (wake)
pad_lateral = 0.5  # fraction of crosswind site width, on each side

# Physical parameters. The flow is specified in real-world units and converted
# to lattice units with ``UnitConvertor`` (as in urban_canopy_3d.py).
wind_speed_mps = 5.0  # physical free-stream wind speed (m/s)
kinematic_viscosity = 1.508e-2  # kinematic viscosity of air (m^2/s)

# ``wind_speed_lbm`` is the lattice free-stream speed: it sets the time step
# (Mach number / compressibility) and how many steps a flow pass takes. Keep it
# small (<= ~0.05) for accuracy. It does NOT change the physical Reynolds number.
wind_speed_lbm = 0.02

flow_passes = 3.0  # number of streamwise domain passes to simulate

# Compute backend and precision.
compute_backend = ComputeBackend.WARP
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, compute_backend=compute_backend)


# -------------------------- Mesh -> 2D footprint --------------------------
def load_obj_triangles(filename):
    """Manually parse an OBJ into (vertices, triangles).

    trimesh's OBJ loader fails on some Rhino exports, so we parse ``v``/``f``
    lines directly. Polygonal faces (e.g. quads) are fan-triangulated and any
    texture/normal indices (``v/vt/vn``) are ignored.
    """
    # NOTE: ``filename`` is a fixed configuration constant in this example, not
    # external/untrusted input, so it is safe to open directly.
    vertices = []
    triangles = []
    with open(filename, "r") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                idx = [int(tok.split("/")[0]) - 1 for tok in parts[1:]]
                for k in range(1, len(idx) - 1):
                    triangles.append([idx[0], idx[k], idx[k + 1]])
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if vertices.size == 0 or triangles.size == 0:
        raise ValueError(f"No geometry parsed from {filename!r}.")
    return vertices, triangles


def rasterize_footprint(verts2d, triangles, grid_shape):
    """Rasterise projected triangles into a boolean obstacle mask.

    ``verts2d`` are vertex coordinates already mapped to (fractional) lattice
    indices in the (streamwise, crosswind) plane. A cell is solid if its centre
    falls inside any projected triangle; vertical walls project to thin lines
    while floor/roof faces fill the building interiors, giving full footprints.
    """
    nx, ny = grid_shape
    mask = np.zeros((nx, ny), dtype=bool)
    for tri in triangles:
        p = verts2d[tri]  # (3, 2) in lattice coordinates
        min_x = max(int(np.floor(p[:, 0].min())), 0)
        max_x = min(int(np.ceil(p[:, 0].max())), nx - 1)
        min_y = max(int(np.floor(p[:, 1].min())), 0)
        max_y = min(int(np.ceil(p[:, 1].max())), ny - 1)
        if max_x < min_x or max_y < min_y:
            continue
        (x1, y1), (x2, y2), (x3, y3) = p
        denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(denom) < 1e-9:
            continue  # degenerate (e.g. an edge-on vertical wall)
        xs = np.arange(min_x, max_x + 1) + 0.5
        ys = np.arange(min_y, max_y + 1) + 0.5
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        a = ((y2 - y3) * (gx - x3) + (x3 - x2) * (gy - y3)) / denom
        b = ((y3 - y1) * (gx - x3) + (x1 - x3) * (gy - y3)) / denom
        c = 1.0 - a - b
        inside = (a >= 0.0) & (b >= 0.0) & (c >= 0.0)
        mask[min_x : max_x + 1, min_y : max_y + 1] |= inside
    return mask


def build_geometry():
    """Load the mesh, choose the grid, and return the obstacle mask + grid shape."""
    vertical_axis = 1  # mesh Y is up
    horizontal_axes = [a for a in (0, 1, 2) if a != vertical_axis]
    crosswind_axis = horizontal_axes[1] if streamwise_axis == horizontal_axes[0] else horizontal_axes[0]

    vertices, triangles = load_obj_triangles(stl_filename)

    # Keep only faces that reach above the ground plane (i.e. building walls and
    # roofs); discard the flat terrain so footprints are per-building.
    tri_top = vertices[triangles, vertical_axis].max(axis=1)
    building_tris = triangles[tri_top >= ground_height_threshold]
    if building_tris.size == 0:
        raise ValueError(
            "No faces above ground_height_threshold; lower the threshold or check the mesh axes."
        )

    # Project onto the horizontal plane: column 0 = streamwise, column 1 = crosswind.
    proj = vertices[:, [streamwise_axis, crosswind_axis]]
    used = np.unique(building_tris)
    site_min = proj[used].min(axis=0)
    site_max = proj[used].max(axis=0)
    site_size = site_max - site_min  # (streamwise, crosswind) extent in mesh units

    # Domain origin (mesh coords of lattice index 0) including upstream/lateral pad.
    origin = np.array(
        [
            site_min[0] - pad_upstream * site_size[0],
            site_min[1] - pad_lateral * site_size[1],
        ]
    )
    domain_size = np.array(
        [
            site_size[0] * (1.0 + pad_upstream + pad_downstream),
            site_size[1] * (1.0 + 2.0 * pad_lateral),
        ]
    )
    grid_shape = (
        int(np.ceil(domain_size[0] / voxel_size)),
        int(np.ceil(domain_size[1] / voxel_size)),
    )

    verts2d = (proj - origin) / voxel_size
    mask = rasterize_footprint(verts2d, building_tris, grid_shape)
    # Vertical walls only rasterise to thin outlines; fill the enclosed interior
    # so each building becomes a solid footprint.
    mask = ndimage.binary_fill_holes(mask)
    return mask, grid_shape, site_size


# Build geometry up front so derived quantities (grid_shape, omega, steps) are available.
obstacle_mask, grid_shape, site_size = build_geometry()

# Convert physical units -> lattice units. ``reference_length`` is the voxel size
# and ``reference_velocity = wind_speed_mps / wind_speed_lbm`` (m/s per lattice
# unit), so the lattice viscosity follows from the physical kinematic viscosity.
unit_convertor = UnitConvertor(
    velocity_lbm_unit=wind_speed_lbm,
    velocity_physical_unit=wind_speed_mps,
    voxel_size_physical_unit=voxel_size,
)
prescribed_vel = wind_speed_lbm  # free-stream speed used by the boundary conditions
nu_lattice = unit_convertor.viscosity_to_lbm(kinematic_viscosity)
omega = 1.0 / (3.0 * nu_lattice + 0.5)

# Diagnostic Reynolds number based on the crosswind extent of the site.
char_length_m = float(site_size[1])
Re = wind_speed_mps * char_length_m / kinematic_viscosity

flow_pass = int(grid_shape[0] / prescribed_vel)
num_steps = int(flow_passes * flow_pass)
post_process_interval = max(1, int(0.1 * flow_pass))


def main() -> None:
    xlb.init(
        velocity_set=velocity_set,
        default_backend=compute_backend,
        default_precision_policy=precision_policy,
    )

    grid = grid_factory(grid_shape, compute_backend=compute_backend)

    box = grid.bounding_box_indices()
    box_no_edge = grid.bounding_box_indices(remove_edges=True)
    inlet = box_no_edge["left"]
    outlet = box_no_edge["right"]
    # Lateral far-field boundaries own the corners (as in flow_past_cylinder_2d).
    lateral = [box["bottom"][i] + box["top"][i] for i in range(velocity_set.d)]
    lateral = np.unique(np.array(lateral), axis=-1).tolist()

    # Obstacle cells; drop any that land on the domain border (shouldn't happen
    # given the padding, but keeps BC index sets disjoint).
    interior = obstacle_mask.copy()
    interior[0, :] = interior[-1, :] = False
    interior[:, 0] = interior[:, -1] = False
    bldg_idx = np.where(interior)
    buildings = [bldg_idx[i].tolist() for i in range(velocity_set.d)]

    free_stream = (prescribed_vel, 0.0)
    bc_inlet = RegularizedBC("velocity", prescribed_value=free_stream, indices=inlet)
    bc_lateral = FullwayBounceBackBC(indices=lateral)
    bc_outlet = DoNothingBC(indices=outlet)
    bc_buildings = FullwayBounceBackBC(indices=buildings)
    # Convention: obstacle BC last, outlet second-to-last.
    boundary_conditions = [bc_lateral, bc_inlet, bc_outlet, bc_buildings]

    stepper = IncompressibleNavierStokesStepper(
        grid=grid,
        boundary_conditions=boundary_conditions,
        collision_type="KBC",
    )
    f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()

    macro = Macroscopic(
        compute_backend=ComputeBackend.JAX,
        precision_policy=precision_policy,
        velocity_set=xlb.velocity_set.D2Q9(
            precision_policy=precision_policy,
            compute_backend=ComputeBackend.JAX,
        ),
    )

    # One-off preview of the rasterised footprint (white = building).
    save_image(obstacle_mask.astype(np.float32), prefix="site_footprint_2d_footprint", cmap="gray")

    def post_process(step: int, f_0) -> None:
        wp.synchronize()
        if not isinstance(f_0, jnp.ndarray):
            # Warp pads 2D domains with a singleton z dimension.
            f_0 = warp_array_to_jax(f_0)[..., 0]
            wp.synchronize()

        _, u = macro(f_0)
        u_mag = jnp.sqrt(u[0] ** 2 + u[1] ** 2)
        # Blank out building interiors so the obstacles read clearly in the image.
        u_mag = jnp.where(jnp.asarray(obstacle_mask), 0.0, u_mag)

        # convert u_mag to physical units
        u_mag_phys = unit_convertor.velocity_to_physical(u_mag)

        save_image(u_mag_phys, timestep=step, prefix="site_footprint_2d_umag", cmap="inferno")

    n_solid = int(obstacle_mask.sum())
    print(
        f"grid_shape={grid_shape}, site_size(streamwise,crosswind)={tuple(np.round(site_size, 1))} m, "
        f"voxel_size={voxel_size} m, solid_cells={n_solid} "
        f"({100.0 * n_solid / obstacle_mask.size:.2f}% of domain)"
    )
    dt_phys = unit_convertor.time_step_physical
    print(
        f"wind_speed={wind_speed_mps} m/s ({prescribed_vel} lbm), nu_air={kinematic_viscosity} m^2/s "
        f"(nu_lattice={nu_lattice:.3e}), Re={Re:,.0f} (crosswind extent {char_length_m:.0f} m)"
    )
    print(
        f"omega={omega:.6f}, dt={dt_phys:.4e} s/step, num_steps={num_steps} "
        f"(~{num_steps * dt_phys:.1f} s physical), backend={compute_backend.name}"
    )

    start_time = time.time()
    pbar = tqdm(range(num_steps), unit="step")
    for step in pbar:
        f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, step)
        f_0, f_1 = f_1, f_0

        if step % post_process_interval == 0 or step == num_steps - 1:
            post_process(step, f_0)
            elapsed = time.time() - start_time
            pbar.set_postfix(chunk_s=f"{elapsed:.1f}")
            start_time = time.time()


if __name__ == "__main__":
    main()
