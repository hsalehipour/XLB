import xlb
import argparse
import time
import warp as wp
import numpy as np
import neon

from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import multires_grid_factory
from xlb.operator.stepper import MultiresIncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import FullwayBounceBackBC, EquilibriumBC
from xlb.mres_perf_optimization_type import MresPerfOptimizationType


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="MLUPS for 3D Lattice Boltzmann Method Simulation with Multi-resolution Grid",
        epilog="""
Examples:
  %(prog)s 100 1000 neon fp32/fp32 2 NAIVE_COLLIDE_STREAM
  %(prog)s 200 500 neon fp64/fp64 3 FUSION_AT_FINEST --report
  %(prog)s 50 2000 neon fp32/fp16 2 NAIVE_COLLIDE_STREAM --export_final_velocity

Valid values:
  compute_backend: neon
  precision: fp32/fp32, fp64/fp64, fp64/fp32, fp32/fp16
  mres_perf_opt: NAIVE_COLLIDE_STREAM, FUSION_AT_FINEST
  velocity_set: D3Q19, D3Q27
  collision_model: BGK, KBC
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # Positional arguments
    parser.add_argument("cube_edge", type=int, help="Length of the edge of the cubic grid (e.g., 100)")
    parser.add_argument("num_steps", type=int, help="Number of timesteps for the simulation (e.g., 1000)")
    parser.add_argument("compute_backend", type=str, help="Backend for the simulation (neon)")
    parser.add_argument("precision", type=str, help="Precision for the simulation (fp32/fp32, fp64/fp64, fp64/fp32, fp32/fp16)")
    parser.add_argument("num_levels", type=int, help="Number of levels for the multiresolution grid (e.g., 2)")
    parser.add_argument("mres_perf_opt", type=MresPerfOptimizationType.from_string, help="Multi-resolution performance optimization strategy (NAIVE_COLLIDE_STREAM, FUSION_AT_FINEST)")

    # Optional arguments
    parser.add_argument("--num_devices", type=int, default=0, help="Number of devices for the simulation (default: 0)")
    parser.add_argument("--velocity_set", type=str, default="D3Q19",
                        help="Lattice type: D3Q19 or D3Q27 (default: D3Q19)")
    parser.add_argument("--collision_model", type=str, default="BGK",
                        help="Collision model: BGK or KBC (default: BGK)")

    parser.add_argument("--report", action="store_true", help="Generate a neon report file (default: disabled)")
    parser.add_argument("--export_final_velocity", action="store_true",
                        help="Export the final velocity field to a vti file (default: disabled)")

    try:
        args = parser.parse_args()
    except SystemExit:
        # Re-raise with custom message
        print("\n" + "="*60)
        print("USAGE EXAMPLES:")
        print("="*60)
        print("python mlups_3d_multires.py 100 1000 neon fp32/fp32 2 NAIVE_COLLIDE_STREAM")
        print("python mlups_3d_multires.py 200 500 neon fp64/fp64 3 FUSION_AT_FINEST --report")
        print("\nVALID VALUES:")
        print("  compute_backend: neon")
        print("  precision: fp32/fp32, fp64/fp64, fp64/fp32, fp32/fp16")
        print("  mres_perf_opt: NAIVE_COLLIDE_STREAM, FUSION_AT_FINEST")
        print("  velocity_set: D3Q19, D3Q27")
        print("  collision_model: BGK, KBC")
        print("="*60)
        raise

    print_args(args)

    if args.compute_backend != "neon":
        raise ValueError("Invalid compute backend specified. Use 'neon' which supports multi-resolution!")

    if args.collision_model not in ["BGK", "KBC"]:
        raise ValueError("Invalid collision model specified. Use 'BGK' or 'KBC'.")

    return args


def print_args(args):
    # Print simulation configuration
    print("=" * 60)
    print("           3D LATTICE BOLTZMANN SIMULATION CONFIG")
    print("=" * 60)
    print(f"Grid Size:            {args.cube_edge}³ ({args.cube_edge:,} × {args.cube_edge:,} × {args.cube_edge:,})")
    print(f"Total Lattice Points: {args.cube_edge ** 3:,}")
    print(f"Time Steps:           {args.num_steps:,}")
    print(f"Number Levels:        {args.num_levels}")
    print(f"Compute Backend:      {args.compute_backend}")
    print(f"Precision Policy:     {args.precision}")
    print(f"Velocity Set:         {args.velocity_set}")
    print(f"Collision Model:      {args.collision_model}")
    print(f"Mres Perf Opt:        {args.mres_perf_opt}")
    print(f"Generate Report:      {'Yes' if args.report else 'No'}")
    print(f"Export Velocity:      {'Yes' if args.export_final_velocity else 'No'}")

    print("=" * 60)
    print("Starting simulation...")
    print()


def setup_simulation(args):
    compute_backend = None
    if args.compute_backend == "neon":
        compute_backend = ComputeBackend.NEON
    else:
        raise ValueError("Invalid compute backend specified. Use 'neon' which supports multi-resolution!")

    precision_policy_map = {
        "fp32/fp32": PrecisionPolicy.FP32FP32,
        "fp64/fp64": PrecisionPolicy.FP64FP64,
        "fp64/fp32": PrecisionPolicy.FP64FP32,
        "fp32/fp16": PrecisionPolicy.FP32FP16,
    }
    precision_policy = precision_policy_map.get(args.precision)
    if precision_policy is None:
        raise ValueError("Invalid precision")

    velocity_set = None
    if args.velocity_set == "D3Q19":
        velocity_set = xlb.velocity_set.D3Q19(precision_policy=precision_policy, compute_backend=compute_backend)
    elif args.velocity_set == "D3Q27":
        velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)
    if velocity_set is None:
        raise ValueError("Invalid velocity set")

    xlb.init(
        velocity_set=velocity_set,
        default_backend=compute_backend,
        default_precision_policy=precision_policy,
    )

    return velocity_set


def problem1(grid_shape, velocity_set, num_levels):
    def peel(dim, idx, peel_level, outwards):
        if outwards:
            xIn = idx.x <= peel_level or idx.x >= dim.x - 1 - peel_level
            yIn = idx.y <= peel_level or idx.y >= dim.y - 1 - peel_level
            zIn = idx.z <= peel_level or idx.z >= dim.z - 1 - peel_level
            return xIn or yIn or zIn
        else:
            xIn = idx.x >= peel_level and idx.x <= dim.x - 1 - peel_level
            yIn = idx.y >= peel_level and idx.y <= dim.y - 1 - peel_level
            zIn = idx.z >= peel_level and idx.z <= dim.z - 1 - peel_level
            return xIn and yIn and zIn

    dim = neon.Index_3d(grid_shape[0], grid_shape[1], grid_shape[2])

    def get_peeled_np(level, width):
        divider = 2 ** level
        m = neon.Index_3d(dim.x // divider, dim.y // divider, dim.z // divider)
        if level == 0:
            m = dim

        mask = np.zeros((m.x, m.y, m.z), dtype=int)
        mask = np.ascontiguousarray(mask, dtype=np.int32)
        # loop over all the elements in mask and set to one any that have x=0 or y=0 or z=0
        for i in range(m.x):
            for j in range(m.y):
                for k in range(m.z):
                    idx = neon.Index_3d(i, j, k)
                    val = 0
                    if peel(m, idx, width, True):
                        val = 1
                    mask[i, j, k] = val
        return mask

    def get_levels(num_levels):
        levels = []
        for i in range(num_levels - 1):
            l = get_peeled_np(i, 8)
            levels.append(l)
        lastLevel = num_levels - 1
        divider = 2 ** lastLevel
        m = neon.Index_3d(dim.x // divider + 1, dim.y // divider + 1, dim.z // divider + 1)
        lastLevel = np.ones((m.x, m.y, m.z), dtype=int)
        lastLevel = np.ascontiguousarray(lastLevel, dtype=np.int32)
        levels.append(lastLevel)
        return levels

    levels = get_levels(num_levels)

    grid = multires_grid_factory(
        grid_shape,
        velocity_set=velocity_set,
        sparsity_pattern_list=levels,
        sparsity_pattern_origins=[neon.Index_3d(0, 0, 0)] * len(levels),
    )

    box = grid.bounding_box_indices()
    box_no_edge = grid.bounding_box_indices(remove_edges=True)
    lid = box_no_edge["top"]
    walls = [box["bottom"][i] + box["left"][i] + box["right"][i] + box["front"][i] + box["back"][i] for i in
             range(len(grid.shape))]
    walls = np.unique(np.array(walls), axis=-1).tolist()
    # convert bc indices to a list of list, where the first entry of the list corresponds to the finest level
    lid = [lid] + [[] for _ in range(num_levels - 1)]
    walls = [walls] + [[] for _ in range(num_levels - 1)]
    return grid, lid, walls


def problem2(grid_shape, velocity_set, num_levels):
    # Example 2: Coarsest at the edges (2 level only)
    level_origins = []
    level_list = []
    for lvl in range(num_levels):
        divider = 2 ** lvl
        growth = 1.5 ** lvl
        shape = grid_shape[0] // divider, grid_shape[1] // divider, grid_shape[2] // divider
        if lvl == num_levels - 1:
            level = np.ascontiguousarray(np.ones(shape, dtype=int), dtype=np.int32)
            box_origin = (0, 0, 0)  # The coarsest level has no origin offset
        else:
            box_size = tuple([int(shape[i] // 4 * growth) for i in range(3)])
            box_origin = tuple([shape[i] // 2 - box_size[i] // 2 for i in range(3)])
            level = np.ascontiguousarray(np.ones(box_size, dtype=int), dtype=np.int32)
        level_list.append(level)
        level_origins.append(neon.Index_3d(*box_origin))

    # Create the multires grid
    grid = multires_grid_factory(
        grid_shape,
        velocity_set=velocity_set,
        sparsity_pattern_list=level_list,
        sparsity_pattern_origins=level_origins,
    )

    box = grid.bounding_box_indices(shape=grid.level_to_shape(num_levels - 1))
    box_no_edge = grid.bounding_box_indices(shape=grid.level_to_shape(1), remove_edges=True)
    lid = box_no_edge["top"]
    walls = [box["bottom"][i] + box["left"][i] + box["right"][i] + box["front"][i] + box["back"][i] for i in
             range(len(grid.shape))]
    walls = np.unique(np.array(walls), axis=-1).tolist()
    # convert bc indices to a list of list, where the first entry of the list corresponds to the finest level
    lid = [[] for _ in range(num_levels - 1)] + [lid]
    walls = [[] for _ in range(num_levels - 1)] + [walls]
    return grid, lid, walls


def run(velocity_set,
        grid_shape,
        num_steps,
        num_levels,
        collision_model,
        export_final_velocity,
        mres_perf_opt, ):
    # Create grid and setup boundary conditions

    # Convert indices to list of indices per level
    # TODO: overlaps emerge if bc indices are orignally specified at the finest grid and they exist at the coarser levels
    # levels_mask = [lvl.astype(bool) for lvl in levels]
    # lid = construct_indices_per_level(grid_shape, lid, levels_mask, level_origins)
    # walls = construct_indices_per_level(grid_shape, walls, levels_mask, level_origins)

    # Example 1: fine to coarse
    # grid, lid, walls = problem1(grid_shape, velocity_set, num_levels)

    # Example 2: Coarse to fine:
    grid, lid, walls = problem1(grid_shape, velocity_set, num_levels)

    prescribed_vel = 0.1
    boundary_conditions = [
        EquilibriumBC(rho=1.0, u=(prescribed_vel, 0.0, 0.0), indices=lid),
        FullwayBounceBackBC(indices=walls),
    ]

    # Problem parameters
    Re = 5000.0
    clength = grid_shape[0] - 1
    visc = prescribed_vel * clength / Re
    omega_finest = 1.0 / (3.0 * visc + 0.5)

    # Define a multi-resolution simulation manager
    sim = xlb.helper.MultiresSimulationManager(omega=omega,
                                               grid=grid,
                                               boundary_conditions=boundary_conditions,
                                               collision_type=collision_model,
                                               mres_perf_opt=mres_perf_opt, )

    # sim.export_macroscopic("Initial_")
    # sim.step()

    print("start timing")
    wp.synchronize()
    start_time = time.time()

    if num_levels == 1:
        num_steps = num_steps // 2 
    
    for i in range(num_steps):
        sim.step()
        # if i % 1000 == 0:
        #     print(f"step {i}")
        #     sim.export_macroscopic("u_lid_driven_cavity_")
    wp.synchronize()
    t = time.time() - start_time
    print(f"Timing  {t}")

    if export_final_velocity:
        sim.export_macroscopic("u_lid_driven_cavity_")

    # sim.export_macroscopic("u_lid_driven_cavity_")
    num_levels = grid.count_levels
    return {"time": t, "num_levels": num_levels}


def calculate_mlups(cube_edge, num_steps, elapsed_time, num_levels):
    num_step_finer = num_steps * 2 ** (num_levels - 1)
    total_lattice_updates = cube_edge ** 3 * num_step_finer
    mlups = (total_lattice_updates / elapsed_time) / 1e6
    return {"EMLUPS": mlups, "finer_steps": num_step_finer}

    # # remove boundary cells
    # rho = rho[:, 1:-1, 1:-1, 1:-1]
    # u = u[:, 1:-1, 1:-1, 1:-1]
    # u_magnitude = (u[0] ** 2 + u[1] ** 2) ** 0.5
    #
    # fields = {"rho": rho[0], "u_x": u[0], "u_y": u[1], "u_magnitude": u_magnitude}
    #
    # # save_fields_vtk(fields, timestep=i, prefix="lid_driven_cavity")
    # ny=fields["u_magnitude"].shape[1]
    # from xlb.utils import  save_image
    # save_image(fields["u_magnitude"][:, ny//2, :], timestep=i, prefix="lid_driven_cavity")


def generate_report(args, stats, mlups_stats):
    """Generate a neon report file with simulation parameters and results"""
    import neon
    import sys

    report = neon.Report("LBM MLUPS Multiresolution LDC")

    # Save the full command line
    command_line = " ".join(sys.argv)
    report.add_member("command_line", command_line)

    report.add_member("velocity_set", args.velocity_set)
    report.add_member("compute_backend", args.compute_backend)
    report.add_member("precision_policy", args.precision)
    report.add_member("collision_model", args.collision_model)
    report.add_member("grid_size", args.cube_edge)
    report.add_member("num_steps", args.num_steps)
    report.add_member("num_levels", stats["num_levels"])
    report.add_member("finer_steps", mlups_stats["finer_steps"])

    # Performance metrics
    report.add_member("elapsed_time", stats["time"])
    report.add_member("emlups", mlups_stats["EMLUPS"])

    report_name = f"mlups_3d_multires_size_{args.cube_edge}_levels_{stats['num_levels']}"
    report.write(report_name, True)
    print("Report generated successfully.")


def main():
    args = parse_arguments()
    velocity_set = setup_simulation(args)
    grid_shape = (args.cube_edge, args.cube_edge, args.cube_edge)
    stats = run(velocity_set, grid_shape, args.num_steps, args.num_levels, args.collision_model,
                args.export_final_velocity,
                mres_perf_opt = args.mres_perf_opt)
    mlups_stats = calculate_mlups(args.cube_edge, args.num_steps, stats["time"], stats["num_levels"])

    print(f"Simulation completed in {stats['time']:.2f} seconds")
    print(f"Number of levels {stats['num_levels']}")
    print(f"Cube edge {args.cube_edge}")
    print(f"Coarse Iterations {args.num_steps}")
    finer_steps = mlups_stats["finer_steps"]
    print(f"Fine Iterations {finer_steps}")
    EMLUPS = mlups_stats["EMLUPS"]
    print(f"EMLUPs: {EMLUPS:.2f}")

    # Generate report if requested
    if args.report:
        generate_report(args, stats, mlups_stats)


if __name__ == "__main__":
    main()
