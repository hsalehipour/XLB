import xlb
import argparse
import time
import warp as wp
import numpy as np
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import FullwayBounceBackBC, EquilibriumBC
from xlb.distribute import distribute
from xlb.operator.macroscopic import Macroscopic


# -------------------------- Simulation Setup --------------------------


def parse_arguments():
    parser = argparse.ArgumentParser(description="MLUPS for 3D Lattice Boltzmann Method Simulation (BGK)")
    parser.add_argument("cube_edge", type=int, help="Length of the edge of the cubic grid")
    parser.add_argument("num_steps", type=int, help="Number of timesteps for the simulation")
    parser.add_argument("compute_backend", type=str, help="Backend for the simulation (jax, warp or neon)")
    parser.add_argument("precision", type=str, help="Precision for the simulation (e.g., fp32/fp32)")
    parser.add_argument(
        "--gpu_devices",
        type=str,
        default=None,
        help="List of the CUDA devices to use (e.g., --gpu_devices=[0,1,2]). This is only used for Neon backend.",
    )
    # add a flat to choose between 19 or 27 velocity set
    parser.add_argument("--velocity_set", type=str, default="D3Q19", help="Lattice type: D3Q19 or D3Q27 (default: D3Q19)")
    # add a flat to choose between multi-gpu occ options based on the neon occ:
    parser.add_argument(
        "--occ", type=str, default="standard", help="Overlapping Communication and Computation option (standard, none) (default: standard)"
    )
    parser.add_argument("--report", action="store_true", help="Generate a neon report file (default: disabled)")
    parser.add_argument("--export_final_velocity", action="store_true", help="Export the final velocity field to a vti file (default: disabled)")
    parser.add_argument("--measure_scalability", action="store_true", help="Measure scalability of the simulation (default: disabled)")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of repetitions for the simulation (default: 1) to get the average MLUPs and standard deviation",
    )

    args = parser.parse_args()

    # Parse gpu_devices string to list
    if args.gpu_devices is not None:
        try:
            import ast

            args.gpu_devices = ast.literal_eval(args.gpu_devices)
            if not isinstance(args.gpu_devices, list):
                args.gpu_devices = [args.gpu_devices]  # Handle single integer case
        except (ValueError, SyntaxError):
            raise ValueError("Invalid gpu_devices format. Use format like [0,1,2] or [0]")

    # Checking the compute backend and covert it to the right type
    compute_backend = None
    if args.compute_backend == "jax":
        compute_backend = ComputeBackend.JAX
    elif args.compute_backend == "warp":
        compute_backend = ComputeBackend.WARP
    elif args.compute_backend == "neon":
        compute_backend = ComputeBackend.NEON
    else:
        raise ValueError("Invalid compute backend specified. Use 'jax', 'warp', or 'neon'.")
    args.compute_backend = compute_backend

    # Checking OCC
    if args.occ not in ["standard", "none"]:
        raise ValueError("Invalid occupancy option. Use 'standard', or 'none'.")
    if args.gpu_devices is None and args.compute_backend == ComputeBackend.NEON:
        print("[Warning] No GPU devices specified. Using default device 0.")
        args.gpu_devices = [0]
    if args.compute_backend == ComputeBackend.NEON:
        import neon

        occ = neon.SkeletonConfig.OCC.from_string(args.occ)
        args.occ = occ

    # Checking precision policy
    precision_policy_map = {
        "fp32/fp32": PrecisionPolicy.FP32FP32,
        "fp64/fp64": PrecisionPolicy.FP64FP64,
        "fp64/fp32": PrecisionPolicy.FP64FP32,
        "fp32/fp16": PrecisionPolicy.FP32FP16,
    }
    precision_policy = precision_policy_map.get(args.precision)
    if precision_policy is None:
        raise ValueError("Invalid precision specified.")
    args.precision_policy = precision_policy

    # Checking velocity set
    if args.velocity_set not in ["D3Q19", "D3Q27"]:
        raise ValueError("Invalid velocity set. Use 'D3Q19' or 'D3Q27'.")

    if args.velocity_set == "D3Q19":
        velocity_set = xlb.velocity_set.D3Q19(precision_policy=args.precision_policy, compute_backend=compute_backend)
    elif args.velocity_set == "D3Q27":
        velocity_set = xlb.velocity_set.D3Q27(precision_policy=args.precision_policy, compute_backend=compute_backend)
    args.velocity_set = velocity_set

    if args.gpu_devices is not None and args.compute_backend != ComputeBackend.NEON:
        raise ValueError("--gpu_devices can be used only with the Neon backend.")

    if args.gpu_devices is None:
        args.gpu_devices = [0]

    print_args(args)

    return args


def print_args(args):
    # Print simulation configuration
    print("=" * 60)
    print("           3D LATTICE BOLTZMANN SIMULATION CONFIG")
    print("=" * 60)
    print(f"Grid Size:            {args.cube_edge}³ ({args.cube_edge:,} × {args.cube_edge:,} × {args.cube_edge:,})")
    print(f"Total Lattice Points: {args.cube_edge**3:,}")
    print(f"Time Steps:           {args.num_steps:,}")
    print(f"Compute Backend:      {args.compute_backend.name}")
    print(f"Precision Policy:     {args.precision}")
    print(f"Velocity Set:         {args.velocity_set.__class__.__name__}")
    print(f"Generate Report:      {'Yes' if args.report else 'No'}")
    print(f"Measure Scalability:  {'Yes' if args.measure_scalability else 'No'}")
    print(f"Export Velocity:      {'Yes' if args.export_final_velocity else 'No'}")
    print(f"Repetitions:          {args.repetitions}")

    if args.compute_backend.name == "NEON":
        print(f"GPU Devices:          {args.gpu_devices}")
        # Convert the neon OCC enum back to string for display
        occ_display = args.occ.to_string() if hasattr(args.occ, "__class__") else args.occ
        print(f"OCC Strategy:         {occ_display}")

    print("=" * 60)
    print("Starting simulation...")
    print()


def init_xlb(args):
    xlb.init(
        velocity_set=args.velocity_set,
        default_backend=args.compute_backend,
        default_precision_policy=args.precision_policy,
    )
    options = None
    if args.compute_backend == ComputeBackend.NEON:
        neon_options = {"occ": args.occ, "device_list": args.gpu_devices}
        options = neon_options
    return args.compute_backend, args.precision_policy, options


def run_simulation(compute_backend, precision_policy, grid_shape, num_steps, options, export_final_velocity, repetitions, num_devices):
    grid = grid_factory(grid_shape, backend_config=options)
    box = grid.bounding_box_indices()
    box_no_edge = grid.bounding_box_indices(remove_edges=True)

    lid = box_no_edge["top"]
    walls = [box["bottom"][i] + box["left"][i] + box["right"][i] + box["front"][i] + box["back"][i] for i in range(len(grid.shape))]
    walls = np.unique(np.array(walls), axis=-1).tolist()

    boundary_conditions = [
        EquilibriumBC(rho=1.0, u=(0.02, 0.0, 0.0), indices=lid),
        FullwayBounceBackBC(indices=walls),
    ]

    stepper = IncompressibleNavierStokesStepper(
        grid=grid,
        boundary_conditions=boundary_conditions,
        collision_type="BGK",
        backend_config=options,
    )

    # Distribute if using JAX
    if compute_backend == ComputeBackend.JAX:
        stepper = distribute(
            stepper,
            grid,
            xlb.velocity_set.D3Q19(precision_policy=precision_policy, compute_backend=compute_backend),
        )

    # Initialize fields
    omega = 1.0
    f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()

    warmup_iterations = 10
    # Warp-up iterations
    for i in range(warmup_iterations):
        f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, i)
        f_0, f_1 = f_1, f_0
    wp.synchronize()
    export_num_steps = warmup_iterations

    elapsed_time_list = []
    for i in range(repetitions):
        start_time = time.time()
        for i in range(num_steps):
            f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, i)
            f_0, f_1 = f_1, f_0
        wp.synchronize()
        elapsed_time = time.time() - start_time
        elapsed_time_list.append(elapsed_time)
        export_num_steps += num_steps

    # Define Macroscopic Calculation
    macro = Macroscopic(
        compute_backend=compute_backend,
        precision_policy=precision_policy,
        velocity_set=xlb.velocity_set.D3Q19(precision_policy=precision_policy, compute_backend=compute_backend),
    )

    if compute_backend == ComputeBackend.NEON:
        if export_final_velocity:
            rho = grid.create_field(cardinality=1, dtype=precision_policy.store_precision)
            u = grid.create_field(cardinality=3, dtype=precision_policy.store_precision)

            macro(f_0, rho, u)
            wp.synchronize()
            u.update_host(0)
            wp.synchronize()
            u.export_vti(f"mlups_3d_size_{grid_shape[0]}_dev_{num_devices}_step_{export_num_steps}.vti", "u")

    return elapsed_time_list


def calculate_mlups(cube_edge, num_steps, elapsed_time):
    total_lattice_updates = cube_edge**3 * num_steps
    mlups = (total_lattice_updates / elapsed_time) / 1e6
    return mlups


def print_summary(args, elapsed_time, mlups):
    """Print comprehensive simulation summary with parameters and performance results"""
    total_lattice_points = args.cube_edge**3
    total_lattice_updates = total_lattice_points * args.num_steps
    lattice_points_per_second = total_lattice_updates / elapsed_time

    print("\n\n\n" + "=" * 70)
    print("                    SIMULATION SUMMARY")
    print("=" * 70)

    # Simulation Parameters
    print("SIMULATION PARAMETERS:")
    print("-" * 25)
    print(f"  Grid Size:              {args.cube_edge}³ ({args.cube_edge:,} × {args.cube_edge:,} × {args.cube_edge:,})")
    print(f"  Total Lattice Points:   {total_lattice_points:,}")
    print(f"  Time Steps:             {args.num_steps:,}")
    print(f"  Total Lattice Updates:  {total_lattice_updates:,}")
    print(f"  Compute Backend:        {args.compute_backend.name}")
    print(f"  Precision Policy:       {args.precision}")
    print(f"  Velocity Set:           {args.velocity_set.__class__.__name__}")
    print(f"  Generate Report:        {'Yes' if args.report else 'No'}")
    print(f"  Measure Scalability:    {'Yes' if args.measure_scalability else 'No'}")

    if args.compute_backend.name == "NEON":
        print(f"  GPU Devices:            {args.gpu_devices}")
        occ_display = str(args.occ).split(".")[-1] if hasattr(args.occ, "__class__") else args.occ
        print(f"  OCC Strategy:           {occ_display}")

    print()

    # Performance Results
    print("PERFORMANCE RESULTS:")
    print("-" * 20)
    print(f"  Time in main loop:      {elapsed_time:.3f} seconds")
    print(f"  MLUPs:                  {mlups:.2f}")
    print(f"  Time per LBM step:      {elapsed_time / args.num_steps * 1000:.3f} ms")

    if args.compute_backend.name == "NEON" and len(args.gpu_devices) > 1:
        mlups_per_gpu = mlups / len(args.gpu_devices)
        print(f"  MLUPs per GPU:          {mlups_per_gpu:.2f}")

    print("=" * 70)


def print_summary_with_stats(args, stats):
    """Print comprehensive simulation summary with statistics from multiple repetitions"""
    total_lattice_points = args.cube_edge**3
    total_lattice_updates = total_lattice_points * args.num_steps

    mean_mlups = stats["mean_mlups"]
    std_mlups = stats["std_dev_mlups"]
    mean_elapsed_time = stats["mean_elapsed_time"]
    std_elapsed_time = stats["std_dev_elapsed_time"]

    print("\n\n\n" + "=" * 70)
    print("                    SIMULATION SUMMARY")
    print("=" * 70)

    # Simulation Parameters
    print("SIMULATION PARAMETERS:")
    print("-" * 25)
    print(f"  Grid Size:              {args.cube_edge}³ ({args.cube_edge:,} × {args.cube_edge:,} × {args.cube_edge:,})")
    print(f"  Total Lattice Points:   {total_lattice_points:,}")
    print(f"  Time Steps:             {args.num_steps:,}")
    print(f"  Total Lattice Updates:  {total_lattice_updates:,}")
    print(f"  Repetitions:            {args.repetitions}")
    print(f"  Compute Backend:        {args.compute_backend.name}")
    print(f"  Precision Policy:       {args.precision}")
    print(f"  Velocity Set:           {args.velocity_set.__class__.__name__}")
    print(f"  Generate Report:        {'Yes' if args.report else 'No'}")
    print(f"  Measure Scalability:    {'Yes' if args.measure_scalability else 'No'}")

    if args.compute_backend.name == "NEON":
        print(f"  GPU Devices:            {args.gpu_devices}")
        occ_display = str(args.occ).split(".")[-1] if hasattr(args.occ, "__class__") else args.occ
        print(f"  OCC Strategy:           {occ_display}")

    print()

    # Raw Data (if multiple repetitions)
    if args.repetitions > 1:
        print("RAW MEASUREMENT DATA:")
        print("-" * 21)
        print(f"{'Run':<6} {'Elapsed Time (s)':<18} {'MLUPs':<12} {'Time/Step (ms)':<15}")
        print("-" * 53)

        raw_elapsed_times = stats["raw_elapsed_times"]
        raw_mlups = stats["raw_mlups"]

        for i, (elapsed_time, mlups) in enumerate(zip(raw_elapsed_times, raw_mlups)):
            time_per_step = elapsed_time / args.num_steps * 1000
            print(f"{i + 1:<6} {elapsed_time:<18.3f} {mlups:<12.2f} {time_per_step:<15.3f}")

        print("-" * 53)
        print()

    # Performance Results (Statistical Summary)
    print("PERFORMANCE RESULTS:")
    print("-" * 20)
    if args.repetitions > 1:
        print(f"  Time in main loop:      {mean_elapsed_time:.3f} ± {std_elapsed_time:.3f} seconds")
        print(f"  MLUPs:                  {mean_mlups:.2f} ± {std_mlups:.2f}")
        print(f"  Time per LBM step:      {mean_elapsed_time / args.num_steps * 1000:.3f} ± {std_elapsed_time / args.num_steps * 1000:.3f} ms")
    else:
        print(f"  Time in main loop:      {mean_elapsed_time:.3f} seconds")
        print(f"  MLUPs:                  {mean_mlups:.2f}")
        print(f"  Time per LBM step:      {mean_elapsed_time / args.num_steps * 1000:.3f} ms")

    if args.compute_backend.name == "NEON" and len(args.gpu_devices) > 1:
        mlups_per_gpu = mean_mlups / len(args.gpu_devices)
        if args.repetitions > 1:
            mlups_per_gpu_std = std_mlups / len(args.gpu_devices)
            print(f"  MLUPs per GPU:          {mlups_per_gpu:.2f} ± {mlups_per_gpu_std:.2f}")
        else:
            print(f"  MLUPs per GPU:          {mlups_per_gpu:.2f}")

    print("=" * 70)


def print_scalability_summary(args, stats_list):
    """Print comprehensive scalability summary with MLUPs statistics for different GPU counts"""
    total_lattice_points = args.cube_edge**3
    total_lattice_updates = total_lattice_points * args.num_steps

    print("\n\n\n" + "=" * 95)
    print("                           SCALABILITY ANALYSIS")
    print("=" * 95)

    # Simulation Parameters
    print("SIMULATION PARAMETERS:")
    print("-" * 25)
    print(f"  Grid Size:              {args.cube_edge}³ ({args.cube_edge:,} × {args.cube_edge:,} × {args.cube_edge:,})")
    print(f"  Total Lattice Points:   {total_lattice_points:,}")
    print(f"  Time Steps:             {args.num_steps:,}")
    print(f"  Total Lattice Updates:  {total_lattice_updates:,}")
    print(f"  Repetitions:            {args.repetitions}")
    print(f"  Compute Backend:        {args.compute_backend.name}")
    print(f"  Precision Policy:       {args.precision}")
    print(f"  Velocity Set:           {args.velocity_set.__class__.__name__}")

    if args.compute_backend.name == "NEON":
        occ_display = str(args.occ).split(".")[-1] if hasattr(args.occ, "__class__") else args.occ
        print(f"  OCC Strategy:           {occ_display}")
        print(f"  Available GPU Devices:  {args.gpu_devices}")

    print()

    # Extract mean MLUPs for calculations
    mlups_means = [stats["mean_mlups"] for stats in stats_list]
    baseline_mlups = mlups_means[0] if mlups_means else 0

    # Scalability Results
    print("SCALABILITY RESULTS:")
    print("-" * 20)
    print(f"{'GPUs':<6} {'MLUPs (mean±std)':<18} {'Speedup':<10} {'Efficiency':<12} {'MLUPs/GPU':<12}")
    print("-" * 68)

    for i, stats in enumerate(stats_list):
        num_gpus = i + 1
        mean_mlups = stats["mean_mlups"]
        std_mlups = stats["std_dev_mlups"]
        speedup = mean_mlups / baseline_mlups if baseline_mlups > 0 else 0
        efficiency = (speedup / num_gpus) if num_gpus > 0 else 0
        mlups_per_gpu = mean_mlups / num_gpus if num_gpus > 0 else 0

        # Format MLUPs with standard deviation
        if args.repetitions > 1:
            mlups_str = f"{mean_mlups:.2f}±{std_mlups:.2f}"
        else:
            mlups_str = f"{mean_mlups:.2f}"

        print(f"{num_gpus:<6} {mlups_str:<18} {speedup:<10.2f} {efficiency:<11.3f} {mlups_per_gpu:<12.2f}")

    print("-" * 68)

    # Summary Statistics
    if len(stats_list) > 1:
        max_mlups = max(mlups_means)
        max_mlups_idx = mlups_means.index(max_mlups)
        max_speedup = max_mlups / baseline_mlups if baseline_mlups > 0 else 0
        best_efficiency_idx = 0
        best_efficiency = 0.0

        for i, mean_mlups in enumerate(mlups_means):
            num_gpus = i + 1
            speedup = mean_mlups / baseline_mlups if baseline_mlups > 0 else 0
            efficiency = (speedup / num_gpus) if num_gpus > 0 else 0
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_efficiency_idx = i

        print()
        print("SUMMARY STATISTICS:")
        print("-" * 19)
        print(f"  Best Performance:       {max_mlups:.2f} MLUPs ({max_mlups_idx + 1} GPUs)")
        if args.repetitions > 1:
            max_std = stats_list[max_mlups_idx]["std_dev_mlups"]
            print(f"  Performance Std Dev:    ±{max_std:.2f} MLUPs")
        print(f"  Maximum Speedup:        {max_speedup:.2f}x")
        print(f"  Best Efficiency:        {best_efficiency:.3f} ({best_efficiency_idx + 1} GPUs)")
        print(f"  Scalability Range:      1-{len(stats_list)} GPUs")

    print("=" * 95)


def report(args, stats):
    import neon
    import sys

    report = neon.Report("LBM MLUPS LDC")
    
    # Save the full command line
    command_line = " ".join(sys.argv)
    report.add_member("command_line", command_line)
    
    report.add_member("velocity_set", args.velocity_set.__class__.__name__)
    report.add_member("compute_backend", args.compute_backend.name)
    report.add_member("precision_policy", args.precision)
    report.add_member("grid_size", args.cube_edge)
    report.add_member("num_steps", args.num_steps)
    report.add_member("repetitions", args.repetitions)

    # Statistical measures
    report.add_member("mean_elapsed_time", stats["mean_elapsed_time"])
    report.add_member("mean_mlups", stats["mean_mlups"])
    report.add_member("std_dev_elapsed_time", stats["std_dev_elapsed_time"])
    report.add_member("std_dev_mlups", stats["std_dev_mlups"])

    # Raw data vectors (if multiple repetitions)
    if args.repetitions > 1:
        report.add_member_vector("raw_elapsed_times", stats["raw_elapsed_times"])
        report.add_member_vector("raw_mlups", stats["raw_mlups"])

    # Legacy fields for backwards compatibility
    report.add_member("elapsed_time", stats["mean_elapsed_time"])
    report.add_member("mlups", stats["mean_mlups"])

    report.add_member("occ", (args.occ.to_string()))
    report.add_member_vector("gpu_devices", args.gpu_devices)
    report.add_member("num_devices", len(args.gpu_devices))
    report.add_member("measure_scalability", args.measure_scalability)

    report_name = "mlups_3d_" + f"size_{args.cube_edge}"
    if args.measure_scalability:
        report_name += f"_dev_{len(args.gpu_devices)}"
    if args.repetitions > 1:
        report_name += f"_rep_{args.repetitions}"
    report.write(report_name, True)


# -------------------------- Simulation Loop --------------------------


def benchmark(args):
    compute_backend, precision_policy, options = init_xlb(args)
    grid_shape = (args.cube_edge, args.cube_edge, args.cube_edge)

    elapsed_time_list = []
    mlups_list = []
    elapsed_time_list = run_simulation(
        compute_backend=compute_backend,
        precision_policy=precision_policy,
        grid_shape=grid_shape,
        num_steps=args.num_steps,
        options=options,
        export_final_velocity=args.export_final_velocity,
        repetitions=args.repetitions,
        num_devices=len(args.gpu_devices),
    )

    for elapsed_time in elapsed_time_list:
        mlups = calculate_mlups(args.cube_edge, args.num_steps, elapsed_time)
        mlups_list.append(mlups)

    mean_mlups = np.mean(mlups_list)
    std_dev_mlups = np.std(mlups_list)
    mean_elapsed_time = np.mean(elapsed_time_list)
    std_dev_elapsed_time = np.std(elapsed_time_list)

    stats = {
        "mean_mlups": mean_mlups,
        "std_dev_mlups": std_dev_mlups,
        "mean_elapsed_time": mean_elapsed_time,
        "std_dev_elapsed_time": std_dev_elapsed_time,
        "num_devices": len(args.gpu_devices),
        "raw_mlups": mlups_list,
        "raw_elapsed_times": elapsed_time_list,
    }
    # Generate report if requested
    if args.report:
        report(args, stats)
        print("Report generated successfully.")

    return stats


def main():
    args = parse_arguments()
    if not args.measure_scalability:
        stats = benchmark(args)
        # For single run, print_summary expects individual values with additional stats
        print_summary_with_stats(args, stats)
        return

    stats_list = []
    for num_devices in range(1, len(args.gpu_devices) + 1):
        import copy

        args_copy = copy.deepcopy(args)
        args_copy.gpu_devices = args_copy.gpu_devices[:num_devices]
        stats = benchmark(args_copy)
        stats_list.append(stats)

    # Print comprehensive scalability analysis
    print_scalability_summary(args, stats_list)


if __name__ == "__main__":
    main()
