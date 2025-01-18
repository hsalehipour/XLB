import xlb
import argparse
import time
import warp as wp
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import EquilibriumBC, HalfwayBounceBackBC, DoNothingBC, FullwayBounceBackBC
from xlb.operator.equilibrium import QuadraticEquilibrium
from typing import Tuple


def initialize_eq(f, grid, velocity_set, backend, rho=None, u=None):
    rho = rho or grid.create_field(cardinality=1, fill_value=1.0, precision=xlb.Precision.FP32)
    u = u or grid.create_field(cardinality=velocity_set.d, fill_value=0.0, precision=xlb.Precision.FP32)
    equilibrium = xlb.operator.equilibrium.QuadraticEquilibrium(
            velocity_set=velocity_set,
            precision_policy=xlb.PrecisionPolicy.FP32FP32,
            compute_backend=backend,
        )

    if backend == ComputeBackend.JAX:
        f = equilibrium(rho, u)

    elif backend == ComputeBackend.WARP:
        f = equilibrium(rho, u, f)

    del rho, u

    return f

def create_nse_fields(
    grid_shape: Tuple[int, int, int], velocity_set= xlb.velocity_set.D3Q19(), compute_backend=ComputeBackend.WARP, precision_policy=PrecisionPolicy.FP32FP32
):
    # Make grid
    grid = xlb.grid.WarpGrid(shape=grid_shape)

    # Make feilds
    f_0 = grid.create_field(cardinality=velocity_set.q, precision=xlb.Precision.FP32)
    f_1 = grid.create_field(cardinality=velocity_set.q, precision=xlb.Precision.FP32)
    boundary_mask = grid.create_field(cardinality=1, precision=xlb.Precision.UINT8)
    missing_mask = grid.create_field(cardinality=velocity_set.q, precision=xlb.Precision.BOOL)

    return grid, f_0, f_1, missing_mask, boundary_mask

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="MLUPS for 3D Lattice Boltzmann Method Simulation (BGK)"
    )
    parser.add_argument("cube_edge", type=int, help="Length of the edge of the cubic grid")
    parser.add_argument("num_steps", type=int, help="Timestep for the simulation")
    parser.add_argument("backend", type=str, help="Backend for the simulation (jax or warp)")
    parser.add_argument("precision", type=str, help="Precision for the simulation (e.g., fp32/fp32)")
    return parser.parse_args()

def setup_simulation(args):
    backend = ComputeBackend.JAX if args.backend == "jax" else ComputeBackend.WARP
    precision_policy_map = {
        "fp32/fp32": PrecisionPolicy.FP32FP32,
        "fp64/fp64": PrecisionPolicy.FP64FP64,
        "fp64/fp32": PrecisionPolicy.FP64FP32,
        "fp32/fp16": PrecisionPolicy.FP32FP16
    }
    precision_policy = precision_policy_map.get(args.precision)
    if precision_policy is None:
        raise ValueError("Invalid precision")

    return backend, precision_policy

def create_grid_and_fields(cube_edge):
    grid_shape = (cube_edge, cube_edge, cube_edge)
    grid, f_0, f_1, missing_mask, boundary_mask = create_nse_fields(grid_shape)

    return grid, f_0, f_1, missing_mask, boundary_mask

def define_boundary_indices(grid):
    lid = grid.boundingBoxIndices['top']
    walls = [grid.boundingBoxIndices['bottom'][i] + grid.boundingBoxIndices['left'][i] + 
            grid.boundingBoxIndices['right'][i] + grid.boundingBoxIndices['front'][i] +
            grid.boundingBoxIndices['back'][i] for i in range(xlb.velocity_set.D3Q19().d)]
    return lid, walls
    
def setup_boundary_conditions(grid):
    lid, walls = define_boundary_indices(grid)    
    # Note: The MLUPs number as based on single A6000 GPU after running this file twice (first clearing cache manually each time)

    # MLUPs: 4443.81
    # return [EquilibriumBC(rho=1.0, u=(0.02, 0.0, 0.0), indices=lid), FullwayBounceBackBC(indices=walls)]

    # MLUPs: 3332.13
    # return [EquilibriumBC(rho=1.0, u=(0.02, 0.0, 0.0), indices=lid), HalfwayBounceBackBC(indices=walls)]

    # MLUPs: 4449.56
    # return [DoNothingBC(indices=walls)]
    
    # MLUPs: 3205.85
    # return [HalfwayBounceBackBC(indices=walls)]

    # MLUPs: 4450.15
    return [EquilibriumBC(rho=1.0, u=(0.02, 0.0, 0.0)), FullwayBounceBackBC(), DoNothingBC()]

def run(stepper, f_0, f_1, backend, grid, boundary_mask, missing_mask, num_steps):
    start_time = time.time()

    for i in range(num_steps):
        f_1 = stepper(f_0, f_1, boundary_mask, missing_mask, i)
        f_0, f_1 = f_1, f_0
    wp.synchronize()

    end_time = time.time()
    return end_time - start_time

def calculate_mlups(cube_edge, num_steps, elapsed_time):
    total_lattice_updates = cube_edge**3 * num_steps
    mlups = (total_lattice_updates / elapsed_time) / 1e6
    return mlups

def main():
    args = parse_arguments()
    backend, precision_policy = setup_simulation(args)
    grid, f_0, f_1, missing_mask, boundary_mask = create_grid_and_fields(args.cube_edge)
    f_0 = initialize_eq(f_0, grid, xlb.velocity_set.D3Q19(), backend)
    nx, ny, nz = args.cube_edge, args.cube_edge, args.cube_edge

    omega = 1
    velocity_set = xlb.velocity_set.D3Q19()
    compute_backend = backend

    collision = xlb.operator.collision.BGK(
        omega=omega,
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )
    equilibrium = xlb.operator.equilibrium.QuadraticEquilibrium(
        velocity_set=velocity_set,
        precision_policy= precision_policy,
        compute_backend= compute_backend,
    )
    macroscopic = xlb.operator.macroscopic.Macroscopic(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )
    stream = xlb.operator.stream.Stream(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )
    equilibrium_bc = xlb.operator.boundary_condition.EquilibriumBC(
            rho=1.0,
            u=(0.02, 0.0, 0.0),
            equilibrium_operator=equilibrium,
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend,
        )
    half_way_bc = xlb.operator.boundary_condition.HalfwayBounceBackBC(
            velocity_set= velocity_set,
            precision_policy= precision_policy,
            compute_backend= compute_backend,
        )
    full_way_bc = xlb.operator.boundary_condition.FullwayBounceBackBC(
        velocity_set= velocity_set,
        precision_policy= precision_policy,
        compute_backend= compute_backend,
        )
    do_nothing_bc = xlb.operator.boundary_condition.DoNothingBC(
        velocity_set= velocity_set,
        precision_policy= precision_policy,
        compute_backend= compute_backend,
    )
    stepper = xlb.operator.stepper.IncompressibleNavierStokesStepper(
        collision= collision,
        equilibrium= equilibrium,
        macroscopic= macroscopic,
        stream= stream,
        boundary_conditions=[
                half_way_bc,
                full_way_bc,
                equilibrium_bc,
                do_nothing_bc
        ],
    )
    planar_boundary_masker = xlb.operator.boundary_masker.PlanarBoundaryMasker(
            velocity_set= velocity_set,
            precision_policy= precision_policy,
            compute_backend= compute_backend,
        )
    
    # Note: The MLUPs number as based on single A6000 GPU after running this file twice (first clearing cache manually each time)

    # MLUPs: 3825.33 (Only fullway)
    # bc1 = full_way_bc
    # bc2 = full_way_bc; bc2.id=0

    # MLUPs: 3742.33 (Only halfway)
    # bc1 = half_way_bc
    # bc2 = full_way_bc; bc2.id=0

    # MLUPS: 3806.58 (no BC)
    # bc1 = half_way_bc; bc1.id = 0
    # bc2 = full_way_bc; bc2.id=0

    # MLUPs: 3713.42 (halfway + fullway)
    bc1 = half_way_bc
    bc2 = full_way_bc

    # Set bc1 (bottom z face)
    lower_bound = (0, 0, 0)
    upper_bound = (nx,  ny, 0)
    direction = (0, 0, 1)
    boundary_mask, missing_mask = planar_boundary_masker(
        lower_bound,
        upper_bound,
        direction,
        bc1.id,
        boundary_mask,
        missing_mask,
        (0, 0, 0)
    )

    # Set bc2 (top z face)
    lower_bound = (0, 0, nz-1)
    upper_bound = (nx, ny, nz-1)
    direction = (0, 0, -1)
    boundary_mask, missing_mask = planar_boundary_masker(
        lower_bound,
        upper_bound,
        direction,
         bc2.id,
         boundary_mask,
         missing_mask,
        (0, 0, 0)
    )

    elapsed_time = run(stepper, f_0, f_1, backend, grid, boundary_mask, missing_mask, args.num_steps)
    mlups = calculate_mlups(args.cube_edge, args.num_steps, elapsed_time)

    print(f"Simulation completed in {elapsed_time:.2f} seconds")
    print(f"MLUPs: {mlups:.2f}")

if __name__ == "__main__":
    main()