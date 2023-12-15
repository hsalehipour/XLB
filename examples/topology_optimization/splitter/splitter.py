from pathlib import Path

import jax.numpy as jnp
import trimesh

from doppler.callbacks.csv_logger import CSVLogger
from doppler.callbacks.shape_checkpoint import ShapeCheckpoint
from doppler.topopt import ALTopOpt, ALConstraint, ALConstraintType
from doppler.objectives import PressureDrop, VolumeFraction
from doppler.geometry.sdf import SDFGrid

from src.utils import *
from src.boundary_conditions import *
from src.adjoint import LBMBaseDifferentiable
from src.models import BGKSim
from src.lattice import LatticeD3Q19

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + '/' + "block.stl"

# class SimulationParameters:
#     lattice: LatticeD3Q27('f32/f32')
#     omega: float = None
#     nx: int = 100
#     ny: int = 100
#     nz: int = 100
#     precision: str = 'f32/f32'
#     io_rate: int = 100
#     print_info_rate: int = 100
#
# config = SimulationParameters()

class Splitter(LBMBaseDifferentiable):
    def __init__(self, sdf, **kwargs):
        self.sdf = sdf
        super().__init__(**kwargs)

    def set_boundary_conditions(self):
        voxel_coordinates = self.sdf.voxel_grid_coordinates()
        bmask = self.sdf.boundary_mask()
        bdry_cord = voxel_coordinates[bmask[..., 0], :]
        bdry_indices = self.sdf.index_from_coord(bdry_cord)
        xx, yy, zz = bdry_cord[:, 0], bdry_cord[:, 1], bdry_cord[:, 2]
        Lx, Ly, Lz = xx.max() - xx.min(), yy.max() - yy.min(), zz.max() - zz.min()
        inlet = (xx == xx.min()) & \
                (yy < yy.min() + 0.8*Ly) & (yy > yy.min() + 0.6*Ly) & \
                (zz < zz.min() + 0.8*Lz) & (zz > zz.min() + 0.2*Lz)
        outlet = (yy == yy.min()) & \
                 (xx < xx.min() + 0.8*Lx) & (xx > xx.min() + 0.6*Lx) & \
                 (zz < zz.min() + 0.8*Lz) & (zz > zz.min() + 0.2*Lz)
        wall = ~(inlet | outlet)

        # Get boundary voxel indices from the above masks
        inlet = bdry_indices[inlet]
        outlet = bdry_indices[outlet]
        wall = bdry_indices[wall]

        # Inlet BC
        vel_inlet = np.zeros(inlet.shape, dtype=self.precisionPolicy.compute_dtype)
        vel_inlet[:, 0] = 0.02
        self.BCs.append(ZouHe(tuple(inlet.T), self.gridInfo, self.precisionPolicy, 'velocity', vel_inlet))

        # Outlet BC
        rho_outlet = np.ones((outlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        self.BCs.append(ZouHe(tuple(outlet.T), self.gridInfo, self.precisionPolicy, 'pressure', rho_outlet))

        # No-slip BC for all no-slip boundaries that are part of optimization
        # sdf_wall = self.sdf.array[tuple(wall.T)]
        implicit_distance = np.array(self.sdf.array[..., 0])
        self.BCs.append(InterpolatedBounceBackDifferentiable(tuple(wall.T),
                                                             implicit_distance, self.gridInfo, self.precisionPolicy))
        self.BCs[-1].needsExtraConfiguration = False
        self.BCs[-1].isSolid = False

def main():
    orig_mesh = trimesh.load(filename)
    extents = orig_mesh.extents
    nx, ny, nz = 96, 96, 16 # tuple((extents * 2).astype(np.int64))
    shape = (nx, ny, nz)
    sdf_grid = SDFGrid.load_from_mesh(orig_mesh, shape, dtype=jnp.float32)
    sdf_grid.pad((8, 8), (8, 8), (8, 8))

    def xlb_instantiator(sdf_grid):
        precision = 'f32/f32'
        lattice = LatticeD3Q19(precision)
        omega = 1.0#

        kwargs = {
            'lattice': lattice,
            'omega': omega,
            'nx': nx,
            'ny': ny,
            'nz': nz,
            'precision': precision,
            'io_rate': 1,
            'print_info_rate': 1,
        }
        return Splitter(sdf_grid, **kwargs)

    # Minimize the variance of the shape
    objectives = [PressureDrop(xlb_instantiator=xlb_instantiator, init_shape=sdf_grid)]

    # subject to a volume constraint to avoid collapsing to a point
    constraints = [ALConstraint(VolumeFraction(init_shape=sdf_grid), target=0.5,
                                constraint_type=ALConstraintType.EqualTo)]

    callbacks = [CSVLogger(Path(__file__).parent / "outputs" / "ex1_shape_variance"),
                 ShapeCheckpoint(Path(__file__).parent / "outputs" / "ex1_shape_variance" / "checkpoints")]

    # Careful with the max_inner_loop_iter here. Setting it to a large value can drive the shape to collapse to a point
    # because the shape variance is minimized to zero.
    topopt = ALTopOpt(sdf_grid, objectives=objectives, constraints=constraints, max_iter=20, max_inner_loop_iter=4,
                      callbacks=callbacks, band_voxels=3)
    # The final shape is in topopt.shape or saved to VTI and PLY files if the ShapeCheckpoint callback was provided
    topopt.run()


if __name__ == "__main__":
    main()
