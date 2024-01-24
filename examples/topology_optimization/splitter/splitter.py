from pathlib import Path
from jax import config

import numpy as np
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

# config.update('jax_enable_x64', True)

class Splitter(LBMBaseDifferentiable):
    def __init__(self, sdf, **kwargs):
        self.sdf = sdf
        super().__init__(**kwargs)

    def boundary(self, sdf: SDFGrid):
        """
        SDFGrid.boundary_mask() returns both voxels around the zero SDF, inside and outside of the fluid domain. The
        purpose of this routine is to isolate those voxels inside the fluid domain with SDF <= 0 but ensure other
        corner cells that are with sdf <-self.spacing, eg. -\sqrt(dim) < sdf/self.spacing < -1, are also included in
        the boundary. This is achieved by inspecting solid voxels with 0 < sdf < self.spacing that would land on SDF <0
        after travelling along all possible LBM lattice directions.
        """
        # get the required information
        voxel_coordinates = sdf.voxel_grid_coordinates()
        sdf_array = sdf.array[..., 0]

        # First let's isolate the obvious fluid boundary cells with negative SDF
        bmask_interior = (sdf_array > -sdf.spacing) & (sdf_array <= 0.)
        bdry_cord = voxel_coordinates[bmask_interior]
        bdry_indices_interior = sdf.index_from_coord(bdry_cord)

        # Next, find the fluid boundary voxels not included in the first set by traversing solid boundary voxels
        bmask_exterior = (sdf_array > 0.0) & (sdf_array < sdf.spacing)
        bdry_cord = voxel_coordinates[bmask_exterior]
        c = np.array(self.lattice.c)
        bdry_indices_lst = []
        for q in range(1, self.lattice.q):
            bdry_cord_nbr = bdry_cord + c[:, q]*sdf.spacing
            idx = sdf.index_from_coord(bdry_cord_nbr)
            mask = sdf_array[tuple(idx.T)] <= 0
            bdry_indices_lst.append(idx[mask])
        bdry_indices_exterior = np.unique(np.vstack(bdry_indices_lst), axis=0)
        return np.unique(np.vstack([bdry_indices_interior, bdry_indices_exterior]), axis=0)


    def set_boundary_conditions(self):
        bdry_indices = self.boundary(self.sdf)
        bdry_cord = self.sdf.coord_from_index(bdry_indices)
        xx, yy, zz = bdry_cord[:, 0], bdry_cord[:, 1], bdry_cord[:, 2]
        # Lx, Ly, Lz = xx.max() - xx.min(), yy.max() - yy.min(), zz.max() - zz.min()
        # inlet = (xx == xx.min()) & \
        #         (yy < yy.min() + 0.8*Ly) & (yy > yy.min() + 0.6*Ly) & \
        #         (zz < zz.min() + 0.8*Lz) & (zz > zz.min() + 0.2*Lz)
        # outlet = (yy == yy.min()) & \
        #          (xx < xx.min() + 0.8*Lx) & (xx > xx.min() + 0.6*Lx) & \
        #          (zz < zz.min() + 0.8*Lz) & (zz > zz.min() + 0.2*Lz)
        Lx, Ly, Lz = 47., 47.0, 7.
        xxmin, yymin, zzmin = 0.5, 0.5, 0.5
        inlet = (xx == xxmin) & \
                (yy < yymin + 0.8*Ly) & (yy > yymin + 0.6*Ly) & \
                (zz < zzmin + 0.8*Lz) & (zz > zzmin + 0.2*Lz)
        outlet = (yy == yymin) & \
                 (xx < xxmin + 0.8*Lx) & (xx > xxmin + 0.6*Lx) & \
                 (zz < zzmin + 0.8*Lz) & (zz > zzmin + 0.2*Lz)
        wall = ~(inlet | outlet)

        # Get boundary voxel indices from the above masks
        inlet = bdry_indices[inlet]
        outlet = bdry_indices[outlet]
        wall = bdry_indices[wall]

        # Inlet BC
        vel_inlet = np.zeros(inlet.shape, dtype=self.precisionPolicy.compute_dtype)
        vel_inlet[:, 0] = 0.08
        self.BCs.append(ZouHe(tuple(inlet.T), self.gridInfo, self.precisionPolicy, 'velocity', vel_inlet))

        # Outlet BC
        rho_outlet = np.ones((outlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        self.BCs.append(ZouHe(tuple(outlet.T), self.gridInfo, self.precisionPolicy, 'pressure', rho_outlet))

        # No-slip BC for all no-slip boundaries that are part of optimization
        # sdf_wall = self.sdf.array[tuple(wall.T)]
        self.BCs.append(InterpolatedBounceBackDifferentiable(tuple(wall.T),
                                                             self.gridInfo, self.precisionPolicy))
        self.BCs[-1].needsExtraConfiguration = False
        self.BCs[-1].isSolid = False

    # def output_data(self, **kwargs):
    #     # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
    #     rho = np.array(kwargs["rho"])
    #     u = np.array(kwargs["u"])
    #     timestep = kwargs["timestep"]
    #     fields = {"rho": rho[..., 0], "u_x": u[..., 0], "u_y": u[..., 1], "u_z": u[..., 2]}
    #     if timestep % 100 == 0:
    #         save_fields_vtk(timestep, fields)
    #         save_BCs_vtk(timestep, self.BCs, self.gridInfo)

# def port_coord(shape: SDFGrid):
#     voxel_coordinates = shape.voxel_grid_coordinates()
#     bmask = shape.boundary_mask()
#     bdry_cord = voxel_coordinates[bmask[..., 0], :]
#     xx, yy, zz = bdry_cord[:, 0], bdry_cord[:, 1], bdry_cord[:, 2]
#     Lx, Ly, Lz = xx.max() - xx.min(), yy.max() - yy.min(), zz.max() - zz.min()
#     inlet = (xx == xx.min()) & \
#             (yy < yy.min() + 0.8*Ly) & (yy > yy.min() + 0.6*Ly) & \
#             (zz < zz.min() + 0.8*Lz) & (zz > zz.min() + 0.2*Lz)
#     outlet = (yy == yy.min()) & \
#              (xx < xx.min() + 0.8*Lx) & (xx > xx.min() + 0.6*Lx) & \
#              (zz < zz.min() + 0.8*Lz) & (zz > zz.min() + 0.2*Lz)
#     inlet_coord = bdry_cord[inlet]
#     outlet_coord = bdry_cord[outlet]
#     port_dic = {'inlet': inlet_coord,
#                 'outlet': outlet_coord}
#     return port_dic

def main():
    orig_mesh = trimesh.load(filename)
    extents = orig_mesh.extents
    nx, ny, nz = 48, 48, 8 # tuple((extents * 2).astype(np.int64))
    shape = (nx, ny, nz)
    pad_width = 4
    sdf_grid = SDFGrid.load_from_mesh(orig_mesh, shape, dtype=jnp.float32, pad_width=pad_width)
    nx += 2*pad_width
    ny += 2*pad_width
    nz += 2*pad_width

    def xlb_instantiator(sdf_grid):
        precision = 'f32/f32'
        lattice = LatticeD3Q19(precision)
        omega = 1.6#

        kwargs = {
            'lattice': lattice,
            'omega': omega,
            'nx': nx,
            'ny': ny,
            'nz': nz,
            'precision': precision,
            'io_rate': 100,
            'print_info_rate': 100,
        }
        return Splitter(sdf_grid, **kwargs)

    os.system('rm -rf ./*.vtk && rm -rf outputs')

    # Minimize the variance of the shape
    objectives = [PressureDrop(xlb_instantiator=xlb_instantiator, init_shape=sdf_grid, max_iter=500)]

    # subject to a volume constraint to avoid collapsing to a point
    constraints = [ALConstraint(VolumeFraction(init_shape=sdf_grid), target=0.5,
                                constraint_type=ALConstraintType.EqualTo)]

    callbacks = [CSVLogger(Path(__file__).parent / "outputs"),
                 ShapeCheckpoint(Path(__file__).parent / "outputs" / "checkpoints")]

    # Careful with the max_inner_loop_iter here. Setting it to a large value can drive the shape to collapse to a point
    # because the shape variance is minimized to zero.
    topopt = ALTopOpt(sdf_grid, objectives=objectives, constraints=constraints, max_iter=20, max_inner_loop_iter=4,
                      callbacks=callbacks, band_voxels=3, line_search_iter=1)
    # The final shape is in topopt.shape or saved to VTI and PLY files if the ShapeCheckpoint callback was provided
    topopt.run()


if __name__ == "__main__":
    main()
