import os, sys
from pathlib import Path
import jax

sys.path.append(os.path.abspath('../doppler'))
sys.path.append(os.path.abspath('../XLB'))

from doppler.callbacks.csv_logger import CSVLogger
from doppler.callbacks.shape_checkpoint import ShapeCheckpoint
from doppler.topopt import ALTopOpt, ALConstraint, ALConstraintType
from doppler.objectives import PressureDrop, VolumeFraction
from doppler.geometry.sdf import SDFGrid

from src.utils import *
from src.boundary_conditions import *
from src.adjoint import LBMBaseDifferentiable
from src.lattice import LatticeD3Q19, LatticeD3Q27
from jax import config

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

config.update('jax_enable_x64', True)

project_name = 'valve'
dir_path = os.path.dirname(os.path.realpath(__file__)) + '/' + project_name

# Currently we have 2 methods of introducing level-set field into the TO pipeleine:
#   v1: through collision operator using tanh function
#   v2: through a differentiable iinterpolation bc 
TO_method = 'v1'

def read_json(dir_path):
    # Read the JSON file
    import json
    with open(dir_path + '/' + 'project.json', 'r') as file:
        data = json.load(file)
    return data

def pad_and_adjust_sdf_shape(sdf_grid):
    # Note: we should ensure the pad_width is enough in case keepIn and keepOuts go beyond.
    nx, ny, nz = sdf_grid.resolution
    sf = 5  # scaling factor
    min_pad = 5
    pads = [nx // sf, ny // sf, nz // sf]

    # Ensure each pad value is at least min_pad
    pads = [max(pad, min_pad) for pad in pads]

    # Assign padding values to respective variables
    pad_x1, pad_x2, pad_y1, pad_y2, pad_z1, pad_z2 = pads * 2

    nDevices = jax.device_count()
    deficit = nDevices - (nx + pad_x1 + pad_x2) % nDevices
    pad_x1 += deficit // 2
    pad_x2 += deficit - deficit // 2

    sdf_grid.pad((pad_x1, pad_x2), (pad_y1, pad_y2), (pad_z1, pad_z2))
    return sdf_grid

class Project(LBMBaseDifferentiable):
    def __init__(self, sdf: SDFGrid, **kwargs):
        super().__init__(sdf, **kwargs)

    def boundary(self, sdf: SDFGrid, side='inside'):
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
        other_side = 'outside' if side == 'inside' else 'inside'

        def get_bmask(side):
            if side == 'inside':
                bmask = (sdf_array > -sdf.spacing) & (sdf_array <= 0.)
            elif side == 'outside':
                bmask = (sdf_array < sdf.spacing) & (sdf_array > 0.)
            else:
                raise ValueError(f"location = {side} not supported! Must be either \"inside\" or \"outside\"")
            return bmask

        # First let's isolate the obvious fluid boundary cells with negative SDF
        bmask1 = get_bmask(side)
        bdry_cord = voxel_coordinates[bmask1]
        bdry_indices_firstSide = sdf.index_from_coord(bdry_cord)

        # Next, find the fluid boundary voxels not included in the first set by traversing solid boundary voxels
        bmask2 = get_bmask(other_side)
        bdry_cord = voxel_coordinates[bmask2]
        c = np.array(self.lattice.c)
        bdry_indices_lst = []
        for q in range(1, self.lattice.q):
            bdry_cord_nbr = bdry_cord + c[:, q]*sdf.spacing
            idx = sdf.index_from_coord(bdry_cord_nbr)
            if side == 'inside':
                mask = sdf_array[tuple(idx.T)] <= 0
            else:
                mask = sdf_array[tuple(idx.T)] > 0
            bdry_indices_lst.append(idx[mask])
        bdry_indices_otherSide = np.unique(np.vstack(bdry_indices_lst), axis=0)
        return np.unique(np.vstack([bdry_indices_firstSide, bdry_indices_otherSide]), axis=0)

    def set_boundary_conditions(self):
        # Set boundary conditions

        # read json file
        data_json = read_json(dir_path)

        # Extract boundaryID and faceIndices
        bdry_indices = self.boundary(self.sdf)
        port_indices = []
        fluid_cases = data_json['fluidProj']['fluidCases']
        for fluid_case in fluid_cases:
            fluid_bcs = fluid_case['fluidBCs']
            for fluid_bc in fluid_bcs:
                bc_indices = self.read_bc_data(fluid_bc, bdry_indices)
                bc_type = self.read_bc_type(fluid_bc)
                port_indices.append(bc_indices)
                if bc_type == "velocity":
                    vel_vec = np.zeros(bc_indices.shape, dtype=self.precisionPolicy.compute_dtype)
                    vel_vec += self.read_vel_bc_value(fluid_bc)
                    self.BCs.append(Regularized(tuple(bc_indices.T), self.gridInfo, self.precisionPolicy,
                                          'velocity', vel_vec))
                elif bc_type == "pressure":
                    rho_outlet = np.ones((bc_indices.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
                    self.BCs.append(Regularized(tuple(bc_indices.T), self.gridInfo, self.precisionPolicy,
                                          'pressure', rho_outlet))
                    # self.BCs.append(ExtrapolationOutflow(tuple(bc_indices.T), self.gridInfo, self.precisionPolicy))

        # Find no-slip boundaries as the remaining boundary voxels
        port_indices = np.vstack(port_indices)
        noslip = np.array(list(set(list(map(tuple, bdry_indices))) - set(list(map(tuple, port_indices)))))

        # No-slip BC for all no-slip boundaries that are part of optimization
        if self.TO_method == 'v1':
            self.BCs.append(BounceBackHalfway(tuple(noslip.T), self.gridInfo, self.precisionPolicy))
        else:
            self.BCs.append(InterpolatedBounceBackDifferentiable(tuple(noslip.T),
                                                                self.gridInfo, self.precisionPolicy))
        self.BCs[-1].needsExtraConfiguration = False
        self.BCs[-1].isSolid = False
        return

    def read_bc_data(self, fluid_bc, bdry_seed_indices):
        """
        This function extracts fluid boundary voxels on a particular axis-aligned surface given by a mesh and a series
        of mesh face indices provided in the JSON file.
        **Note:
            Important caveat: the current implementation only works for axis-aligned BC faces.
        """
        # Read the mesh and create submesh
        boundary_id = fluid_bc['boundaryID']
        face_indices = fluid_bc['faceIndices']
        mesh = trimesh.load(dir_path + '/' + boundary_id)
        sm = mesh.submesh([face_indices], append=True)

        # Step 1: Find the cartesian index associated with the face normal axis
        _, axis_index = np.nonzero(sm.face_normals)
        # ensure that accidentally some ids from other dimensions are not included.
        axis_index = np.argmax(np.bincount(axis_index))

        # Step 2: Exclude the boundary indices that are beyond 1 voxel spacing away from the mesh indices
        bdry_seed_coord = self.sdf.coord_from_index(bdry_seed_indices)
        mesh_face_coord = sm.vertices[0, axis_index]
        idx = np.nonzero(abs(bdry_seed_coord[:, axis_index] - mesh_face_coord) <= self.sdf.spacing)[0]
        bindex = bdry_seed_indices[idx, :]

        # Step 3: Ensure boundary voxels are inside the port sdf not outside of it.
        port_sdf = SDFGrid.load_from_mesh(mesh, self.sdf.resolution,
                                          spacing=self.sdf.spacing,
                                          origin=self.sdf.origin,
                                          dtype=jnp.float32)
        mask = port_sdf.array[..., 0][tuple(bindex.T)] <= 0
        return bindex[mask]

    def read_bc_type(self, fluid_bc):
        return fluid_bc['fluidDef']['bcType']

    def read_vel_bc_value(self, fluid_bc):
        return np.array(list(fluid_bc['fluidDef']['velocity'].values()))

    # def output_data(self, **kwargs):
    #     # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
    #     rho = np.array(kwargs["rho"])
    #     u = np.array(kwargs["u"])
    #     timestep = kwargs["timestep"]
    #     fields = {"rho": rho[..., 0], "u_x": u[..., 0], "u_y": u[..., 1], "u_z": u[..., 2],
    #               "umag": np.sqrt(u[..., 0]**2+u[..., 1]**2+u[..., 2]**2)}
    #     save_fields_vtk(timestep, fields)
    #     save_BCs_vtk(timestep, self.BCs, self.gridInfo)


# Helper function to specify a parabolic poiseuille profile
poiseuille_profile  = lambda x,x0,d,umax: np.maximum(0.,4.*umax/(d**2)*((x-x0)*d-(x-x0)**2))

def main():

    # read json file
    data_json = read_json(dir_path)

    # Create seed geometry and its sdf object
    filename = dir_path + '/' + data_json['seedGeom']
    orig_mesh = trimesh.load(filename)
    extents = orig_mesh.extents
    nx, ny, nz = tuple([int(0.25*n) for n in extents])
    sdf_grid = SDFGrid.load_from_mesh(orig_mesh, (nx, ny, nz), dtype=jnp.float32)
    sdf_grid = pad_and_adjust_sdf_shape(sdf_grid)
    shape = sdf_grid.resolution

    # Create SDFGrid objects for keep_ins and keep_outs
    filename_keep_ins = data_json['keepIns']
    filename_keep_outs = data_json['keepOuts']
    keepins, keepouts = [], []
    for filename in filename_keep_ins:
        mesh = trimesh.load(dir_path + '/' + filename)
        keepins.append(SDFGrid.load_from_mesh(mesh,
                                              shape,
                                              spacing=sdf_grid.spacing,
                                              origin=sdf_grid.origin,
                                              dtype=jnp.float32))
    for filename in filename_keep_outs:
        mesh = trimesh.load(dir_path + '/' + filename)
        keepouts.append(SDFGrid.load_from_mesh(mesh,
                                               shape,
                                               spacing=sdf_grid.spacing,
                                               origin=sdf_grid.origin,
                                               dtype=jnp.float32))

    # Combine the keepins with the shape
    for keepin in keepins:
        sdf_grid = sdf_grid.boolean_union(keepin)
    for keepout in keepouts:
        sdf_grid = sdf_grid.boolean_difference(keepout)

    def xlb_instantiator(sdf_grid):
        precision = 'f64/f32'
        lattice = LatticeD3Q27(precision)
        omega = 1.95
        nx, ny, nz = sdf_grid.resolution
        kwargs = {
            'lattice': lattice,
            'omega': omega,
            'nx': nx,
            'ny': ny,
            'nz': nz,
            'precision': precision,
            'io_rate': 500,
            'print_info_rate': 1000,
            'keepins': keepins,
            'TO_method': TO_method,
            'collision_model': 'kbc',
        }
        return Project(sdf_grid, **kwargs)

    # clean up previous files
    os.system('rm -rf ' + dir_path + '/*.vtk && rm -rf ' + dir_path + '/outputs')

    # Set the objective function
    objectives = [PressureDrop(xlb_instantiator=xlb_instantiator, init_shape=sdf_grid, max_iter=2000)]

    # subject to a volume constraint
    constraints = [ALConstraint(VolumeFraction(init_shape=sdf_grid), target=0.45,
                                constraint_type=ALConstraintType.EqualTo)]

    callbacks = [CSVLogger(Path(dir_path) / "outputs"),
                 ShapeCheckpoint(Path(dir_path) / "outputs" / "checkpoints")]

    # Create the TO object to perform level-set based TO
    topopt = ALTopOpt(sdf_grid, keepins, keepouts, objectives=objectives, constraints=constraints, max_iter=40,
                      max_inner_loop_iter=8, callbacks=callbacks, band_voxels=3, line_search_iter=3,
                      line_search_method='golden-backtracking')
    # The final shape is in topopt.shape or saved to VTI and PLY files if the ShapeCheckpoint callback was provided
    # topopt.shape.isosurface().export(Path(dir_path) / "outputs" / "checkpoints" / "seed_boolean.ply")
    topopt.run()


if __name__ == "__main__":
    main()
