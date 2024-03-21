import os, sys
from pathlib import Path


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
from src.lattice import LatticeD3Q19


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

project_name = 'splitter'
dir_path = os.path.dirname(os.path.realpath(__file__)) + '/' + project_name

def read_json(dir_path):
    # Read the JSON file
    import json
    with open(dir_path + '/' + 'project.json', 'r') as file:
        data = json.load(file)
    return data

class Splitter(LBMBaseDifferentiable):
    def __init__(self, sdf, **kwargs):
        self.sdf = sdf
        super().__init__(**kwargs)

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
                    self.BCs.append(ZouHe(tuple(bc_indices.T), self.gridInfo, self.precisionPolicy,
                                          'velocity', vel_vec))
                elif bc_type == "pressure":
                    rho_outlet = np.ones((bc_indices.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
                    self.BCs.append(ZouHe(tuple(bc_indices.T), self.gridInfo, self.precisionPolicy,
                                          'pressure', rho_outlet))

        # Find no-slip boundaries as the remaining boundary voxels
        port_indices = np.vstack(port_indices)
        noslip = np.array(list(set(list(map(tuple, bdry_indices))) - set(list(map(tuple, port_indices)))))

        # No-slip BC for all no-slip boundaries that are part of optimization
        self.BCs.append(InterpolatedBounceBackDifferentiable(tuple(noslip.T),
                                                             self.gridInfo, self.precisionPolicy))
        self.BCs[-1].needsExtraConfiguration = False
        self.BCs[-1].isSolid = False
        return

    def read_bc_data(self, fluid_bc, bdry_indices_seed):
        """
        This function extracts fluid boundary voxels on a particular axis-aligned surface given by a mesh and a series
        of mesh face indices provided in the JSON file.
        **Note:
            Important caveat: the current implementation only works for axis-aligned BC faces.
        """
        boundary_id = fluid_bc['boundaryID']
        face_indices = fluid_bc['faceIndices']
        mesh = trimesh.load(dir_path + '/' + boundary_id)
        sm = mesh.submesh([face_indices], append=True).subdivide()
        mesh_indices = self.sdf.index_from_coord(sm.vertices)
        idx, count = np.unique(mesh_indices, return_counts=True)
        voxelized_face_axis_id = idx[count.argmax()]
        _, axis_index = np.nonzero(mesh_indices == voxelized_face_axis_id)
        # ensure that accidentally some ids from other dimensions are not included.
        axis_index = np.argmax(np.bincount(axis_index))

        # Step 1: custruct SDF pf the boundary port (subdivide to make sure SDF has good quality)
        port_sdf = SDFGrid.load_from_mesh(mesh.subdivide(), self.sdf.resolution,
                                          spacing=self.sdf.spacing,
                                          origin=self.sdf.origin,
                                          dtype=jnp.float32)
        # Step 2: find common indices
        bdry_indices_port = self.boundary(port_sdf, side="inside")
        bc_indices = np.array(list(set(list(map(tuple, np.array(bdry_indices_seed)))).
                                   intersection(set(list(map(tuple, bdry_indices_port))))))

        # Step 3: Exclude the common indices that are beyond 1 voxel away from the mesh indices
        idx = np.nonzero(abs(bc_indices[:, axis_index] - voxelized_face_axis_id) <= 1)[0]
        return bc_indices[idx, :]

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
    nx, ny, nz = tuple([int(n) for n in extents])
    shape = (nx, ny, nz)
    pad_width = 8
    sdf_grid = SDFGrid.load_from_mesh(orig_mesh, shape, dtype=jnp.float32, pad_width=pad_width)
    nx += 2*pad_width
    ny += 2*pad_width
    nz += 2*pad_width
    shape = (nx, ny, nz)

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

    def xlb_instantiator(sdf_grid):
        precision = 'f32/f32'
        lattice = LatticeD3Q19(precision)
        omega = 1.78

        kwargs = {
            'lattice': lattice,
            'omega': omega,
            'nx': nx,
            'ny': ny,
            'nz': nz,
            'precision': precision,
            'io_rate': 500,
            'print_info_rate': 500,
        }
        return Splitter(sdf_grid, **kwargs)

    # clean up previous files
    os.system('rm -rf ' + dir_path + '/*.vtk && rm -rf ' + dir_path + '/outputs')

    # Minimize the variance of the shape
    objectives = [PressureDrop(xlb_instantiator=xlb_instantiator, init_shape=sdf_grid, max_iter=1000)]

    # subject to a volume constraint to avoid collapsing to a point
    constraints = [ALConstraint(VolumeFraction(init_shape=sdf_grid), target=0.15,
                                constraint_type=ALConstraintType.EqualTo)]

    callbacks = [CSVLogger(Path(dir_path) / "outputs"),
                 ShapeCheckpoint(Path(dir_path) / "outputs" / "checkpoints")]

    # Careful with the max_inner_loop_iter here. Setting it to a large value can drive the shape to collapse to a point
    # because the shape variance is minimized to zero.
    topopt = ALTopOpt(sdf_grid, keepins, keepouts, objectives=objectives, constraints=constraints, max_iter=40,
                      max_inner_loop_iter=8, callbacks=callbacks, band_voxels=1, line_search_iter=3,
                      line_search_method='golden')
    # The final shape is in topopt.shape or saved to VTI and PLY files if the ShapeCheckpoint callback was provided
    topopt.run()


if __name__ == "__main__":
    main()