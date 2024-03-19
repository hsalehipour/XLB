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

    def read_bc_data(self, fluid_bc, bdry_indices):
        import itertools
        boundary_id = fluid_bc['boundaryID']
        face_indices = fluid_bc['faceIndices']
        mesh = trimesh.load(dir_path + '/' + boundary_id)
        sm = mesh.submesh([face_indices], append=True)
        mesh_indices = self.sdf.index_from_coord(sm.vertices)
        mesh_indices = np.unique(mesh_indices, axis=0)

        # Step 1: given the boundary indices of the seed geometry, construct full sweep of mesh face indices.
        # It is important to clip the indices because the vertices of bc faces do not coincide with the voxel center.
        _min, _max = mesh_indices.min(axis=0), mesh_indices.max(axis=0)
        _min_global, _max_global = bdry_indices.min(axis=0), bdry_indices.max(axis=0)
        _min = np.clip(_min, _min_global, _max_global)
        _max = np.clip(_max, _min_global, _max_global)
        xx = np.arange(_min[0], _max[0] + 1)
        yy = np.arange(_min[1], _max[1] + 1)
        zz = np.arange(_min[2], _max[2] + 1)
        bc_indices_fullsweep = np.array(list(itertools.product(xx, yy, zz)))

        # Step 2: find common indices
        bc_indices = np.array(list(set(list(map(tuple, np.array(bdry_indices)))).
                                   intersection(set(list(map(tuple, bc_indices_fullsweep))))))

        return bc_indices

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
        omega = 1.6

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
