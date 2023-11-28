import math
from time import time
from src.boundary_conditions import *
from jax.config import config
from src.utils import *
import numpy as np
from src.lattice import LatticeD3Q19, LatticeD3Q27
from src.models import KBCSim
import jax.numpy as jnp
import os

# Use 8 CPU devices
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
import jax

# disable JIt compilation
# jax.config.update('jax_disable_jit', True)

precision = 'f32/f32'

drone_fname_dic = {
    'prop1': {'fname': 'stl-files/kaizen/Prop 1 v1.stl', 'axis': [0, 0, -1], 'translate': [0, 0, -2]},
    'prop2': {'fname': 'stl-files/kaizen/Prop 2 v1.stl', 'axis': [0, 0, 1], 'translate': [0, 0, 2]},
    'prop3': {'fname': 'stl-files/kaizen/Prop 3 v1.stl', 'axis': [0, 0, -1], 'translate': [0, 0, -2]},
    'prop4': {'fname': 'stl-files/kaizen/Prop 4 v1.stl', 'axis': [0, 0, 1], 'translate': [0, 0, 2]},
    'prop5': {'fname': 'stl-files/kaizen/Prop 5 v1.stl', 'axis': [0, 0, -1], 'translate': [0, 0, -2]},
    'prop6': {'fname': 'stl-files/kaizen/Prop 6 v1.stl', 'axis': [0, 0, 1], 'translate': [0, 0, 2]},
    'prop7': {'fname': 'stl-files/kaizen/Prop 7 v1.stl', 'axis': [0, 0, -1], 'translate': [0, 0, -2]},
    'prop8': {'fname': 'stl-files/kaizen/Prop 8 v1.stl', 'axis': [0, 0, 1], 'translate': [0, 0, 2]},
    'main_body': {'fname': 'stl-files/kaizen/main body.stl', 'translate': [0, 0, 0]}
}


class PropellorBC(BounceBackMoving):
    def __init__(self, indices, gridInfo, precision_policy, **kwargs):
        super().__init__(indices, gridInfo, precision_policy, **kwargs)

    def update_function(self, time):
        """
        A user-defined function to be invoked at every iteration after the LBM-step
        In this class, this step_user function rotates the indices specified by rotation_kwargs
        """
        # axis = [0, 1, 0]
        angle = -self.angularVelocity * time  # \Delta theta = omega * \Delta t = omega
        prop_indices_rotated = rotate_geometry(self.indices, self.rotationOrigin, self.rotationAxis, angle)
        vel = jnp.cross(jnp.array(self.rotationAxis) * angularVelocity,
                        jnp.array(self.indices).T - self.rotationOrigin)
        return prop_indices_rotated, vel

# main code
class Drone(KBCSim):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def shift_assembly_indices(self, mesh_voxelized, reference_origin):
        mesh_matrix = mesh_voxelized.matrix
        origin = mesh_voxelized.points_to_indices(np.array([0, 0, 0]))
        origin_shift = reference_origin - origin
        indices = np.argwhere(mesh_matrix) + origin_shift
        return indices

    def initialize_macroscopic_fields(self):
        rho = self.precisionPolicy.cast_to_output(1.0)
        vel = np.zeros((self.nx, self.ny, self.nz, self.dim))
        vel[..., 2] = u_inlet
        u = self.distributed_array_init((self.nx, self.ny, self.nz, self.dim),
                                        self.precisionPolicy.output_dtype, init_val=vel)
        u = self.precisionPolicy.cast_to_output(u)
        return rho, u

    def set_boundary_conditions(self):
        print('Voxelizing mesh...')
        stl_filename = drone_fname_dic['main_body']['fname']
        body_voxelized, pitch = voxelize_stl(stl_filename, 5 * prop_radius_lbm)
        tx, ty, tz = np.array([nx, ny, nz]) - body_voxelized.matrix.shape
        shift = [tx // 2, ty // 2, tz // 4]
        body_indices = np.argwhere(body_voxelized.matrix) + shift + drone_fname_dic['main_body']['translate']
        reference_origin = body_voxelized.points_to_indices(np.array([0, 0, 0])) + shift
        self.BCs.append(BounceBack(tuple(body_indices.T), self.gridInfo, self.precisionPolicy))

        for key, value in drone_fname_dic.items():
            time_start = time()
            stl_filename = value.get('fname')
            prop_voxelized, _ = voxelize_stl(stl_filename, prop_radius_lbm, pitch=pitch)
            prop_matrix = prop_voxelized.matrix
            print(key + ' Voxelization time for pitch={}: {} seconds'.format(pitch, time() - time_start))
            print(key + ' matrix shape: ', prop_matrix.shape)
            prop_indices = self.shift_assembly_indices(prop_voxelized, reference_origin) + value['translate']
            rotationOrigin = np.rint(prop_indices.mean(axis=0)).astype('int')
            rotationAxis = value.get('axis')
            if rotationAxis is not None:
                self.BCs.append(PropellorBC(tuple(prop_indices.T), self.gridInfo, self.precisionPolicy,
                                            rotationAxis=rotationAxis,
                                            rotationOrigin=rotationOrigin,
                                            angularVelocity=angularVelocity))

        wall = np.concatenate((self.boundingBoxIndices['front'], self.boundingBoxIndices['back'],
                               self.boundingBoxIndices['left'], self.boundingBoxIndices['right']))
        self.BCs.append(BounceBack(tuple(wall.T), self.gridInfo, self.precisionPolicy))

        outlet = self.boundingBoxIndices['top']
        self.BCs.append(DoNothing(tuple(outlet.T), self.gridInfo, self.precisionPolicy))
        self.BCs[-1].implementationStep = 'PostCollision'
        # rho_outlet = np.ones(outlet.shape[0], dtype=self.precision_policy.compute_dtype)
        # self.BCs.append(Regularized(tuple(outlet.T),
        #                                          self.grid_info,
        #                                          self.precision_policy,
        #                                          'pressure', rho_outlet))

        inlet = self.boundingBoxIndices['bottom']
        rho_inlet = np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        vel_inlet = np.zeros(inlet.shape, dtype=self.precisionPolicy.compute_dtype)

        vel_inlet[:, 2] = u_inlet
        self.BCs.append(EquilibriumBC(tuple(inlet.T), self.gridInfo, self.precisionPolicy, rho_inlet, vel_inlet))
        # self.BCs.append(ZouHe(tuple(inlet.T),
        #                                          self.grid_info,
        #                                          self.precision_policy,
        #                                          'velocity', vel_inlet))
        return


    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        iter = kwargs['timestep']
        rho = np.array(kwargs['rho'][1:-1, 1:-1, :, 0])
        u = np.array(kwargs['u'][1:-1, 1:-1, ...])
        u_prev = kwargs['u_prev'][1:-1, 1:-1, ...]
        u_old = np.linalg.norm(u_prev, axis=2)
        u_new = np.linalg.norm(u, axis=2)

        err = np.sum(np.abs(u_old - u_new))
        print('error= {:07.6f}'.format(err))
        fields = {"rho": rho, "u_x": u[..., 0], "u_y": u[..., 1], "u_z": u[..., 2]}
        save_fields_vtk(iter, fields)


if __name__ == '__main__':
    lattice = LatticeD3Q27(precision)

    # Problem dependent dimensional quantities
    rpm = 3000  # round per minute
    vel_angular_phy = 2.0 * math.pi * rpm / 60  # rad / sec
    vel_transl_phy = 15  # m / sec
    prop_radius_phy = 0.5  # meter
    prop_radius_lbm = 20

    # Computational domain size
    nx = 24 * prop_radius_lbm
    ny = 24 * prop_radius_lbm
    nz = 24 * prop_radius_lbm

    # Non-dimensional LBM quantities
    Re = 100000.0
    u_inlet = 0.003
    clength = 2 * prop_radius_lbm
    visc = u_inlet * clength / Re
    omega = 1.0 / (3. * visc + 0.5)

    # Problem dependent non-dimensionalization
    u_prop_tip = u_inlet * vel_angular_phy * prop_radius_phy / vel_transl_phy
    angularVelocity = u_prop_tip / prop_radius_lbm

    os.system("rm -rf ./*.vtk && rm -rf ./*.png")

    kwargs = {
        'lattice': lattice,
        'omega': omega,
        'nx': nx,
        'ny': ny,
        'nz': nz,
        'precision': precision,
        'io_rate': 100,
        'print_info_rate': 100,
        'restore_checkpoint': False,
    }

    sim = Drone(**kwargs)
    sim.run(5000)

