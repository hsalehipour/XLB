import math
from time import time
from src.boundary_conditions import *
from jax import config
from src.utils import *
import numpy as np
from src.lattice import LatticeD3Q19, LatticeD3Q27
from src.models import KBCSim
import jax.numpy as jnp
import os
import phantomgaze as pg

# Use 8 CPU devices
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
import jax

# disable JIt compilation
# jax.config.update('jax_disable_jit', True)

precision = 'f32/f32'

dir_path = os.path.dirname(os.path.realpath(__file__)) + '/stl-files/kaizen/'
drone_fname_dic = {
    'prop1': {'fname': dir_path + 'Prop 1 v1.stl', 'axis': [0, 0, -1], 'translate': [0, 0, -2]},
    'prop2': {'fname': dir_path + 'Prop 2 v1.stl', 'axis': [0, 0, 1], 'translate': [0, 0, 2]},
    'prop3': {'fname': dir_path + 'Prop 3 v1.stl', 'axis': [0, 0, -1], 'translate': [0, 0, -2]},
    'prop4': {'fname': dir_path + 'Prop 4 v1.stl', 'axis': [0, 0, 1], 'translate': [0, 0, 2]},
    'prop5': {'fname': dir_path + 'Prop 5 v1.stl', 'axis': [0, 0, -1], 'translate': [0, 0, -2]},
    'prop6': {'fname': dir_path + 'Prop 6 v1.stl', 'axis': [0, 0, 1], 'translate': [0, 0, 2]},
    'prop7': {'fname': dir_path + 'Prop 7 v1.stl', 'axis': [0, 0, -1], 'translate': [0, 0, -2]},
    'prop8': {'fname': dir_path + 'Prop 8 v1.stl', 'axis': [0, 0, 1], 'translate': [0, 0, 2]},
    'main_body': {'fname': dir_path + 'main body.stl', 'translate': [0, 0, 0]}
}


class PropellorBC(Regularized):
    def __init__(self, indices, gridInfo, precision_policy, **kwargs):
        shape = len(indices), len(indices[0])
        vel_prescribed = np.zeros(shape, dtype=precision_policy.compute_dtype)
        super().__init__(indices, gridInfo, precision_policy, 'velocity', vel_prescribed) 
        self.needsExtraConfiguration = False
        self.isDynamic = True
        self.__dict__.update(kwargs)

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
    
    @partial(jit, static_argnums=(0,))
    def apply(self, fout, fin, time):
        """
        Applies the a regularized type boundary condition with dynamically changing indices 
        and without using non-equilibrium bounceback that requires imissing and iknown 
        reconstruction at every time step.
        """
        indices, vel = self.update_function(time)
        
        # set the unknown f populations based on the non-equilibrium bounce-back method
        fbd = fout[indices]

        # compute the equilibrium based on prescribed values and the type of BC
        rho = jnp.sum(fbd, axis=-1, keepdims=True)
        feq = self.equilibrium(rho, vel)

        # Regularize the boundary fpop
        fbd = self.regularize_fpop(fbd, feq)
        return fout.at[indices].set(fbd)
    

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
        shift = [tx // 2, ty // 2, tz // 8]
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

        # side walls
        wall = np.concatenate((self.boundingBoxIndices['front'], self.boundingBoxIndices['back'],
                               self.boundingBoxIndices['left'], self.boundingBoxIndices['right']))
        self.BCs.append(BounceBack(tuple(wall.T), self.gridInfo, self.precisionPolicy))

        # Outlet
        outlet = self.boundingBoxIndices['top']
        self.BCs.append(ExtrapolationOutflow(tuple(outlet.T), self.gridInfo, self.precisionPolicy))

        # inlet
        inlet = self.boundingBoxIndices['bottom']
        rho_inlet = np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        vel_inlet = np.zeros(inlet.shape, dtype=self.precisionPolicy.compute_dtype)
        vel_inlet[:, 2] = u_inlet
        self.BCs.append(Regularized(tuple(inlet.T), self.gridInfo, self.precisionPolicy, 'velocity', vel_inlet))
        return


    def output_data(self, **kwargs):

        # get time
        time = kwargs['timestep']

        # Store viz box
        viz_box = jnp.zeros((self.nx, self.ny, self.nz), dtype=jnp.float32)
        for bc in self.BCs[:-3]:
            idx = bc.update_function(time)[0] if bc.isDynamic else bc.indices
            viz_box = viz_box.at[idx].set(1.0)

        # Get velocity field
        u = kwargs['u']
        rho = kwargs['rho']

        # output the vtk file
        # fields = {"rho": rho[..., 0], "u_x": u[..., 0], "u_y": u[..., 1], "u_z": u[..., 2], 'umag': np.sqrt(np.sum(u**2, axis=-1))}
        # save_fields_vtk(time, fields)

        # Compute q-criterion and vorticity using finite differences
        # vorticity and q-criterion
        norm_mu, q = q_criterion(u)

        # Make phantomgaze volume
        dx = 0.01
        origin = (0.0, 0.0, 0.0)
        upper_bound = (viz_box.shape[0] * dx, viz_box.shape[1] * dx, viz_box.shape[2] * dx)
        q_volume = pg.objects.Volume(
            q,
            spacing=(dx, dx, dx),
            origin=origin,
        )
        norm_mu_volume = pg.objects.Volume(
            norm_mu,
            spacing=(dx, dx, dx),
            origin=origin,
        )
        boundary_volume = pg.objects.Volume(
            viz_box,
            spacing=(dx, dx, dx),
            origin=origin,
        )

        # Make colormap for norm_mu
        colormap = pg.Colormap("jet", vmin=0.0, vmax=0.05)

        # Get camera parameters
        focal_point = (viz_box.shape[0] * dx / 2, viz_box.shape[1] * dx / 2, viz_box.shape[2] * dx / 4)
        radius = 0.1*prop_radius_lbm
        camera_position = (focal_point[0] + radius, focal_point[1] + radius, focal_point[2] - 0.75*radius)

        # Rotate camera
        camera = pg.Camera(position=camera_position, focal_point=focal_point, view_up=(0.0, 0.0, -1.0), max_depth=30.0, width=1920, height=1920, background=pg.SolidBackground(color=(0.0, 0.0, 0.0)))

        # Make wireframe
        # screen_buffer = pg.render.wireframe(lower_bound=origin, upper_bound=upper_bound, thickness=0.01, camera=camera)

        # Render axes
        # screen_buffer = pg.render.axes(size=0.1, center=(0.0, 0.0, 1.1), camera=camera, screen_buffer=screen_buffer)

        # Render q-criterion
        screen_buffer = pg.render.contour(q_volume, threshold=0.00003, color=norm_mu_volume, colormap=colormap, camera=camera)

        # Render boundary
        boundary_colormap = pg.Colormap("bone_r", vmin=0.0, vmax=3.0, opacity=np.linspace(0.0, 100.0, 256))
        screen_buffer = pg.render.volume(boundary_volume, camera=camera, colormap=boundary_colormap, screen_buffer=screen_buffer)

        # Show the rendered image
        plt.imsave('q_criterion_' + str(kwargs['timestep']).zfill(7) + '.png', np.minimum(screen_buffer.image.get(), 1.0))
        return


if __name__ == '__main__':
    lattice = LatticeD3Q27(precision)

    # Problem dependent dimensional quantities
    rpm = 500  # round per minute
    vel_angular_phy = 2.0 * math.pi * rpm / 60  # rad / sec
    vel_transl_phy = 15  # m / sec
    prop_radius_phy = 0.5  # meter
    prop_radius_lbm = 20

    # Computational domain size
    nx = 24 * prop_radius_lbm
    ny = 24 * prop_radius_lbm
    nz = 48 * prop_radius_lbm

    # Problem dependent non-dimensionalization
    u_inlet = 0.04
    u_prop_tip = u_inlet * vel_angular_phy * prop_radius_phy / vel_transl_phy
    angularVelocity = u_prop_tip / prop_radius_lbm

    # Non-dimensional LBM quantities
    Re = 10000.0
    clength = 2 * prop_radius_lbm
    visc = u_prop_tip * clength / Re
    omega = 1.0 / (3. * visc + 0.5)

    os.system("rm -rf ./*.vtk && rm -rf ./*.png")

    kwargs = {
        'lattice': lattice,
        'omega': omega,
        'nx': nx,
        'ny': ny,
        'nz': nz,
        'precision': precision,
        'io_rate': 50,
        'print_info_rate': 1000,
        'restore_checkpoint': False,
    }

    sim = Drone(**kwargs)
    sim.run(50000)

