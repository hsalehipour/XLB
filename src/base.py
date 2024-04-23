# Standard Libraries
import os
import time

# Third-Party Libraries
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from termcolor import colored

# JAX-related imports
from jax import jit, lax, vmap
from jax.experimental import mesh_utils
from jax.experimental.multihost_utils import process_allgather
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec, PositionalSharding, Mesh
import orbax.checkpoint as orb

# functools imports
from functools import partial

# Local/Custom Libraries
from src.utils import downsample_field
from src.boundary_conditions import BounceBackHalfway

jax.config.update("jax_spmd_mode", 'allow_all')
# Disables annoying TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class LBMBase(object):
    """
    LBMBase: A class that represents a base for Lattice Boltzmann Method simulation.
    
    Parameters
    ----------
        lattice (object): The lattice object that contains the lattice structure and weights.
        omega (float): The relaxation parameter for the LBM simulation.
        nx (int): Number of grid points in the x-direction.
        ny (int): Number of grid points in the y-direction.
        nz (int, optional): Number of grid points in the z-direction. Defaults to 0.
        precision (str, optional): A string specifying the precision used for the simulation. Defaults to "f32/f32".
    """
    
    def __init__(self, **kwargs):
        self.omega = kwargs.get("omega")
        self.nx = kwargs.get("nx")
        self.ny = kwargs.get("ny")
        self.nz = kwargs.get("nz")

        self.precision = kwargs.get("precision")
        computedType, storedType = self.set_precisions(self.precision)
        self.precisionPolicy = jmp.Policy(compute_dtype=computedType,
                                            param_dtype=computedType, output_dtype=storedType)
        
        self.lattice = kwargs.get("lattice")
        self.checkpointRate = kwargs.get("checkpoint_rate", 0)
        self.checkpointDir = kwargs.get("checkpoint_dir", './checkpoints')
        self.downsamplingFactor = kwargs.get("downsampling_factor", 1)
        self.printInfoRate = kwargs.get("print_info_rate", 100)
        self.ioRate = kwargs.get("io_rate", 0)
        self.returnFpost = kwargs.get("return_fpost", False)
        self.computeMLUPS = kwargs.get("compute_MLUPS", False)
        self.restore_checkpoint = kwargs.get("restore_checkpoint", False)
        self.nDevices = jax.device_count()
        self.backend = jax.default_backend()

        if self.computeMLUPS:
            self.restore_checkpoint = False
            self.ioRate = 0
            self.checkpointRate = 0
            self.printInfoRate = 0

        # Check for distributed mode
        if self.nDevices > jax.local_device_count():
            print("WARNING: Running in distributed mode. Make sure that jax.distributed.initialize is called before performing any JAX computations.")
                    
        self.c = self.lattice.c
        self.q = self.lattice.q
        self.w = self.lattice.w
        self.dim = self.lattice.d

        # Set the checkpoint manager
        if self.checkpointRate > 0:
            mngr_options = orb.CheckpointManagerOptions(save_interval_steps=self.checkpointRate, max_to_keep=1)
            self.mngr = orb.CheckpointManager(self.checkpointDir, orb.PyTreeCheckpointer(), options=mngr_options)
        else:
            self.mngr = None
        
        # Adjust the number of grid points in the x direction, if necessary.
        # If the number of grid points is not divisible by the number of devices
        # it increases the number of grid points to the next multiple of the number of devices.
        # This is done in order to accommodate the domain sharding per XLA device
        nx, ny, nz = kwargs.get("nx"), kwargs.get("ny"), kwargs.get("nz")
        if None in {nx, ny, nz}:
            raise ValueError("nx, ny, and nz must be provided. For 2D examples, nz must be set to 0.")
        self.nx = nx
        if nx % self.nDevices:
            self.nx = nx + (self.nDevices - nx % self.nDevices)
            print("WARNING: nx increased from {} to {} in order to accommodate domain sharding per XLA device.".format(nx, self.nx))
        self.ny = ny
        self.nz = nz

        self.show_simulation_parameters()
    
        # Store grid information
        self.gridInfo = {
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "dim": self.lattice.d,
            "lattice": self.lattice
        }

        # Define the right permutation
        self.rightPerm = [(i, (i + 1) % self.nDevices) for i in range(self.nDevices)]
        # Define the left permutation
        self.leftPerm = [((i + 1) % self.nDevices, i) for i in range(self.nDevices)]

        # Set up the sharding for 2D and 3D simulations
        if self.dim == 2:
            self.devices = mesh_utils.create_device_mesh((self.nDevices, 1, 1))
            self.mesh = Mesh(self.devices, axis_names=("x", "y", "value"))
            self.sharding = NamedSharding(self.mesh, PartitionSpec("x", "y", "value"))
            inout_specs = PartitionSpec("x", None, None)
        elif self.dim == 3:
            self.devices = mesh_utils.create_device_mesh((self.nDevices, 1, 1, 1))
            self.mesh = Mesh(self.devices, axis_names=("x", "y", "z", "value"))
            self.sharding = NamedSharding(self.mesh, PartitionSpec("x", "y", "z", "value"))
            inout_specs = PartitionSpec("x", None, None, None)
        else:
            raise ValueError(f"dim = {self.dim} not supported")

        # Set up streaming
        self.streaming = jit(shard_map(self.streaming_m, mesh=self.mesh,
                                       in_specs=inout_specs, out_specs=inout_specs, check_rep=False))

        # Compute the bounding box indices for boundary conditions
        self.boundingBoxIndices= self.bounding_box_indices()
        # Create boundary data for the simulation
        self._create_boundary_data()
        self.force = self.get_force()

        # Set Smagorinsky constant
        self.smagorinskyConstant = 0.1**2

    @property
    def lattice(self):
        return self._lattice

    @lattice.setter
    def lattice(self, value):
        if value is None:
            raise ValueError("Lattice type must be provided.")
        if self.nz == 0 and value.name not in ['D2Q9']:
            raise ValueError("For 2D simulations, lattice type must be LatticeD2Q9.")
        if self.nz != 0 and value.name not in ['D3Q19', 'D3Q27']:
            raise ValueError("For 3D simulations, lattice type must be LatticeD3Q19, or LatticeD3Q27.")
                            
        self._lattice = value

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, value):
        if value is None:
            raise ValueError("omega must be provided")
        if not isinstance(value, float):
            raise TypeError("omega must be a float")
        self._omega = value

    @property
    def nx(self):
        return self._nx

    @nx.setter
    def nx(self, value):
        if value is None:
            raise ValueError("nx must be provided")
        if not isinstance(value, int):
            raise TypeError("nx must be an integer")
        self._nx = value

    @property
    def ny(self):
        return self._ny

    @ny.setter
    def ny(self, value):
        if value is None:
            raise ValueError("ny must be provided")
        if not isinstance(value, int):
            raise TypeError("ny must be an integer")
        self._ny = value

    @property
    def nz(self):
        return self._nz

    @nz.setter
    def nz(self, value):
        if value is None:
            raise ValueError("nz must be provided")
        if not isinstance(value, int):
            raise TypeError("nz must be an integer")
        self._nz = value

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, value):
        if not isinstance(value, str):
            raise TypeError("precision must be a string")
        self._precision = value

    @property
    def checkpointRate(self):
        return self._checkpointRate

    @checkpointRate.setter
    def checkpointRate(self, value):
        if not isinstance(value, int):
            raise TypeError("checkpointRate must be an integer")
        self._checkpointRate = value

    @property
    def checkpointDir(self):
        return self._checkpointDir

    @checkpointDir.setter
    def checkpointDir(self, value):
        if not isinstance(value, str):
            raise TypeError("checkpointDir must be a string")
        self._checkpointDir = value

    @property
    def downsamplingFactor(self):
        return self._downsamplingFactor

    @downsamplingFactor.setter
    def downsamplingFactor(self, value):
        if not isinstance(value, int):
            raise TypeError("downsamplingFactor must be an integer")
        self._downsamplingFactor = value

    @property
    def printInfoRate(self):
        return self._printInfoRate

    @printInfoRate.setter
    def printInfoRate(self, value):
        if not isinstance(value, int):
            raise TypeError("printInfoRate must be an integer")
        self._printInfoRate = value

    @property
    def ioRate(self):
        return self._ioRate

    @ioRate.setter
    def ioRate(self, value):
        if not isinstance(value, int):
            raise TypeError("ioRate must be an integer")
        self._ioRate = value

    @property
    def returnFpost(self):
        return self._returnFpost

    @returnFpost.setter
    def returnFpost(self, value):
        if not isinstance(value, bool):
            raise TypeError("returnFpost must be a boolean")
        self._returnFpost = value

    @property
    def computeMLUPS(self):
        return self._computeMLUPS

    @computeMLUPS.setter
    def computeMLUPS(self, value):
        if not isinstance(value, bool):
            raise TypeError("computeMLUPS must be a boolean")
        self._computeMLUPS = value

    @property
    def restore_checkpoint(self):
        return self._restore_checkpoint

    @restore_checkpoint.setter
    def restore_checkpoint(self, value):
        if not isinstance(value, bool):
            raise TypeError("restore_checkpoint must be a boolean")
        self._restore_checkpoint = value

    @property
    def nDevices(self):
        return self._nDevices

    @nDevices.setter
    def nDevices(self, value):
        if not isinstance(value, int):
            raise TypeError("nDevices must be an integer")
        self._nDevices = value

    def show_simulation_parameters(self):
        attributes_to_show = [
            'omega', 'nx', 'ny', 'nz', 'dim', 'precision', 'lattice', 
            'checkpointRate', 'checkpointDir', 'downsamplingFactor', 
            'printInfoRate', 'ioRate', 'computeMLUPS', 
            'restore_checkpoint', 'backend', 'nDevices'
        ]

        descriptive_names = {
            'omega': 'Omega',
            'nx': 'Grid Points in X',
            'ny': 'Grid Points in Y',
            'nz': 'Grid Points in Z',
            'dim': 'Dimensionality',
            'precision': 'Precision Policy',
            'lattice': 'Lattice Type',
            'checkpointRate': 'Checkpoint Rate',
            'checkpointDir': 'Checkpoint Directory',
            'downsamplingFactor': 'Downsampling Factor',
            'printInfoRate': 'Print Info Rate',
            'ioRate': 'I/O Rate',
            'computeMLUPS': 'Compute MLUPS',
            'restore_checkpoint': 'Restore Checkpoint',
            'backend': 'Backend',
            'nDevices': 'Number of Devices'
        }
        simulation_name = self.__class__.__name__
        
        print(colored(f'\n**** Simulation Parameters for {simulation_name} ****', 'green'))
                
        header = f"{colored('Parameter', 'blue'):>30} | {colored('Value', 'yellow')}"
        print(header)
        print('-' * 50)
        
        for attr in attributes_to_show:
            value = getattr(self, attr, 'Attribute not set')
            descriptive_name = descriptive_names.get(attr, attr)  # Use the attribute name as a fallback
            row = f"{colored(descriptive_name, 'blue'):>30} | {colored(value, 'yellow')}"
            print(row)

    def get_solid_voxels(self):
        # Accumulate the indices of all BCs to create the grid mask with FALSE along directions that
        # stream into a boundary voxel.
        solid_list = [np.array(bc.indices).T for bc in self.BCs if bc.isSolid]
        solid_voxels = np.unique(np.vstack(solid_list), axis=0) if solid_list else None
        return solid_voxels

    def _create_boundary_data(self):
        """
        Create boundary data for the Lattice Boltzmann simulation by setting boundary conditions,
        creating grid mask, and preparing local masks and normal arrays.
        """
        self.BCs = []
        self.set_boundary_conditions()
        solid_voxels = self.get_solid_voxels()

        # Create the grid mask on each process
        start = time.time()
        grid_mask = self.create_grid_mask(solid_voxels)
        print("Time to create the grid mask:", time.time() - start)

        start = time.time()
        removed_voxels_list = []
        for bc in self.BCs:
            assert bc.implementationStep in ['PostStreaming', 'PostCollision']
            bc.create_local_mask_and_normal_arrays(grid_mask)
            if bc.removed_voxels is not None:
                removed_voxels_list.append(bc.removed_voxels)

        # No-slip BC for all removed voxels
        # TODO: is there a better/cleaner way to do this!
        if removed_voxels_list:
            noslip = np.hstack(removed_voxels_list)
            bc = BounceBackHalfway(tuple(noslip), self.gridInfo, self.precisionPolicy)
            bc.needsExtraConfiguration = False
            bc.isSolid = False
            bc.create_local_mask_and_normal_arrays(grid_mask)
            self.BCs.append(bc)

        print("Time to create the local masks and normal arrays:", time.time() - start)
        return

    # This is another non-JITed way of creating the distributed arrays. It is not used at the moment.
    # def distributed_array_init(self, shape, type, init_val=None):
    #     sharding_dim = shape[0] // self.nDevices
    #     sharded_shape = (self.nDevices, sharding_dim,  *shape[1:])
    #     device_shape = sharded_shape[1:]
    #     arrays = []

    #     for d, index in self.sharding.addressable_devices_indices_map(sharded_shape).items():
    #         jax.default_device = d
    #         if init_val is None:
    #             x = jnp.zeros(shape=device_shape, dtype=type)
    #         else:
    #             x = jnp.full(shape=device_shape, fill_value=init_val, dtype=type)  
    #         arrays += [jax.device_put(x, d)] 
    #     jax.default_device = jax.devices()[0]
    #     return jax.make_array_from_single_device_arrays(shape, self.sharding, arrays)

    @partial(jit, static_argnums=(0, 1, 2, 4))
    def distributed_array_init(self, shape, type, init_val=0, sharding=None):
        """
        Initialize a distributed array using JAX, with a specified shape, data type, and initial value.
        Optionally, provide a custom sharding strategy.

        Parameters
        ----------
            shape (tuple): The shape of the array to be created.
            type (dtype): The data type of the array to be created.
            init_val (scalar, optional): The initial value to fill the array with. Defaults to 0.
            sharding (Sharding, optional): The sharding strategy to use. Defaults to `self.sharding`.

        Returns
        -------
            jax.numpy.ndarray: A JAX array with the specified shape, data type, initial value, and sharding strategy.
        """
        if sharding is None:
            sharding = self.sharding
        x = jnp.full(shape=shape, fill_value=init_val, dtype=type)        
        return jax.lax.with_sharding_constraint(x, sharding)
    
    @partial(jit, static_argnums=(0,))
    def create_grid_mask(self, solid_halo_voxels):
        """
        This function creates a mask for the background grid that accounts for the location of the boundaries.
        
        Parameters
        ----------
            solid_halo_voxels: A numpy array representing the voxels in the halo of the solid object.
            
        Returns
        -------
            A JAX array representing the grid mask of the grid.
        """
        # Halo width (hw_x is different to accommodate the domain sharding per XLA device)
        hw_x = self.nDevices
        hw_y = hw_z = 1
        if self.dim == 2:
            grid_mask = self.distributed_array_init((self.nx + 2 * hw_x, self.ny + 2 * hw_y, self.lattice.q), jnp.bool_, init_val=True)
            grid_mask = grid_mask.at[(slice(hw_x, -hw_x), slice(hw_y, -hw_y), slice(None))].set(False)
            if solid_halo_voxels is not None:
                solid_halo_voxels = solid_halo_voxels.at[:, 0].add(hw_x)
                solid_halo_voxels = solid_halo_voxels.at[:, 1].add(hw_y)
                grid_mask = grid_mask.at[tuple(solid_halo_voxels.T)].set(True)  

            grid_mask = self.streaming(grid_mask)
            return lax.with_sharding_constraint(grid_mask, self.sharding)

        elif self.dim == 3:
            grid_mask = self.distributed_array_init((self.nx + 2 * hw_x, self.ny + 2 * hw_y, self.nz + 2 * hw_z, self.lattice.q), jnp.bool_, init_val=True)
            grid_mask = grid_mask.at[(slice(hw_x, -hw_x), slice(hw_y, -hw_y), slice(hw_z, -hw_z), slice(None))].set(False)
            if solid_halo_voxels is not None:
                solid_halo_voxels = solid_halo_voxels.at[:, 0].add(hw_x)
                solid_halo_voxels = solid_halo_voxels.at[:, 1].add(hw_y)
                solid_halo_voxels = solid_halo_voxels.at[:, 2].add(hw_z)
                grid_mask = grid_mask.at[tuple(solid_halo_voxels.T)].set(True)
            grid_mask = self.streaming(grid_mask)
            return lax.with_sharding_constraint(grid_mask, self.sharding)

    def bounding_box_indices(self):
        """
        This function calculates the indices of the bounding box of a 2D or 3D grid.
        The bounding box is defined as the set of grid points on the outer edge of the grid.

        Returns
        -------
            boundingBox (dict): A dictionary where keys are the names of the bounding box faces
            ("bottom", "top", "left", "right" for 2D; additional "front", "back" for 3D), and values
            are numpy arrays of indices corresponding to each face.
        """
        if self.dim == 2:
            # For a 2D grid, the bounding box consists of four edges: bottom, top, left, and right.
            # Each edge is represented as an array of indices. For example, the bottom edge includes
            # all points where the y-coordinate is 0, so its indices are [[i, 0] for i in range(self.nx)].
            bounding_box = {"bottom": np.array([[i, 0] for i in range(self.nx)], dtype=int),
                           "top": np.array([[i, self.ny - 1] for i in range(self.nx)], dtype=int),
                           "left": np.array([[0, i] for i in range(self.ny)], dtype=int),
                           "right": np.array([[self.nx - 1, i] for i in range(self.ny)], dtype=int)}
                            
            return bounding_box

        elif self.dim == 3:
            # For a 3D grid, the bounding box consists of six faces: bottom, top, left, right, front, and back.
            # Each face is represented as an array of indices. For example, the bottom face includes all points
            # where the z-coordinate is 0, so its indices are [[i, j, 0] for i in range(self.nx) for j in range(self.ny)].
            bounding_box = {
                "bottom": np.array([[i, j, 0] for i in range(self.nx) for j in range(self.ny)], dtype=int),
                "top": np.array([[i, j, self.nz - 1] for i in range(self.nx) for j in range(self.ny)],dtype=int),
                "left": np.array([[0, j, k] for j in range(self.ny) for k in range(self.nz)], dtype=int),
                "right": np.array([[self.nx - 1, j, k] for j in range(self.ny) for k in range(self.nz)], dtype=int),
                "front": np.array([[i, 0, k] for i in range(self.nx) for k in range(self.nz)], dtype=int),
                "back": np.array([[i, self.ny - 1, k] for i in range(self.nx) for k in range(self.nz)], dtype=int)}

            return bounding_box

    def set_precisions(self, precision):
        """
        This function sets the precision of the computations. The precision is defined by a pair of values,
        representing the precision of the computation and the precision of the storage, respectively.

        Parameters
        ----------
            precision (str): A string representing the desired precision. The string should be in the format
            "computation/storage", where "computation" and "storage" are either "f64", "f32", or "f16",
            representing 64-bit, 32-bit, or 16-bit floating point numbers, respectively.

        Returns
        -------
            tuple: A pair of jax.numpy data types representing the computation and storage precisions, respectively.
            If the input string does not match any of the predefined options, the function defaults to (jnp.float32, jnp.float32).
        """
        return {
            "f64/f64": (jnp.float64, jnp.float64),
            "f32/f32": (jnp.float32, jnp.float32),
            "f32/f16": (jnp.float32, jnp.float16),
            "f16/f16": (jnp.float16, jnp.float16),
            "f64/f32": (jnp.float64, jnp.float32),
            "f64/f16": (jnp.float64, jnp.float16),
        }.get(precision, (jnp.float32, jnp.float32))

    def initialize_macroscopic_fields(self):
        """
        This function initializes the macroscopic fields (density and velocity) to their default values.
        The default density is 1 and the default velocity is 0.

        Note: This function is a placeholder and should be overridden in a subclass or in an instance of the class
        to provide specific initial conditions.

        Returns
        -------
            None, None: The default density and velocity, both None. This indicates that the actual values should be set elsewhere.
        """
        print("WARNING: Default initial conditions assumed: density = 1, velocity = 0")
        print("         To set explicit initial density and velocity, use self.initialize_macroscopic_fields.")
        return None, None

    def assign_fields_sharded(self, init_val=None):
        """
        This function is used to initialize the simulation by assigning the macroscopic fields and populations.

        The function first initializes the macroscopic fields, which are the density (rho0) and velocity (u0).
        Depending on the dimension of the simulation (2D or 3D), it then sets the shape of the array that will hold the 
        distribution functions (f).

        If the density or velocity are not provided, the function initializes the distribution functions with a default 
        value (self.w), representing density=1 and velocity=0. Otherwise, it uses the provided density and velocity to initialize the populations.

        Parameters
        ----------
        None

        Returns
        -------
        f: a distributed JAX array of shape (nx, ny, nz, q) or (nx, ny, q) holding the distribution functions for the simulation.
        """
        rho0, u0 = self.initialize_macroscopic_fields()

        if self.dim == 2:
            shape = (self.nx, self.ny, self.lattice.q)
        if self.dim == 3:
            shape = (self.nx, self.ny, self.nz, self.lattice.q)
    
        if rho0 is None or u0 is None:
            if init_val is None:
                init_val = self.w
            f = self.distributed_array_init(shape, self.precisionPolicy.output_dtype, init_val=init_val)
        else:
            f = self.initialize_populations(rho0, u0)

        return f
    
    def initialize_populations(self, rho0, u0):
        """
        This function initializes the populations (distribution functions) for the simulation.
        It uses the equilibrium distribution function, which is a function of the macroscopic 
        density and velocity.

        Parameters
        ----------
        rho0: jax.numpy.ndarray
            The initial density field.
        u0: jax.numpy.ndarray
            The initial velocity field.

        Returns
        -------
        f: jax.numpy.ndarray
            The array holding the initialized distribution functions for the simulation.
        """
        return self.equilibrium(rho0, u0)

    def send_right(self, x, axis_name):
        """
        This function sends the data to the right neighboring process in a parallel computing environment.
        It uses a permutation operation provided by the LAX library.

        Parameters
        ----------
        x: jax.numpy.ndarray
            The data to be sent.
        axis_name: str
            The name of the axis along which the data is sent.

        Returns
        -------
        jax.numpy.ndarray
            The data after being sent to the right neighboring process.
        """
        return lax.ppermute(x, perm=self.rightPerm, axis_name=axis_name)
   
    def send_left(self, x, axis_name):
        """
        This function sends the data to the left neighboring process in a parallel computing environment.
        It uses a permutation operation provided by the LAX library.

        Parameters
        ----------
        x: jax.numpy.ndarray
            The data to be sent.
        axis_name: str
            The name of the axis along which the data is sent.

        Returns
        -------
            The data after being sent to the left neighboring process.
        """
        return lax.ppermute(x, perm=self.leftPerm, axis_name=axis_name)
    
    def streaming_m(self, f):
        """
        This function performs the streaming step in the Lattice Boltzmann Method and propagates
        the distribution functions along lattice directions.

        To enable multi-GPU/TPU functionality, it extracts the left and right boundary slices of the
        distribution functions that need to be communicated to the neighboring processes.

        The function then sends the left boundary slice to the right neighboring process and the right 
        boundary slice to the left neighboring process. The received data is then set to the 
        corresponding indices in the receiving domain.

        Parameters
        ----------
        f: jax.numpy.ndarray
            The array holding the distribution functions for the simulation.

        Returns
        -------
        jax.numpy.ndarray
            The distribution functions after the streaming operation.
        """
        f = self.streaming_p(f, self.c)
        left_comm, right_comm = f[:1, ..., self.lattice.right_indices], f[-1:, ..., self.lattice.left_indices]

        left_comm, right_comm = self.send_right(left_comm, 'x'), self.send_left(right_comm, 'x')
        f = f.at[:1, ..., self.lattice.right_indices].set(left_comm)
        f = f.at[-1:, ..., self.lattice.left_indices].set(right_comm)
        return f

    @partial(jit, static_argnums=(0,))
    def streaming_p(self, f, c):
        """
        Perform streaming operation on a partitioned (in the x-direction) distribution function.
        
        The function uses the vmap operation provided by the JAX library to vectorize the computation 
        over all lattice directions.

        Parameters
        ----------
            f: The distribution function.
            c: The streaming vector in all lattice directions.

        Returns
        -------
            The updated distribution function after streaming.
        """
        def streaming_i(f, ci):
            """
            Perform individual streaming operation in a direction.

            Parameters
            ----------
                f : The distribution function.
                ci: The streaming vector along i-th lattice direction.

            Returns
            -------
                jax.numpy.ndarray
                The updated distribution function after streaming.
            """
            if self.dim == 2:
                return jnp.roll(f, (ci[0], ci[1]), axis=(0, 1))
            elif self.dim == 3:
                return jnp.roll(f, (ci[0], ci[1], ci[2]), axis=(0, 1, 2))

        return vmap(streaming_i, in_axes=(-1, 0), out_axes=-1)(f, c.T)

    @partial(jit, static_argnums=(0, 3), inline=True)
    def equilibrium(self, rho, u, cast_output=True):
        """
        This function computes the equilibrium distribution function in the Lattice Boltzmann Method.
        The equilibrium distribution function is a function of the macroscopic density and velocity.

        The function first casts the density and velocity to the compute precision if the cast_output flag is True.
        The function finally casts the equilibrium distribution function to the output precision if the cast_output 
        flag is True.

        Parameters
        ----------
        rho: jax.numpy.ndarray
            The macroscopic density.
        u: jax.numpy.ndarray
            The macroscopic velocity.
        cast_output: bool, optional
            A flag indicating whether to cast the density, velocity, and equilibrium distribution function to the 
            compute and output precisions. Default is True.

        Returns
        -------
        feq: ja.numpy.ndarray
            The equilibrium distribution function.
        """
        # Cast the density and velocity to the compute precision if the cast_output flag is True
        if cast_output:
            rho, u = self.precisionPolicy.cast_to_compute((rho, u))

        # Cast c to compute precision so that XLA call FXX matmul, 
        # which is faster (it is faster in some older versions of JAX, newer versions are smart enough to do this automatically)
        c = jnp.array(self.c, dtype=self.precisionPolicy.compute_dtype)
        cu = 3.0 * jnp.dot(u, c)
        usqr = 1.5 * jnp.sum(jnp.square(u), axis=-1, keepdims=True)
        feq = rho * self.w * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)

        if cast_output:
            return self.precisionPolicy.cast_to_output(feq)
        else:
            return feq

    @partial(jit, static_argnums=(0,))
    def momentum_flux(self, fneq):
        """
        This function computes the momentum flux, which is the product of the non-equilibrium 
        distribution functions (fneq) and the lattice moments (cc).

        The momentum flux is used in the computation of the stress tensor in the Lattice Boltzmann 
        Method (LBM).

        Parameters
        ----------
        fneq: jax.numpy.ndarray
            The non-equilibrium distribution functions.

        Returns
        -------
        jax.numpy.ndarray
            The computed momentum flux.
        """
        return jnp.dot(fneq, self.lattice.cc)

    @partial(jit, static_argnums=(0,), inline=True)
    def update_macroscopic(self, f):
        """
        This function computes the macroscopic variables (density and velocity) based on the 
        distribution functions (f).

        The density is computed as the sum of the distribution functions over all lattice directions. 
        The velocity is computed as the dot product of the distribution functions and the lattice 
        velocities, divided by the density.

        Parameters
        ----------
        f: jax.numpy.ndarray
            The distribution functions.

        Returns
        -------
        rho: jax.numpy.ndarray
            Computed density.
        u: jax.numpy.ndarray
            Computed velocity.
        """
        rho =jnp.sum(f, axis=-1, keepdims=True)
        c = jnp.array(self.c, dtype=self.precisionPolicy.compute_dtype).T
        u = jnp.dot(f, c) / rho

        return rho, u
    
    @partial(jit, static_argnums=(0, 5), inline=True)
    def apply_bc(self, fout, fin, timestep, sdf, implementation_step):
        """
        This function applies the boundary conditions to the distribution functions.

        It iterates over all boundary conditions (BCs) and checks if the implementation step of the 
        boundary condition matches the provided implementation step. If it does, it applies the 
        boundary condition to the post-streaming distribution functions (fout).

        Parameters
        ----------
        fout: jax.numpy.ndarray
            The post-collision distribution functions.
        fin: jax.numpy.ndarray
            The post-streaming distribution functions.
        sdf: jax.numpy.ndarray, optional
            signed distance field.
        implementation_step: str
            The implementation step at which the boundary conditions should be applied.

        Returns
        -------
        ja.numpy.ndarray
            The output distribution functions after applying the boundary conditions.
        """
        for bc in self.BCs:
            fout = bc.prepare_populations(fout, fin, implementation_step)
            if bc.implementationStep == implementation_step:
                if bc.isDynamic:
                    fout = bc.apply(fout, fin, timestep)
                else:
                    fout = fout.at[bc.indices].set(bc.apply(fout, fin, sdf))
                    
        return fout

    @partial(jit, static_argnums=(0, 4))
    def step(self, f_poststreaming, timestep, sdf, return_fpost=False):
        """
        This function performs a single step of the LBM simulation.

        It first performs the collision step, which is the relaxation of the distribution functions 
        towards their equilibrium values. It then applies the respective boundary conditions to the 
        post-collision distribution functions.

        The function then performs the streaming step, which is the propagation of the distribution 
        functions in the lattice. It then applies the respective boundary conditions to the post-streaming 
        distribution functions.

        Parameters
        ----------
        f_poststreaming: jax.numpy.ndarray
            The post-streaming distribution functions.
        timestep: int
            The current timestep of the simulation.
        sdf: jax.numpy.ndarray, optional
            signed distance field.
        return_fpost: bool, optional
            If True, the function also returns the post-collision distribution functions.

        Returns
        -------
        f_poststreaming: jax.numpy.ndarray
            The post-streaming distribution functions after the simulation step.
        f_postcollision: jax.numpy.ndarray or None
            The post-collision distribution functions after the simulation step, or None if 
            return_fpost is False.
        """
        f_postcollision = self.collision(f_poststreaming, sdf)
        f_postcollision = self.apply_bc(f_postcollision, f_poststreaming, timestep, sdf, "PostCollision")
        f_poststreaming = self.streaming(f_postcollision)
        f_poststreaming = self.apply_bc(f_poststreaming, f_postcollision, timestep, sdf, "PostStreaming")

        if return_fpost:
            return f_poststreaming, f_postcollision
        else:
            return f_poststreaming, None

    def run(self, t_max, sdf=None, f=None):
        """
        This function runs the LBM simulation for a specified number of time steps.

        It first initializes the distribution functions and then enters a loop where it performs the 
        simulation steps (collision, streaming, and boundary conditions) for each time step.

        The function can also print the progress of the simulation, save the simulation data, and 
        compute the performance of the simulation in million lattice updates per second (MLUPS).

        Parameters
        ----------
        t_max: int
            The total number of time steps to run the simulation.
        sdf: jax.numpy.ndarray, optional
            a signed distance field (if provide) used in imposing interpolated boundary condition
        Returns
        -------
        f: jax.numpy.ndarray
            The distribution functions after the simulation.
        """
        f = self.assign_fields_sharded(f)
        start_step = 0
        if self.restore_checkpoint:
            latest_step = self.mngr.latest_step()
            if latest_step is not None:  # existing checkpoint present
                # Assert that the checkpoint manager is not None
                assert self.mngr is not None, "Checkpoint manager does not exist."
                state = {'f': f}
                shardings = jax.tree_map(lambda x: x.sharding, state)
                restore_args = orb.checkpoint_utils.construct_restore_args(state, shardings)
                try:
                    f = self.mngr.restore(latest_step, restore_kwargs={'restore_args': restore_args})['f']
                    print(f"Restored checkpoint at step {latest_step}.")
                except ValueError:
                    raise ValueError(f"Failed to restore checkpoint at step {latest_step}.")
                
                start_step = latest_step + 1
                if not (t_max > start_step):
                    raise ValueError(f"Simulation already exceeded maximum allowable steps (t_max = {t_max}). Consider increasing t_max.")
        if self.computeMLUPS:
            start = time.time()
        # Loop over all time steps
        for timestep in range(start_step, t_max + 1):
            io_flag = self.ioRate > 0 and (timestep % self.ioRate == 0 or timestep == t_max)
            print_iter_flag = self.printInfoRate> 0 and timestep % self.printInfoRate== 0
            checkpoint_flag = self.checkpointRate > 0 and timestep % self.checkpointRate == 0

            if io_flag:
                # Update the macroscopic variables and save the previous values (for error computation)
                rho_prev, u_prev = self.update_macroscopic(f)
                rho_prev = downsample_field(rho_prev, self.downsamplingFactor)
                u_prev = downsample_field(u_prev, self.downsamplingFactor)
                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                rho_prev = process_allgather(rho_prev)
                u_prev = process_allgather(u_prev)


            # Perform one time-step (collision, streaming, and boundary conditions)
            f, fstar = self.step(f, timestep, sdf, return_fpost=self.returnFpost)
            # Print the progress of the simulation
            if print_iter_flag:
                print(colored("Timestep ", 'blue') + colored(f"{timestep}", 'green') + colored(" of ", 'blue') + colored(f"{t_max}", 'green') + colored(" completed", 'blue'))

            if io_flag:
                # Save the simulation data
                print(f"Saving data at timestep {timestep}/{t_max}")
                rho, u = self.update_macroscopic(f)
                rho = downsample_field(rho, self.downsamplingFactor)
                u = downsample_field(u, self.downsamplingFactor)
                
                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                rho = process_allgather(rho)
                u = process_allgather(u)

                # Save the data
                self.handle_io_timestep(timestep, f, fstar, rho, u, rho_prev, u_prev)
            
            if checkpoint_flag:
                # Save the checkpoint
                print(f"Saving checkpoint at timestep {timestep}/{t_max}")
                state = {'f': f}
                self.mngr.save(timestep, state)
            
            # Start the timer for the MLUPS computation after the first timestep (to remove compilation overhead)
            if self.computeMLUPS and timestep == 1:
                jax.block_until_ready(f)
                start = time.time()

        if self.computeMLUPS:
            # Compute and print the performance of the simulation in MLUPS
            jax.block_until_ready(f)
            end = time.time()
            if self.dim == 2:
                print(colored("Domain: ", 'blue') + colored(f"{self.nx} x {self.ny}", 'green') if self.dim == 2 else colored(f"{self.nx} x {self.ny} x {self.nz}", 'green'))
                print(colored("Number of voxels: ", 'blue') + colored(f"{self.nx * self.ny}", 'green') if self.dim == 2 else colored(f"{self.nx * self.ny * self.nz}", 'green'))
                print(colored("MLUPS: ", 'blue') + colored(f"{self.nx * self.ny * t_max / (end - start) / 1e6}", 'red'))

            elif self.dim == 3:
                print(colored("Domain: ", 'blue') + colored(f"{self.nx} x {self.ny} x {self.nz}", 'green'))
                print(colored("Number of voxels: ", 'blue') + colored(f"{self.nx * self.ny * self.nz}", 'green'))
                print(colored("MLUPS: ", 'blue') + colored(f"{self.nx * self.ny * self.nz * t_max / (end - start) / 1e6}", 'red'))

        if self.returnFpost:
            return f, fstar
        else:
            return f

    def handle_io_timestep(self, timestep, f, fstar, rho, u, rho_prev, u_prev):
        """
        This function handles the input/output (I/O) operations at each time step of the simulation.

        It prepares the data to be saved and calls the output_data function, which can be overwritten 
        by the user to customize the I/O operations.

        Parameters
        ----------
        timestep: int
            The current time step of the simulation.
        f: jax.numpy.ndarray
            The post-streaming distribution functions at the current time step.
        fstar: jax.numpy.ndarray
            The post-collision distribution functions at the current time step.
        rho: jax.numpy.ndarray
            The density field at the current time step.
        u: jax.numpy.ndarray
            The velocity field at the current time step.
        """
        kwargs = {
            "timestep": timestep,
            "rho": rho,
            "rho_prev": rho_prev,
            "u": u,
            "u_prev": u_prev,
            "f_poststreaming": f,
            "f_postcollision": fstar
        }
        self.output_data(**kwargs)

    def output_data(self, **kwargs):
        """
        This function is intended to be overwritten by the user to customize the input/output (I/O) 
        operations of the simulation.

        By default, it does nothing. When overwritten, it could save the simulation data to files, 
        display the simulation results in real time, send the data to another process for analysis, etc.

        Parameters
        ----------
        **kwargs: dict
            A dictionary containing the simulation data to be outputted. The keys are the names of the 
            data fields, and the values are the data fields themselves.
        """
        pass

    def set_boundary_conditions(self):
        """
        This function sets the boundary conditions for the simulation.

        It is intended to be overwritten by the user to specify the boundary conditions according to 
        the specific problem being solved.

        By default, it does nothing. When overwritten, it could set periodic boundaries, no-slip 
        boundaries, inflow/outflow boundaries, etc.
        """
        pass

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, fin):
        """
        This function performs the collision step in the Lattice Boltzmann Method.

        It is intended to be overwritten by the user to specify the collision operator according to 
        the specific LBM model being used.

        By default, it does nothing. When overwritten, it could implement the BGK collision operator,
        the MRT collision operator, etc.

        Parameters
        ----------
        fin: jax.numpy.ndarray
            The pre-collision distribution functions.

        Returns
        -------
        fin: jax.numpy.ndarray
            The post-collision distribution functions.
        """
        return fin

    def get_force(self):
        """
        This function computes the force to be applied to the fluid in the Lattice Boltzmann Method.

        It is intended to be overwritten by the user to specify the force according to the specific 
        problem being solved.

        By default, it does nothing and returns None. When overwritten, it could implement a constant 
        force term.

        Returns
        -------
        force: jax.numpy.ndarray
            The force to be applied to the fluid.
        """
        pass

    @partial(jit, static_argnums=(0,), inline=True)
    def apply_force(self, f_postcollision, feq, rho, u):
        """
        add force based on exact-difference method due to Kupershtokh

        Parameters
        ----------
        f_postcollision: jax.numpy.ndarray
            The post-collision distribution functions.
        feq: jax.numpy.ndarray
            The equilibrium distribution functions.
        rho: jax.numpy.ndarray
            The density field.

        u: jax.numpy.ndarray
            The velocity field.
        
        Returns
        -------
        f_postcollision: jax.numpy.ndarray
            The post-collision distribution functions with the force applied.
        
        References
        ----------
        Kupershtokh, A. (2004). New method of incorporating a body force term into the lattice Boltzmann equation. In
        Proceedings of the 5th International EHD Workshop (pp. 241-246). University of Poitiers, Poitiers, France.
        Chikatamarla, S. S., & Karlin, I. V. (2013). Entropic lattice Boltzmann method for turbulent flow simulations:
        Boundary conditions. Physica A, 392, 1925-1930.
        Krüger, T., et al. (2017). The lattice Boltzmann method. Springer International Publishing, 10.978-3, 4-15.
        """
        delta_u = self.get_force()
        feq_force = self.equilibrium(rho, u + delta_u, cast_output=False)
        f_postcollision = f_postcollision + feq_force - feq
        return f_postcollision


    @partial(jit, static_argnums=(0,), inline=True)
    def tensor_inner_product(self, A, B):
        """
        computes the inner product of two tensors A and B, sum_i sum_j A_{ij}B_{ij}
        The shapes of "A" and "B" are similar with their axis=0 having a shape of [dim*(dim+1)/2]
        """
        # Set the diagonal and off-diagonal components of a symmetric rank 2 tensor as per definition in "momentum_flux"
        # if A.shape[-1] == 3:
        #     dim = 2
        # else:
        #     dim = 3
        if self.dim == 3:
            diagonal    = (0, 3, 5)
            offdiagonal = (1, 2, 4)
        else:
            # dim ==2:
            diagonal    = (0, 2)
            offdiagonal = (1,)

        AB = jnp.sum(A[..., diagonal]*B[..., diagonal], axis=-1, keepdims=True) + \
             jnp.sum(2.0 * A[..., offdiagonal]*B[..., offdiagonal], axis=-1, keepdims=True)
        return AB

    @partial(jit, static_argnums=(0,), inline=True)
    def tensor_modulus(self, T):
        """returning |T| = sqrt (2TijTij)"""
        return jnp.sqrt(2.*self.tensor_inner_product(T, T))

    @partial(jit, static_argnums=(0,), inline=True)
    def strain_rate(self, fneq, tau, rho):
        """
        compute the strain rate tensor (considering symmetry of the tensor)
        f_neq: non-equilibrium distribution function
        tau:  total eddy viscosity (equal to molecular viscosity tau0 if LES Smagorinsky is not invoked)
        return: strain rate tensor (symmetric tensor, 3 elements in 2D and 6 elements in 3D)
        strain_rate_tensor = 0.5 [ \nabla \vec{vel} + (\nabla \vec{vel}).T ]
                           =-(1/(2*c_s^2*\rho*\tau))* \Pi^{1}
                           ~-(1/(2*c_s^2*\rho*\tau))* \Pi^{Neq}
        """
        PiNeq = self.momentum_flux(fneq)
        Sij = -1.5*PiNeq/rho/tau
        return Sij

    @partial(jit, static_argnums=(0,), inline=True)
    def viscous_dissipation(self, fneq, rho):
        """
        Calculating "viscous dissipation" based on its microscopic definition:
        epsilon = 2* \nu * |strain_rate_tensor|^2
        strain_rate_tensor = 0.5 [ \nabla \vec{vel} + (\nabla \vec{vel}).T ]
                           =-(1/(2*c_s^2*\rho*\tau))* \Pi^{1}
                           ~-(1/(2*c_s^2*\rho*\tau))* \Pi^{Neq}
        which gives:
        epsilon = 0.5* \nu * /(c_s^4*\rho^2*\tau^2) * |\Pi^{Neq}|^2
        """
        tau = 1./self.omega
        viscosity = (tau - 0.5)/3.
        Sij = self.strain_rate(fneq, tau, rho)
        visc_dissip = 2. * viscosity * self.tensor_inner_product(Sij, Sij)
        return visc_dissip

    @partial(jit, static_argnums=(0,), inline=True)
    def turbulent_relaxation(self, fneq, tau0):
        """
        fneq: non-equilibrium distribution function
        tau0: the relaxation time associated with the molecular viscosity (single scalar)
        return: Turbulent relaxation time, \tau_t = 3*eddy_viscosity

        Ref:
        [1] Stiebler, M., Krafczyk, M., Freudiger, S. and Geier, M., 2011. Lattice Boltzmann large eddy simulation
            of subcritical flows around a sphere on non-uniform grids. Computers & Mathematics with Applications,
            61(12), pp.3475-3484.
        """

        # compute the momentum flux tensor
        PiNeq = self.momentum_flux(fneq)

        # Compute the modulus of the momentum flux |Pi_Neq Pi_Neq|
        momentum_flux_modulus = self.tensor_modulus(PiNeq)

        # Turbulent relaxation time, \tau_t (Eq 14b [2])
        tau_t = 0.5*(jnp.sqrt(tau0**2 + 18.*self.smagorinskyConstant*momentum_flux_modulus) - tau0)

        return tau_t


    @partial(jit, static_argnums=(0,), inline=True)
    def fdecompose_shear_d2q9(self, fneq):
        """
        Decompose fneq into shear components for D2Q9 lattice.

        Parameters
        ----------
        fneq : jax.numpy.array
            Non-equilibrium distribution function.

        Returns
        -------
        jax.numpy.array
            Shear components of fneq.
        """
        Pi = self.momentum_flux(fneq)
        N = Pi[..., 0] - Pi[..., 2]
        s = jnp.zeros_like(fneq)
        s = s.at[..., 6].set(N)
        s = s.at[..., 3].set(N)
        s = s.at[..., 2].set(-N)
        s = s.at[..., 1].set(-N)
        s = s.at[..., 8].set(Pi[..., 1])
        s = s.at[..., 4].set(-Pi[..., 1])
        s = s.at[..., 5].set(-Pi[..., 1])
        s = s.at[..., 7].set(Pi[..., 1])
        return s

    @partial(jit, static_argnums=(0,), inline=True)
    def fdecompose_shear_d3q27(self, fneq):
        """
        Decompose fneq into shear components for D3Q27 lattice.

        Parameters
        ----------
        fneq : jax.numpy.ndarray
            Non-equilibrium distribution function.

        Returns
        -------
        jax.numpy.ndarray
            Shear components of fneq.
        """
        # if self.grid.dim == 3:
        #     diagonal    = (0, 3, 5)
        #     offdiagonal = (1, 2, 4)
        # elif self.grid.dim == 2:
        #     diagonal    = (0, 2)
        #     offdiagonal = (1,)

        # c=
        # array([[0, 0, 0],-----0
        #        [0, 0, -1],----1
        #        [0, 0, 1],-----2
        #        [0, -1, 0],----3
        #        [0, -1, -1],---4
        #        [0, -1, 1],----5
        #        [0, 1, 0],-----6
        #        [0, 1, -1],----7
        #        [0, 1, 1],-----8
        #        [-1, 0, 0],----9
        #        [-1, 0, -1],--10
        #        [-1, 0, 1],---11
        #        [-1, -1, 0],--12
        #        [-1, -1, -1],-13
        #        [-1, -1, 1],--14
        #        [-1, 1, 0],---15
        #        [-1, 1, -1],--16
        #        [-1, 1, 1],---17
        #        [1, 0, 0],----18
        #        [1, 0, -1],---19
        #        [1, 0, 1],----20
        #        [1, -1, 0],---21
        #        [1, -1, -1],--22
        #        [1, -1, 1],---23
        #        [1, 1, 0],----24
        #        [1, 1, -1],---25
        #        [1, 1, 1]])---26
        Pi = self.momentum_flux(fneq)
        Nxz = Pi[..., 0] - Pi[..., 5]
        Nyz = Pi[..., 3] - Pi[..., 5]

        # For c = (i, 0, 0), c = (0, j, 0) and c = (0, 0, k)
        s = jnp.zeros_like(fneq)
        s = s.at[..., 9].set((2.0 * Nxz - Nyz) / 6.0)
        s = s.at[..., 18].set((2.0 * Nxz - Nyz) / 6.0)
        s = s.at[..., 3].set((-Nxz + 2.0 * Nyz) / 6.0)
        s = s.at[..., 6].set((-Nxz + 2.0 * Nyz) / 6.0)
        s = s.at[..., 1].set((-Nxz - Nyz) / 6.0)
        s = s.at[..., 2].set((-Nxz - Nyz) / 6.0)

        # For c = (i, j, 0)
        s = s.at[..., 12].set(Pi[..., 1] / 4.0)
        s = s.at[..., 24].set(Pi[..., 1] / 4.0)
        s = s.at[..., 21].set(-Pi[..., 1] / 4.0)
        s = s.at[..., 15].set(-Pi[..., 1] / 4.0)

        # For c = (i, 0, k)
        s = s.at[..., 10].set(Pi[..., 2] / 4.0)
        s = s.at[..., 20].set(Pi[..., 2] / 4.0)
        s = s.at[..., 19].set(-Pi[..., 2] / 4.0)
        s = s.at[..., 11].set(-Pi[..., 2] / 4.0)

        # For c = (0, j, k)
        s = s.at[..., 8].set(Pi[..., 4] / 4.0)
        s = s.at[..., 4].set(Pi[..., 4] / 4.0)
        s = s.at[..., 7].set(-Pi[..., 4] / 4.0)
        s = s.at[..., 5].set(-Pi[..., 4] / 4.0)
        return s