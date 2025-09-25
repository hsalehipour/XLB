import neon
import warp as wp
import numpy as np
import time
import os
import re
import matplotlib.pyplot as plt
import trimesh
import shutil

import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import multires_grid_factory
from xlb.operator.boundary_condition import (
    FullwayBounceBackBC,
    HalfwayBounceBackBC,
    RegularizedBC,
    ExtrapolationOutflowBC,
    DoNothingBC,
    ZouHeBC,
    HybridBC,
)
from xlb.operator.boundary_masker import MeshVoxelizationMethod
#from xlb.utils.mesher import make_cuboid_mesh, MultiresIO
from xlb.utils.makemesh import generate_mesh
from xlb.operator.force import MultiresMomentumTransfer
from xlb.helper.initializers import CustomMultiresInitializer
from xlb import MresPerfOptimizationType
from typing import Any




def adjust_bbox(cuboid_max, cuboid_min, voxel_size_up):
    """
    Adjust the bounding box to the nearest points of one level finer grid that encloses the desired region.

    Args:
        cuboid_min (np.ndarray): Desired minimum coordinates of the bounding box.
        cuboid_max (np.ndarray): Desired maximum coordinates of the bounding box.
        voxel_size_up (float): Voxel size of one level higher (finer) grid.

    Returns:
        tuple: (adjusted_min, adjusted_max) snapped to grid points of one level higher.
    """

    adjusted_min = np.round(cuboid_min / voxel_size_up) * voxel_size_up
    adjusted_max = np.round(cuboid_max / voxel_size_up) * voxel_size_up
    return adjusted_min, adjusted_max

def prepare_sparsity_pattern(level_data):
    """
    Prepare the sparsity pattern for the multiresolution grid based on the level data.
    """
    sparsity_pattern = []
    level_origins = []
    for lvl in range(len(level_data)):
        level_mask = level_data[lvl][0]
        level_mask = np.ascontiguousarray(level_mask, dtype=np.int32)
        sparsity_pattern.append(level_mask)
        level_origins.append(level_data[lvl][2])
    return sparsity_pattern, level_origins

def make_cuboid_mesh(voxel_size, cuboids, stl_filename):
    """
    Create a strongly-balanced multi-level cuboid mesh with a sequence of bounding boxes.
    Outputs mask arrays that are set to True only in regions not covered by finer levels.

    Args:
        voxel_size (float): Voxel size of the finest grid .
        cuboids (list): List of multipliers defining each level's domain.
        stl_name (str): Path to the STL file.

    Returns:
        list: Level data with mask arrays, voxel sizes, origins, and levels.
    """
    # Load the mesh and get its bounding box
    mesh = trimesh.load_mesh(stl_filename, process=False)
    assert not mesh.is_empty, ValueError("Loaded mesh is empty or invalid.")

    mesh_vertices = mesh.vertices
    min_bound = mesh_vertices.min(axis=0)
    max_bound = mesh_vertices.max(axis=0)
    partSize = max_bound - min_bound

    level_data = []
    adjusted_bboxes = []
    max_voxel_size = voxel_size * pow(2, (len(cuboids) - 1))
    # Step 1: Generate all levels and store their data
    for level in range(len(cuboids)):
        # Compute desired bounding box for this level
        cuboid_min = np.array(
            [
                min_bound[0] - cuboids[level][0] * partSize[0],
                min_bound[1] - cuboids[level][2] * partSize[1],
                min_bound[2] - cuboids[level][4] * partSize[2],
            ],
            dtype=float,
        )

        cuboid_max = np.array(
            [
                max_bound[0] + cuboids[level][1] * partSize[0],
                max_bound[1] + cuboids[level][3] * partSize[1],
                max_bound[2] + cuboids[level][5] * partSize[2],
            ],
            dtype=float,
        )

        # Set voxel size for this level
        voxel_size_level = max_voxel_size / pow(2, level)

        # Adjust bounding box to align with one level up (finer grid)
        if level > 0:
            voxel_level_up = max_voxel_size / pow(2, level - 1)
        else:
            voxel_level_up = voxel_size_level
        adjusted_min, adjusted_max = adjust_bbox(cuboid_max, cuboid_min, voxel_level_up)

        xmin, ymin, zmin = adjusted_min
        xmax, ymax, zmax = adjusted_max
        
        # Compute number of voxels based on level-specific voxel size
        nx = int(np.round((xmax - xmin) / voxel_size_level))
        ny = int(np.round((ymax - ymin) / voxel_size_level))
        nz = int(np.round((zmax - zmin) / voxel_size_level))
        print(f"Domain {nx}, {ny}, {nz}  Origin {adjusted_min}  Voxel Size {voxel_size_level} Voxel Level Up {voxel_level_up}")
    
        voxel_matrix = np.ones((nx, ny, nz), dtype=bool)

        origin = adjusted_min
        level_data.append((voxel_matrix, voxel_size_level, origin, level))
        adjusted_bboxes.append((adjusted_min, adjusted_max))

    # Step 2: Adjust coarser levels to exclude regions covered by finer levels
    for k in range(len(level_data) - 1):  # Exclude the finest level
        # Current level's data
        voxel_matrix_k = level_data[k][0]
        origin_k = level_data[k][2]
        voxel_size_k = level_data[k][1]
        nx, ny, nz = voxel_matrix_k.shape

        # Next finer level's bounding box
        adjusted_min_k1, adjusted_max_k1 = adjusted_bboxes[k + 1]

        # Compute index ranges in level k that overlap with level k+1's bounding box
        # Use epsilon (1e-10) to handle floating-point precision
        i_start = max(0, int(np.ceil((adjusted_min_k1[0] - origin_k[0] - 1e-10) / voxel_size_k)))
        i_end = min(nx, int(np.floor((adjusted_max_k1[0] - origin_k[0] + 1e-10) / voxel_size_k)))
        j_start = max(0, int(np.ceil((adjusted_min_k1[1] - origin_k[1] - 1e-10) / voxel_size_k)))
        j_end = min(ny, int(np.floor((adjusted_max_k1[1] - origin_k[1] + 1e-10) / voxel_size_k)))
        k_start = max(0, int(np.ceil((adjusted_min_k1[2] - origin_k[2] - 1e-10) / voxel_size_k)))
        k_end = min(nz, int(np.floor((adjusted_max_k1[2] - origin_k[2] + 1e-10) / voxel_size_k)))

        # Set overlapping region to zero
        voxel_matrix_k[i_start:i_end, j_start:j_end, k_start:k_end] = 0

    # Step 3 Convert to Indices from STL units
    num_levels = len(level_data)
    level_data = [(dr, int(v / voxel_size), np.round(dOrigin / v).astype(int), num_levels - 1 - l) for dr, v, dOrigin, l in level_data]

    level_data =  list(reversed(level_data))
    sparsity_pattern, level_origins = prepare_sparsity_pattern(level_data)
    
    return level_data, sparsity_pattern, level_origins


class MultiresIO(object):
    def __init__(self, field_name_cardinality_dict, levels_data, scale=1, offset=(0.0, 0.0, 0.0), store_precision=None, timestep_size=1):
        """
        Initialize the MultiresIO object.

        Parameters
        ----------
        field_name_cardinality_dict : dict
            A dictionary mapping field names to their cardinalities.
            Example: {'velocity_x': 1, 'velocity_y': 1, 'velocity': 3, 'density': 1}
        levels_data : list of tuples
            Each tuple contains (data, voxel_size, origin, level).
        scale : float or tuple, optional
            Scale factor for the coordinates. Typically smallest voxel size
        offset : tuple, optional
            Offset to be applied to the coordinates.
        store_precision : str, optional
            The precision policy for storing data.
        timestep_size: float
            Scale factor to convert velocities to model units. Typically smallest timestep size
        """
        # Process the multires geometry and extract coordinates and connectivity in the coordinate system of the finest level
        coordinates, connectivity, level_id_field, total_cells = self.process_geometry(levels_data, scale)

        # Ensure that coordinates and connectivity are not empty
        assert coordinates.size != 0, "Error: No valid data to process. Check the input levels_data."

        # Merge duplicate points
        coordinates, connectivity = self._merge_duplicates(coordinates, connectivity, levels_data)

        # Apply scale and offset
        coordinates = self._transform_coordinates(coordinates, scale, offset)

        # Assign to self
        self.field_name_cardinality_dict = field_name_cardinality_dict
        self.levels_data = levels_data
        self.coordinates = coordinates
        self.connectivity = connectivity
        self.level_id_field = level_id_field
        self.total_cells = total_cells
        self.centroids = np.mean(coordinates[connectivity], axis=1)

        #For convertin velocities to model units
        if scale != 1 and timestep_size !=1:            
            self.conversion = scale / timestep_size
        else: 
            self.conversion = 1

        # Set the default precision policy if not provided
        from xlb import DefaultConfig

        if store_precision is None:
            self.store_precision = DefaultConfig.default_precision_policy.store_precision
            self.store_dtype = DefaultConfig.default_precision_policy.store_precision.wp_dtype

        # Prepare and allocate the inputs for the NEON container
        self.field_warp_dict, self.origin_list = self._prepare_container_inputs()

        # Construct the NEON container for exporting multi-resolution data
        self.container = self._construct_neon_container()

    def process_geometry(self, levels_data, scale):
        num_voxels_per_level = [np.sum(data) for data, _, _, _ in levels_data]
        num_points_per_level = [8 * nv for nv in num_voxels_per_level]
        point_id_offsets = np.cumsum([0] + num_points_per_level[:-1])

        all_corners = []
        all_connectivity = []
        level_id_field = []
        total_cells = 0

        for level_idx, (data, voxel_size, origin, level) in enumerate(levels_data):
            origin = origin * voxel_size
            corners_list, conn_list, _ = self._process_level(data, voxel_size, origin, level, point_id_offsets[level_idx])

            if corners_list:
                print(f"\tProcessing level {level}: Voxel size {voxel_size * scale}, Origin {origin}, Shape {data.shape}")
                all_corners.extend(corners_list)
                all_connectivity.extend(conn_list)
                num_cells = sum(c.shape[0] for c in conn_list)
                level_id_field.extend([level] * num_cells)
                total_cells += num_cells
            else:
                print(f"\tSkipping level {level} (no unique data)")

        # Stacking coordinates and connectivity
        coordinates = np.concatenate(all_corners, axis=0).astype(np.float32)
        connectivity = np.concatenate(all_connectivity, axis=0).astype(np.int32)
        level_id_field = np.array(level_id_field, dtype=np.uint8)

        return coordinates, connectivity, level_id_field, total_cells

    def _process_level(self, data, voxel_size, origin, level, point_id_offset):
        """
        Given a voxel grid, returns all corners and connectivity in NumPy for this resolution level.
        """
        true_indices = np.argwhere(data)
        if true_indices.size == 0:
            return [], [], level

        max_voxels_per_chunk = 268_435_450
        chunks = np.array_split(true_indices, max(1, (len(true_indices) + max_voxels_per_chunk - 1) // max_voxels_per_chunk))

        all_corners = []
        all_connectivity = []
        pid_offset = point_id_offset

        for chunk in chunks:
            if chunk.size == 0:
                continue
            corners, connectivity = self._process_voxel_chunk(chunk, np.asarray(origin, dtype=np.float32), voxel_size, pid_offset)
            all_corners.append(corners)
            all_connectivity.append(connectivity)
            pid_offset += len(chunk) * 8

        return all_corners, all_connectivity, level

    def _process_voxel_chunk(self, true_indices, origin, voxel_size, point_id_offset):
        """
        Given a set of voxel indices, returns 8 corners and connectivity for each cube using NumPy.
        """
        true_indices = np.asarray(true_indices, dtype=np.float32)
        mins = origin + true_indices * voxel_size
        offsets = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.float32,
        )

        corners = (mins[:, None, :] + offsets[None, :, :] * voxel_size).reshape(-1, 3).astype(np.float32)
        base_ids = point_id_offset + np.arange(len(true_indices), dtype=np.int32) * 8
        connectivity = (base_ids[:, None] + np.arange(8, dtype=np.int32)).astype(np.int32)

        return corners, connectivity

    def optimal_chunk_shape(self, shape, dtype, target_mb=4, min_chunks=64, max_chunks=4096, max_chunk_mb=64):
        """
        Choose a row-major HDF5 chunk shape for compression:
          - target_mb: desired uncompressed bytes per chunk (e.g., 4 for gzip, 8–16 for lzf/lz4)
          - keeps total number of chunks roughly in [min_chunks, max_chunks]
          - caps chunk size at max_chunk_mb to limit per-chunk memory

        Returns a tuple suitable for h5py.create_dataset(..., chunks=...).
        """
        
        n0 = int(shape[0]) if len(shape) else 1
        itemsize = np.dtype(dtype).itemsize
        row_elems = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
        row_bytes = max(1, row_elems * itemsize)

        # initial rows by target size
        target_bytes = int(target_mb * 1024 * 1024)
        rows = max(1, target_bytes // row_bytes)

        # clamp by desired chunk-count window
        lower_rows = max(1, (n0 + max_chunks - 1) // max_chunks)   # ensures <= max_chunks
        upper_rows = max(1, n0 // max_chunks if min_chunks == 0 else n0 // min_chunks)  # ensures >= min_chunks
        rows = min(max(rows, lower_rows), max(1, upper_rows))

        # cap by max bytes per chunk
        max_bytes = int(max_chunk_mb * 1024 * 1024)
        rows = min(rows, max(1, max_bytes // row_bytes))

        return (min(n0, rows),) + tuple(shape[1:])
        
    def save_xdmf(self, h5_filename, xmf_filename, total_cells, num_points, fields={}):
        # Generate an XDMF file to accompany the HDF5 file
        print(f"\tGenerating XDMF file: {xmf_filename}")
        hdf5_rel_path = h5_filename.split("/")[-1]
        with open(xmf_filename, "w") as xmf:
            xmf.write(f'''<?xml version="1.0" ?>
    <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
    <Xdmf Version="3.0">
        <Domain>
            <Grid Name="VoxelMesh" GridType="Uniform">
                <Topology TopologyType="Hexahedron" NumberOfElements="{total_cells}">
                    <DataItem Dimensions="{total_cells} 8" NumberType="Int" Format="HDF">
                        {hdf5_rel_path}:/Mesh/Connectivity
                    </DataItem>
                </Topology>
                <Geometry GeometryType="XYZ">
                    <DataItem Dimensions="{num_points} 3" NumberType="Float" Precision="4" Format="HDF">
                        {hdf5_rel_path}:/Mesh/Points
                    </DataItem>
                </Geometry>
                <Attribute Name="Level" AttributeType="Scalar" Center="Cell">
                    <DataItem Dimensions="{total_cells}" NumberType="UInt8" Format="HDF">
                        {hdf5_rel_path}:/Mesh/Level
                    </DataItem>
                </Attribute>
        ''')
            for field_name in fields.keys():
                xmf.write(f'''
            <Attribute Name="{field_name}" AttributeType="Scalar" Center="Cell">
                <DataItem Dimensions="{total_cells}" NumberType="Float" Precision="4" Format="HDF">
                {hdf5_rel_path}:/Fields/{field_name}
                </DataItem>
            </Attribute>
            ''')
            xmf.write("""
                </Grid>
            </Domain>
        </Xdmf>
        """)
        print("\tXDMF file written successfully")
        return

    def save_hdf5_file(self, filename, coordinates, connectivity, level_id_field, fields_data, compression="gzip", compression_opts=0):
        """Write the processed mesh data to an HDF5 file.
        Parameters
        ----------
        filename : str
            The name of the output HDF5 file.
        coordinates : numpy.ndarray
            An array of all coordinates.
        connectivity : numpy.ndarray
            An array of all connectivity data.
        level_id_field : numpy.ndarray
            An array of all level data.
        fields_data : dict
            A dictionary of all field data.
        compression : str, optional
            The compression method to use for the HDF5 file.
        compression_opts : int, optional
            The compression options to use for the HDF5 file.
        """
        import h5py

        pts_chunks  = self.optimal_chunk_shape(coordinates.shape,  np.float32, target_mb=4)
        conn_chunks = self.optimal_chunk_shape(connectivity.shape, np.int32,   target_mb=4)
        lvl_chunks  = self.optimal_chunk_shape(level_id_field.shape, np.uint8, target_mb=4)
        fld_chunks  = self.optimal_chunk_shape((self.total_cells,), np.float32, target_mb=4)
        
        with h5py.File(filename + ".h5", "w") as f:
            f.create_dataset("/Mesh/Points", data=coordinates, compression=compression, compression_opts=compression_opts, chunks=pts_chunks, shuffle=True)
            f.create_dataset("/Mesh/Connectivity", data=connectivity, compression=compression, compression_opts=compression_opts, chunks=conn_chunks, shuffle=True)
            f.create_dataset("/Mesh/Level", data=level_id_field, compression=compression, compression_opts=compression_opts, chunks=lvl_chunks, shuffle=True)
            fg = f.create_group("/Fields")
            for fname, fdata in fields_data.items():
                #Convert lbm velocity to model velocity
                if "velocity" in fname.lower():
                    fdata = fdata * self.conversion
                fg.create_dataset(fname, data=fdata.astype(np.float32), compression=compression, compression_opts=compression_opts, chunks=fld_chunks, shuffle=True)

    def _merge_duplicates(self, coordinates, connectivity, levels_data):
        # Merging duplicate points
        tolerance = 0.01
        chunk_size = 10_000_000  # Adjust based on GPU memory
        num_points = coordinates.shape[0]
        unique_points = []
        mapping = np.zeros(num_points, dtype=np.int32)
        unique_idx = 0

        # Get the grid shape of computational box at the finest level from the levels_data
        num_levels = len(levels_data)
        grid_shape_finest = np.array(levels_data[-1][0].shape) * 2 ** (num_levels - 1)

        for start in range(0, num_points, chunk_size):
            end = min(start + chunk_size, num_points)
            coords_chunk = coordinates[start:end]

            # Simple hashing: grid coordinates as tuple keys
            grid_coords = np.round(coords_chunk / tolerance).astype(np.int64)
            hash_keys = grid_coords[:, 0] + grid_coords[:, 1] * grid_shape_finest[0] + grid_coords[:, 2] * grid_shape_finest[0] * grid_shape_finest[1]
            unique_hash, inverse = np.unique(hash_keys, return_inverse=True)
            unique_hash, unique_indices, inverse = np.unique(hash_keys, return_index=True, return_inverse=True)
            unique_chunk = coords_chunk[unique_indices]

            unique_points.append(unique_chunk)
            mapping[start:end] = inverse + unique_idx
            unique_idx += len(unique_hash)

        coordinates = np.concatenate(unique_points)
        connectivity = mapping[connectivity]
        return coordinates, connectivity

    def _transform_coordinates(self, coordinates, scale, offset):
        scale = np.array([scale] * 3 if isinstance(scale, (int, float)) else scale, dtype=np.float32)
        offset = np.array(offset, dtype=np.float32)
        return coordinates * scale + offset

    def _prepare_container_inputs(self):
        # load necessary modules
        from xlb.compute_backend import ComputeBackend
        from xlb.grid import grid_factory

        # Get the number of levels from the levels_data
        num_levels = len(self.levels_data)

        # Prepare lists to hold warp fields and origins allocated for each level
        field_warp_dict = {}
        origin_list = []
        for field_name, cardinality in self.field_name_cardinality_dict.items():
            field_warp_dict[field_name] = []
            for level in range(num_levels):
                # get the shape of the grid at this level
                box_shape = self.levels_data[level][0].shape

                # Use the warp backend to create dense fields to be written in multi-res NEON fields
                grid_dense = grid_factory(box_shape, compute_backend=ComputeBackend.WARP)
                field_warp_dict[field_name].append(grid_dense.create_field(cardinality=cardinality, dtype=self.store_precision))
                origin_list.append(wp.vec3i(*([int(x) for x in self.levels_data[level][2]])))

        return field_warp_dict, origin_list

    def _construct_neon_container(self):
        """
        Constructs a NEON container for exporting multi-resolution data to HDF5.
        This container will be used to transfer multi-resolution NEON fields into stacked warp fields.
        """

        @neon.Container.factory(name="HDF5MultiresExporter")
        def container(
            field_neon: Any,
            field_warp: Any,
            origin: Any,
            level: Any,
        ):
            def launcher(loader: neon.Loader):
                loader.set_mres_grid(field_neon.get_grid(), level)
                field_neon_hdl = loader.get_mres_read_handle(field_neon)
                refinement = 2**level

                @wp.func
                def kernel(index: Any):
                    cIdx = wp.neon_global_idx(field_neon_hdl, index)
                    # Get local indices by dividing the global indices (associated with the finest level) by 2^level
                    # Subtract the origin to get the local indices in the warp field
                    lx = wp.neon_get_x(cIdx) // refinement - origin[0]
                    ly = wp.neon_get_y(cIdx) // refinement - origin[1]
                    lz = wp.neon_get_z(cIdx) // refinement - origin[2]

                    # write the values to the warp field
                    cardinality = field_warp.shape[0]
                    for card in range(cardinality):
                        field_warp[card, lx, ly, lz] = self.store_dtype(wp.neon_read(field_neon_hdl, index, card))

                loader.declare_kernel(kernel)

            return launcher

        return container

    def get_fields_data(self, field_neon_dict):
        """
        Extracts and prepares the fields data from the NEON fields for export.
        """
        # Check if the field_neon_dict is empty
        if not field_neon_dict:
            return {}

        # Ensure that this operator is called on multires grids
        grid_mres = next(iter(field_neon_dict.values())).get_grid()
        assert grid_mres.name== "mGrid", f"Operation {self.__class__.__name} is only applicable to multi-resolution cases"

        for field_name in field_neon_dict.keys():
            assert field_name in self.field_name_cardinality_dict.keys(), (
                f"Field {field_name} is not provided in the instantiation of the MultiresIO class!"
            )

        # number of levels
        num_levels = grid_mres.num_levels
        assert num_levels == len(self.levels_data), "Error: Inconsistent number of levels!"

        # Prepare the fields dictionary to be written by transfering multi-res NEON fields into stacked warp fields and then numpy arrays
        fields_data = {}
        for field_name, cardinality in self.field_name_cardinality_dict.items():
            if field_name not in field_neon_dict:
                continue
            for card in range(cardinality):
                fields_data[f"{field_name}_{card}"] = []

        # Iterate over each field and level to fill the dictionary with numpy fields
        for field_name, cardinality in self.field_name_cardinality_dict.items():
            if field_name not in field_neon_dict:
                continue
            for level in range(num_levels):
                # Create the container and run it to fill the warp fields
                c = self.container(field_neon_dict[field_name], self.field_warp_dict[field_name][level], self.origin_list[level], level)
                c.run(0, container_runtime=neon.Container.ContainerRuntime.neon)

                # Ensure all operations are complete before converting to JAX and Numpy arrays
                wp.synchronize()

                # Convert the warp fields to numpy arrays and use level's mask to filter the data
                mask = self.levels_data[level][0]
                field_np = self.field_warp_dict[field_name][level].numpy()
                for card in range(cardinality):
                    field_np_card = field_np[card][mask]
                    fields_data[f"{field_name}_{card}"].append(field_np_card)

        # Concatenate all field data
        for field_name in fields_data.keys():
            fields_data[field_name] = np.concatenate(fields_data[field_name])
            assert fields_data[field_name].size == self.total_cells, f"Error: Field {field_name} size mismatch!"

        return fields_data

    def to_hdf5(self, output_filename, field_neon_dict, compression="gzip", compression_opts=0, store_precision=None):
        """
        Export the multi-resolution mesh data to an HDF5 file.
        Parameters
        ----------
        output_filename : str
            The name of the output HDF5 file (without extension).
        field_neon_dict : a dictionary of neon mGrid Fields
            Eg. The NEON fields containing velocity and density data as { "velocity": velocity_neon, "density": density_neon}
        compression : str, optional
            The compression method to use for the HDF5 file.
        compression_opts : int, optional
            The compression options to use for the HDF5 file.
        store_precision : str, optional
            The precision policy for storing data in the HDF5 file.
        """
        import time

        # Get the fields data from the NEON fields
        fields_data = self.get_fields_data(field_neon_dict)

        # Save XDMF file
        self.save_xdmf(output_filename + ".h5", output_filename + ".xmf", self.total_cells, len(self.coordinates), fields_data)

        # Writing HDF5 file
        print("\tWriting HDF5 file")
        tic_write = time.perf_counter()
        self.save_hdf5_file(output_filename, self.coordinates, self.connectivity, self.level_id_field, fields_data, compression, compression_opts)
        toc_write = time.perf_counter()
        print(f"\tHDF5 file written in {toc_write - tic_write:0.1f} seconds")

    def to_slice_image(
        self,
        output_filename,
        field_neon_dict,
        plane_point,
        plane_normal,
        slice_thickness=1.0,
        bounds=[0, 1, 0, 1],
        grid_res=512,
        cmap=None,
        component=None,
        show_axes=True,
        show_colorbar=True,
        normalize=1.0,
        output=None,
        **kwargs,
    ):
        """
        Export an arbitrary-plane slice from unstructured point data to PNG.

        Parameters
        ----------
        output_filename : str
            Output PNG filename (without extension).
        field_neon_dict : dict
            A dictionary of NEON fields containing the data to be plotted.
            Example: {"velocity": velocity_neon, "density": density_neon}
        plane_point : array_like
            A point [x, y, z] on the plane.
        plane_normal : array_like
            Plane normal vector [nx, ny, nz].
        slice_thickness : float
            How thick (in units of the coordinate system) the slice should be.
        grid_resolution : tuple
            Resolution of output image (pixels in plane u, v directions).
        grid_size : tuple
            Physical size of slice grid (width, height).
        cmap : str
            Matplotlib colormap.
        normalize : float
            Factor to scale and normalize data to ensure consistent images
        output: str
            None = png output, 'array' = no PNG and returns array of results, 'both' = png and returns array
        """
  
        # Get the fields data from the NEON fields
        assert len(field_neon_dict.keys()) == 1, "Error: This function is designed to plot a single field at a time."
        fields_data = self.get_fields_data(field_neon_dict)

        # Check if the component is within the valid range
        if component is None:
            print("\tCreating slice image of the field magnitude!") 
            cell_data = list(fields_data.values())
            squared = [comp**2 for comp in cell_data]
            cell_data = np.sqrt(sum(squared))
            field_name = list(fields_data.keys())[0].split("_")[0] + "_magnitude"
        else:
            assert component < max(self.field_name_cardinality_dict.values()), (
                f"Error: Component {component} is out of range for the provided fields."
            )
            print(f"\tCreating slice image for component {component} of the input field!")
            field_name = list(fields_data.keys())[component]
            cell_data = fields_data[field_name]
        if "velocity" in field_name.lower():
            cell_data = cell_data * self.conversion
        if normalize != 1.0:  
            cell_data = np.clip(cell_data / normalize,0,1)
        else:   
            cell_data = cell_data  
        # Plot each field in the dictionary
        result =  self._to_slice_image_single_field(
            f"{output_filename}_{field_name}",
            cell_data,
            plane_point,
            plane_normal,
            slice_thickness=slice_thickness,
            bounds=bounds,
            grid_res=grid_res,
            cmap=cmap,
            show_axes=show_axes,
            show_colorbar=show_colorbar,
            output=output,
            **kwargs,
        )
        if output == 'array':
            return result            
        elif output == 'both':
            print(f"\tSlice image for field {field_name} saved as {output_filename}.png")
            return result            
        else:            
            print(f"\tSlice image for field {field_name} saved as {output_filename}.png")
 
    def _to_slice_image_single_field(
        self,
        output_filename,
        field_data,
        plane_point,
        plane_normal,
        slice_thickness,
        bounds,
        grid_res,
        cmap,
        show_axes,
        show_colorbar,
        output,
        **kwargs,
    ):
        """
        Helper function to create a slice image for a single field.
        """
        from matplotlib import cm
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.spatial import cKDTree
        
        
        # field data are associated with the cells centers
        cell_values = field_data

        # get the normalized plane normal
        plane_normal = np.asarray(np.abs(plane_normal))
        n = plane_normal / np.linalg.norm(plane_normal)

        # Compute signed distances of each cell center to the plane
        plane_point *= plane_normal
        sdf = np.dot(self.centroids - plane_point, n)

        # Filter: cells with centroid near plane
        mask = np.abs(sdf) <= slice_thickness / 2
        if not np.any(mask):
            raise ValueError("No cells intersect the plane within thickness.")

        # Project centroids to plane
        centroids_slice = self.centroids[mask]
        sdf_slice = sdf[mask]
        proj = centroids_slice - np.outer(sdf_slice, n)

        values = cell_values[mask]

        # Build in-plane basis
        if np.allclose(n, [1, 0, 0]):
            u1 = np.array([0, 1, 0])
        else:
            u1 = np.array([1, 0, 0])
        u2 = np.abs(np.cross(n, u1))

        local_x = np.dot(proj - plane_point, u1)
        local_y = np.dot(proj - plane_point, u2)

        # Define extent of the plot
        xmin, xmax, ymin, ymax = local_x.min(), local_x.max(), local_y.min(), local_y.max()
        Lx = xmax - xmin
        Ly = ymax - ymin
        extent = np.array([xmin + bounds[0] * Lx, xmin + bounds[1] * Lx, ymin + bounds[2] * Ly, ymin + bounds[3] * Ly])
        mask_bounds = (extent[0] <= local_x) & (local_x <= extent[1]) & (extent[2] <= local_y) & (local_y <= extent[3])

        if cmap is None:
            cmap = cm.nipy_spectral

        # Adjust vertical resolution based on bounds
        bounded_x_min = local_x[mask_bounds].min()
        bounded_x_max = local_x[mask_bounds].max()
        bounded_y_min = local_y[mask_bounds].min()
        bounded_y_max = local_y[mask_bounds].max()
        width_x = bounded_x_max - bounded_x_min
        height_y = bounded_y_max - bounded_y_min        
        aspect_ratio = height_y / width_x        
        grid_resY = max(1, int(np.round(grid_res*aspect_ratio)))
        # Create grid
        grid_x = np.linspace(bounded_x_min, bounded_x_max, grid_res)
        grid_y = np.linspace(bounded_y_min, bounded_y_max, grid_resY)
        xv, yv = np.meshgrid(grid_x, grid_y, indexing="xy")
        
        # Fast KDTree-based interpolation
        points = np.column_stack((local_x[mask_bounds], local_y[mask_bounds]))
        tree = cKDTree(points)
        
        # Query points
        query_points = np.column_stack((xv.ravel(), yv.ravel()))
        
        # Find k nearest neighbors for smoother interpolation
        k = min(4, len(points))  # Use 4 neighbors or less if not enough points
        distances, indices = tree.query(query_points, k=k, workers=-1) #-1 uses all cores
        
        # Inverse distance weighting
        epsilon = 1e-10
        weights = 1.0 / (distances + epsilon)
        weights /= weights.sum(axis=1, keepdims=True)
        
        # Interpolate values
        neighbor_values = values[mask_bounds][indices]
        grid_field = (neighbor_values * weights).sum(axis=1).reshape(grid_resY, grid_res)
        
        if output == 'array':
            return grid_field
        # Plot
        if show_colorbar or show_axes:
            dpi = 300
            plt.imshow(
                grid_field,
                extent=[bounded_x_min, bounded_x_max, bounded_y_min, bounded_y_max],
                cmap=cmap,
                origin="lower",
                aspect="equal",
                **kwargs,
            )        
            if show_colorbar:
                plt.colorbar()
            if not show_axes:
                plt.axis('off')
            plt.savefig(output_filename + ".png", dpi=dpi, bbox_inches="tight", pad_inches=0)
            plt.close()
        else:
            plt.imsave(output_filename + ".png", grid_field, cmap=cmap, origin="lower")
            
        if output == 'both':
            return grid_field
            
    def _to_slice_image_single_field2(
        self,
        output_filename,
        field_data,
        plane_point,
        plane_normal,
        slice_thickness,
        bounds,
        grid_res,
        cmap,
        show_axes,
        show_colorbar,
        **kwargs,
    ):
        """
        Helper function to create a slice image for a single field.
        """
        from matplotlib import cm
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.interpolate import griddata
        from scipy.spatial import cKDTree
        tic = time.time()
        # field data are associated with the cells centers
        cell_values = field_data

        # get the normalized plane normal
        plane_normal = np.asarray(np.abs(plane_normal))
        n = plane_normal / np.linalg.norm(plane_normal)

        # Compute centroids (K = 8 for hexahedral cells)
        #cell_points = self.coordinates[self.connectivity]
        #centroids = np.mean(cell_points, axis=1)
        centroids = self.centroids

        # Compute signed distances of each cell center to the plane
        plane_point *= plane_normal
        sdf = np.dot(centroids - plane_point, n)

        # Filter: cells with centroid near plane
        mask = np.abs(sdf) <= slice_thickness / 2
        if not np.any(mask):
            raise ValueError("No cells intersect the plane within thickness.")

        # Project centroids to plane
        centroids_slice = centroids[mask]
        sdf_slice = sdf[mask]
        proj = centroids_slice - np.outer(sdf_slice, n)

        values = cell_values[mask]

        # Build in-plane basis
        if np.allclose(n, [1, 0, 0]):
            u1 = np.array([0, 1, 0])
        else:
            u1 = np.array([1, 0, 0])
        u2 = np.abs(np.cross(n, u1))

        local_x = np.dot(proj - plane_point, u1)
        local_y = np.dot(proj - plane_point, u2)

        # Define extent of the plot
        xmin, xmax, ymin, ymax = local_x.min(), local_x.max(), local_y.min(), local_y.max()
        Lx = xmax - xmin
        Ly = ymax - ymin
        extent = np.array([xmin + bounds[0] * Lx, xmin + bounds[1] * Lx, ymin + bounds[2] * Ly, ymin + bounds[3] * Ly])
        mask_bounds = (extent[0] <= local_x) & (local_x <= extent[1]) & (extent[2] <= local_y) & (local_y <= extent[3])

        if cmap is None:
            cmap = cm.nipy_spectral

        #Adjust vertical resolution based on bounds
        # Compute bounded ranges
        bounded_x_min = local_x[mask_bounds].min()
        bounded_x_max = local_x[mask_bounds].max()
        bounded_y_min = local_y[mask_bounds].min()
        bounded_y_max = local_y[mask_bounds].max()
        width_x = bounded_x_max - bounded_x_min
        height_y = bounded_y_max - bounded_y_min        
        aspect_ratio = height_y / width_x        
        grid_resY = max(1, int(np.round(grid_res*aspect_ratio)))
        print(f" ******Time to slice griddata {time.time()-tic}")
        # Rasterize: scatter cell centers to 2D grid
        grid_x = np.linspace(local_x[mask_bounds].min(), local_x[mask_bounds].max(), grid_res)
        grid_y = np.linspace(local_y[mask_bounds].min(), local_y[mask_bounds].max(), grid_resY)
        xv, yv = np.meshgrid(grid_x, grid_y, indexing="xy")

        # Linear interpolation for each grid point
        grid_field = griddata(points=(local_x, local_y), values=values, xi=(xv, yv), method="linear", fill_value=np.nan)        
        print(f" *****Time to after griddata {time.time()-tic}")
        # Plot
        if show_colorbar or show_axes:
            # Plot
            dpi = 300
            plt.imshow(
                grid_field,
                extent=[xmin, xmax, ymin, ymax],
                cmap=cmap,
                origin="lower",
                aspect="equal",
                **kwargs,
            )        
            if show_colorbar:
                plt.colorbar()
            if not show_axes:
                plt.axis('off')
            plt.savefig(output_filename + ".png", dpi=300, bbox_inches="tight", pad_inches=0)
            plt.close()
        else:
            plt.imsave(output_filename + ".png", grid_field, cmap=cmap, origin="lower")
    
    def to_line(self, 
        output_filename, 
        field_neon_dict, 
        start_point, 
        end_point, 
        resolution, 
        component=None,
        radius=1.0,
        **kwargs,):
        """
        Extract field data along a line between start_point and end_point and save to a CSV file.

        This function performs two main steps:
        1. Extracts field data from field_neon_dict, handling components or computing magnitude.
        2. Interpolates the field values along a line defined by start_point and end_point,
           then saves the results (coordinates and field values) to a CSV file.

        Parameters
        ----------
        output_filename : str
            The name of the output CSV file (without extension). Example: "velocity_profile".
        field_neon_dict : dict
            A dictionary containing the field data to extract, with a single key-value pair.
            The key is the field name (e.g., "velocity"), and the value is the NEON data object
            containing the field values. Example: {"velocity": velocity_neon}.
        start_point : array_like
            The starting point of the line in 3D space (e.g., [x0, y0, z0]).
            Units must match the coordinate system used in the class (voxel units if untransformed,
            or model units if scale/offset are applied).
        end_point : array_like
            The ending point of the line in 3D space (e.g., [x1, y1, z1]).
            Units must match the coordinate system used in the class.
        resolution : int
            The number of points along the line where the field will be interpolated.
            Example: 100 for 100 evenly spaced points.
        component : int, optional
            The specific component of the field to extract (e.g., 0 for x-component, 1 for y-component).
            If None, the magnitude of the field is computed. Default is None.
        radius : int
            The specified distance (in units of the coordinate system) to prefilter and query for line plot 

        Returns
        -------
        None
            The function writes the output to a CSV file and prints a confirmation message.

        Notes
        -----
        - The output CSV file will contain columns: 'x', 'y', 'z', and the value of the field name (e.g., 'velocity_x' or 'velocity_magnitude').
        """
        
       
        # Get the fields data from the NEON fields
        assert len(field_neon_dict.keys()) == 1, "Error: This function is designed to plot a single field at a time."
        fields_data = self.get_fields_data(field_neon_dict)
        
        # Check if the component is within the valid range
        if component is None:
            print("\tCreating csv plot of the field magnitude!")
            cell_data = list(fields_data.values())
            squared = [comp**2 for comp in cell_data]
            cell_data = np.sqrt(sum(squared))
            field_name = list(fields_data.keys())[0].split("_")[0] + "_magnitude"
            
        else:
            assert component < max(self.field_name_cardinality_dict.values()), (
                f"Error: Component {component} is out of range for the provided fields."
            )
            print(f"\tCreating csv plot for component {component} of the input field!")
            field_name = list(fields_data.keys())[component]
            cell_data = fields_data[field_name]
           
        if "velocity" in field_name.lower():
            cell_data = cell_data * self.conversion
       
        # Plot each field in the dictionary
        self._to_line_field(
            f"{output_filename}_{field_name}",
            cell_data,
            start_point,
            end_point,
            resolution,
            radius=radius,
            **kwargs,
        )
        print(f"\tLine Plot for field {field_name} saved as {output_filename}.csv")
    
    def _to_line_field(
        self, 
        output_filename, 
        cell_data, 
        start_point, 
        end_point, 
        resolution,
        radius,
        **kwargs,
        ):
        """
        Helper function to create a line plot for a single field.
        """
        import numpy as np
        
        #cell_points = self.coordinates[self.connectivity]  # Shape: (M, K, 3), where M is num cells, K is nodes per cell
        #centroids = np.mean(cell_points, axis=1)  # Shape: (M, 3)
        centroids = self.centroids
        p0 = np.array(start_point, dtype=np.float32)
        p1 = np.array(end_point, dtype=np.float32)
        
        # direction and parameter t for each centroid
        d = (p1 - p0)
        L = np.linalg.norm(d)
        d_unit = d / L
        v = centroids - p0
        t = v.dot(d_unit)
        closest = p0 + np.outer(t, d_unit)
        perp_dist = np.linalg.norm(centroids-closest, axis=1)

        # optionally mask to [0,L] or a small perp-radius
        mask = (t >= 0) & (t <= L) & (perp_dist <= radius)
        t, data = t[mask], cell_data[mask]

        # sort by t
        idx = np.argsort(t)
        t_sorted = t[idx]
        data_sorted = data[idx]

        # target samples
        t_line = np.linspace(0, L, resolution)

        # 1D linear interpolation
        vals_line = np.interp(t_line, t_sorted, data_sorted, left=np.nan, right=np.nan)

        # reconstruct (x,y,z)
        line_xyz = p0[None,:] + t_line[:,None]*d_unit[None,:]

        # vectorized CSV dump
        out = np.hstack([line_xyz, vals_line[:,None]])
        np.savetxt(
            output_filename + '.csv',
            out,
            delimiter=',',
            header='x,y,z,value',
            comments=''
        )

wp.clear_kernel_cache()
wp.config.quiet = True

# User Configuration
# =================
# Physical and simulation parameters
voxel_size = 0.002  # Finest voxel size in meters
ulb = 0.05         # Lattice velocity
u_physical = 38.0  # Physical inlet velocity in m/s (user input)
flow_passes = 3    # Domain flow passes
kinematic_viscosity = 1.508e-5  # Kinematic viscosity of air in m^2/s 1.508e-5

trim = True
trim_voxels = 5

# STL filename
stl_filename = "examples/stl/drivaer_fb_engine.stl"
script_name = "Drivaer_Fastback_2mm_grad_makemesh"

# I/O settings
print_interval_percentage = 1   # Print every 1% of iterations
file_output_crossover_percentage = 80  # Crossover at 50% of iterations
num_file_outputs_pre_crossover = 3    # Outputs before crossover
num_file_outputs_post_crossover = 10   # Outputs after crossover

# Other setup parameters
compute_backend = ComputeBackend.NEON
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)

# Choose mesher type
mesher_type = "makemesh"  # Options: "makemesh" or "cuboid"

# Mesh Generation Functions
# =========================
def generate_makemesh_mesh(stl_filename, voxel_size, trim, trim_voxels, ground_refinement_level=-1, ground_voxel_height=6):
    """
    Generate a makemesh mesh based on the provided voxel size in meters, domain multipliers, and padding values.
    """
    # Number of requested refinement levels
    num_levels = 10

    # Domain multipliers for the full domain
    domain_multiplier = {
        "-x": 2.5,
        "x": 3.5,
        "-y": 1.75,
        "y": 1.75,
        "-z": 0.0,
        "z": 4,
    }

    padding_values = [         
        #[25, 80, 30, 30, 30, 50],
        [15, 60, 15, 15, 15, 15],
        [10, 40, 10, 10, 10, 10],
        [8, 20, 8, 8, 8, 8],
        [8, 20, 8, 8, 8, 8],
        [6, 6, 6, 6, 6, 6],
        [6, 6, 6, 6, 6, 6],
        [6, 6, 6, 6, 6, 6],
        [6, 6, 6, 6, 6, 6],
        [6, 6, 6, 6, 6, 6],
        
    ]
    

    # Load the mesh
    mesh = trimesh.load_mesh(stl_filename, process=False)
    if mesh.is_empty:
        raise ValueError("Loaded mesh is empty or invalid.")

    # Compute original bounds
    min_bound = mesh.vertices.min(axis=0)
    max_bound = mesh.vertices.max(axis=0)
    partSize = max_bound - min_bound
    x0 = [max_bound[0]-0.603, min_bound[1]+(0.5*partSize[1]), min_bound[2]] #Center of wheelbase for Drivaer
    

    # Compute translation to put mesh into first octant of the domain
    shift = np.array(
        [
            domain_multiplier["-x"] * partSize[0] - min_bound[0],
            domain_multiplier["-y"] * partSize[1] - min_bound[1],
            domain_multiplier["-z"] * partSize[2] - min_bound[2],
        ],
        dtype=float,
    )
  
    # Apply translation and save out temp STL
    mesh.apply_translation(shift)
    _ = mesh.vertex_normals
    mesh.export("temp.stl")
     # Generate mesh using make_cuboid_mesh
    # Generate mesh using generate_mesh with ground refinement
    level_data, _, sparsity_pattern, level_origins = generate_mesh(
        num_levels,
        "temp.stl",
        voxel_size,
        padding_values,
        domain_multiplier,
        ground_refinement_level=ground_refinement_level,
        ground_voxel_height=ground_voxel_height,
    )

    if trim == True:
        zShift = trim_voxels
        plane_origin = np.array([0, 0, mesh.bounds[0][2]+(zShift* voxel_size)])
        plane_normal = np.array([0, 0, 1])  # Upward pointing normal
        # Slice the mesh using the defined plane.
        # With cap=True, the open slice is automatically closed off.
        mesh_above = mesh.slice_plane(plane_origin=plane_origin,
                    plane_normal=plane_normal,
                    cap=True)
        mesh_above.export('temp.stl')
        body_stl =  'temp.stl'
        mesh = trimesh.load_mesh(body_stl, process=False)
        mesh_vertices = np.asarray(mesh.vertices) / voxel_size
    else:
        mesh_vertices = np.asarray(mesh.vertices) / voxel_size
    

    actual_num_levels = len(level_data)
    grid_shape_finest = tuple([int(i * 2 ** (actual_num_levels - 1)) for i in level_data[-1][0].shape])
    print(f"Requested levels: {num_levels}, Actual levels: {actual_num_levels}")
    print(f"Full shape based on finest voxel size is {grid_shape_finest}")
    #os.remove("temp.stl")

    return level_data, mesh_vertices, tuple([int(a) for a in grid_shape_finest]), partSize, actual_num_levels, shift, sparsity_pattern, level_origins, x0

def generate_cuboid_mesh(stl_filename, voxel_size, trim, trim_voxels):
    """
    Alternative cuboid mesh generation based on Apolo's method with domain multipliers per level.
    """
    # Domain multipliers for each refinement level
    #domain_multiplier = [
     #   [3.0, 4.0,  2.5,  2.5,  0.0, 4.0],  # -x, x, -y, y, -z, z0.17361
     #   [1.2, 1.25, 1.75, 1.75, 0.0, 1.5],  # -x, x, -y, y, -z, z
     #   [0.8, 1.0,  1.25, 1.25, 0.0, 1.2],  # -x, x, -y, y, -z, z
     #   [0.4, 0.4,  0.25, 0.25, 0.0, 0.25],
        
    #]

    domain_multiplier = [
        [2.5,  3.5,  1.5,  1.5,  0.0, 3.7],  # -x, x, -y, y, -z, z
        [1.8,  1.6, 1.2,  1.2 , 0.0, 2.0],  # -x, x, -y, y, -z, z
        [1.4,  1.25, 1.0, 1.0, 0.0, 1.6],  # -x, x, -y, y, -z, z
        [0.8,  1.0,  0.6, 0.6, 0.0, 1.2],
        #[0.4, 0.4,  0.25, 0.25, 0.0, 0.25],  # -x, x, -y, y, -z, z
        [0.55,  0.65, 0.65,  0.65, 0.0, 0.65],
        [0.25, 0.25, 0.22, 0.22, 0.0, 0.25],
        
    ]

    # Load the mesh
    mesh = trimesh.load_mesh(stl_filename, process=False)
    if mesh.is_empty:
        raise ValueError("Loaded mesh is empty or invalid.")

    # Compute original bounds
    min_bound = mesh.vertices.min(axis=0)
    max_bound = mesh.vertices.max(axis=0)
    partSize = max_bound - min_bound
    x0 = [max_bound[0]-0.603, min_bound[1]+(0.5*partSize[1]), min_bound[2]] #Center of wheelbase for Drivaer
    # Compute translation to put mesh into first octant of the domain
    shift = np.array(
        [
            domain_multiplier[0][0] * partSize[0] - min_bound[0],
            domain_multiplier[0][2] * partSize[1] - min_bound[1],
            domain_multiplier[0][4] * partSize[2] - min_bound[2],
        ],
        dtype=float,
    )

    # Apply translation and save out temp STL
    mesh.apply_translation(shift)
    _ = mesh.vertex_normals
    mesh.export("temp.stl")
     # Generate mesh using make_cuboid_mesh
    level_data, sparsity_pattern, level_origins = make_cuboid_mesh(
        voxel_size,
        domain_multiplier,
        "temp.stl",
    )
    if trim == True:
        zShift = trim_voxels
        plane_origin = np.array([0, 0, mesh.bounds[0][2]+(zShift* voxel_size)])
        plane_normal = np.array([0, 0, 1])  # Upward pointing normal
        # Slice the mesh using the defined plane.
        # With cap=True, the open slice is automatically closed off.
        mesh_above = mesh.slice_plane(plane_origin=plane_origin,
                    plane_normal=plane_normal,
                    cap=True)
        mesh_above.export('temp.stl')
        body_stl =  'temp.stl'
        mesh = trimesh.load_mesh(body_stl, process=False)
        mesh_vertices = np.asarray(mesh.vertices) / voxel_size
    else:
        mesh_vertices = np.asarray(mesh.vertices) / voxel_size
        
    

   
    actual_num_levels = len(level_data)
    grid_shape_finest = tuple([int(i * 2 ** (actual_num_levels - 1)) for i in level_data[-1][0].shape])
    print(f"Requested levels: {len(domain_multiplier)}, Actual levels: {actual_num_levels}")
    print(f"Full shape based on finest voxel size is {grid_shape_finest}")
    #os.remove("temp.stl")

    return level_data, mesh_vertices, tuple([int(a) for a in grid_shape_finest]), partSize, actual_num_levels, shift, sparsity_pattern, level_origins, x0

# Boundary Conditions Setup
# =========================
def setup_boundary_conditions(grid, level_data, body_vertices, ulb, compute_backend=ComputeBackend.NEON):
    """
    Set up boundary conditions for the simulation.
    """
    num_levels = len(level_data)
    coarsest_level = num_levels - 1
    box = grid.bounding_box_indices(shape=grid.level_to_shape(coarsest_level))
    left_indices = grid.boundary_indices_across_levels(level_data, box_side="left", remove_edges=True)
    right_indices = grid.boundary_indices_across_levels(level_data, box_side="right", remove_edges=True)
    top_indices = grid.boundary_indices_across_levels(level_data, box_side="top", remove_edges=False)
    bottom_indices = grid.boundary_indices_across_levels(level_data, box_side="bottom", remove_edges=False)
    front_indices = grid.boundary_indices_across_levels(level_data, box_side="front", remove_edges=False)
    back_indices = grid.boundary_indices_across_levels(level_data, box_side="back", remove_edges=False)

    # Filter front and back indices to remove overlaps with top and bottom at each level
    filtered_front_indices = []
    filtered_back_indices = []
    filtered_top_indices = []
    filtered_bottom_indices = []
    for level in range(num_levels):
        left_set = set(zip(*left_indices[level])) if left_indices[level] else set()
        right_set = set(zip(*right_indices[level])) if right_indices[level] else set()
        top_set = set(zip(*top_indices[level])) if top_indices[level] else set()
        bottom_set = set(zip(*bottom_indices[level])) if bottom_indices[level] else set()
        front_set = set(zip(*front_indices[level])) if front_indices[level] else set()
        back_set = set(zip(*back_indices[level])) if back_indices[level] else set()
        filtered_front_set = front_set - (top_set | bottom_set )
        filtered_back_set = back_set - (top_set | bottom_set )
        filtered_top_set = top_set - (left_set | right_set)
        filtered_bottom_set = bottom_set - (left_set | right_set)
        filtered_front_indices.append(
            [list(coords) for coords in zip(*filtered_front_set)] if filtered_front_set else []
        )
        filtered_back_indices.append(
            [list(coords) for coords in zip(*filtered_back_set)] if filtered_back_set else []
        )
        filtered_top_indices.append(
            [list(coords) for coords in zip(*filtered_top_set)] if filtered_top_set else []
        )
        filtered_bottom_indices.append(
            [list(coords) for coords in zip(*filtered_bottom_set)] if filtered_bottom_set else []
        )

    # Turbulent Flow Profile
    def bc_profile_taper(taper_fraction=0.07):
        assert compute_backend == ComputeBackend.NEON
        _, ny, nz = grid_shape_zip
        dtype = precision_policy.compute_precision.wp_dtype
        H_y = dtype(ny-1)
        H_z = dtype(nz-1)
        two = dtype(2.0)
        ulb_wp = dtype(ulb)
        taper_frac = dtype(taper_fraction)
        core_frac = dtype(1.0 - 2.0 * taper_fraction)
        _u_vec = wp.vec(velocity_set.d,dtype=dtype)

        @wp.func
        def bc_profile_warp(index: wp.vec3i):
            y = dtype(index[1])
            z = dtype(index[2])
            y_center = wp.abs(y - (H_y / two))
            z_center = wp.abs(z - (H_z / two))
            y_norm = two * y_center / H_y
            z_norm = two * z_center / H_z
            max_norm = wp.max(y_norm, z_norm)
            velocity = ulb_wp
            if max_norm > core_frac:
                velocity = ulb_wp * (dtype(1.0) - (max_norm - core_frac) / taper_frac)
            velocity = wp.max(dtype(0.0), velocity)
            return wp.vec(velocity, length=1)

        return bc_profile_warp

    # Initialize boundary conditions


    #bc_inlet = HybridBC(
    #    bc_method="nonequilibrium_regularized",        
    #    prescribed_value=(ulb, 0.0, 0.0),
    #    indices=left_indices
    #)

    bc_inlet = RegularizedBC(
        "velocity",
        #profile=bc_profile_taper(),
        prescribed_value=(ulb, 0.0, 0.0),
       indices=left_indices,
    )

    bc_outlet = DoNothingBC(indices=right_indices)

    #bc_top = FullwayBounceBackBC(indices=top_indices)
    bc_top = HybridBC(bc_method="nonequilibrium_regularized",prescribed_value=(ulb, 0.0, 0.0),indices=top_indices)
    
    bc_bottom = HybridBC(bc_method="nonequilibrium_regularized",prescribed_value=(ulb, 0.0, 0.0),indices=bottom_indices)
    #bc_bottom = FullwayBounceBackBC(indices=bottom_indices)
    #bc_front = FullwayBounceBackBC(indices=filtered_front_indices)
    bc_front = HybridBC(bc_method="nonequilibrium_regularized",prescribed_value=(ulb, 0.0, 0.0),indices=filtered_front_indices)
    #bc_back = FullwayBounceBackBC(indices=filtered_back_indices)
    bc_back = HybridBC(bc_method="nonequilibrium_regularized",prescribed_value=(ulb, 0.0, 0.0),indices=filtered_back_indices)

    bc_body = HybridBC(
        bc_method="bounceback_grads",
        mesh_vertices=body_vertices,
        voxelization_method=MeshVoxelizationMethod("AABB_CLOSE", close_voxels=3),
        use_mesh_distance=True,
    )

    return [bc_top, bc_bottom, bc_front, bc_back, bc_inlet, bc_outlet, bc_body] # Body must be last. Outlet must be second to last
    # return [bc_walls, bc_inlet, bc_outlet, bc_body]


# Simulation Initialization
# =========================
def initialize_simulation(grid, boundary_conditions, omega, initializer, collision_type="KBC", mres_perf_opt=xlb.MresPerfOptimizationType.FUSION_AT_FINEST):
    """
    Initialize the multiresolution simulation manager.
    """
    sim = xlb.helper.MultiresSimulationManager(
        omega=omega,
        grid=grid,
        boundary_conditions=boundary_conditions,
        collision_type=collision_type,
        initializer=initializer,
        mres_perf_opt=mres_perf_opt,
    )
    return sim

# Utility Functions
# =================
def print_lift_drag(sim, step, momentum_transfer, ulb, reference_area, voxel_size):
    """
    Calculate and print lift and drag coefficients.
    """
    boundary_force = momentum_transfer(sim.f_0, sim.f_1, sim.bc_mask, sim.missing_mask)
    drag = boundary_force[0]
    lift = boundary_force[2]
    cd = 2.0 * drag / (ulb**2 * reference_area)
    cl = 2.0 * lift / (ulb**2 * reference_area)
    if np.isnan(cd) or np.isnan(cl):
        raise ValueError(f"NaN detected in coefficients at step {step}: Cd={cd}, Cl={cl}")
    drag_values.append([cd, cl])
    # print(f"CD={cd:.3f}, CL={cl:.3f}, Drag Force (lattice units)={drag:.6f}")
    return cd, cl, drag

def plot_drag_lift(drag_values, output_dir, print_interval, script_name, percentile_range=(15, 85), use_log_scale=False):
    """
    Plot CD and CL over time and save the plot to the output directory.
    """
    drag_values_array = np.array(drag_values)
    steps = np.arange(0, len(drag_values) * print_interval, print_interval)
    cd_values = drag_values_array[:, 0]
    cl_values = drag_values_array[:, 1]
    y_min = np.percentile(cd_values, percentile_range[0])
    y_max = np.percentile(cd_values, percentile_range[1])
    padding = (y_max - y_min) * 0.1
    y_min, y_max = y_min - padding, y_max + padding
    if use_log_scale:
        y_min = max(y_min, 1e-6)
    plt.figure(figsize=(10, 6))
    plt.plot(steps, cd_values, label='Drag Coefficient (Cd)', color='blue')
    plt.xlabel('Simulation Step')
    plt.ylabel('Coefficient')
    plt.title(f'{script_name}: Drag Coefficients Over Time')
    plt.legend()
    plt.grid(True)
    plt.ylim(y_min, y_max)
    if use_log_scale:
        plt.yscale('log')
    plt.savefig(os.path.join(output_dir, 'drag_lift_plot.png'))
    plt.close()
    

def compute_voxel_statistics_and_reference_area(sim, bc_mask_exporter, level_data, actual_num_levels, sparsity_pattern, boundary_conditions, voxel_size):
    """
    Compute active/solid voxels, totals, lattice updates, and reference area based on simulation data.
    """
    # Compute macro fields
    sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
    fields_data = bc_mask_exporter.get_fields_data({"bc_mask": sim.bc_mask})
    bc_mask_data = fields_data["bc_mask_0"]
    level_id_field = bc_mask_exporter.level_id_field

    # Compute solid voxels per level (assuming 255 is the solid marker)
    solid_voxels = []
    for lvl in range(actual_num_levels):
        level_mask = level_id_field == lvl
        solid_voxels.append(np.sum(bc_mask_data[level_mask] == 255))

    # Compute active voxels (total non-zero in sparsity minus solids)
    active_voxels = [np.count_nonzero(mask) for mask in sparsity_pattern]
    active_voxels = [max(0, active_voxels[lvl] - solid_voxels[lvl]) for lvl in range(actual_num_levels)]

    # Totals
    total_voxels = sum(active_voxels)
    total_lattice_updates_per_step = sum(active_voxels[lvl] * (2 ** (actual_num_levels - 1 - lvl)) for lvl in range(actual_num_levels))

    # Compute reference area (projected on YZ plane at finest level)
    finest_level = 0
    mask_finest = level_id_field == finest_level
    bc_mask_finest = bc_mask_data[mask_finest]
    active_indices_finest = np.argwhere(level_data[0][0])
    bc_body_id = boundary_conditions[-1].id  # Assuming last BC is bc_body
    solid_voxels_indices = active_indices_finest[bc_mask_finest == bc_body_id]
    unique_jk = np.unique(solid_voxels_indices[:, 1:3], axis=0)
    reference_area = unique_jk.shape[0]
    reference_area_physical = reference_area * (voxel_size ** 2)

    return {
        "active_voxels": active_voxels,
        "solid_voxels": solid_voxels,
        "total_voxels": total_voxels,
        "total_lattice_updates_per_step": total_lattice_updates_per_step,
        "reference_area": reference_area,
        "reference_area_physical": reference_area_physical
    }


def plot_data(x0, output_dir, delta_x_coarse, sim, IOexporter, prefix='Drivaer_Fastback'):
    '''       
        Drivear Car Model
        https://repository.lboro.ac.uk/articles/dataset/DrivAer_Experimental_Aerodynamic_Dataset/12881213
        Profiles on symmetry plane (y=0) covering entire field
        Origin of coordinate system: 
            x=0: center of the car, y=0: symmetry plane, z=0: ground plane
            
        Coordaintes in meters 
        Velocity data in m/s 
                                       
        Key is Xlocation
        Value X is vx
        Value Y is z
        '''
    
    def _load_sim_line(csv_path):
        """
        Read a CSV exported by IOexporter.to_line without pandas.
        Returns (z, Ux).
        """
        # Read with header as column names
        data = np.genfromtxt(
            csv_path,
            delimiter=',',
            names=True,         # use header
            autostrip=True,
            dtype=None,         # let numpy infer dtypes
            encoding='utf-8'    # handle any non-ascii names
        )
        if data.size == 0:
            raise ValueError(f"No data in {csv_path}")

        names = data.dtype.names
        lower = {n: n.lower() for n in names}

        # Find z-like column (fallback: first column)
        z_candidates = [
            n for n in names
            if lower[n] == 'z'
            or lower[n] in ('s', 'distance', 'arc_length', 'arclength')
            or 'z' == lower[n].split('_')[-1]
        ]
        z_name = z_candidates[0] if z_candidates else names[0]

        # Find velocity-x column (fallback: last column)
        vel_candidates = [n for n in names if any(k in lower[n] for k in ('value', 'u', 'velocity'))]
        # Prefer an x-component if present (common patterns after numpy sanitizes names)
        vel_x_pref = [n for n in vel_candidates if any(k in lower[n] for k in ('x', '_0', '0'))]
        vel_name = vel_x_pref[0] if vel_x_pref else (vel_candidates[0] if vel_candidates else names[-1])

        z = np.asarray(data[z_name], dtype=float)
        ux = np.asarray(data[vel_name], dtype=float)
        return z, ux
    
    testData = { 
         '-0.781' : { 'x' : [0,38.69,38.88,38.52,38.77,38.52,38.38,38.38,38.52,38.62,38.32,38.4,38.36,37.74,37.69,38,38.14,38.27,38.57,38.7,38.78,38.73,38.53,38.04,37.68,37.84,37.42,37.74,37.83,37.91,38.19,38.03,38.1,38.02,37.82,37.78,37.75,38.45,38.23,37.2,37.12,38.45,38.24,37.66,38.23,38.22,37.56,36.48,37.27,37.36,37.92,37.74,37.78,37.56,37.05,36.83,37.29,37.08,37.2,36.96,36.5,36.6,36.28,36.51,36.48,36.3,35.47,36.33,36.68,37.33,35.64,35.99,34.22,34.96,36.35,36.36,36.36,36.62,36.61,36.25,36.62,35.95,35.82,35.89,35.76,36.49,35.92,35.93,35.29,35.53,36.28,35.64,35.43,35.34,34.74,35.46,36.21,35.57,35.5,36.77,35.58,36.84,37.26,35.79,35.1,35.51,35.06,35,35.56,36.56,37.21,35.19,35.36,35.42,35.82,35.01,35.39,34.09,35.42,35.05,34.69,34.93,34.09,33.77,34.17,33.86,35.67,34.73,34.54,33.34,34.54,33.99,34.96,35.28,34.85,35.63,35.3,35.4,35.31,36.33,34.85,35.2,34.94,34.85,34.65,34.47,33.75,32.88,33.09,34.19,33.23,33.33,33.52,33.29,32.43,31.72,31.78,33.61,34.13,33.8,32.92,32.64,32.42,29.84,29.66,28.73,29.5,27.54,28.6,26.47,25.18,25.09,23.95,22.08,0,0,0,0,0,0], 'y' : [0.322,0.32,0.319,0.317,0.315,0.314,0.312,0.31,0.308,0.307,0.305,0.303,0.302,0.3,0.298,0.297,0.295,0.293,0.291,0.29,0.288,0.286,0.285,0.283,0.281,0.279,0.278,0.276,0.274,0.273,0.271,0.269,0.268,0.266,0.264,0.262,0.261,0.259,0.257,0.256,0.254,0.252,0.251,0.249,0.247,0.245,0.244,0.242,0.24,0.239,0.237,0.235,0.234,0.232,0.23,0.228,0.227,0.225,0.223,0.222,0.22,0.218,0.216,0.215,0.213,0.211,0.21,0.208,0.206,0.205,0.203,0.201,0.199,0.198,0.196,0.194,0.193,0.191,0.189,0.188,0.186,0.184,0.182,0.181,0.179,0.177,0.176,0.174,0.172,0.17,0.169,0.167,0.165,0.164,0.162,0.16,0.159,0.157,0.155,0.153,0.152,0.15,0.148,0.147,0.145,0.143,0.142,0.14,0.138,0.136,0.135,0.133,0.131,0.13,0.128,0.126,0.125,0.123,0.121,0.119,0.118,0.116,0.114,0.113,0.111,0.109,0.107,0.106,0.104,0.102,0.101,0.099,0.097,0.096,0.094,0.092,0.09,0.089,0.087,0.085,0.084,0.082,0.08,0.079,0.077,0.075,0.073,0.072,0.07,0.068,0.067,0.065,0.063,0.061,0.06,0.058,0.056,0.055,0.053,0.051,0.05,0.048,0.046,0.044,0.043,0.041,0.039,0.038,0.036,0.034,0.033,0.031,0.029,0.027,0.026,0.024,0.022,0.021,0.019,0.017]},
         '-0.614' : { 'x' : [0,38.37,38.32,38.21,38.4,38.15,38.12,38.18,38.06,38.09,38.07,38.07,37.73,37.33,37.33,37.65,37.59,37.46,37.42,37.47,37.53,37.41,37.14,37.21,37.18,37.24,37.17,37.03,36.94,36.88,36.8,36.57,36.73,36.65,36.4,36.13,35.88,35.76,35.91,36.02,35.94,35.7,35.64,35.45,35.47,35.4,35.38,35.35,35.39,35.18,34.88,35.11,34.86,34.68,34.53,34.53,34.58,34.33,33.94,33.68,33.77,33.61,33.62,33.48,33.29,33.28,33.13,33.21,33.12,32.82,32.35,32.32,32.22,31.65,31.58,31.6,31.59,31.49,31.37,30.98,30.88,30.51,30.53,30.32,30.26,29.86,29.89,29.4,29.47,29.19,28.68,28.89,28.72,28.48,28.67,28.21,28.06,27.84,27.57,27.27,27.16,27.2,27.34,26.38,26.34,26.08,26.65,26,25.89,26.11,25.49,25.94,25.92,26.07,25.69,25.83,25.37,25.17,25.28,25.54,25.04,24.4,24.41,24.47,24.72,24.94,24.53,25.11,24.17,24.7,24.93,24.95,23.83,25.55,25.19,24.93,25.03,24.96,24.37,24.63,25.09,25.21,25.39,25.7,25.68,25.78,25.07,25.23,25.31,24.38,24.81,24.74,24.49,24.7,24.95,25.47,26.38,26.54,25.68,25.22,25.71,25.05,24.82,25.4,25.97,27.11,26.61,25.06,24.13,21.25,21,20.64,20.02,21.65,0,0,0,0,0,0], 'y' : [0.322,0.32,0.319,0.317,0.315,0.314,0.312,0.31,0.308,0.307,0.305,0.303,0.302,0.3,0.298,0.297,0.295,0.293,0.291,0.29,0.288,0.286,0.285,0.283,0.281,0.279,0.278,0.276,0.274,0.273,0.271,0.269,0.268,0.266,0.264,0.262,0.261,0.259,0.257,0.256,0.254,0.252,0.251,0.249,0.247,0.245,0.244,0.242,0.24,0.239,0.237,0.235,0.234,0.232,0.23,0.228,0.227,0.225,0.223,0.222,0.22,0.218,0.216,0.215,0.213,0.211,0.21,0.208,0.206,0.205,0.203,0.201,0.199,0.198,0.196,0.194,0.193,0.191,0.189,0.188,0.186,0.184,0.182,0.181,0.179,0.177,0.176,0.174,0.172,0.17,0.169,0.167,0.165,0.164,0.162,0.16,0.159,0.157,0.155,0.153,0.152,0.15,0.148,0.147,0.145,0.143,0.142,0.14,0.138,0.136,0.135,0.133,0.131,0.13,0.128,0.126,0.125,0.123,0.121,0.119,0.118,0.116,0.114,0.113,0.111,0.109,0.107,0.106,0.104,0.102,0.101,0.099,0.097,0.096,0.094,0.092,0.09,0.089,0.087,0.085,0.084,0.082,0.08,0.079,0.077,0.075,0.073,0.072,0.07,0.068,0.067,0.065,0.063,0.061,0.06,0.058,0.056,0.055,0.053,0.051,0.05,0.048,0.046,0.044,0.043,0.041,0.039,0.038,0.036,0.034,0.033,0.031,0.029,0.027,0.026,0.024,0.022,0.021,0.019,0.017]},
         '-0.521' : { 'x' : [0,40.11,39.99,39.96,40,40.26,40.07,39.93,39.86,40.14,40.18,39.82,39.77,39.87,40,39.85,39.9,39.93,39.91,40.29,40.14,40.19,40.12,40.08,40.16,40.14,39.86,39.98,39.9,40.03,40.02,40.07,40.22,40.05,40.13,39.87,39.96,39.8,39.79,39.75,40.04,40.04,40.16,40.13,40.06,40,40.03,40.14,40.18,40.24,40.39,40.38,40.11,40.18,40.12,40.1,40.25,40.46,40.25,40.37,40.28,40.15,40.39,40.67,40.78,40.76,40.93,40.95,40.71,40.79,41,41.11,41.19,41.38,41.44,41.6,41.7,41.51,40.83,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'y' : [0.322,0.32,0.319,0.317,0.315,0.314,0.312,0.31,0.308,0.307,0.305,0.303,0.302,0.3,0.298,0.297,0.295,0.293,0.291,0.29,0.288,0.286,0.285,0.283,0.281,0.279,0.278,0.276,0.274,0.273,0.271,0.269,0.268,0.266,0.264,0.262,0.261,0.259,0.257,0.256,0.254,0.252,0.251,0.249,0.247,0.245,0.244,0.242,0.24,0.239,0.237,0.235,0.234,0.232,0.23,0.228,0.227,0.225,0.223,0.222,0.22,0.218,0.216,0.215,0.213,0.211,0.21,0.208,0.206,0.205,0.203,0.201,0.199,0.198,0.196,0.194,0.193,0.191,0.189,0.188,0.186,0.184,0.182,0.181,0.179,0.177,0.176,0.174,0.172,0.17,0.169,0.167,0.165,0.164,0.162,0.16,0.159,0.157,0.155,0.153,0.152,0.15,0.148,0.147,0.145,0.143,0.142,0.14,0.138,0.136,0.135,0.133,0.131,0.13,0.128,0.126,0.125,0.123,0.121,0.119,0.118,0.116,0.114,0.113,0.111,0.109,0.107,0.106,0.104,0.102,0.101,0.099,0.097,0.096,0.094,0.092,0.09,0.089,0.087,0.085,0.084,0.082,0.08,0.079,0.077,0.075,0.073,0.072,0.07,0.068,0.067,0.065,0.063,0.061,0.06,0.058,0.056,0.055,0.053,0.051,0.05,0.048,0.046,0.044,0.043,0.041,0.039,0.038,0.036,0.034,0.033,0.031,0.029,0.027,0.026,0.024,0.022,0.021,0.019,0.017]},
         '-0.48' : { 'x' : [0,40.29,40.5,40.7,40.7,40.86,41.1,41.04,40.77,40.8,41.01,41.18,41.2,41.07,40.9,40.97,41.34,41.57,41.52,41.41,41.77,41.59,41.5,41.47,41.6,41.81,41.65,41.66,41.53,41.69,41.89,41.98,41.92,42.11,42.16,42.33,42.35,42.21,42.18,42.61,42.46,42.59,42.82,43.05,43.14,43.17,43.05,43.27,43.48,43.22,43.59,43.79,43.93,43.88,43.97,44.49,44.5,44.34,44.37,44.71,44.91,45.25,45.41,45.68,45.81,46.08,46.36,45.88,43.96,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'y' : [0.322,0.32,0.319,0.317,0.315,0.314,0.312,0.31,0.308,0.307,0.305,0.303,0.302,0.3,0.298,0.297,0.295,0.293,0.291,0.29,0.288,0.286,0.285,0.283,0.281,0.279,0.278,0.276,0.274,0.273,0.271,0.269,0.268,0.266,0.264,0.262,0.261,0.259,0.257,0.256,0.254,0.252,0.251,0.249,0.247,0.245,0.244,0.242,0.24,0.239,0.237,0.235,0.234,0.232,0.23,0.228,0.227,0.225,0.223,0.222,0.22,0.218,0.216,0.215,0.213,0.211,0.21,0.208,0.206,0.205,0.203,0.201,0.199,0.198,0.196,0.194,0.193,0.191,0.189,0.188,0.186,0.184,0.182,0.181,0.179,0.177,0.176,0.174,0.172,0.17,0.169,0.167,0.165,0.164,0.162,0.16,0.159,0.157,0.155,0.153,0.152,0.15,0.148,0.147,0.145,0.143,0.142,0.14,0.138,0.136,0.135,0.133,0.131,0.13,0.128,0.126,0.125,0.123,0.121,0.119,0.118,0.116,0.114,0.113,0.111,0.109,0.107,0.106,0.104,0.102,0.101,0.099,0.097,0.096,0.094,0.092,0.09,0.089,0.087,0.085,0.084,0.082,0.08,0.079,0.077,0.075,0.073,0.072,0.07,0.068,0.067,0.065,0.063,0.061,0.06,0.058,0.056,0.055,0.053,0.051,0.05,0.048,0.046,0.044,0.043,0.041,0.039,0.038,0.036,0.034,0.033,0.031,0.029,0.027,0.026,0.024,0.022,0.021,0.019,0.017]},
         '-0.435' : { 'x' : [0,41.85,41.96,41.94,41.55,41.8,42.07,41.44,41.43,41.63,41.85,41.89,41.55,42.02,42.03,42.33,42.15,41.94,41.88,42.13,42.34,42.17,42.44,42.53,42.72,42.54,42.81,42.82,42.87,43.03,42.94,43.23,43.52,43.53,43.52,43.23,43.32,43.45,43.53,43.62,43.7,43.66,43.94,43.98,44.24,44.24,44.26,44.28,44.46,44.64,44.5,44.16,44.58,44.8,44.68,44.96,45.38,45.62,45.62,45.45,43.99,39.58,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'y' : [0.322,0.32,0.319,0.317,0.315,0.314,0.312,0.31,0.308,0.307,0.305,0.303,0.302,0.3,0.298,0.297,0.295,0.293,0.291,0.29,0.288,0.286,0.285,0.283,0.281,0.279,0.278,0.276,0.274,0.273,0.271,0.269,0.268,0.266,0.264,0.262,0.261,0.259,0.257,0.256,0.254,0.252,0.251,0.249,0.247,0.245,0.244,0.242,0.24,0.239,0.237,0.235,0.234,0.232,0.23,0.228,0.227,0.225,0.223,0.222,0.22,0.218,0.216,0.215,0.213,0.211,0.21,0.208,0.206,0.205,0.203,0.201,0.199,0.198,0.196,0.194,0.193,0.191,0.189,0.188,0.186,0.184,0.182,0.181,0.179,0.177,0.176,0.174,0.172,0.17,0.169,0.167,0.165,0.164,0.162,0.16,0.159,0.157,0.155,0.153,0.152,0.15,0.148,0.147,0.145,0.143,0.142,0.14,0.138,0.136,0.135,0.133,0.131,0.13,0.128,0.126,0.125,0.123,0.121,0.119,0.118,0.116,0.114,0.113,0.111,0.109,0.107,0.106,0.104,0.102,0.101,0.099,0.097,0.096,0.094,0.092,0.09,0.089,0.087,0.085,0.084,0.082,0.08,0.079,0.077,0.075,0.073,0.072,0.07,0.068,0.067,0.065,0.063,0.061,0.06,0.058,0.056,0.055,0.053,0.051,0.05,0.048,0.046,0.044,0.043,0.041,0.039,0.038,0.036,0.034,0.033,0.031,0.029,0.027,0.026,0.024,0.022,0.021,0.019,0.017]},
         '0.251' : { 'x' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6.74,34.57,38.8,41.9,44.27,45.66,46.29,46.68,46.86,46.92,46.92,46.94,47.09,47.08,47.09,47.13,47.19,47.15,47.16,47.21,47.22,47.21,47.28,47.21,47.19,47.22,37.85,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'y' : [0.003,0.005,0.007,0.008,0.01,0.011,0.013,0.015,0.016,0.018,0.019,0.021,0.023,0.024,0.026,0.028,0.029,0.031,0.032,0.034,0.036,0.037,0.039,0.04,0.042,0.044,0.045,0.047,0.049,0.05,0.052,0.053,0.055,0.057,0.058,0.06,0.061,0.063,0.065,0.066,0.068,0.07,0.071,0.073,0.074,0.076,0.078,0.079,0.081,0.083,0.084,0.086,0.087,0.089,0.091,0.092,0.094,0.095,0.097,0.099,0.1,0.102,0.104,0.105,0.107,0.108,0.11,0.112,0.113,0.115,0.116,0.118,0.12,0.121,0.123,0.125,0.126,0.128,0.129,0.131,0.133,0.134,0.136,0.138,0.139,0.141,0.142,0.144,0.146,0.147,0.149,0.15,0.152,0.154,0.155,0.157,0.159,0.16,0.162,0.163,0.165,0.167,0.168,0.17,0.171,0.173,0.175,0.176,0.178,0.18,0.181,0.183,0.184,0.186,0.188,0.189,0.191,0.193,0.194,0.196,0.197,0.199,0.201,0.202,0.204,0.205,0.207,0.209,0.21,0.212,0.214,0.215,0.217,0.218,0.22,0.222,0.223,0.225,0.226,0.228,0.23,0.231,0.233,0.235,0.236,0.238,0.239,0.241,0.243,0.244,0.246,0.247,0.249,0.251,0.252,0.254,0.256,0.257,0.259,0.26,0.262,0.264,0.265,0.267,0.269,0.27,0.272,0.273,0.275,0.277,0.278,0.28,0.281,0.283,0.285,0.286,0.288,0.29,0.291,0.293,0.294,0.296,0.298,0.299,0.301,0.302,0.304,0.306,0.307,0.309,0.311,0.312,0.314,0.315,0.317,0.319,0.32,0.322,0.324,0.325,0.327,0.328,0.33,0.332,0.333,0.335,0.336,0.338,0.34,0.341,0.343,0.345,0.346,0.348,0.349,0.351,0.353,0.354,0.356,0.357,0.359,0.361,0.362,0.364,0.366,0.367,0.369,0.37,0.372,0.374,0.375,0.377,0.378,0.38,0.382,0.383,0.385,0.387,0.388,0.39,0.391,0.393,0.395,0.396,0.398,0.4,0.401,0.403,0.404,0.406,0.408,0.409,0.411,0.412,0.414,0.416,0.417,0.419,0.421,0.422,0.424,0.425,0.427,0.429,0.43]},
         '0.358' : { 'x' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3.94,20.73,29.28,32.38,35.13,37.59,39.75,41.55,42.9,43.76,44.35,44.66,44.83,44.96,45.03,45.08,45.12,45.17,45.23,45.25,45.25,45.29,45.31,45.32,45.35,45.37,45.4,45.4,45.42,45.42,45.43,45.43,45.42,45.41,45.42,45.42,45.41,45.42,36.4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'y' : [0.003,0.005,0.007,0.008,0.01,0.011,0.013,0.015,0.016,0.018,0.019,0.021,0.023,0.024,0.026,0.028,0.029,0.031,0.032,0.034,0.036,0.037,0.039,0.04,0.042,0.044,0.045,0.047,0.049,0.05,0.052,0.053,0.055,0.057,0.058,0.06,0.061,0.063,0.065,0.066,0.068,0.07,0.071,0.073,0.074,0.076,0.078,0.079,0.081,0.083,0.084,0.086,0.087,0.089,0.091,0.092,0.094,0.095,0.097,0.099,0.1,0.102,0.104,0.105,0.107,0.108,0.11,0.112,0.113,0.115,0.116,0.118,0.12,0.121,0.123,0.125,0.126,0.128,0.129,0.131,0.133,0.134,0.136,0.138,0.139,0.141,0.142,0.144,0.146,0.147,0.149,0.15,0.152,0.154,0.155,0.157,0.159,0.16,0.162,0.163,0.165,0.167,0.168,0.17,0.171,0.173,0.175,0.176,0.178,0.18,0.181,0.183,0.184,0.186,0.188,0.189,0.191,0.193,0.194,0.196,0.197,0.199,0.201,0.202,0.204,0.205,0.207,0.209,0.21,0.212,0.214,0.215,0.217,0.218,0.22,0.222,0.223,0.225,0.226,0.228,0.23,0.231,0.233,0.235,0.236,0.238,0.239,0.241,0.243,0.244,0.246,0.247,0.249,0.251,0.252,0.254,0.256,0.257,0.259,0.26,0.262,0.264,0.265,0.267,0.269,0.27,0.272,0.273,0.275,0.277,0.278,0.28,0.281,0.283,0.285,0.286,0.288,0.29,0.291,0.293,0.294,0.296,0.298,0.299,0.301,0.302,0.304,0.306,0.307,0.309,0.311,0.312,0.314,0.315,0.317,0.319,0.32,0.322,0.324,0.325,0.327,0.328,0.33,0.332,0.333,0.335,0.336,0.338,0.34,0.341,0.343,0.345,0.346,0.348,0.349,0.351,0.353,0.354,0.356,0.357,0.359,0.361,0.362,0.364,0.366,0.367,0.369,0.37,0.372,0.374,0.375,0.377,0.378,0.38,0.382,0.383,0.385,0.387,0.388,0.39,0.391,0.393,0.395,0.396,0.398,0.4,0.401,0.403,0.404,0.406,0.408,0.409,0.411,0.412,0.414,0.416,0.417,0.419,0.421,0.422,0.424,0.425,0.427,0.429,0.43]},
         '0.46' : { 'x' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.62,3.27,5.68,7.33,9.09,11.06,13.29,15.72,18.33,21,23.82,26.69,29.4,31.74,33.71,35.38,36.78,37.94,38.86,39.55,40.05,40.4,40.67,40.88,41.01,41.16,41.3,41.37,41.42,41.52,41.69,41.78,41.87,41.94,42.03,42.12,42.17,42.26,42.34,42.37,42.4,42.46,42.54,42.6,42.65,42.71,42.77,42.79,42.84,42.91,42.94,42.93,42.92,42.95,42.99,43.03,43.05,43.07,43.06,43.05,35.53,42.53,42.53,42.53,42.61,42.85,42.97,42.96,42.82,42.94,42.99,42.91,42.96,42.96,42.97,42.96,42.93,43.03,43.06,43.04,42.96,42.97,42.88,36.36,6.99,0], 'y' : [0.003,0.005,0.007,0.008,0.01,0.011,0.013,0.015,0.016,0.018,0.019,0.021,0.023,0.024,0.026,0.028,0.029,0.031,0.032,0.034,0.036,0.037,0.039,0.04,0.042,0.044,0.045,0.047,0.049,0.05,0.052,0.053,0.055,0.057,0.058,0.06,0.061,0.063,0.065,0.066,0.068,0.07,0.071,0.073,0.074,0.076,0.078,0.079,0.081,0.083,0.084,0.086,0.087,0.089,0.091,0.092,0.094,0.095,0.097,0.099,0.1,0.102,0.104,0.105,0.107,0.108,0.11,0.112,0.113,0.115,0.116,0.118,0.12,0.121,0.123,0.125,0.126,0.128,0.129,0.131,0.133,0.134,0.136,0.138,0.139,0.141,0.142,0.144,0.146,0.147,0.149,0.15,0.152,0.154,0.155,0.157,0.159,0.16,0.162,0.163,0.165,0.167,0.168,0.17,0.171,0.173,0.175,0.176,0.178,0.18,0.181,0.183,0.184,0.186,0.188,0.189,0.191,0.193,0.194,0.196,0.197,0.199,0.201,0.202,0.204,0.205,0.207,0.209,0.21,0.212,0.214,0.215,0.217,0.218,0.22,0.222,0.223,0.225,0.226,0.228,0.23,0.231,0.233,0.235,0.236,0.238,0.239,0.241,0.243,0.244,0.246,0.247,0.249,0.251,0.252,0.254,0.256,0.257,0.259,0.26,0.262,0.264,0.265,0.267,0.269,0.27,0.272,0.273,0.275,0.277,0.278,0.28,0.281,0.283,0.285,0.286,0.288,0.29,0.291,0.293,0.294,0.296,0.298,0.299,0.301,0.302,0.304,0.306,0.307,0.309,0.311,0.312,0.314,0.315,0.317,0.319,0.32,0.322,0.324,0.325,0.327,0.328,0.33,0.332,0.333,0.335,0.336,0.338,0.34,0.341,0.343,0.345,0.346,0.348,0.349,0.351,0.353,0.354,0.356,0.357,0.359,0.361,0.362,0.364,0.366,0.367,0.369,0.37,0.372,0.374,0.375,0.377,0.378,0.38,0.382,0.383,0.385,0.387,0.388,0.39,0.391,0.393,0.395,0.396,0.398,0.4,0.401,0.403,0.404,0.406,0.408,0.409,0.411,0.412,0.414,0.416,0.417,0.419,0.421,0.422,0.424,0.425,0.427,0.429,0.43]},
         '0.56' : { 'x' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.8,4.1,5.12,5.52,5.96,6.53,7.23,8.1,9.14,10.25,11.44,12.84,14.33,15.88,17.52,19.19,20.87,22.59,24.4,26.23,28.04,29.84,31.6,33.31,34.81,36.13,37.25,38.11,38.82,39.34,39.69,39.98,40.2,40.28,40.38,40.5,40.48,40.52,40.49,40.71,40.81,40.79,40.81,40.87,40.98,41.04,41,41.04,41.27,41.33,41.33,41.35,41.45,41.52,41.59,41.71,41.76,41.72,41.75,41.74,41.72,41.75,41.77,41.79,41.87,41.91,41.9,41.86,41.91,41.95,41.89,41.91,41.92,42.03,42.03,42.01,42.09,42.16,42.18,42.22,42.23,42.31,42.28,42.35,42.37,42.39,42.47,42.47,42.43,42.54,42.55,42.52,42.5,42.52,42.62,42.69,42.61,42.62,42.69,42.67,42.67,42.69,42.65,42.52,36.01,6.93,0], 'y' : [0.003,0.005,0.007,0.008,0.01,0.011,0.013,0.015,0.016,0.018,0.019,0.021,0.023,0.024,0.026,0.028,0.029,0.031,0.032,0.034,0.036,0.037,0.039,0.04,0.042,0.044,0.045,0.047,0.049,0.05,0.052,0.053,0.055,0.057,0.058,0.06,0.061,0.063,0.065,0.066,0.068,0.07,0.071,0.073,0.074,0.076,0.078,0.079,0.081,0.083,0.084,0.086,0.087,0.089,0.091,0.092,0.094,0.095,0.097,0.099,0.1,0.102,0.104,0.105,0.107,0.108,0.11,0.112,0.113,0.115,0.116,0.118,0.12,0.121,0.123,0.125,0.126,0.128,0.129,0.131,0.133,0.134,0.136,0.138,0.139,0.141,0.142,0.144,0.146,0.147,0.149,0.15,0.152,0.154,0.155,0.157,0.159,0.16,0.162,0.163,0.165,0.167,0.168,0.17,0.171,0.173,0.175,0.176,0.178,0.18,0.181,0.183,0.184,0.186,0.188,0.189,0.191,0.193,0.194,0.196,0.197,0.199,0.201,0.202,0.204,0.205,0.207,0.209,0.21,0.212,0.214,0.215,0.217,0.218,0.22,0.222,0.223,0.225,0.226,0.228,0.23,0.231,0.233,0.235,0.236,0.238,0.239,0.241,0.243,0.244,0.246,0.247,0.249,0.251,0.252,0.254,0.256,0.257,0.259,0.26,0.262,0.264,0.265,0.267,0.269,0.27,0.272,0.273,0.275,0.277,0.278,0.28,0.281,0.283,0.285,0.286,0.288,0.29,0.291,0.293,0.294,0.296,0.298,0.299,0.301,0.302,0.304,0.306,0.307,0.309,0.311,0.312,0.314,0.315,0.317,0.319,0.32,0.322,0.324,0.325,0.327,0.328,0.33,0.332,0.333,0.335,0.336,0.338,0.34,0.341,0.343,0.345,0.346,0.348,0.349,0.351,0.353,0.354,0.356,0.357,0.359,0.361,0.362,0.364,0.366,0.367,0.369,0.37,0.372,0.374,0.375,0.377,0.378,0.38,0.382,0.383,0.385,0.387,0.388,0.39,0.391,0.393,0.395,0.396,0.398,0.4,0.401,0.403,0.404,0.406,0.408,0.409,0.411,0.412,0.414,0.416,0.417,0.419,0.421,0.422,0.424,0.425,0.427,0.429,0.43]},
         '0.67' : { 'x' : [0,0,0,0,1.04,15.87,20.99,23.7,24.83,25.68,26.54,26.88,27.21,27.38,27.23,27.19,27.13,27.08,27.03,26.97,26.92,26.68,26.42,26.16,25.82,25.48,25.09,24.67,24.25,23.82,23.39,22.89,22.37,21.82,21.27,20.69,20.03,19.27,18.49,17.81,17.1,16.34,15.53,14.64,13.75,12.84,11.96,11.08,10.19,9.28,8.33,7.39,6.5,5.64,4.8,4.02,3.15,2.27,1.41,0.7,0.01,-0.69,-1.35,-1.97,-2.54,-3.03,-3.5,-3.93,-4.29,-4.59,-4.82,-4.98,-5.16,-5.35,-5.52,-5.67,-5.81,-5.96,-5.97,-5.94,-5.88,-5.88,-5.85,-5.77,-5.65,-5.55,-5.48,-5.4,-5.31,-5.19,-5.03,-4.88,-4.72,-4.53,-4.33,-4.12,-3.9,-3.68,-3.46,-3.23,-2.98,-2.73,-2.48,-2.21,-1.94,-1.67,-1.43,-1.18,-0.9,-0.57,-0.24,0.1,0.42,0.78,1.15,1.53,1.97,2.44,2.93,3.45,3.97,4.46,4.98,5.48,5.95,6.33,6.76,7.23,7.66,8.17,8.76,9.28,9.78,10.26,10.71,11.25,11.84,12.4,12.96,13.49,13.97,14.41,14.85,15.2,15.49,15.81,16.13,16.46,16.82,17.2,17.59,18.02,18.49,19,19.62,20.32,21.07,21.88,22.75,23.68,24.7,25.71,26.75,27.84,28.95,30.06,31.18,32.29,33.38,34.54,35.61,36.6,37.48,38.2,38.81,39.31,39.68,39.96,40.11,40.22,40.34,40.47,40.52,40.59,40.72,40.74,40.79,40.93,40.97,40.98,41.02,41.03,41.05,41.12,41.18,41.23,41.25,41.13,41.14,41.18,41.33,41.37,41.33,41.33,41.36,41.41,41.39,41.46,41.41,41.37,41.44,41.48,41.48,41.52,41.59,41.62,41.59,41.6,41.7,41.71,41.72,41.74,41.67,41.71,41.72,41.78,41.82,41.88,41.93,41.92,41.94,41.95,41.9,41.91,41.98,41.84,41.83,41.91,42,41.89,41.82,41.79,41.9,42.04,42,42.09,42.12,42.09,42.01,42,41.96,41.86,41.87,41.93,42.04,42.12,42.2,42.13,42.16,42.26,42.18,42.17,35.63,6.85,0], 'y' : [0.003,0.005,0.007,0.008,0.01,0.011,0.013,0.015,0.016,0.018,0.019,0.021,0.023,0.024,0.026,0.028,0.029,0.031,0.032,0.034,0.036,0.037,0.039,0.04,0.042,0.044,0.045,0.047,0.049,0.05,0.052,0.053,0.055,0.057,0.058,0.06,0.061,0.063,0.065,0.066,0.068,0.07,0.071,0.073,0.074,0.076,0.078,0.079,0.081,0.083,0.084,0.086,0.087,0.089,0.091,0.092,0.094,0.095,0.097,0.099,0.1,0.102,0.104,0.105,0.107,0.108,0.11,0.112,0.113,0.115,0.116,0.118,0.12,0.121,0.123,0.125,0.126,0.128,0.129,0.131,0.133,0.134,0.136,0.138,0.139,0.141,0.142,0.144,0.146,0.147,0.149,0.15,0.152,0.154,0.155,0.157,0.159,0.16,0.162,0.163,0.165,0.167,0.168,0.17,0.171,0.173,0.175,0.176,0.178,0.18,0.181,0.183,0.184,0.186,0.188,0.189,0.191,0.193,0.194,0.196,0.197,0.199,0.201,0.202,0.204,0.205,0.207,0.209,0.21,0.212,0.214,0.215,0.217,0.218,0.22,0.222,0.223,0.225,0.226,0.228,0.23,0.231,0.233,0.235,0.236,0.238,0.239,0.241,0.243,0.244,0.246,0.247,0.249,0.251,0.252,0.254,0.256,0.257,0.259,0.26,0.262,0.264,0.265,0.267,0.269,0.27,0.272,0.273,0.275,0.277,0.278,0.28,0.281,0.283,0.285,0.286,0.288,0.29,0.291,0.293,0.294,0.296,0.298,0.299,0.301,0.302,0.304,0.306,0.307,0.309,0.311,0.312,0.314,0.315,0.317,0.319,0.32,0.322,0.324,0.325,0.327,0.328,0.33,0.332,0.333,0.335,0.336,0.338,0.34,0.341,0.343,0.345,0.346,0.348,0.349,0.351,0.353,0.354,0.356,0.357,0.359,0.361,0.362,0.364,0.366,0.367,0.369,0.37,0.372,0.374,0.375,0.377,0.378,0.38,0.382,0.383,0.385,0.387,0.388,0.39,0.391,0.393,0.395,0.396,0.398,0.4,0.401,0.403,0.404,0.406,0.408,0.409,0.411,0.412,0.414,0.416,0.417,0.419,0.421,0.422,0.424,0.425,0.427,0.429,0.43]},
         '0.77' : { 'x' : [0,0,0,0,0,0,8.78,20.92,22.19,22.91,23.37,23.89,24.11,24.22,24.29,24.32,24.26,24.18,24.1,24.02,23.78,23.41,23.15,22.96,22.33,21.78,21.3,20.81,20.35,19.91,19.46,19,18.55,18.06,17.47,16.86,16.31,15.73,15.13,14.49,13.85,13.21,12.6,12.12,11.62,11.09,10.45,9.87,9.31,8.69,8.1,7.52,6.91,6.34,5.77,5.24,4.77,4.31,3.82,3.41,3.02,2.66,2.2,1.8,1.46,1.15,0.91,0.71,0.55,0.46,0.4,0.38,0.25,0.11,0.11,0.11,0.08,-0.01,0.01,0.04,0.04,0.19,0.42,0.71,0.94,1.18,1.43,1.68,2.03,2.46,2.89,3.33,3.78,4.18,4.59,5,5.37,5.81,6.29,6.82,7.2,7.6,8.07,8.58,9.09,9.63,10.24,10.71,11.11,11.67,12.16,12.54,12.88,13.24,13.63,14.05,14.47,14.89,15.3,15.69,16.05,16.39,16.84,17.23,17.56,17.88,18.17,18.47,18.81,19.1,19.42,19.74,20.06,20.38,20.64,20.88,21.12,21.53,21.94,22.38,22.91,23.46,23.97,24.5,25.15,25.84,26.52,27.26,28,28.79,29.53,30.26,31,31.73,32.45,33.15,33.88,34.52,35.16,35.82,36.37,36.82,37.19,37.53,37.76,37.89,38,38.11,38.22,38.28,38.4,38.57,38.63,38.75,38.87,38.97,38.99,38.97,39.11,39.27,39.45,39.51,39.63,39.74,39.79,39.62,39.59,39.62,39.72,39.8,39.88,39.99,40.07,40.18,40.16,40.17,40.19,40.19,40.2,40.18,40.26,40.29,40.34,40.4,40.44,40.43,40.56,40.55,40.48,40.44,40.44,40.46,40.43,40.51,40.57,40.57,40.51,40.58,40.67,40.72,40.74,40.82,40.89,40.89,40.87,40.82,40.78,40.8,40.82,40.87,40.94,40.97,41.08,41.12,41.21,41.1,40.99,40.93,41.02,41.1,41.21,41.24,41.21,41.15,41.2,41.3,41.23,41.03,41.09,41.12,41.18,41.13,41.13,41.24,41.11,41.06,41.1,41.12,41.11,41.25,41.26,41.12,34.77,6.69,0], 'y' : [0.003,0.005,0.007,0.008,0.01,0.011,0.013,0.015,0.016,0.018,0.019,0.021,0.023,0.024,0.026,0.028,0.029,0.031,0.032,0.034,0.036,0.037,0.039,0.04,0.042,0.044,0.045,0.047,0.049,0.05,0.052,0.053,0.055,0.057,0.058,0.06,0.061,0.063,0.065,0.066,0.068,0.07,0.071,0.073,0.074,0.076,0.078,0.079,0.081,0.083,0.084,0.086,0.087,0.089,0.091,0.092,0.094,0.095,0.097,0.099,0.1,0.102,0.104,0.105,0.107,0.108,0.11,0.112,0.113,0.115,0.116,0.118,0.12,0.121,0.123,0.125,0.126,0.128,0.129,0.131,0.133,0.134,0.136,0.138,0.139,0.141,0.142,0.144,0.146,0.147,0.149,0.15,0.152,0.154,0.155,0.157,0.159,0.16,0.162,0.163,0.165,0.167,0.168,0.17,0.171,0.173,0.175,0.176,0.178,0.18,0.181,0.183,0.184,0.186,0.188,0.189,0.191,0.193,0.194,0.196,0.197,0.199,0.201,0.202,0.204,0.205,0.207,0.209,0.21,0.212,0.214,0.215,0.217,0.218,0.22,0.222,0.223,0.225,0.226,0.228,0.23,0.231,0.233,0.235,0.236,0.238,0.239,0.241,0.243,0.244,0.246,0.247,0.249,0.251,0.252,0.254,0.256,0.257,0.259,0.26,0.262,0.264,0.265,0.267,0.269,0.27,0.272,0.273,0.275,0.277,0.278,0.28,0.281,0.283,0.285,0.286,0.288,0.29,0.291,0.293,0.294,0.296,0.298,0.299,0.301,0.302,0.304,0.306,0.307,0.309,0.311,0.312,0.314,0.315,0.317,0.319,0.32,0.322,0.324,0.325,0.327,0.328,0.33,0.332,0.333,0.335,0.336,0.338,0.34,0.341,0.343,0.345,0.346,0.348,0.349,0.351,0.353,0.354,0.356,0.357,0.359,0.361,0.362,0.364,0.366,0.367,0.369,0.37,0.372,0.374,0.375,0.377,0.378,0.38,0.382,0.383,0.385,0.387,0.388,0.39,0.391,0.393,0.395,0.396,0.398,0.4,0.401,0.403,0.404,0.406,0.408,0.409,0.411,0.412,0.414,0.416,0.417,0.419,0.421,0.422,0.424,0.425,0.427,0.429,0.43]},
         '0.88' : { 'x' : [0,0,0,0,0,0,6.03,14.35,15.53,16.04,16.39,16.85,17.5,18.12,18.39,18.37,18.38,18.49,18.7,18.97,19.19,19.02,18.9,18.82,18.58,18.23,17.88,17.71,17.56,17.45,17.21,16.91,16.55,16.04,15.83,15.63,15.36,15.07,14.8,14.56,14.4,14.1,13.7,13.25,13,12.92,12.79,12.59,12.36,12.17,11.9,11.66,11.58,11.46,11.37,11.34,11.26,11.19,11.21,11.32,11.44,11.54,11.55,11.63,11.77,11.79,11.89,12.08,12.26,12.48,12.66,12.63,12.83,13.08,13.32,13.55,13.8,14.12,14.32,14.61,15.07,15.38,15.66,15.75,16.05,16.19,16.32,16.55,16.96,17.49,17.7,17.95,18.29,18.72,18.82,19,19.32,19.52,19.74,19.92,20.14,20.4,20.59,20.84,21.05,21.27,21.56,21.88,22.2,22.18,22.17,22.15,22.6,22.87,23.2,23.43,23.53,23.71,23.95,24.05,24.25,24.79,25.28,25.66,25.81,26.17,26.5,26.83,27.15,27.48,27.9,28.38,28.94,29.47,29.91,30.46,30.94,31.44,31.9,32.26,32.5,32.91,33.31,33.65,33.97,34.2,34.23,34.72,35.07,35.33,35.63,35.91,36.19,36.46,36.66,36.73,36.68,36.64,36.75,36.82,37.04,37.24,37.24,37.1,37.06,37.26,37.43,37.47,37.35,37.13,37.04,36.73,36.75,37.17,37.45,37.55,37.7,37.93,37.88,37.79,37.81,38.11,38.18,38.22,38.2,38.29,38.26,38.2,38.2,38.35,38.46,38.6,38.66,38.65,38.84,38.98,39.1,39.15,39.09,38.92,38.92,39.01,39.04,39.2,39.25,39.38,39.36,39.39,39.3,39.34,39.4,39.41,39.52,39.58,39.59,39.48,39.53,39.52,39.59,39.59,39.62,39.58,39.65,39.76,39.77,39.55,39.69,39.77,39.84,39.74,39.81,40.13,40.1,40.16,40.23,40.09,40.04,40.15,40.12,39.98,39.82,39.57,39.49,39.49,39.7,39.89,40.13,40.47,40.34,40.23,40,39.89,39.88,40.03,40.31,40.4,40.36,40.19,40.04,39.88,40.24,39.8,33.53,6.45,0], 'y' : [0.003,0.005,0.007,0.008,0.01,0.011,0.013,0.015,0.016,0.018,0.019,0.021,0.023,0.024,0.026,0.028,0.029,0.031,0.032,0.034,0.036,0.037,0.039,0.04,0.042,0.044,0.045,0.047,0.049,0.05,0.052,0.053,0.055,0.057,0.058,0.06,0.061,0.063,0.065,0.066,0.068,0.07,0.071,0.073,0.074,0.076,0.078,0.079,0.081,0.083,0.084,0.086,0.087,0.089,0.091,0.092,0.094,0.095,0.097,0.099,0.1,0.102,0.104,0.105,0.107,0.108,0.11,0.112,0.113,0.115,0.116,0.118,0.12,0.121,0.123,0.125,0.126,0.128,0.129,0.131,0.133,0.134,0.136,0.138,0.139,0.141,0.142,0.144,0.146,0.147,0.149,0.15,0.152,0.154,0.155,0.157,0.159,0.16,0.162,0.163,0.165,0.167,0.168,0.17,0.171,0.173,0.175,0.176,0.178,0.18,0.181,0.183,0.184,0.186,0.188,0.189,0.191,0.193,0.194,0.196,0.197,0.199,0.201,0.202,0.204,0.205,0.207,0.209,0.21,0.212,0.214,0.215,0.217,0.218,0.22,0.222,0.223,0.225,0.226,0.228,0.23,0.231,0.233,0.235,0.236,0.238,0.239,0.241,0.243,0.244,0.246,0.247,0.249,0.251,0.252,0.254,0.256,0.257,0.259,0.26,0.262,0.264,0.265,0.267,0.269,0.27,0.272,0.273,0.275,0.277,0.278,0.28,0.281,0.283,0.285,0.286,0.288,0.29,0.291,0.293,0.294,0.296,0.298,0.299,0.301,0.302,0.304,0.306,0.307,0.309,0.311,0.312,0.314,0.315,0.317,0.319,0.32,0.322,0.324,0.325,0.327,0.328,0.33,0.332,0.333,0.335,0.336,0.338,0.34,0.341,0.343,0.345,0.346,0.348,0.349,0.351,0.353,0.354,0.356,0.357,0.359,0.361,0.362,0.364,0.366,0.367,0.369,0.37,0.372,0.374,0.375,0.377,0.378,0.38,0.382,0.383,0.385,0.387,0.388,0.39,0.391,0.393,0.395,0.396,0.398,0.4,0.401,0.403,0.404,0.406,0.408,0.409,0.411,0.412,0.414,0.416,0.417,0.419,0.421,0.422,0.424,0.425,0.427,0.429,0.43]},
        
         }
                  
    xData =[-0.781, -0.614, -0.521, -0.48, -0.435, 0.251, 0.358, 0.46, 0.56, 0.67, .77, 0.88]
    
    
    for i in xData:
        #Extract y dimension 
        refY = np.array(testData[str(i)]['y'])
        #u is already converted to model units (m/s) no need to convert reference velocity
        #Ref 
        refX = np.array(testData[str(i)]['x'])
    
        #From reference x0 (rear of body) find x1 for plot            
        x1 = x0[0] + i
        
        
        print(f' Start x1 is {x1}, {x0[1]}, {x0[2]}')
        print(f' End x1 is {x1}, {x0[1]}, {x0[2]+1.0}')
        sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
        filename = os.path.join(output_dir, f"{prefix}_{str(i)}")
        wp.synchronize()                 
        IOexporter.to_line(
            filename,
            {"velocity": sim.u},
            start_point=(x1, x0[1], x0[2]),
            end_point=(x1,  x0[1], x0[2]+1.0),            
            resolution=250,   
            component=0,
            radius=delta_x_coarse #needed with model units
        )
        # read the CSV written by the exporter
        csv_path = filename + "_velocity_0.csv"  # adjust if your exporter uses another extension
        print(f"CSV path is {csv_path}")
        
        try:
            sim_z, sim_ux = _load_sim_line(csv_path)
        except Exception as e:
            print(f"Failed to read {csv_path}: {e}")
            continue

        # plot reference vs simulation
        plt.figure(figsize=(4.5, 6))
        plt.plot(refX, refY, 'o', mfc='none', label='Experimental)')
        plt.plot(sim_ux, sim_z, '-', lw=2, label='Simulation')
        plt.xlim(np.min(refX)*.9, np.max(refX)*1.1)
        plt.ylim(np.min(refY), np.max(refY))
        plt.xlabel('Ux [m/s]')
        plt.ylabel('z [m]')
        plt.title(f'Velocity Plot at {i:+.3f}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename + ".png", dpi=150)
        plt.close()
        
        
  
# Main Script
# ===========
# Initialize XLB

xlb.init(
    velocity_set=velocity_set,
    default_backend=compute_backend,
    default_precision_policy=precision_policy,
)

# Generate mesh
if mesher_type == "makemesh":
    level_data, body_vertices, grid_shape_zip, partSize, actual_num_levels, shift, sparsity_pattern, level_origins, x0 = generate_makemesh_mesh(
        stl_filename, voxel_size, trim, trim_voxels
    )
elif mesher_type == "cuboid":
    level_data, body_vertices, grid_shape_zip, partSize, actual_num_levels, shift, sparsity_pattern, level_origins, x0 = generate_cuboid_mesh(
        stl_filename, voxel_size, trim, trim_voxels
    )
else:
    raise ValueError(f"Invalid mesher_type: {mesher_type}. Must be 'makemesh' or 'cuboid'.")

# Characteristic length
L = partSize[0]
L = float(L)  # Cast to built-in float to avoid NumPy type propagation issues with Warp

# Compute Re
Re = u_physical * L / kinematic_viscosity

# Calculate lattice parameters
delta_x_coarse = voxel_size * 2 ** (actual_num_levels - 1)
delta_t = voxel_size * ulb / u_physical
nu_lattice = kinematic_viscosity * delta_t / (voxel_size ** 2)
omega = 1.0 / (3.0 * nu_lattice + 0.5)

# Create output directory
current_dir = os.path.join(os.path.dirname(__file__))
output_dir = os.path.join(current_dir, script_name)
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# Define exporter objects
field_name_cardinality_dict = {"velocity": 3, "density": 1}
h5exporter = MultiresIO(
    field_name_cardinality_dict,
    level_data,
    scale=voxel_size,
    offset=-shift,
    timestep_size=delta_t,
)
bc_mask_exporter = MultiresIO({"bc_mask": 1}, level_data, scale=voxel_size, offset=-shift,)

# Create grid
grid = multires_grid_factory(
    grid_shape_zip,
    velocity_set=velocity_set,
    sparsity_pattern_list=sparsity_pattern,
    sparsity_pattern_origins=[neon.Index_3d(*box_origin) for box_origin in level_origins],
)

# Calculate num_steps
coarsest_level = grid.count_levels - 1
grid_shape_x_coarsest = grid.level_to_shape(coarsest_level)[0]
num_steps = int(flow_passes * (grid_shape_x_coarsest / ulb))

# Calculate print and file output intervals
print_interval = max(1, int(num_steps * (print_interval_percentage / 100.0)))
crossover_step = int(num_steps * (file_output_crossover_percentage / 100.0))
file_output_interval_pre_crossover = max(1, int(crossover_step / num_file_outputs_pre_crossover)) if num_file_outputs_pre_crossover > 0 else num_steps + 1
file_output_interval_post_crossover = max(1, int((num_steps - crossover_step) / num_file_outputs_post_crossover)) if num_file_outputs_post_crossover > 0 else num_steps + 1
final_print_interval = max(1, int((num_steps-crossover_step) * (print_interval_percentage / 100.0)))
# Setup boundary conditions
boundary_conditions = setup_boundary_conditions(grid, level_data, body_vertices, ulb, compute_backend)

# Create initializer
initializer = CustomMultiresInitializer(
    bc_id=boundary_conditions[-2].id,  # bc_outlet
    constant_velocity_vector=(ulb, 0.0, 0.0),
    velocity_set=velocity_set,
    precision_policy=precision_policy,
    compute_backend=compute_backend,
)

# Initialize simulation
sim = initialize_simulation(grid, boundary_conditions, omega, initializer)

# Compute voxel statistics and reference area
stats = compute_voxel_statistics_and_reference_area(sim, bc_mask_exporter, level_data, actual_num_levels, sparsity_pattern, boundary_conditions, voxel_size)
active_voxels = stats["active_voxels"]
solid_voxels = stats["solid_voxels"]
total_voxels = stats["total_voxels"]
total_lattice_updates_per_step = stats["total_lattice_updates_per_step"]
reference_area = stats["reference_area"]
reference_area_physical = stats["reference_area_physical"]

# Save initial bc_mask
filename = os.path.join(output_dir, f"{script_name}_initial_bc_mask")
bc_mask_exporter.to_hdf5(filename, {"bc_mask": sim.bc_mask}, compression="gzip", compression_opts=1)
wp.synchronize()


# Setup momentum transfer
# momentum_transfer = MultiresMomentumTransfer(boundary_conditions[-1], compute_backend=compute_backend)  # bc_body

momentum_transfer = MultiresMomentumTransfer(
    boundary_conditions[-1],
    mres_perf_opt=xlb.MresPerfOptimizationType.FUSION_AT_FINEST,
    compute_backend=compute_backend,
)

# Print simulation info
print("\n" + "=" * 50 + "\n")
print(f"Simulation Configuration for Re = {Re}:")
# print(f"Grid shape at finest level: {grid_shape_zip}")
# print(f"Grid shape at coarsest level: {grid.level_to_shape(coarsest_level)}")
print(f"Number of flow passes: {flow_passes}")
print(f"Calculated iterations: {num_steps:,}")
# print(f"Output directory: {output_dir}")
# print(f"Print interval: {print_interval} steps (every {print_interval_percentage}% of iterations)")
# print(f"File output interval pre-crossover (0-{file_output_crossover_percentage}%): {file_output_interval_pre_crossover} steps")
# print(f"File output interval post-crossover ({file_output_crossover_percentage}-100%): {file_output_interval_post_crossover} steps")
print(f"Finest voxel size: {voxel_size} meters")
print(f"Coarsest voxel size: {delta_x_coarse} meters")
print(f"Total voxels: {sum(np.count_nonzero(mask) for mask in sparsity_pattern):,}")
print(f"Total active voxels: {total_voxels:,}")
print(f"Active voxels per level: {active_voxels}")
print(f"Solid voxels per level: {solid_voxels}")
print(f"Total lattice updates per global step: {total_lattice_updates_per_step:,}")
print(f"Actual number of refinement levels: {actual_num_levels}")
print(f"Physical inlet velocity: {u_physical:.4f} m/s")
print(f"Lattice velocity (ulb): {ulb}")
print(f"Characteristic length: {L: .4f} meters")
# print(f"Kinematic viscosity: {kinematic_viscosity} m^2/s")
print(f"Computed reference area (bc_mask): {reference_area} lattice units")
print(f"Physical reference area (bc_mask): {reference_area_physical:.6f} m^2")
print(f"Reynolds number: {Re:,.2f}")
# print(f"Lattice viscosity: {nu_lattice:.5f}")
print(f"Relaxation parameter (omega): {omega:.5f}")
print("\n" + "=" * 50 + "\n")

# -------------------------- Simulation Loop --------------------------
wp.synchronize()
start_time = time.time()
compute_time = 0.0
steps_since_last_print = 0
drag_values = []

for step in range(num_steps):
    step_start = time.time()
    sim.step()
    wp.synchronize()
    compute_time += time.time() - step_start
    steps_since_last_print += 1
    if (step % print_interval == 0 and step < crossover_step) or step == num_steps - 1:
        sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
        wp.synchronize()
        cd, cl, drag = print_lift_drag(sim, step, momentum_transfer, ulb, reference_area, voxel_size)
        filename = os.path.join(output_dir, f"{script_name}_{step:04d}")
        h5exporter.to_slice_image(
            filename,
            {"velocity": sim.u},
            plane_point=(1, 0, 0),
            plane_normal=(0, 1, 0),
            grid_res=2000,
            bounds=(0, 1, 0, 1),
            show_axes=False,
            show_colorbar=False,
            slice_thickness=delta_x_coarse, #needed when using model units
            normalize = u_physical*1.5, #eventually we could have the 1.5 read from json as we did before
        )
        end_time = time.time()
        elapsed = end_time - start_time
        total_lattice_updates = total_lattice_updates_per_step * steps_since_last_print
        MLUPS = total_lattice_updates / compute_time / 1e6 if compute_time > 0 else 0.0
        current_flow_passes = step * ulb / grid_shape_x_coarsest
        remaining_steps = num_steps - step - 1
        time_remaining = 0.0 if MLUPS == 0 else (total_lattice_updates_per_step * remaining_steps) / (MLUPS * 1e6)
        hours, rem = divmod(time_remaining, 3600)
        minutes, seconds = divmod(rem, 60)
        time_remaining_str = f"{int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s"
        percent_complete = (step + 1) / num_steps * 100
        print(f"Completed step {step}/{num_steps} ({percent_complete:.2f}% complete)")
        print(f"  Flow Passes: {current_flow_passes:.2f}")
        print(f"  Time elapsed: {elapsed:.1f}s, Compute time: {compute_time:.1f}s, ETA: {time_remaining_str}")
        print(f"  MLUPS: {MLUPS:.1f}")
        print(f"  Cd= {cd:.3f}, Cl= {cl:.3f}, Drag Force (lattice units)={drag:.3f}")
        start_time = time.time()
        compute_time = 0.0
        steps_since_last_print = 0
    file_output_interval = file_output_interval_pre_crossover if step < crossover_step else file_output_interval_post_crossover
    if step % file_output_interval == 0 or step == num_steps - 1:
        sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
        filename = os.path.join(output_dir, f"{script_name}_{step:04d}")
        h5exporter.to_hdf5(filename, {"velocity": sim.u, "density": sim.rho}, compression="gzip", compression_opts=1)
        
        wp.synchronize()
    if step >= crossover_step and step % final_print_interval ==0 :
        sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
        wp.synchronize()
        cd, cl, drag = print_lift_drag(sim, step, momentum_transfer, ulb, reference_area, voxel_size)
        filename = os.path.join(output_dir, f"{script_name}_{step:04d}")
        h5exporter.to_slice_image(
            filename,
            {"velocity": sim.u},
            plane_point=(1, 0, 0),
            plane_normal=(0, 1, 0),
            grid_res=2000,
            bounds=(0, 1, 0, 1),
            show_axes=False,
            show_colorbar=False,
            slice_thickness=delta_x_coarse, #needed when using model units
            normalize = u_physical*1.5, #eventually we could have the 1.5 read from json as we did before
        )
        print(f"Completed step {step}/{num_steps} ")
        print(f"  Cd= {cd:.3f}, Cl= {cl:.3f}, Drag Force (lattice units)={drag:.3f}")
        
    if step == num_steps - 1:
        plot_data(x0, output_dir, delta_x_coarse, sim, h5exporter, prefix='Drivaer_Fastback')

# Save drag and lift data to CSV
if len(drag_values) > 0:
    with open(os.path.join(output_dir, "drag_lift.csv"), 'w') as fd:
        fd.write("Step,Cd,Cl\n")
        for i, (cd, cl) in enumerate(drag_values):
            fd.write(f"{i * print_interval},{cd},{cl}\n")
    plot_drag_lift(drag_values, output_dir, print_interval, script_name)

# Calculate and print average Cd and Cl for the last 50%
drag_values_array = np.array(drag_values)
if len(drag_values) > 0:
    start_index = int(len(drag_values) * file_output_crossover_percentage/100)
    last_half = drag_values_array[start_index:, :]
    avg_cd = np.mean(last_half[:, 0])
    avg_cl = np.mean(last_half[:, 1])
    print(f"Average Drag Coefficient (Cd) for last {(100-file_output_crossover_percentage)}%: {avg_cd:.6f}")
    print(f"Average Lift Coefficient (Cl) for last {(100-file_output_crossover_percentage)}%: {avg_cl:.6f}")
    print(f"Experimental Drag Coefficient (Cd): {0.3088}")  
    print(f"Error Drag Coefficient (Cd): {((avg_cd-0.3088)/0.3088)*100:.2f}%")  
    
else:
    print("No drag or lift data collected.")
    
