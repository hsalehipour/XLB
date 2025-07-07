import numpy as np
import open3d as o3d
from typing import Any

import neon
import warp as wp


def adjust_bbox(cuboid_max, cuboid_min, voxel_size_coarsest):
    """
    Adjust the bounding box to the nearest level 0 grid points that enclose the desired region.

    Args:
        cuboid_min (np.ndarray): Desired minimum coordinates of the bounding box.
        cuboid_max (np.ndarray): Desired maximum coordinates of the bounding box.
        voxel_size_coarsest (float): Voxel size of the coarsest grid (level 0).

    Returns:
        tuple: (adjusted_min, adjusted_max) snapped to level 0 grid points.
    """
    adjusted_min = np.round(cuboid_min / voxel_size_coarsest) * voxel_size_coarsest
    adjusted_max = np.round(cuboid_max / voxel_size_coarsest) * voxel_size_coarsest
    return adjusted_min, adjusted_max


def make_cuboid_mesh(voxel_size, cuboids, stl_name):
    """
    Create a multi-level cuboid mesh with bounding boxes aligned to the level 0 grid.
    Voxel matrices are set to ones only in regions not covered by finer levels.

    Args:
        voxel_size (float): Voxel size of the finest grid .
        cuboids (list): List of multipliers defining each level's domain.
        stl_name (str): Path to the STL file.

    Returns:
        list: Level data with voxel matrices, voxel sizes, origins, and levels.
    """
    # Load the mesh and get its bounding box
    mesh = o3d.io.read_triangle_mesh(stl_name)
    if mesh.is_empty():
        raise ValueError("Loaded mesh is empty or invalid.")

    aabb = mesh.get_axis_aligned_bounding_box()
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()
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
        if level > 0:
            voxel_level_up = max_voxel_size / pow(2, level - 1)
        else:
            voxel_level_up = voxel_size_level
        # Adjust bounding box to align with level 0 grid
        adjusted_min, adjusted_max = adjust_bbox(cuboid_max, cuboid_min, voxel_level_up)

        xmin, ymin, zmin = adjusted_min
        xmax, ymax, zmax = adjusted_max

        cuboid = adjusted_max - adjusted_min

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

    return list(reversed(level_data))


class ExportMultiresHDF5(object):
    def __init__(self, levels_data, scale=1, offset=(0.0, 0.0, 0.0)):
        """
        Initialize the ExportMultiresHDF5 object.

        Parameters
        ----------
        levels_data : list of tuples
            Each tuple contains (data, voxel_size, origin, level).
        filename : str
            The name of the output HDF5 file.
        fields : dict, optional
            A dictionary of fields to be included in the HDF5 file.
        scale : float or tuple, optional
            Scale factor for the coordinates.
        offset : tuple, optional
            Offset to be applied to the coordinates.
        compression : str, optional
            Compression method for the HDF5 datasets.
        compression_opts : int, optional
            Compression options for the HDF5 datasets.
        """
        # Process the multires geometry and extract coordinates and connectivity in the coordinate system of the finest level
        coordinates, connectivity, level_id_field, total_cells = self.process_geometry(levels_data, scale)

        # Ensure that coordinates and connectivity are not empty
        assert coordinates.size != 0, "Error: No valid data to process. Check the input levels_data."

        # Merge duplicate points
        coordinates, connectivity = self._merge_duplicates(coordinates, connectivity)

        # Apply scale and offset
        coordinates = self._transform_coordinates(coordinates, scale, offset)

        # Assign to self
        self.levels_data = levels_data
        self.coordinates = coordinates
        self.connectivity = connectivity
        self.level_id_field = level_id_field
        self.total_cells = total_cells

        # Prepare and allocate the inputs for the NEON container
        self.velocity_warp_list, self.density_warp_list, self.origin_list = self._prepare_container_inputs()

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
                {h5_filename}:/Fields/{field_name}
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

    def write_hdf5_file(self, filename, coordinates, connectivity, level_id_field, field_data, compression="gzip", compression_opts=0):
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
        field_data : dict
            A dictionary of all field data.
        compression : str, optional
            The compression method to use for the HDF5 file.
        compression_opts : int, optional
            The compression options to use for the HDF5 file.
        """
        import h5py

        with h5py.File(filename + ".h5", "w") as f:
            f.create_dataset("/Mesh/Points", data=coordinates, compression=compression, compression_opts=compression_opts, chunks=(100000, 3))
            f.create_dataset(
                "/Mesh/Connectivity",
                data=connectivity,
                compression=compression,
                compression_opts=compression_opts,
                chunks=(30000, 8),
            )
            f.create_dataset("/Mesh/Level", data=level_id_field, compression=compression, compression_opts=compression_opts)
            fg = f.create_group("/Fields")
            for fname, fdata in field_data.items():
                fg.create_dataset(fname, data=fdata.astype(np.float32), compression=compression, compression_opts=compression_opts)

    def _merge_duplicates(self, coordinates, connectivity):
        # Merging duplicate points
        tolerance = 0.01
        grid_coords = np.round(coordinates / tolerance).astype(np.int64)
        hash_keys = grid_coords[:, 0] + grid_coords[:, 1] * 1_000_000 + grid_coords[:, 2] * 1_000_000_000_000

        _, unique_indices, inverse = np.unique(hash_keys, return_index=True, return_inverse=True)
        coordinates = coordinates[unique_indices]
        connectivity = inverse[connectivity]
        return coordinates, connectivity

    def _transform_coordinates(self, coordinates, scale, offset):
        scale = np.array([scale] * 3 if isinstance(scale, (int, float)) else scale, dtype=np.float32)
        offset = np.array(offset, dtype=np.float32)
        return coordinates * scale + offset

    def _prepare_container_inputs(self, store_precision=None):
        # load necessary modules
        from xlb.compute_backend import ComputeBackend
        from xlb.grid import grid_factory
        from xlb import DefaultConfig

        # Get the number of levels from the levels_data
        num_levels = len(self.levels_data)

        # Set the default precision policy if not provided
        if store_precision is None:
            store_precision = DefaultConfig.default_precision_policy.store_precision

        # Prepare lists to hold warp fields and origins allocated for each level
        velocity_warp_list = []
        density_warp_list = []
        origin_list = []
        for level in range(num_levels):
            # get the shape of the grid at this level
            box_shape = self.levels_data[level][0].shape

            # Use the warp backend to create dense fields to be written in multi-res NEON fields
            grid_dense = grid_factory(box_shape, compute_backend=ComputeBackend.WARP)
            velocity_warp_list.append(grid_dense.create_field(cardinality=3, dtype=store_precision))
            density_warp_list.append(grid_dense.create_field(cardinality=1, dtype=store_precision))
            origin_list.append(wp.vec3i(*([int(x) for x in self.levels_data[level][2]])))

        return velocity_warp_list, density_warp_list, origin_list

    def _construct_neon_container(self):
        """
        Constructs a NEON container for exporting multi-resolution data to HDF5.
        This container will be used to transfer multi-resolution NEON fields into stacked warp fields.
        """

        @neon.Container.factory(name="HDF5MultiresExporter")
        def container(
            velocity_neon: Any,
            density_neon: Any,
            velocity_warp: Any,
            density_warp: Any,
            origin: Any,
            level: Any,
        ):
            def launcher(loader: neon.Loader):
                loader.set_mres_grid(velocity_neon.get_grid(), level)
                velocity_neon_hdl = loader.get_mres_read_handle(velocity_neon)
                density_neon_hdl = loader.get_mres_read_handle(density_neon)
                refinement = 2**level

                @wp.func
                def kernel(index: Any):
                    cIdx = wp.neon_global_idx(velocity_neon_hdl, index)
                    # Get local indices by dividing the global indices (associated with the finest level) by 2^level
                    # Subtract the origin to get the local indices in the warp field
                    lx = wp.neon_get_x(cIdx) // refinement - origin[0]
                    ly = wp.neon_get_y(cIdx) // refinement - origin[1]
                    lz = wp.neon_get_z(cIdx) // refinement - origin[2]

                    # write the values to the warp field
                    density_warp[0, lx, ly, lz] = wp.neon_read(density_neon_hdl, index, 0)
                    for card in range(3):
                        velocity_warp[card, lx, ly, lz] = wp.neon_read(velocity_neon_hdl, index, card)

                loader.declare_kernel(kernel)

            return launcher

        return container

    def __call__(self, filename, velocity_neon, density_neon, compression="gzip", compression_opts=0, store_precision=None):
        import time

        # Ensure that this operator is called on multires grids
        grid_mres = velocity_neon.get_grid()
        assert grid_mres.get_name() == "mGrid", f"Operation {self.__class__.__name} is only applicable to multi-resolution cases"

        # number of levels
        num_levels = grid_mres.get_num_levels()
        assert num_levels == len(self.levels_data), "Error: Inconsistent number of levels!"

        # Prepare the fields to be written by transfering multi-res NEON fields into stacked warp fields
        fields_data = {
            "velocity_x": [],
            "velocity_y": [],
            "velocity_z": [],
            "density": [],
        }
        for level in range(num_levels):

            # Create the container and run it to fill the warp fields
            c = self.container(
                velocity_neon,
                density_neon,
                self.velocity_warp_list[level],
                self.density_warp_list[level],
                self.origin_list[level],
                level
            )
            c.run(0, container_runtime=neon.Container.ContainerRuntime.neon)

            # Convert the warp fields to numpy arrays and use level's mask to filter the data
            mask = self.levels_data[level][0]
            velocity_np = np.array(wp.to_jax(self.velocity_warp_list[level]))
            rho = np.array(wp.to_jax(self.density_warp_list[level]))[0][mask]
            vx, vy, vz = velocity_np[0][mask], velocity_np[1][mask], velocity_np[2][mask]
            fields_data["velocity_x"].append(vx)
            fields_data["velocity_y"].append(vy)
            fields_data["velocity_z"].append(vz)
            fields_data["density"].append(rho)

        # Concatenate all field data
        for field_name in fields_data.keys():
            fields_data[field_name] = np.concatenate(fields_data[field_name])
            assert fields_data[field_name].size == self.total_cells, f"Error: Field {field_name} size mismatch!"

        # Save XDMF file
        self.save_xdmf(filename + ".h5", filename + ".xmf", self.total_cells, len(self.coordinates), fields_data)

        # Writing HDF5 file
        print("\tWriting HDF5 file")
        tic_write = time.perf_counter()
        self.write_hdf5_file(filename, self.coordinates, self.connectivity, self.level_id_field, fields_data, compression, compression_opts)
        toc_write = time.perf_counter()
        print(f"\tHDF5 file written in {toc_write - tic_write:0.1f} seconds")
