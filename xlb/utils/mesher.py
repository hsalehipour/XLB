import numpy as np
import trimesh
from typing import Any

import neon
import warp as wp


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

    return list(reversed(level_data))


class MultiresIO(object):
    def __init__(self, field_name_cardinality_dict, levels_data, scale=1, offset=(0.0, 0.0, 0.0), store_precision=None):
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
            Scale factor for the coordinates.
        offset : tuple, optional
            Offset to be applied to the coordinates.
        store_precision : str, optional
            The precision policy for storing data.
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

        with h5py.File(filename + ".h5", "w") as f:
            f.create_dataset("/Mesh/Points", data=coordinates, compression=compression, compression_opts=compression_opts, chunks=True)
            f.create_dataset(
                "/Mesh/Connectivity",
                data=connectivity,
                compression=compression,
                compression_opts=compression_opts,
                chunks=True,
            )
            f.create_dataset("/Mesh/Level", data=level_id_field, compression=compression, compression_opts=compression_opts)
            fg = f.create_group("/Fields")
            for fname, fdata in fields_data.items():
                fg.create_dataset(fname, data=fdata.astype(np.float32), compression=compression, compression_opts=compression_opts, chunks=True)

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
        assert grid_mres.get_name() == "mGrid", f"Operation {self.__class__.__name} is only applicable to multi-resolution cases"

        for field_name in field_neon_dict.keys():
            assert field_name in self.field_name_cardinality_dict.keys(), (
                f"Field {field_name} is not provided in the instantiation of the MultiresIO class!"
            )

        # number of levels
        num_levels = grid_mres.get_num_levels()
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
                field_np = np.array(wp.to_jax(self.field_warp_dict[field_name][level]))
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

        # Plot each field in the dictionary
        self._to_slice_image_single_field(
            f"{output_filename}_{field_name}",
            cell_data,
            plane_point,
            plane_normal,
            slice_thickness=slice_thickness,
            bounds=bounds,
            grid_res=grid_res,
            cmap=cmap,
            **kwargs,
        )
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
        **kwargs,
    ):
        """
        Helper function to create a slice image for a single field.
        """
        from matplotlib import cm
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.interpolate import griddata

        # field data are associated with the cells centers
        cell_values = field_data

        # get the normalized plane normal
        plane_normal = np.asarray(np.abs(plane_normal))
        n = plane_normal / np.linalg.norm(plane_normal)

        # Compute centroids (K = 8 for hexahedral cells)
        cell_points = self.coordinates[self.connectivity]  # shape (M, K, 3)
        centroids = np.mean(cell_points, axis=1)  # (M, 3)

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

        # Rasterize: scatter cell centers to 2D grid
        grid_x = np.linspace(local_x[mask_bounds].min(), local_x[mask_bounds].max(), grid_res)
        grid_y = np.linspace(local_y[mask_bounds].min(), local_y[mask_bounds].max(), grid_res)
        xv, yv = np.meshgrid(grid_x, grid_y, indexing="xy")

        # Linear interpolation for each grid point
        grid_field = griddata(points=(local_x, local_y), values=values, xi=(xv, yv), method="linear", fill_value=np.nan)

        # Plot
        plt.imshow(
            grid_field,
            extent=[xmin, xmax, ymin, ymax],
            cmap=cmap,
            origin="lower",
            aspect="equal",
            **kwargs,
        )
        plt.colorbar()
        plt.savefig(output_filename + ".png", dpi=300, bbox_inches="tight")
        plt.close()
