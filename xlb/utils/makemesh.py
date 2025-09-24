import numpy as np
import open3d as o3d
from typing import Any
import time
from pathlib import Path
from tabulate import tabulate

import neon
import warp as wp

DEVICE = "cuda"

def generate_mesh(
    levels,
    stl_name,
    voxSize,
    padding_table=None,
    domainMultiplier=None,
    close=True,
    ground_refinement_level=-1,
    ground_voxel_height=4,
    downsample=-1,
):
    """
    Generate a multi-resolution voxel grid based on an STL file.

    This function serves as a high-level interface to create a multi-resolution voxel grid by
    voxelizing an STL file and processing it across multiple resolution levels.

    Parameters
    ----------
    levels : int
        The number of resolution levels in the multi-resolution grid. Must be a positive integer.
    stl_name : str
        The file path or name of the STL file to be voxelized (e.g., "sphere.stl").
    voxSize : float
        The voxel size at the finest resolution level, in meters. Must be positive.
    padding_table : list of lists
        A list where each inner list contains six integers [xn, xp, yn, yp, zn, zp]
        representing padding in negative and positive x, y, z directions for each level.
    domainMultiplier : dictionary
        A 6-element dictionary array specifying scale in a given bounding box direction 'x', '-x', etc
    close : bool, optional
        If True, applies a closing operation to fill gaps or islands in the voxel grid at each level.
        Default is True.
    ground_refinement_level : int, optional
        The level at which to apply ground refinement (e.g., adding a solid ground layer).
        If -1, no ground refinement is performed. Default is -1.
    ground_voxel_height : int, optional
        The number of voxels to use during ground_refinement.
    downsample : int, optional
        The highest level (inclusive) to downsample when saving data, doubling voxel sizes for levels
        0 to `downsample`. If -1, no downsampling is applied. Default is -1.

    Returns
    -------
    level_data : list of tuples
        A list where each tuple corresponds to a resolution level and contains:
        - dr : numpy.ndarray
            The voxel matrix (3D boolean array) for the level.
        - v : float
            The voxel size for the level, in meters (adjusted for level).
        - dOrigin : numpy.ndarray
            The origin coordinates (x, y, z) of the voxel grid for the level, in meters.
        - l : int
            The level number (0 to `levels - 1`).
    """
    if not padding_table:
        padding_table = [
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
        ]
    if not domainMultiplier:
        domainMultiplier = {
            "-x": 0,
            "x": 0,
            "-y": 0,
            "y": 0,
            "z": 0,
            "-z": 0,
        }
    kernel = calculate_kernel(padding_table)
    return makeMesh(
        levels,
        stl_name,
        voxSize,
        kernel,
        domainMultiplier,
        close=close,
        ground_refinement_level=ground_refinement_level,
        ground_voxel_height=ground_voxel_height,
        downsample=downsample,
    )


# Warp kernels for voxel operations: grow, fill, remove, and crop


# Copy input matrix into a padded matrix on GPU
@wp.kernel
def copy_to_padded_kernel(input: wp.array3d(dtype=wp.uint8), padded: wp.array3d(dtype=wp.uint8), pad_x: int, pad_y: int, pad_z: int):
    i, j, k = wp.tid()
    if i < input.shape[0] and j < input.shape[1] and k < input.shape[2]:
        padded[i + pad_x, j + pad_y, k + pad_z] = input[i, j, k]


# Apply convolution to a padded matrix on GPU
@wp.kernel
def convolution_kernel(
    padded: wp.array3d(dtype=wp.uint8), kernel: wp.array3d(dtype=wp.uint8), output: wp.array3d(dtype=wp.uint8), kx: int, ky: int, kz: int
):
    i, j, k = wp.tid()
    if i >= padded.shape[0] or j >= padded.shape[1] or k >= padded.shape[2]:
        return
    sum_val = wp.uint8(0)
    for di in range(kx):
        for dj in range(ky):
            for dk in range(kz):
                if kernel[di, dj, dk] != 0:
                    ii = i + di - kx // 2
                    jj = j + dj - ky // 2
                    kk = k + dk - kz // 2
                    if 0 <= ii < padded.shape[0] and 0 <= jj < padded.shape[1] and 0 <= kk < padded.shape[2]:
                        sum_val += padded[ii, jj, kk]
    if sum_val > 0:
        output[i, j, k] = wp.uint8(1)
    else:
        output[i, j, k] = wp.uint8(0)


# Expand a voxel matrix using convolution-based growth on GPU
def grow_gpu(matrix, voxSize, origin, kernel):
    pad = (np.array(kernel.shape) * 0.5).astype(int)
    print("    Grow padding: ", pad)
    wp_matrix = wp.array(matrix.astype(np.uint8), dtype=wp.uint8, device=DEVICE)
    wp_kernel = wp.array(kernel.astype(np.uint8), dtype=wp.uint8, device=DEVICE)
    padded_shape = tuple(np.array(matrix.shape) + 2 * pad)
    padded = wp.zeros(padded_shape, dtype=wp.uint8, device=DEVICE)
    nx, ny, nz = matrix.shape
    wp.launch(kernel=copy_to_padded_kernel, dim=(nx, ny, nz), inputs=[wp_matrix, padded, pad[0], pad[1], pad[2]], device=DEVICE)
    output = wp.zeros(padded_shape, dtype=wp.uint8, device=DEVICE)
    nx, ny, nz = padded_shape
    kx, ky, kz = kernel.shape
    wp.launch(kernel=convolution_kernel, dim=(nx, ny, nz), inputs=[padded, wp_kernel, output, kx, ky, kz], device=DEVICE)
    wp.synchronize()
    r = output.numpy().astype(bool)
    kernel_shape = np.array(kernel.shape)
    originPad = origin - (kernel_shape - 1) * voxSize * 0.5
    return r, originPad


# Compute OR of 2x2x2 blocks in a padded matrix on GPU
@wp.kernel
def compute_or_blocks(padded: wp.array3d(dtype=wp.uint8), or_results: wp.array3d(dtype=wp.uint8)):
    bx, by, bz = wp.tid()
    if bx >= or_results.shape[0] or by >= or_results.shape[1] or bz >= or_results.shape[2]:
        return
    result = wp.uint8(0)
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                if padded[bx * 2 + di, by * 2 + dj, bz * 2 + dk] != 0:
                    result = wp.uint8(1)
                    break
            if result != 0:
                break
        if result != 0:
            break
    or_results[bx, by, bz] = result


# Perform binary dilation with a cross-shaped element on GPU
@wp.kernel
def binary_dilation_cross(input: wp.array3d(dtype=wp.uint8), output: wp.array3d(dtype=wp.uint8)):
    i, j, k = wp.tid()
    if i >= input.shape[0] or j >= input.shape[1] or k >= input.shape[2]:
        return
    result = wp.uint8(0)
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                if (dx == 0 and dy == 0 and dz == 0) or (abs(dx) + abs(dy) + abs(dz) == 1):
                    ii = wp.clamp(i + dx, 0, input.shape[0] - 1)
                    jj = wp.clamp(j + dy, 0, input.shape[1] - 1)
                    kk = wp.clamp(k + dz, 0, input.shape[2] - 1)
                    if input[ii, jj, kk] != 0:
                        result = wp.uint8(1)
                        break
            if result != 0:
                break
        if result != 0:
            break
    output[i, j, k] = result


# Perform binary erosion with a cross-shaped element on GPU
@wp.kernel
def binary_erosion_cross(input: wp.array3d(dtype=wp.uint8), output: wp.array3d(dtype=wp.uint8)):
    i, j, k = wp.tid()
    if i >= input.shape[0] or j >= input.shape[1] or k >= input.shape[2]:
        return
    result = wp.uint8(1)
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                if (dx == 0 and dy == 0 and dz == 0) or (abs(dx) + abs(dy) + abs(dz) == 1):
                    ii = wp.clamp(i + dx, 0, input.shape[0] - 1)
                    jj = wp.clamp(j + dy, 0, input.shape[1] - 1)
                    kk = wp.clamp(k + dz, 0, input.shape[2] - 1)
                    if input[ii, jj, kk] == 0:
                        result = wp.uint8(0)
                        break
            if result == 0:
                break
        if result == 0:
            break
    output[i, j, k] = result


# Expand OR results back to the padded matrix size on GPU
@wp.kernel
def expand_or_results(or_results: wp.array3d(dtype=wp.uint8), padded: wp.array3d(dtype=wp.uint8)):
    i, j, k = wp.tid()
    if i >= padded.shape[0] or j >= padded.shape[1] or k >= padded.shape[2]:
        return
    bx = i // 2
    by = j // 2
    bz = k // 2
    padded[i, j, k] = or_results[bx, by, bz]


# Extract the central region of a padded matrix on GPU
@wp.kernel
def extract_center(input: wp.array3d(dtype=wp.uint8), output: wp.array3d(dtype=wp.uint8), px: int, py: int, pz: int):
    i, j, k = wp.tid()
    if i < output.shape[0] and j < output.shape[1] and k < output.shape[2]:
        output[i, j, k] = input[i + px, j + py, k + pz]


# Pad an array with zeros on GPU
def pad_array_zeros_gpu(input_arr, pad_width):
    px, py, pz = pad_width
    shape = input_arr.shape
    padded_shape = (shape[0] + 2 * px, shape[1] + 2 * py, shape[2] + 2 * pz)
    padded = wp.zeros(padded_shape, dtype=wp.uint8, device=DEVICE)
    nx, ny, nz = shape
    wp.launch(kernel=copy_to_padded_kernel, dim=(nx, ny, nz), inputs=[input_arr, padded, px, py, pz], device=DEVICE)
    return padded


# Fill a voxel matrix with optional closing operation on GPU
def fill_gpu(matrix, voxSize, origin, close):
    a = (origin / voxSize) % 2
    inds = np.isclose(a, np.round(a), atol=1e-8)
    a[inds] = np.round(a[inds])
    paddingLo = np.floor(a).astype(int)
    paddingHi = np.round((matrix.shape + paddingLo) % 2).astype(int)
    originPad = origin - paddingLo * voxSize
    wp_matrix = wp.array(matrix.astype(np.uint8), dtype=wp.uint8, device=DEVICE)
    padded_shape = tuple(np.array(matrix.shape) + paddingLo + paddingHi)
    padded = wp.zeros(padded_shape, dtype=wp.uint8, device=DEVICE)
    nx, ny, nz = matrix.shape
    wp.launch(kernel=copy_to_padded_kernel, dim=(nx, ny, nz), inputs=[wp_matrix, padded, paddingLo[0], paddingLo[1], paddingLo[2]], device=DEVICE)
    or_results_shape = tuple(np.array(padded_shape) // 2)
    or_results = wp.zeros(or_results_shape, dtype=wp.uint8, device=DEVICE)
    nx, ny, nz = or_results_shape
    wp.launch(kernel=compute_or_blocks, dim=(nx, ny, nz), inputs=[padded, or_results], device=DEVICE)
    if close:
        padded_or_results = pad_array_zeros_gpu(or_results, (2, 2, 2))
        nx, ny, nz = padded_or_results.shape
        dilated = wp.zeros(padded_or_results.shape, dtype=wp.uint8, device=DEVICE)
        wp.launch(kernel=binary_dilation_cross, dim=(nx, ny, nz), inputs=[padded_or_results, dilated], device=DEVICE)
        eroded = wp.zeros(padded_or_results.shape, dtype=wp.uint8, device=DEVICE)
        wp.launch(kernel=binary_erosion_cross, dim=(nx, ny, nz), inputs=[dilated, eroded], device=DEVICE)
        nx, ny, nz = or_results_shape
        new_or_results = wp.zeros(or_results_shape, dtype=wp.uint8, device=DEVICE)
        wp.launch(kernel=extract_center, dim=(nx, ny, nz), inputs=[eroded, new_or_results, 2, 2, 2], device=DEVICE)
        or_results = new_or_results
    nx, ny, nz = padded_shape
    wp.launch(kernel=expand_or_results, dim=(nx, ny, nz), inputs=[or_results, padded], device=DEVICE)
    wp.synchronize()
    m = padded.numpy().astype(bool)
    return m, originPad


# Set specified voxel indices to False on GPU
@wp.kernel
def set_false_kernel(matrix: wp.array3d(dtype=wp.uint8), indices: wp.array(dtype=wp.int32, ndim=2), offset: wp.array(dtype=wp.int32, ndim=1)):
    tid = wp.tid()
    if tid >= indices.shape[0]:
        return
    ix = indices[tid, 0] + offset[0]
    iy = indices[tid, 1] + offset[1]
    iz = indices[tid, 2] + offset[2]
    if 0 <= ix < matrix.shape[0] and 0 <= iy < matrix.shape[1] and 0 <= iz < matrix.shape[2]:
        matrix[ix, iy, iz] = wp.uint8(0)


# Remove specified voxels from a matrix on GPU
def remove_gpu(matrix, origin, removeMat, removeOrigin, voxSize):
    offset = np.round((removeOrigin - origin) / voxSize).astype(int)
    removeIndices = np.argwhere(removeMat)
    if len(removeIndices) == 0:
        return np.copy(matrix)
    wp_matrix = wp.array(matrix.astype(np.uint8), dtype=wp.uint8, device=DEVICE)
    wp_indices = wp.array(removeIndices, dtype=wp.int32, device=DEVICE)
    wp_offset = wp.array(offset, dtype=wp.int32, device=DEVICE)
    wp.launch(kernel=set_false_kernel, dim=len(removeIndices), inputs=[wp_matrix, wp_indices, wp_offset], device=DEVICE)
    wp.synchronize()
    mat = wp_matrix.numpy().astype(bool)
    return mat


# Copy a cropped region from a matrix on GPU
@wp.kernel
def copy_cropped(mat: wp.array3d(dtype=wp.uint8), cropped: wp.array3d(dtype=wp.uint8), offset_x: int, offset_y: int, offset_z: int):
    i, j, k = wp.tid()
    if i < cropped.shape[0] and j < cropped.shape[1] and k < cropped.shape[2]:
        cropped[i, j, k] = mat[i + offset_x, j + offset_y, k + offset_z]


# Crop a voxel matrix to a specified domain on GPU
def crop_gpu(mat, origin, domainMin, domainMax, v):
    cMin = np.round((domainMin - origin) / v).astype(int)
    cMax = np.round((domainMax - origin) / v).astype(int)
    cropMin = np.maximum(cMin, 0)
    cropMax = np.minimum(cMax, mat.shape)
    cropped_shape = tuple(cropMax - cropMin)
    if any(s <= 0 for s in cropped_shape):
        return np.empty((0, 0, 0), dtype=bool), origin
    wp_mat = wp.array(mat.astype(np.uint8), dtype=wp.uint8, device=DEVICE)
    wp_cropped = wp.zeros(cropped_shape, dtype=wp.uint8, device=DEVICE)
    nx, ny, nz = cropped_shape
    wp.launch(kernel=copy_cropped, dim=(nx, ny, nz), inputs=[wp_mat, wp_cropped, cropMin[0], cropMin[1], cropMin[2]], device=DEVICE)
    wp.synchronize()
    cropMat = wp_cropped.numpy().astype(bool)
    origin = origin + cropMin * v
    return cropMat, origin


def pad_to_even(grid):
    shape = grid.shape
    # Calculate padding: 0 if even, 1 if odd, added to the upper side
    pad_width = [(0, 1 if s % 2 != 0 else 0) for s in shape]
    padded_grid = np.pad(grid, pad_width, mode="constant", constant_values=False)
    # Check if any padding was added
    padding_added = [pw[1] for pw in pad_width]  # Extract upper padding for each dimension
    if any(p > 0 for p in padding_added):
        # Map axis indices to x, y, z and report only dimensions with padding
        padding_str = ", ".join([f"{'xyz'[i]} by {p}" for i, p in enumerate(padding_added) if p > 0])
        print(f"    Padded {padding_str}")
    return padded_grid


# Voxelize an STL file using Open3D
def voxelize_stl_open3d(stl_filename, length_lbm_unit):
    """
    Voxelize an STL file using Open3D.

    Args:
        stl_filename (str): Path to the STL file.
        length_lbm_unit (float): Voxel size in meters.

    Returns:
        tuple: (voxel_matrix, origin, partDomain)
            - voxel_matrix: Boolean 3D array representing the voxel grid.
            - origin: Coordinates of the grid origin.
            - partDomain: List of [max_bound, min_bound].
    """
    tic = time.perf_counter()
    mesh = o3d.io.read_triangle_mesh(stl_filename)
    if len(mesh.vertices) == 0:
        raise ValueError("The mesh is empty or invalid.")
    print(f"    Number of vertices: {len(mesh.vertices):,}")
    print(f"    Number of triangles: {len(mesh.triangles):,}")
    toc = time.perf_counter()
    print(f"    Model read in {toc - tic:0.1f} seconds")
    
    tic = time.perf_counter()
    
    # Compute the bounds of the mesh
    min_bound = np.asarray(mesh.get_min_bound())
    max_bound = np.asarray(mesh.get_max_bound())
    
    # Snap bounds to voxel grid
    min_bound = np.floor(min_bound / length_lbm_unit) * length_lbm_unit
    max_bound = np.ceil(max_bound / length_lbm_unit) * length_lbm_unit
    
    # Compute grid size with padding to avoid index errors
    grid_size = np.ceil((max_bound - min_bound) / length_lbm_unit).astype(int) + 1  # Add 1 for padding
    
    # Translate mesh to align min_bound with origin
    mesh.translate(-min_bound)
    
    # Voxelize the mesh
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=length_lbm_unit)
    voxels = voxel_grid.get_voxels()
    voxel_indices = np.array([v.grid_index for v in voxels], dtype=int) if voxels else np.empty((0, 3), dtype=int)
    
    # Ensure indices are within bounds
    voxel_indices = np.clip(
        voxel_indices,
        [0, 0, 0],
        grid_size - 1  # Clip to max valid index
    )
    
    # Create the voxel matrix
    voxel_matrix = np.zeros(grid_size, dtype=bool)
    if voxel_indices.size > 0:
        voxel_matrix[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = True
    
    origin = min_bound
    partDomain = [max_bound, min_bound]
    
    toc = time.perf_counter()
    print(f"    Grid created in {toc - tic:0.1f} seconds")
    
    return voxel_matrix, origin, partDomain


# Print a table of padding values for each level
def print_padding_table(padding_values):
    headers = ["Level", "X-", "X+", "Y-", "Y+", "Z-", "Z+"]
    table = [[level] + values for level, values in enumerate(padding_values)]
    print(tabulate(table, headers=headers, tablefmt="grid"))


# Generate a multi-level voxel mesh from an STL file
def makeMesh(levels, filename, voxSize, kernel, domainMultiplier, close=True, ground_refinement_level=-1, ground_voxel_height=4, downsample=-1):
    stem = Path(filename).stem
    tic = time.perf_counter()

    matrix, origin, partDomain = voxelize_stl_open3d(filename, voxSize)
    partSize = partDomain[0] - partDomain[1]

    domainMin = np.array([0, 0, 0], float)
    domainMax = np.array(
        [
            partDomain[0][0] + (domainMultiplier["x"] * partSize[0]),
            partDomain[0][1] + (domainMultiplier["y"] * partSize[1]),
            partDomain[0][2] + (domainMultiplier["z"] * partSize[2]),
        ],
        float,
    )

    # Store original bounds
    orig_domainMin = domainMin.copy()
    orig_domainMax = domainMax.copy()

    domainSize = domainMax - domainMin

    # Calculate the smallest domain dimension
    min_domain_size = np.min(domainSize)
    threshold = min_domain_size / 8.0  # 1/8th of the smallest dimension
    ratio = threshold / voxSize

    # Check if the base voxel size is valid
    if ratio <= 0:
        raise ValueError("Voxel size is larger than 1/8th of the smallest domain dimension.")

    # Calculate maximum allowable levels
    max_levels_allowed = max(1, int(np.floor(np.log2(ratio))) + 1)

    # Adjust levels if necessary
    if levels > max_levels_allowed:
        print(f"Reducing levels from {levels} to {max_levels_allowed} to satisfy voxel size constraint.")
        levels = max_levels_allowed

    maxVoxSize = voxSize * pow(2, levels - 1)
    domainMin = np.round(domainMin / maxVoxSize) * maxVoxSize
    domainMax = np.ceil(domainMax / maxVoxSize) * maxVoxSize

    print("\n" + "=" * 100 + "\n")
    print("Meshing Configuration:")
    print(f"Model: {filename}")
    print(f"Finest level: {voxSize} meters")
    print(f"Number of levels: {levels}")
    print(f"Close islands: {close}")
    if ground_refinement_level != -1:
        print(f"Ground refinement level: {ground_refinement_level}")
    print("Adjusted domain coordinates: ", domainMin, ", ", domainMax)
    print("Voxel growth strategy:")
    #print_padding_table(padding_values)
    print("\n" + "=" * 100 + "\n")

    domainSize = domainMax - domainMin
    print("/// Make Mesh started... " + stem)
    v = voxSize

    level_data = []
    print("/// Level 0 voxel size: ", v)
    ticLevel = time.perf_counter()
    if levels == 1:
        # Calculate full domain shape
        full_shape = np.round(domainSize / voxSize).astype(int)
        df = np.ones(full_shape, dtype=bool)
        df = pad_to_even(df)  # Pad to ensure even shape
        dr = df.copy()
        dOrigin = domainMin
        level_data.append((dr, v, dOrigin, 0))
    else:
        g, origin = grow_gpu(matrix, voxSize, origin, kernel[0])
        f, origin = fill_gpu(g, voxSize, origin, close)
        df, origin = crop_gpu(f, origin, domainMin, domainMax, v)
        df = pad_to_even(df)
        dOrigin = np.copy(origin)
        dr = df.copy()  # dr is the final matrix for this level

        level_data.append((dr, v, dOrigin, 0))
        tocLevel = time.perf_counter()
        print(f"    Level defined in {tocLevel - ticLevel:0.1f} seconds")

        ground_z = domainMin[2]

        for l in range(1, levels):
            ticLevel = time.perf_counter()
            d = df[::2, ::2, ::2]
            v = v * 2
            print("/// Level", l, "voxel size:", v)
            full_shape = np.round(domainSize / v).astype(int)

            if l < levels - 1:
                dg, dOrigin = grow_gpu(d, v, dOrigin, kernel[l])
                df_natural, dOrigin = fill_gpu(dg, v, dOrigin, close)
                df_natural, dOrigin = crop_gpu(df_natural, dOrigin, domainMin, domainMax, v)
                df_natural = pad_to_even(df_natural)
            else:
                df_natural = np.ones(tuple(full_shape), bool)
                dOrigin = domainMin

            if ground_refinement_level != -1 and l == ground_refinement_level:
                df_ground = np.zeros(tuple(full_shape), bool)
                dOrigin_ground = domainMin
                ground_z_index = int(np.round((ground_z - dOrigin_ground[2]) / v))
                # Ground Voxel Thickness
                n_thick = ground_voxel_height
                if 0 <= ground_z_index < full_shape[2]:
                    end_z = min(ground_z_index + n_thick, full_shape[2])
                    df_ground[:, :, ground_z_index:end_z] = True

                offset = np.round((dOrigin - dOrigin_ground) / v).astype(int)
                x0, y0, z0 = offset
                x1, y1, z1 = x0 + df_natural.shape[0], y0 + df_natural.shape[1], z0 + df_natural.shape[2]
                x0, y0, z0 = np.maximum([x0, y0, z0], 0)
                x1, y1, z1 = np.minimum([x1, y1, z1], full_shape)
                df = np.zeros(tuple(full_shape), bool)
                if x1 > x0 and y1 > y0 and z1 > z0:
                    df[x0:x1, y0:y1, z0:z1] = df_natural[: (x1 - x0), : (y1 - y0), : (z1 - z0)]
                df |= df_ground
                dOrigin = dOrigin_ground
            else:
                df = df_natural

            dr = remove_gpu(df, dOrigin, d, origin, v)
            level_data.append((dr, v, dOrigin, l))
            tocLevel = time.perf_counter()
            print(f"    Level defined in {tocLevel - ticLevel:0.1f} seconds")
            origin = np.copy(dOrigin)

    toc = time.perf_counter()

    print()
    print("/// Mesh Data Report")
    finest_possible_voxels = int(np.prod(domainSize / voxSize))
    total_voxels_billions = finest_possible_voxels / 1e9
    print(f"    Total domain size: {total_voxels_billions:.2f} billion voxels (full dense at {voxSize} m)")

    total_voxel_count = sum(np.sum(dr) for dr, _, _, _ in level_data)
    total_voxel_count_millions = total_voxel_count / 1e6
    print(f"    Total voxel count: {total_voxel_count_millions:.2f} million")

    percentage_reduction = ((finest_possible_voxels - total_voxel_count) / finest_possible_voxels) * 100 if finest_possible_voxels > 0 else 0
    print(f"    Percentage reduction: {percentage_reduction:.4f}% (vs. uniform dense grid)")

    print("    Voxel distribution per level:")
    headers = ["Level", "Voxel Size (m)", "Voxels (M)", "Percentage (%)", "Computation (%)"]
    table_data = []
    # Calculate computational work
    num_levels = len(level_data)
    comp_work = []
    for l, (dr, v, _, _) in enumerate(level_data):
        voxel_count = np.sum(dr)
        # Inner iterations: finest level (l=0) has 2^(num_levels-1), coarsest (l=num_levels-1) has 2^0=1
        inner_iterations = 2 ** (num_levels - 1 - l)
        work = voxel_count * inner_iterations
        comp_work.append(work)
    total_work = sum(comp_work)
    for l, (dr, v, _, _) in enumerate(level_data):
        voxel_count = np.sum(dr)
        voxel_count_millions = voxel_count / 1e6
        percentage = (voxel_count / total_voxel_count) * 100 if total_voxel_count > 0 else 0
        comp_percentage = (comp_work[l] / total_work) * 100 if total_work > 0 else 0
        table_data.append([l, v, f"{voxel_count_millions:.2f}", f"{percentage:.2f}", f"{comp_percentage:.2f}"])
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print()

    # Downsample levels up to (and including) the given 'downsample' threshold.
    if downsample >= 0:
        print(f"Downsampling levels 0 to {downsample} for file saving; doubling their voxel sizes.")
        for i, (dr, v, dOrigin, lev) in enumerate(level_data):
            if lev <= downsample:
                dr_down = dr[::2, ::2, ::2]
                v_down = v * 2
                level_data[i] = (dr_down, v_down, dOrigin, lev)

    ##NEW SHIFT TO FIRST OCTANT
    level_data = [(dr, int(v / voxSize), np.round(dOrigin / v).astype(int), l) for dr, v, dOrigin, l in level_data]

    # Domain Adjustment Report
    print("/// Domain Extension Report")
    print(f"    Original domain: {orig_domainMin} to {orig_domainMax}")
    print(f"    Adjusted domain: {domainMin} to {domainMax}")

    # Calculate extensions in terms of finest voxels
    extension_pos = (domainMax - orig_domainMax) / voxSize
    extension_neg = (orig_domainMin - domainMin) / voxSize

    print("    Extension in positive directions (in finest voxels):")
    print(f"     +x: {extension_pos[0]:.2f}")
    print(f"     +y: {extension_pos[1]:.2f}")
    print(f"     +z: {extension_pos[2]:.2f}")
    print("    Extension in negative directions (in finest voxels):")
    print(f"     -x: {extension_neg[0]:.2f}")
    print(f"     -y: {extension_neg[1]:.2f}")
    print(f"     -z: {extension_neg[2]:.2f}")
    print()

    print("/// Level Shapes Report")
    headers = ["Level", "Shape", "Origin"]
    table_data = []
    for l, (dr, _, dOrigin, _) in enumerate(level_data):
        shape_str = f"({dr.shape[0]}, {dr.shape[1]}, {dr.shape[2]})"
        origin_str = f"({dOrigin[0]}, {dOrigin[1]}, {dOrigin[2]})"
        table_data.append([l, shape_str, origin_str])
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    print(f"/// Mesh Generation Complete in {toc - tic:0.2f} seconds")
    print()
    
    sparsity_pattern, level_origins = prepare_sparsity_pattern(level_data)
    
    return level_data, domainSize / voxSize, sparsity_pattern, level_origins

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


# Calculate convolution kernels based on padding values
def calculate_kernel(padding_values):
    kernels = []
    for level, values in enumerate(padding_values):
        xn, xp, yn, yp, zn, zp = values
        x_dim = max(xp, xn) * 2 + 1
        y_dim = max(yp, yn) * 2 + 1
        z_dim = max(zp, zn) * 2 + 1
        ones_x = xp + xn + 1
        ones_y = yp + yn + 1
        ones_z = zp + zn + 1
        mid_x = (x_dim - 1) // 2
        x1 = mid_x - xp
        x2 = x_dim - (mid_x - xn)
        mid_y = (y_dim - 1) // 2
        y1 = mid_y - yp
        y2 = y_dim - (mid_y - yn)
        mid_z = (z_dim - 1) // 2
        z1 = mid_z - zp
        z2 = z_dim - (mid_z - zn)
        kernel = np.zeros((x_dim, y_dim, z_dim), bool)
        kernel[x1:x2, y1:y2, z1:z2] = np.ones((x2 - x1, y2 - y1, z2 - z1), bool)
        kernels.append(kernel)
    return kernels