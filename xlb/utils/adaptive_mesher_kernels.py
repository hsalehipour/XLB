"""
Warp kernels for surface-adaptive mesh generation.

All hot-path grid operations use NVIDIA Warp (no JAX/CuPy).
"""

from __future__ import annotations

import warp as wp


@wp.func
def unsigned_mesh_distance(mesh_id: wp.uint64, pos: wp.vec3d, max_dist: wp.float64) -> wp.float64:
    """Unsigned distance from ``pos`` to the closest point on the mesh surface."""
    query = wp.mesh_query_point_sign_winding_number(mesh_id, wp.vec3f(pos), wp.float32(max_dist))
    if not query.result:
        return max_dist
    closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
    delta = wp.vec3d(
        wp.float64(closest[0]) - pos[0],
        wp.float64(closest[1]) - pos[1],
        wp.float64(closest[2]) - pos[2],
    )
    return wp.length(delta)


@wp.func
def assign_level_from_distance(
    dist: wp.float64,
    d_band: wp.float64,
    log_ratio: wp.float64,
    num_levels: wp.int32,
) -> wp.int32:
    if dist <= wp.float64(0.0):
        return wp.int32(0)
    ratio = dist / d_band + wp.float64(1.0)
    lvl = wp.int32(wp.floor(wp.log(ratio) / log_ratio))
    if lvl < wp.int32(0):
        return wp.int32(0)
    if lvl >= num_levels:
        return num_levels - wp.int32(1)
    return lvl


@wp.kernel
def kernel_conservative_coarse_targets(
    mesh_id: wp.uint64,
    origin: wp.vec3d,
    voxel_size: wp.float64,
    max_dist: wp.float64,
    d0: wp.float64,
    log_ratio: wp.float64,
    num_levels: wp.int32,
    targets: wp.array3d(dtype=wp.int32),
):
    i, j, k = wp.tid()
    base = wp.vec3d(
        origin[0] + wp.float64(i) * voxel_size,
        origin[1] + wp.float64(j) * voxel_size,
        origin[2] + wp.float64(k) * voxel_size,
    )
    min_dist = max_dist
    for ci in range(9):
        ox = wp.float64(0.0)
        oy = wp.float64(0.0)
        oz = wp.float64(0.0)
        if ci == 1:
            ox = voxel_size
        elif ci == 2:
            oy = voxel_size
        elif ci == 3:
            ox = voxel_size
            oy = voxel_size
        elif ci == 4:
            oz = voxel_size
        elif ci == 5:
            ox = voxel_size
            oz = voxel_size
        elif ci == 6:
            oy = voxel_size
            oz = voxel_size
        elif ci == 7:
            ox = voxel_size
            oy = voxel_size
            oz = voxel_size
        elif ci == 8:
            ox = wp.float64(0.5) * voxel_size
            oy = wp.float64(0.5) * voxel_size
            oz = wp.float64(0.5) * voxel_size
        pos = wp.vec3d(base[0] + ox, base[1] + oy, base[2] + oz)
        d = unsigned_mesh_distance(mesh_id, pos, max_dist)
        if d < min_dist:
            min_dist = d
    targets[i, j, k] = assign_level_from_distance(min_dist, d0, log_ratio, num_levels)


@wp.kernel
def kernel_batched_distances(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3d),
    max_dist: wp.float64,
    distances: wp.array(dtype=wp.float64),
):
    tid = wp.tid()
    distances[tid] = unsigned_mesh_distance(mesh_id, points[tid], max_dist)


@wp.kernel
def kernel_assign_levels_1d(
    distances: wp.array(dtype=wp.float64),
    d0: wp.float64,
    log_ratio: wp.float64,
    num_levels: wp.int32,
    assigned: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    assigned[tid] = assign_level_from_distance(distances[tid], d0, log_ratio, num_levels)


@wp.kernel
def kernel_minimum_filter_3x3(
    field_in: wp.array3d(dtype=wp.int32),
    field_out: wp.array3d(dtype=wp.int32),
):
    i, j, k = wp.tid()
    nx = field_in.shape[0]
    ny = field_in.shape[1]
    nz = field_in.shape[2]
    center = field_in[i, j, k]
    min_val = center
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                ni = i + di
                nj = j + dj
                nk = k + dk
                if ni >= 0 and nj >= 0 and nk >= 0 and ni < nx and nj < ny and nk < nz:
                    v = field_in[ni, nj, nk]
                    if v < min_val:
                        min_val = v
    field_out[i, j, k] = min_val


@wp.kernel
def kernel_maximum_filter_3x3(
    field_in: wp.array3d(dtype=wp.int32),
    field_out: wp.array3d(dtype=wp.int32),
):
    i, j, k = wp.tid()
    nx = field_in.shape[0]
    ny = field_in.shape[1]
    nz = field_in.shape[2]
    center = field_in[i, j, k]
    max_val = center
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                ni = i + di
                nj = j + dj
                nk = k + dk
                if ni >= 0 and nj >= 0 and nk >= 0 and ni < nx and nj < ny and nk < nz:
                    v = field_in[ni, nj, nk]
                    if v > max_val:
                        max_val = v
    field_out[i, j, k] = max_val


# ---------------------------------------------------------------------------
# Euclidean Distance Transform (Felzenszwalb & Huttenlocher 2012)
# ---------------------------------------------------------------------------

EDT_INF = wp.constant(1.0e20)


@wp.kernel
def kernel_edt_init_from_mask(
    mask: wp.array3d(dtype=wp.uint8),
    out: wp.array3d(dtype=wp.float32),
):
    """Initialize squared EDT input: 0 at features (mask==0), inf elsewhere."""
    i, j, k = wp.tid()
    if mask[i, j, k] != wp.uint8(0):
        out[i, j, k] = EDT_INF
    else:
        out[i, j, k] = wp.float32(0.0)


@wp.kernel
def kernel_edt_sqrt(
    sq: wp.array3d(dtype=wp.float32),
    out: wp.array3d(dtype=wp.float32),
):
    i, j, k = wp.tid()
    out[i, j, k] = wp.sqrt(sq[i, j, k])


@wp.kernel
def kernel_edt_pass_x(
    f: wp.array3d(dtype=wp.float32),
    scratch_v: wp.array2d(dtype=wp.int32),
    scratch_z: wp.array2d(dtype=wp.float32),
):
    """Felzenszwalb 1D squared EDT along the x axis (one thread per line)."""
    j, k = wp.tid()
    nx = f.shape[0]
    nz = f.shape[2]
    line_id = j * nz + k

    k_par = wp.int32(0)
    scratch_v[line_id, 0] = wp.int32(0)
    scratch_z[line_id, 0] = -EDT_INF
    scratch_z[line_id, 1] = EDT_INF

    for q in range(1, nx):
        fq = f[q, j, k]
        vk = scratch_v[line_id, k_par]
        s = (fq + wp.float32(q * q) - (f[vk, j, k] + wp.float32(vk * vk))) / (wp.float32(2.0) * wp.float32(q - vk))
        while s <= scratch_z[line_id, k_par]:
            k_par = k_par - wp.int32(1)
            vk = scratch_v[line_id, k_par]
            s = (fq + wp.float32(q * q) - (f[vk, j, k] + wp.float32(vk * vk))) / (wp.float32(2.0) * wp.float32(q - vk))
        k_par = k_par + wp.int32(1)
        scratch_v[line_id, k_par] = q
        scratch_z[line_id, k_par] = s
        scratch_z[line_id, k_par + wp.int32(1)] = EDT_INF

    k_par = wp.int32(0)
    for q in range(nx):
        while scratch_z[line_id, k_par + wp.int32(1)] < wp.float32(q):
            k_par = k_par + wp.int32(1)
        vk = scratch_v[line_id, k_par]
        diff = q - vk
        f[q, j, k] = wp.float32(diff * diff) + f[vk, j, k]


@wp.kernel
def kernel_edt_pass_y(
    f: wp.array3d(dtype=wp.float32),
    scratch_v: wp.array2d(dtype=wp.int32),
    scratch_z: wp.array2d(dtype=wp.float32),
):
    i, k = wp.tid()
    ny = f.shape[1]
    nz = f.shape[2]
    line_id = i * nz + k

    k_par = wp.int32(0)
    scratch_v[line_id, 0] = wp.int32(0)
    scratch_z[line_id, 0] = -EDT_INF
    scratch_z[line_id, 1] = EDT_INF

    for q in range(1, ny):
        fq = f[i, q, k]
        vk = scratch_v[line_id, k_par]
        s = (fq + wp.float32(q * q) - (f[i, vk, k] + wp.float32(vk * vk))) / (wp.float32(2.0) * wp.float32(q - vk))
        while s <= scratch_z[line_id, k_par]:
            k_par = k_par - wp.int32(1)
            vk = scratch_v[line_id, k_par]
            s = (fq + wp.float32(q * q) - (f[i, vk, k] + wp.float32(vk * vk))) / (wp.float32(2.0) * wp.float32(q - vk))
        k_par = k_par + wp.int32(1)
        scratch_v[line_id, k_par] = q
        scratch_z[line_id, k_par] = s
        scratch_z[line_id, k_par + wp.int32(1)] = EDT_INF

    k_par = wp.int32(0)
    for q in range(ny):
        while scratch_z[line_id, k_par + wp.int32(1)] < wp.float32(q):
            k_par = k_par + wp.int32(1)
        vk = scratch_v[line_id, k_par]
        diff = q - vk
        f[i, q, k] = wp.float32(diff * diff) + f[i, vk, k]


@wp.kernel
def kernel_edt_pass_z(
    f: wp.array3d(dtype=wp.float32),
    scratch_v: wp.array2d(dtype=wp.int32),
    scratch_z: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    nz = f.shape[2]
    line_id = i * f.shape[1] + j

    k_par = wp.int32(0)
    scratch_v[line_id, 0] = wp.int32(0)
    scratch_z[line_id, 0] = -EDT_INF
    scratch_z[line_id, 1] = EDT_INF

    for q in range(1, nz):
        fq = f[i, j, q]
        vk = scratch_v[line_id, k_par]
        s = (fq + wp.float32(q * q) - (f[i, j, vk] + wp.float32(vk * vk))) / (wp.float32(2.0) * wp.float32(q - vk))
        while s <= scratch_z[line_id, k_par]:
            k_par = k_par - wp.int32(1)
            vk = scratch_v[line_id, k_par]
            s = (fq + wp.float32(q * q) - (f[i, j, vk] + wp.float32(vk * vk))) / (wp.float32(2.0) * wp.float32(q - vk))
        k_par = k_par + wp.int32(1)
        scratch_v[line_id, k_par] = q
        scratch_z[line_id, k_par] = s
        scratch_z[line_id, k_par + wp.int32(1)] = EDT_INF

    k_par = wp.int32(0)
    for q in range(nz):
        while scratch_z[line_id, k_par + wp.int32(1)] < wp.float32(q):
            k_par = k_par + wp.int32(1)
        vk = scratch_v[line_id, k_par]
        diff = q - vk
        f[i, j, q] = wp.float32(diff * diff) + f[i, j, vk]
