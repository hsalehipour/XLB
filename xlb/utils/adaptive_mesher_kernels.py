"""
Warp kernels for surface-adaptive mesh generation.

All hot-path grid operations use NVIDIA Warp (no JAX/CuPy).
"""

from __future__ import annotations

import warp as wp

wp.init()

TILE_3 = wp.constant(3)
TILE_HALF_3 = wp.constant(1)


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
    d0: wp.float64,
    log_ratio: wp.float64,
    num_levels: wp.int32,
) -> wp.int32:
    if dist < d0:
        return wp.int32(0)
    ratio = dist / d0
    if ratio <= wp.float64(0.0):
        return wp.int32(0)
    lvl = wp.int32(wp.floor(wp.log(ratio) / log_ratio))
    if lvl < wp.int32(0):
        return wp.int32(0)
    if lvl >= num_levels:
        return num_levels - wp.int32(1)
    return lvl


@wp.kernel
def kernel_dense_distances(
    mesh_id: wp.uint64,
    origin: wp.vec3d,
    voxel_size: wp.float64,
    max_dist: wp.float64,
    distances: wp.array3d(dtype=wp.float64),
):
    i, j, k = wp.tid()
    pos = wp.vec3d(
        origin[0] + (wp.float64(i) + wp.float64(0.5)) * voxel_size,
        origin[1] + (wp.float64(j) + wp.float64(0.5)) * voxel_size,
        origin[2] + (wp.float64(k) + wp.float64(0.5)) * voxel_size,
    )
    distances[i, j, k] = unsigned_mesh_distance(mesh_id, pos, max_dist)


@wp.kernel
def kernel_assign_levels(
    distances: wp.array3d(dtype=wp.float64),
    d0: wp.float64,
    log_ratio: wp.float64,
    num_levels: wp.int32,
    assigned: wp.array3d(dtype=wp.int32),
):
    i, j, k = wp.tid()
    assigned[i, j, k] = assign_level_from_distance(distances[i, j, k], d0, log_ratio, num_levels)


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


@wp.kernel
def kernel_apply_balance_refine(
    field_in: wp.array3d(dtype=wp.int32),
    neighbor_min: wp.array3d(dtype=wp.int32),
    field_out: wp.array3d(dtype=wp.int32),
):
    i, j, k = wp.tid()
    v = field_in[i, j, k]
    nmin = neighbor_min[i, j, k]
    if v > nmin + wp.int32(1):
        field_out[i, j, k] = nmin + wp.int32(1)
    else:
        field_out[i, j, k] = v


@wp.kernel
def kernel_apply_balance_coarsen(
    field_in: wp.array3d(dtype=wp.int32),
    neighbor_max: wp.array3d(dtype=wp.int32),
    field_out: wp.array3d(dtype=wp.int32),
):
    i, j, k = wp.tid()
    v = field_in[i, j, k]
    nmax = neighbor_max[i, j, k]
    if v < nmax - wp.int32(1):
        field_out[i, j, k] = nmax - wp.int32(1)
    else:
        field_out[i, j, k] = v


@wp.kernel
def kernel_minimum_with_floor(
    a: wp.array3d(dtype=wp.int32),
    b: wp.array3d(dtype=wp.int32),
    out: wp.array3d(dtype=wp.int32),
):
    i, j, k = wp.tid()
    v = a[i, j, k]
    f = b[i, j, k]
    if v < f:
        out[i, j, k] = v
    else:
        out[i, j, k] = f


@wp.kernel
def kernel_refine_transition(
    field_in: wp.array3d(dtype=wp.int32),
    neighbor_min: wp.array3d(dtype=wp.int32),
    level: wp.int32,
    field_out: wp.array3d(dtype=wp.int32),
):
    i, j, k = wp.tid()
    v = field_in[i, j, k]
    nmin = neighbor_min[i, j, k]
    if v >= level + wp.int32(2) and nmin <= level:
        field_out[i, j, k] = level + wp.int32(1)
    else:
        field_out[i, j, k] = v


@wp.kernel
def kernel_dense_distances_tiled(
    mesh_id: wp.uint64,
    origin: wp.vec3d,
    voxel_size: wp.float64,
    max_dist: wp.float64,
    i_off: wp.int32,
    j_off: wp.int32,
    k_off: wp.int32,
    distances: wp.array3d(dtype=wp.float64),
):
    i, j, k = wp.tid()
    gi = i_off + i
    gj = j_off + j
    gk = k_off + k
    pos = wp.vec3d(
        origin[0] + (wp.float64(gi) + wp.float64(0.5)) * voxel_size,
        origin[1] + (wp.float64(gj) + wp.float64(0.5)) * voxel_size,
        origin[2] + (wp.float64(gk) + wp.float64(0.5)) * voxel_size,
    )
    distances[i, j, k] = unsigned_mesh_distance(mesh_id, pos, max_dist)


@wp.kernel
def kernel_binary_dilate_cube(
    mask_in: wp.array3d(dtype=wp.uint8),
    mask_out: wp.array3d(dtype=wp.uint8),
    width: wp.int32,
):
    """Binary dilation matching ``scipy.ndimage.binary_dilation(..., structure=ones((w,w,w)))``."""
    i, j, k = wp.tid()
    nx = mask_in.shape[0]
    ny = mask_in.shape[1]
    nz = mask_in.shape[2]
    origin = -(width // wp.int32(2))
    max_val = mask_in[i, j, k]
    for di in range(width):
        for dj in range(width):
            for dk in range(width):
                ni = i - di - origin
                nj = j - dj - origin
                nk = k - dk - origin
                if ni >= 0 and nj >= 0 and nk >= 0 and ni < nx and nj < ny and nk < nz:
                    v = mask_in[ni, nj, nk]
                    if v > max_val:
                        max_val = v
    mask_out[i, j, k] = max_val


@wp.kernel
def kernel_binary_dilate_3x3(
    mask_in: wp.array3d(dtype=wp.uint8),
    mask_out: wp.array3d(dtype=wp.uint8),
):
    i, j, k = wp.tid()
    nx = mask_in.shape[0]
    ny = mask_in.shape[1]
    nz = mask_in.shape[2]
    max_val = mask_in[i, j, k]
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                ni = i + di
                nj = j + dj
                nk = k + dk
                if ni >= 0 and nj >= 0 and nk >= 0 and ni < nx and nj < ny and nk < nz:
                    v = mask_in[ni, nj, nk]
                    if v > max_val:
                        max_val = v
    mask_out[i, j, k] = max_val


@wp.kernel
def kernel_promote_near_fine(
    owner_in: wp.array3d(dtype=wp.int32),
    dist: wp.array3d(dtype=wp.float32),
    level: wp.int32,
    transition: wp.int32,
    band_width: wp.float32,
    owner_out: wp.array3d(dtype=wp.int32),
):
    i, j, k = wp.tid()
    v = owner_in[i, j, k]
    d = dist[i, j, k]
    if v >= level + wp.int32(2) and d > wp.float32(0.0) and d <= band_width:
        owner_out[i, j, k] = transition
    else:
        owner_out[i, j, k] = v


@wp.kernel
def kernel_widen_shell(
    owner_in: wp.array3d(dtype=wp.int32),
    dilated_shell: wp.array3d(dtype=wp.uint8),
    level: wp.int32,
    transition: wp.int32,
    owner_out: wp.array3d(dtype=wp.int32),
):
    i, j, k = wp.tid()
    v = owner_in[i, j, k]
    if dilated_shell[i, j, k] != wp.uint8(0) and v > level and v >= transition:
        owner_out[i, j, k] = transition
    else:
        owner_out[i, j, k] = v


@wp.kernel
def kernel_block_uniformity_level(
    owner_in: wp.array3d(dtype=wp.int32),
    stride: wp.int32,
    owner_out: wp.array3d(dtype=wp.int32),
):
    bi, bj, bk = wp.tid()
    sx = owner_in.shape[0] // stride
    sy = owner_in.shape[1] // stride
    sz = owner_in.shape[2] // stride
    if bi >= sx or bj >= sy or bk >= sz:
        return
    bmin = owner_in[bi * stride, bj * stride, bk * stride]
    bmax = bmin
    for di in range(stride):
        for dj in range(stride):
            for dk in range(stride):
                v = owner_in[bi * stride + di, bj * stride + dj, bk * stride + dk]
                if v < bmin:
                    bmin = v
                if v > bmax:
                    bmax = v
    if bmax > bmin:
        for di in range(stride):
            for dj in range(stride):
                for dk in range(stride):
                    owner_out[bi * stride + di, bj * stride + dj, bk * stride + dk] = bmin
    else:
        for di in range(stride):
            for dj in range(stride):
                for dk in range(stride):
                    owner_out[bi * stride + di, bj * stride + dj, bk * stride + dk] = owner_in[
                        bi * stride + di, bj * stride + dj, bk * stride + dk
                    ]


@wp.kernel
def kernel_extract_level_mask(
    owner: wp.array3d(dtype=wp.int32),
    level: wp.int32,
    stride: wp.int32,
    mask: wp.array3d(dtype=wp.uint8),
):
    ci, cj, ck = wp.tid()
    sx = owner.shape[0] // stride
    sy = owner.shape[1] // stride
    sz = owner.shape[2] // stride
    if ci >= sx or cj >= sy or ck >= sz:
        return
    uniform = wp.uint8(1)
    ref = owner[ci * stride, cj * stride, ck * stride]
    if ref != level:
        uniform = wp.uint8(0)
    else:
        for di in range(stride):
            for dj in range(stride):
                for dk in range(stride):
                    if owner[ci * stride + di, cj * stride + dj, ck * stride + dk] != level:
                        uniform = wp.uint8(0)
    mask[ci, cj, ck] = uniform


@wp.kernel
def kernel_paint_owner_block(
    target_level: wp.int32,
    i0: wp.int32,
    j0: wp.int32,
    k0: wp.int32,
    stride: wp.int32,
    owner: wp.array3d(dtype=wp.int32),
):
    bi, bj, bk = wp.tid()
    i = i0 + bi
    j = j0 + bj
    k = k0 + bk
    if i < owner.shape[0] and j < owner.shape[1] and k < owner.shape[2]:
        cur = owner[i, j, k]
        if target_level < cur:
            owner[i, j, k] = target_level


@wp.kernel
def kernel_copy_int3d(
    src: wp.array3d(dtype=wp.int32),
    dst: wp.array3d(dtype=wp.int32),
):
    i, j, k = wp.tid()
    dst[i, j, k] = src[i, j, k]


@wp.kernel
def kernel_any_changed(
    a: wp.array3d(dtype=wp.int32),
    b: wp.array3d(dtype=wp.int32),
    changed: wp.array(dtype=wp.int32),
):
    i, j, k = wp.tid()
    if a[i, j, k] != b[i, j, k]:
        wp.atomic_add(changed, 0, wp.int32(1))


EDT_INF = wp.constant(1.0e20)


@wp.func
def _stride_for_level(level: wp.int32) -> wp.int32:
    s = wp.int32(1)
    for _ in range(level):
        s = s * wp.int32(2)
    return s


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
        s = (fq + wp.float32(q * q) - (f[vk, j, k] + wp.float32(vk * vk))) / (
            wp.float32(2.0) * wp.float32(q - vk)
        )
        while s <= scratch_z[line_id, k_par]:
            k_par = k_par - wp.int32(1)
            vk = scratch_v[line_id, k_par]
            s = (fq + wp.float32(q * q) - (f[vk, j, k] + wp.float32(vk * vk))) / (
                wp.float32(2.0) * wp.float32(q - vk)
            )
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
        s = (fq + wp.float32(q * q) - (f[i, vk, k] + wp.float32(vk * vk))) / (
            wp.float32(2.0) * wp.float32(q - vk)
        )
        while s <= scratch_z[line_id, k_par]:
            k_par = k_par - wp.int32(1)
            vk = scratch_v[line_id, k_par]
            s = (fq + wp.float32(q * q) - (f[i, vk, k] + wp.float32(vk * vk))) / (
                wp.float32(2.0) * wp.float32(q - vk)
            )
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
        s = (fq + wp.float32(q * q) - (f[i, j, vk] + wp.float32(vk * vk))) / (
            wp.float32(2.0) * wp.float32(q - vk)
        )
        while s <= scratch_z[line_id, k_par]:
            k_par = k_par - wp.int32(1)
            vk = scratch_v[line_id, k_par]
            s = (fq + wp.float32(q * q) - (f[i, j, vk] + wp.float32(vk * vk))) / (
                wp.float32(2.0) * wp.float32(q - vk)
            )
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


@wp.kernel
def kernel_build_fine_mask(
    owner: wp.array3d(dtype=wp.int32),
    level: wp.int32,
    fine: wp.array3d(dtype=wp.uint8),
):
    i, j, k = wp.tid()
    if owner[i, j, k] <= level:
        fine[i, j, k] = wp.uint8(0)
    else:
        fine[i, j, k] = wp.uint8(1)


@wp.kernel
def kernel_build_shell_mask(
    owner: wp.array3d(dtype=wp.int32),
    transition: wp.int32,
    shell: wp.array3d(dtype=wp.uint8),
):
    i, j, k = wp.tid()
    if owner[i, j, k] == transition:
        shell[i, j, k] = wp.uint8(1)
    else:
        shell[i, j, k] = wp.uint8(0)


@wp.kernel
def kernel_paint_blocks_level(
    target_level: wp.int32,
    block_i: wp.array(dtype=wp.int32),
    block_j: wp.array(dtype=wp.int32),
    block_k: wp.array(dtype=wp.int32),
    fi_min: wp.int32,
    fj_min: wp.int32,
    fk_min: wp.int32,
    stride: wp.int32,
    owner: wp.array3d(dtype=wp.int32),
):
    block_id, di, dj, dk = wp.tid()
    n_blocks = block_i.shape[0]
    if block_id >= n_blocks:
        return
    i = block_i[block_id] * stride + di - fi_min
    j = block_j[block_id] * stride + dj - fj_min
    k = block_k[block_id] * stride + dk - fk_min
    if i >= 0 and j >= 0 and k >= 0 and i < owner.shape[0] and j < owner.shape[1] and k < owner.shape[2]:
        cur = owner[i, j, k]
        if target_level < cur:
            owner[i, j, k] = target_level


@wp.kernel
def kernel_balance_need_refine(
    field: wp.array3d(dtype=wp.int32),
    neighbor_min: wp.array3d(dtype=wp.int32),
    flags: wp.array3d(dtype=wp.uint8),
):
    i, j, k = wp.tid()
    if field[i, j, k] > neighbor_min[i, j, k] + wp.int32(1):
        flags[i, j, k] = wp.uint8(1)
    else:
        flags[i, j, k] = wp.uint8(0)


@wp.kernel
def kernel_balance_need_coarsen(
    field: wp.array3d(dtype=wp.int32),
    neighbor_max: wp.array3d(dtype=wp.int32),
    flags: wp.array3d(dtype=wp.uint8),
):
    i, j, k = wp.tid()
    if field[i, j, k] < neighbor_max[i, j, k] - wp.int32(1):
        flags[i, j, k] = wp.uint8(1)
    else:
        flags[i, j, k] = wp.uint8(0)


@wp.kernel
def kernel_any_nonzero_uint8(
    flags: wp.array3d(dtype=wp.uint8),
    counter: wp.array(dtype=wp.int32),
):
    i, j, k = wp.tid()
    if flags[i, j, k] != wp.uint8(0):
        wp.atomic_add(counter, 0, wp.int32(1))


@wp.kernel
def kernel_owners_from_masks(
    num_levels: wp.int32,
    mask0: wp.array3d(dtype=wp.uint8),
    mask1: wp.array3d(dtype=wp.uint8),
    mask2: wp.array3d(dtype=wp.uint8),
    mask3: wp.array3d(dtype=wp.uint8),
    mask4: wp.array3d(dtype=wp.uint8),
    mask5: wp.array3d(dtype=wp.uint8),
    mask6: wp.array3d(dtype=wp.uint8),
    mask7: wp.array3d(dtype=wp.uint8),
    owners: wp.array3d(dtype=wp.int32),
):
    i, j, k = wp.tid()
    owners[i, j, k] = wp.int32(-1)
    for level in range(num_levels - wp.int32(1), wp.int32(-1), wp.int32(-1)):
        stride = _stride_for_level(level)
        ci = i // stride
        cj = j // stride
        ck = k // stride
        active = wp.uint8(0)
        if level == wp.int32(0):
            if ci < mask0.shape[0] and cj < mask0.shape[1] and ck < mask0.shape[2]:
                active = mask0[ci, cj, ck]
        elif level == wp.int32(1):
            if ci < mask1.shape[0] and cj < mask1.shape[1] and ck < mask1.shape[2]:
                active = mask1[ci, cj, ck]
        elif level == wp.int32(2):
            if ci < mask2.shape[0] and cj < mask2.shape[1] and ck < mask2.shape[2]:
                active = mask2[ci, cj, ck]
        elif level == wp.int32(3):
            if ci < mask3.shape[0] and cj < mask3.shape[1] and ck < mask3.shape[2]:
                active = mask3[ci, cj, ck]
        elif level == wp.int32(4):
            if ci < mask4.shape[0] and cj < mask4.shape[1] and ck < mask4.shape[2]:
                active = mask4[ci, cj, ck]
        elif level == wp.int32(5):
            if ci < mask5.shape[0] and cj < mask5.shape[1] and ck < mask5.shape[2]:
                active = mask5[ci, cj, ck]
        elif level == wp.int32(6):
            if ci < mask6.shape[0] and cj < mask6.shape[1] and ck < mask6.shape[2]:
                active = mask6[ci, cj, ck]
        elif level == wp.int32(7):
            if ci < mask7.shape[0] and cj < mask7.shape[1] and ck < mask7.shape[2]:
                active = mask7[ci, cj, ck]
        if active != wp.uint8(0):
            owners[i, j, k] = level


@wp.func
def _mark_subdivide(
    level: wp.int32,
    ci: wp.int32,
    cj: wp.int32,
    ck: wp.int32,
    sub0: wp.array3d(dtype=wp.uint8),
    sub1: wp.array3d(dtype=wp.uint8),
    sub2: wp.array3d(dtype=wp.uint8),
    sub3: wp.array3d(dtype=wp.uint8),
    sub4: wp.array3d(dtype=wp.uint8),
    sub5: wp.array3d(dtype=wp.uint8),
    sub6: wp.array3d(dtype=wp.uint8),
    sub7: wp.array3d(dtype=wp.uint8),
):
    if level == wp.int32(0):
        if ci < sub0.shape[0] and cj < sub0.shape[1] and ck < sub0.shape[2]:
            sub0[ci, cj, ck] = wp.uint8(1)
    elif level == wp.int32(1):
        if ci < sub1.shape[0] and cj < sub1.shape[1] and ck < sub1.shape[2]:
            sub1[ci, cj, ck] = wp.uint8(1)
    elif level == wp.int32(2):
        if ci < sub2.shape[0] and cj < sub2.shape[1] and ck < sub2.shape[2]:
            sub2[ci, cj, ck] = wp.uint8(1)
    elif level == wp.int32(3):
        if ci < sub3.shape[0] and cj < sub3.shape[1] and ck < sub3.shape[2]:
            sub3[ci, cj, ck] = wp.uint8(1)
    elif level == wp.int32(4):
        if ci < sub4.shape[0] and cj < sub4.shape[1] and ck < sub4.shape[2]:
            sub4[ci, cj, ck] = wp.uint8(1)
    elif level == wp.int32(5):
        if ci < sub5.shape[0] and cj < sub5.shape[1] and ck < sub5.shape[2]:
            sub5[ci, cj, ck] = wp.uint8(1)
    elif level == wp.int32(6):
        if ci < sub6.shape[0] and cj < sub6.shape[1] and ck < sub6.shape[2]:
            sub6[ci, cj, ck] = wp.uint8(1)
    elif level == wp.int32(7):
        if ci < sub7.shape[0] and cj < sub7.shape[1] and ck < sub7.shape[2]:
            sub7[ci, cj, ck] = wp.uint8(1)


@wp.kernel
def kernel_mark_subdivide_offset(
    owners: wp.array3d(dtype=wp.int32),
    di: wp.int32,
    dj: wp.int32,
    dk: wp.int32,
    sub0: wp.array3d(dtype=wp.uint8),
    sub1: wp.array3d(dtype=wp.uint8),
    sub2: wp.array3d(dtype=wp.uint8),
    sub3: wp.array3d(dtype=wp.uint8),
    sub4: wp.array3d(dtype=wp.uint8),
    sub5: wp.array3d(dtype=wp.uint8),
    sub6: wp.array3d(dtype=wp.uint8),
    sub7: wp.array3d(dtype=wp.uint8),
):
    i, j, k = wp.tid()
    o = owners[i, j, k]
    if o < wp.int32(0):
        return
    ni = i + di
    nj = j + dj
    nk = k + dk
    if ni < 0 or nj < 0 or nk < 0 or ni >= owners.shape[0] or nj >= owners.shape[1] or nk >= owners.shape[2]:
        return
    no = owners[ni, nj, nk]
    if no < wp.int32(0):
        return
    if no - o > wp.int32(1):
        stride = _stride_for_level(no)
        _mark_subdivide(no, ni // stride, nj // stride, nk // stride, sub0, sub1, sub2, sub3, sub4, sub5, sub6, sub7)
    if o - no > wp.int32(1):
        stride = _stride_for_level(o)
        _mark_subdivide(o, i // stride, j // stride, k // stride, sub0, sub1, sub2, sub3, sub4, sub5, sub6, sub7)


@wp.kernel
def kernel_apply_subdivide_level(
    subdivide: wp.array3d(dtype=wp.uint8),
    parent_mask: wp.array3d(dtype=wp.uint8),
    child_mask: wp.array3d(dtype=wp.uint8),
):
    ci, cj, ck = wp.tid()
    if subdivide[ci, cj, ck] == wp.uint8(0):
        return
    if parent_mask[ci, cj, ck] == wp.uint8(0):
        return
    parent_mask[ci, cj, ck] = wp.uint8(0)
    child_mask[ci * 2, cj * 2, ck * 2] = wp.uint8(1)
    child_mask[ci * 2 + 1, cj * 2, ck * 2] = wp.uint8(1)
    child_mask[ci * 2, cj * 2 + 1, ck * 2] = wp.uint8(1)
    child_mask[ci * 2 + 1, cj * 2 + 1, ck * 2] = wp.uint8(1)
    child_mask[ci * 2, cj * 2, ck * 2 + 1] = wp.uint8(1)
    child_mask[ci * 2 + 1, cj * 2, ck * 2 + 1] = wp.uint8(1)
    child_mask[ci * 2, cj * 2 + 1, ck * 2 + 1] = wp.uint8(1)
    child_mask[ci * 2 + 1, cj * 2 + 1, ck * 2 + 1] = wp.uint8(1)


@wp.kernel
def kernel_mark_coverage(
    num_levels: wp.int32,
    mask0: wp.array3d(dtype=wp.uint8),
    mask1: wp.array3d(dtype=wp.uint8),
    mask2: wp.array3d(dtype=wp.uint8),
    mask3: wp.array3d(dtype=wp.uint8),
    mask4: wp.array3d(dtype=wp.uint8),
    mask5: wp.array3d(dtype=wp.uint8),
    mask6: wp.array3d(dtype=wp.uint8),
    mask7: wp.array3d(dtype=wp.uint8),
    covered: wp.array3d(dtype=wp.uint8),
):
    i, j, k = wp.tid()
    covered[i, j, k] = wp.uint8(0)
    for level in range(num_levels):
        stride = _stride_for_level(level)
        ci = i // stride
        cj = j // stride
        ck = k // stride
        active = wp.uint8(0)
        if level == wp.int32(0):
            if ci < mask0.shape[0] and cj < mask0.shape[1] and ck < mask0.shape[2]:
                active = mask0[ci, cj, ck]
        elif level == wp.int32(1):
            if ci < mask1.shape[0] and cj < mask1.shape[1] and ck < mask1.shape[2]:
                active = mask1[ci, cj, ck]
        elif level == wp.int32(2):
            if ci < mask2.shape[0] and cj < mask2.shape[1] and ck < mask2.shape[2]:
                active = mask2[ci, cj, ck]
        elif level == wp.int32(3):
            if ci < mask3.shape[0] and cj < mask3.shape[1] and ck < mask3.shape[2]:
                active = mask3[ci, cj, ck]
        elif level == wp.int32(4):
            if ci < mask4.shape[0] and cj < mask4.shape[1] and ck < mask4.shape[2]:
                active = mask4[ci, cj, ck]
        elif level == wp.int32(5):
            if ci < mask5.shape[0] and cj < mask5.shape[1] and ck < mask5.shape[2]:
                active = mask5[ci, cj, ck]
        elif level == wp.int32(6):
            if ci < mask6.shape[0] and cj < mask6.shape[1] and ck < mask6.shape[2]:
                active = mask6[ci, cj, ck]
        elif level == wp.int32(7):
            if ci < mask7.shape[0] and cj < mask7.shape[1] and ck < mask7.shape[2]:
                active = mask7[ci, cj, ck]
        if active != wp.uint8(0):
            covered[i, j, k] = wp.uint8(1)


@wp.kernel
def kernel_fill_coverage_gaps(
    covered: wp.array3d(dtype=wp.uint8),
    coarsest_mask: wp.array3d(dtype=wp.uint8),
    coarse_stride: wp.int32,
):
    i, j, k = wp.tid()
    if covered[i, j, k] != wp.uint8(0):
        return
    ci = i // coarse_stride
    cj = j // coarse_stride
    ck = k // coarse_stride
    if ci < coarsest_mask.shape[0] and cj < coarsest_mask.shape[1] and ck < coarsest_mask.shape[2]:
        coarsest_mask[ci, cj, ck] = wp.uint8(1)


@wp.kernel
def kernel_any_uncovered(
    covered: wp.array3d(dtype=wp.uint8),
    counter: wp.array(dtype=wp.int32),
):
    i, j, k = wp.tid()
    if covered[i, j, k] == wp.uint8(0):
        wp.atomic_add(counter, 0, wp.int32(1))
