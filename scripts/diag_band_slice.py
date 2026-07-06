"""Diagnostic: reconstruct the per-cell level field and dump vertical slices.

Runs the Warp octree mesher for a small config, rebuilds the dense finest-level
owner field from the extracted masks, and saves x-z slices as a PNG so we can
visually inspect the finest-band coherence.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import trimesh

from xlb.utils.adaptive_mesher import AdaptiveMeshConfig, _compute_domain
from xlb.utils.adaptive_mesher_warp import make_masks_octree_warp, make_masks_dense_warp

STL = "examples/cfd/stl-files/07022026_SEPULVEDA_SITE_MODEL_FORMA_NOTREES.stl"


def reconstruct_levels(masks, num_levels, grid_shape):
    nx, ny, nz = grid_shape
    owners = np.full((nx, ny, nz), -1, dtype=np.int32)
    for level in range(num_levels - 1, -1, -1):
        stride = 2 ** level
        mask = masks[level]
        if not np.any(mask):
            continue
        up = np.repeat(np.repeat(np.repeat(mask, stride, 0), stride, 1), stride, 2)
        ex, ey, ez = up.shape
        sub = owners[:ex, :ey, :ez]
        sub[up] = level
        owners[:ex, :ey, :ez] = sub
    return owners


def main():
    import sys
    big = "--big" in sys.argv
    voxel = 4.0 if big else 12.0
    num_levels = 6 if big else 4
    padding = (0.5, 0.5, 0.5, 0.5, 0.05, 3.0) if big else (0.5, 0.5, 0.5, 0.5, 0.25, 1.0)
    config = AdaptiveMeshConfig(
        voxel_size=voxel,
        num_levels=num_levels,
        expansion_ratio=2.0,
        finest_band_cells=2,
        domain_padding=padding,
        stl_filename=STL,
    )
    mesh, origin_phys, grid_shape = _compute_domain(config)
    nx, ny, nz = grid_shape
    factor = 2 ** (num_levels - 1)
    pad = [(factor - s % factor) % factor for s in grid_shape]
    grid_shape = tuple(grid_shape[i] + pad[i] for i in range(3))
    align = 2 ** num_levels
    grid_shape = tuple(((n + align - 1) // align) * align for n in grid_shape)

    masks, mask_origins = make_masks_octree_warp(mesh, origin_phys, grid_shape, config)
    owners = reconstruct_levels(masks, num_levels, grid_shape)
    nx, ny, nz = grid_shape

    # Analyze L0 band: for each (x,y) column, count contiguous L0 in z, and
    # detect isolated L0 cells (protrusions) whose neighbors are not L0.
    l0 = owners == 0
    # Count L0 cells that have < 3 of 6 face-neighbors also L0 (isolated-ish)
    neigh = np.zeros_like(l0, dtype=np.int32)
    neigh[1:] += l0[:-1]
    neigh[:-1] += l0[1:]
    neigh[:, 1:] += l0[:, :-1]
    neigh[:, :-1] += l0[:, 1:]
    neigh[:, :, 1:] += l0[:, :, :-1]
    neigh[:, :, :-1] += l0[:, :, 1:]
    isolated = l0 & (neigh <= 1)
    print(f"L0 cells: {l0.sum():,}")
    print(f"L0 cells with <=1 L0 face-neighbor (protrusions): {isolated.sum():,}")

    # Save a few x-z slices at different y.
    ys = [ny // 4, ny // 2, 3 * ny // 4]
    fig, axes = plt.subplots(len(ys), 1, figsize=(14, 10))
    for ax, y in zip(axes, ys):
        sl = owners[:, y, :].T  # z rows, x cols
        im = ax.imshow(sl, origin="lower", aspect="auto", cmap="viridis",
                       vmin=-1, vmax=num_levels - 1, interpolation="nearest")
        ax.set_title(f"x-z slice at y={y}")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        plt.colorbar(im, ax=ax, label="level (-1=empty)")
    plt.tight_layout()
    out = "scripts/diag_band_slice.png"
    plt.savefig(out, dpi=110)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
