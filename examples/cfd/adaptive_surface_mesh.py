"""
Build and inspect a surface-adaptive multi-resolution mesh from any STL/OBJ file.

Generates a distance-driven, strongly-balanced multires domain with fine cells near
the surface and progressively coarser cells outward.  Output is compatible with
:func:`xlb.utils.mesher.prepare_sparsity_pattern` and Neon multires grid construction.

Examples
--------
Debug mesh on the checked-in sphere (small, fast):

    python examples/cfd/adaptive_surface_mesh.py \\
        --stl examples/cfd/stl-files/sphere.stl \\
        --voxel-size 2 --num-levels 3 \\
        --domain-padding 2 2 2 2 2 2 \\
        --no-shift-stl

Large model with STL shift and ParaView export (coordinates in the original STL frame):

    python examples/cfd/adaptive_surface_mesh.py \\
        --stl path/to/large_model.stl \\
        --voxel-size 12 --num-levels 4 \\
        --domain-padding 0.5 0.5 0.5 0.5 0.25 1.0 \\
        --export-mesh

Compare adaptive vs cuboid cell counts:

    python examples/cfd/adaptive_surface_mesh.py \\
        --stl examples/cfd/stl-files/sphere.stl \\
        --voxel-size 2 --num-levels 3 --compare-cuboid \\
        --no-shift-stl

Domain padding is ``[-x, +x, -y, +y, -z, +z]`` multiples of the geometry extent.
Level shell thickness is controlled by ``--finest-band-cells`` (inner finest shell)
and ``--expansion-ratio`` (geometric growth between outer shells).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from typing import List, Sequence, Tuple

import numpy as np
import trimesh

from xlb.utils.adaptive_mesher import make_adaptive_surface_mesh, validate_level_data
from xlb.utils.mesher import make_cuboid_mesh, prepare_sparsity_pattern, MultiresIO


def grid_shape_finest(level_data: list) -> Tuple[int, int, int]:
    """Finest lattice shape implied by coarsest-level mask and level count."""
    num_levels = len(level_data)
    return tuple(int(level_data[-1][0].shape[i] * 2 ** (num_levels - 1)) for i in range(3))


def default_cuboid_multipliers(domain_padding: Sequence[float], num_levels: int) -> List[List[float]]:
    """
    Build nested cuboid domain multipliers from the outer padding.

    Scales each axis of ``domain_padding`` toward the geometry for finer levels.
    """
    if num_levels < 1:
        raise ValueError("num_levels must be at least 1.")
    if num_levels == 1:
        return [list(domain_padding)]

    scale_steps = [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.7, 0.7, 0.7, 0.7, 0.8, 0.8],
        [0.4, 0.4, 0.4, 0.4, 0.6, 0.6],
        [0.2, 0.2, 0.2, 0.2, 0.4, 0.4],
    ]
    multipliers: List[List[float]] = []
    for level in range(num_levels):
        if level == 0:
            multipliers.append(list(domain_padding))
            continue
        step = scale_steps[min(level, len(scale_steps) - 1)]
        multipliers.append([domain_padding[i] * step[i] for i in range(6)])
    return multipliers


def load_and_shift_stl(stl_path: str, domain_padding: Sequence[float]) -> Tuple[str, np.ndarray]:
    """
    Load geometry and translate it into the mesh-domain frame.

    The mesh origin is placed so the geometry sits inside the outer padded domain
    starting near the positive octant.  Returns a temporary STL path and the
    translation vector applied (for mapping exports back to the original frame).
    """
    loaded = trimesh.load(stl_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            raise ValueError(f"Loaded mesh is empty: {stl_path}")
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    else:
        mesh = loaded

    min_bound = mesh.vertices.min(axis=0)
    max_bound = mesh.vertices.max(axis=0)
    part_size = max_bound - min_bound

    stl_shift = np.array(
        [
            domain_padding[0] * part_size[0] - min_bound[0],
            domain_padding[2] * part_size[1] - min_bound[1],
            domain_padding[4] * part_size[2] - min_bound[2],
        ],
        dtype=float,
    )
    mesh.apply_translation(stl_shift)
    _ = mesh.vertex_normals

    fd, temp_path = tempfile.mkstemp(suffix=".stl", prefix="adaptive_mesh_")
    os.close(fd)
    mesh.export(temp_path)
    return temp_path, stl_shift


def print_mesh_statistics(label: str, level_data: list, grid_shape_finest_grid: Tuple[int, int, int]) -> Tuple[int, int]:
    """Print per-level counts, validation flags, and return total active / equiv. finest."""
    num_levels = len(level_data)
    sparsity_pattern, level_origins = prepare_sparsity_pattern(level_data)

    print(f"\n{label}")
    print("=" * len(label))
    print(f"Finest grid shape: {grid_shape_finest_grid}")
    for lvl in range(num_levels):
        active = int(np.count_nonzero(sparsity_pattern[lvl]))
        equiv_finest = active * (2 ** (num_levels - 1 - lvl))
        print(
            f"  Level {lvl}: active={active:,}, "
            f"mask shape={sparsity_pattern[lvl].shape}, "
            f"origin={level_origins[lvl]}, "
            f"equiv. finest cells={equiv_finest:,}"
        )

    total_active = sum(int(np.count_nonzero(m)) for m in sparsity_pattern)
    total_equiv_finest = sum(
        int(np.count_nonzero(sparsity_pattern[lvl])) * (2 ** (num_levels - 1 - lvl))
        for lvl in range(num_levels)
    )
    print(f"  Total active cells: {total_active:,}")
    print(f"  Total equivalent finest cells: {total_equiv_finest:,}")

    stats = validate_level_data(level_data, grid_shape_finest_grid)
    print(
        f"  Validation: non_overlapping={stats['non_overlapping']}, "
        f"fully_covering={stats['fully_covering']}, "
        f"strongly_balanced={stats['strongly_balanced']}"
    )
    return total_active, total_equiv_finest


def export_mesh_xdmf(
    level_data: list,
    voxel_size: float,
    output_basename: str,
    export_offset: Tuple[float, float, float],
    original_stl: str,
) -> None:
    """Write HDF5/XDMF geometry for ParaView inspection."""
    exporter = MultiresIO.__new__(MultiresIO)
    exporter.unit_convertor = None
    coords, conn, level_ids, n_cells = MultiresIO.process_geometry(exporter, level_data)
    coords, conn = MultiresIO._merge_duplicates(exporter, coords, conn, level_data)
    coords = MultiresIO._transform_coordinates(exporter, coords * voxel_size, export_offset)
    MultiresIO.save_xdmf(exporter, f"{output_basename}.h5", f"{output_basename}.xmf", n_cells, len(coords), fields={})
    MultiresIO.save_hdf5_file(exporter, output_basename, coords, conn, level_ids, fields_data={})
    print(f"\nExported {n_cells:,} cells to {output_basename}.xmf")
    if export_offset != (0.0, 0.0, 0.0):
        print(f"  Coordinates in original STL frame (offset applied: {export_offset})")
        print(f"  Overlay in ParaView with: {original_stl}")
    else:
        print(f"  Overlay in ParaView with: {original_stl}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and inspect a surface-adaptive multires mesh from an STL/OBJ file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Level shells: finest band width = finest_band_cells * voxel_size; "
            "each outer shell grows by expansion_ratio."
        ),
    )
    parser.add_argument(
        "--stl",
        required=True,
        help="Path to input STL/OBJ geometry.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=4.0,
        help="Finest cell size in physical units (default: 4.0).",
    )
    parser.add_argument(
        "--num-levels",
        type=int,
        default=4,
        help="Number of refinement levels; 0 is finest (default: 4).",
    )
    parser.add_argument(
        "--expansion-ratio",
        type=float,
        default=2.0,
        help="Geometric ratio between consecutive distance shells (default: 2.0).",
    )
    parser.add_argument(
        "--finest-band-cells",
        type=int,
        default=3,
        help="Finest shell thickness in finest-cell counts (default: 3).",
    )
    parser.add_argument(
        "--domain-padding",
        type=float,
        nargs=6,
        metavar=("MX", "PX", "MY", "PY", "MZ", "PZ"),
        default=[0.5, 0.5, 0.5, 0.5, 0.25, 1.0],
        help=(
            "Outer domain padding as [-x, +x, -y, +y, -z, +z] multiples of "
            "geometry extent (default: 0.5 0.5 0.5 0.5 0.25 1.0)."
        ),
    )
    parser.add_argument(
        "--shift-stl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Translate geometry into the mesh-domain frame before meshing "
            "(default: on). Disable for small models already in mesh coordinates."
        ),
    )
    parser.add_argument(
        "--compare-cuboid",
        action="store_true",
        help="Also build a nested cuboid mesh for cell-count comparison.",
    )
    parser.add_argument(
        "--export-mesh",
        action="store_true",
        help="Export mesh geometry to HDF5/XDMF for ParaView.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output basename for --export-mesh (default: <stl_stem>_adaptive_mesh).",
    )
    parser.add_argument(
        "--max-dense-cells",
        type=int,
        default=128**3,
        help="Use dense distance field when finest grid is smaller than this (default: 128^3).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    stl_path = os.path.abspath(args.stl)
    if not os.path.isfile(stl_path):
        print(f"STL not found: {stl_path}", file=sys.stderr)
        return 1

    domain_padding = list(args.domain_padding)
    output_basename = args.output
    if output_basename is None:
        output_basename = f"{os.path.splitext(os.path.basename(stl_path))[0]}_adaptive_mesh"

    print("Surface-adaptive mesh generation")
    print(f"  STL: {stl_path}")
    print(f"  voxel_size={args.voxel_size}, num_levels={args.num_levels}")
    print(f"  expansion_ratio={args.expansion_ratio}, finest_band_cells={args.finest_band_cells}")
    print(f"  domain_padding={domain_padding}")
    print(f"  shift_stl={args.shift_stl}")

    temp_stl: str | None = None
    stl_shift = np.zeros(3, dtype=float)
    mesh_stl = stl_path

    try:
        if args.shift_stl:
            temp_stl, stl_shift = load_and_shift_stl(stl_path, domain_padding)
            mesh_stl = temp_stl
            print(f"  STL shift applied: {stl_shift}")

        t0 = time.perf_counter()
        level_data = make_adaptive_surface_mesh(
            voxel_size=args.voxel_size,
            num_levels=args.num_levels,
            stl_filename=mesh_stl,
            domain_padding=domain_padding,
            expansion_ratio=args.expansion_ratio,
            finest_band_cells=args.finest_band_cells,
            max_dense_cells=args.max_dense_cells,
        )
        print(f"\nAdaptive mesh built in {time.perf_counter() - t0:.1f} s")

        gs = grid_shape_finest(level_data)
        adaptive_total, adaptive_equiv = print_mesh_statistics("Adaptive surface mesh", level_data, gs)

        if args.compare_cuboid:
            cuboid_multipliers = default_cuboid_multipliers(domain_padding, args.num_levels)
            t0 = time.perf_counter()
            cuboid_data = make_cuboid_mesh(args.voxel_size, cuboid_multipliers, mesh_stl)
            print(f"\nCuboid mesh built in {time.perf_counter() - t0:.1f} s")

            cuboid_gs = grid_shape_finest(cuboid_data)
            cuboid_total, cuboid_equiv = print_mesh_statistics("Cuboid mesh (comparison)", cuboid_data, cuboid_gs)

            print("\nComparison")
            print("==========")
            print(f"  Adaptive finest-level cells: {int(np.count_nonzero(level_data[0][0])):,}")
            print(f"  Cuboid finest-level cells:   {int(np.count_nonzero(cuboid_data[0][0])):,}")
            print(f"  Adaptive total active:       {adaptive_total:,}")
            print(f"  Cuboid total active:         {cuboid_total:,}")
            print(f"  Adaptive equiv. finest:      {adaptive_equiv:,}")
            print(f"  Cuboid equiv. finest:        {cuboid_equiv:,}")
            if cuboid_equiv > 0:
                savings = 100.0 * (1.0 - adaptive_equiv / cuboid_equiv)
                print(f"  Equivalent-finest savings:   {savings:.1f}%")

        if args.export_mesh:
            export_offset = tuple(-stl_shift) if args.shift_stl else (0.0, 0.0, 0.0)
            export_mesh_xdmf(level_data, args.voxel_size, output_basename, export_offset, stl_path)

    finally:
        if temp_stl is not None and os.path.isfile(temp_stl):
            os.remove(temp_stl)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
