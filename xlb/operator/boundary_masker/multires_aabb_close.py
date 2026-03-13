"""
Multi-resolution AABB-Close boundary masker with morphological closing.

Extends the AABB-Close masker for Neon multi-resolution grids, applying
dilate-then-erode operations to fill narrow channels with solid voxels.
"""

import warp as wp
import neon
from typing import Any
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.boundary_masker import MeshMaskerAABBClose
from xlb.operator.operator import Operator
from xlb.cell_type import BC_SOLID


class MultiresMeshMaskerAABBClose(MeshMaskerAABBClose):
    """
    Operator for creating boundary missing_mask from mesh using Axis-Aligned Bounding Box (AABB) voxelization
    in multiresolution simulations (NEON backend). It takes in a number of close_voxels to perform morphological
    operations (dilate followed by erode) to ensure small channels are filled with solid voxels.

    This version provides NEON-specific functionals working on multires partitions (mPartition) and bIndex.
    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        close_voxels: int = None,
    ):
        super().__init__(velocity_set, precision_policy, compute_backend, close_voxels)
        if self.compute_backend in [ComputeBackend.JAX, ComputeBackend.WARP]:
            raise NotImplementedError(f"Operator {self.__class__.__name__} not supported in {self.compute_backend} backend.")

        # Build and store NEON dicts
        self.neon_functional_dict, self.neon_container_dict = self._construct_neon()

    def _construct_neon(self):
        # Use the warp functionals from the base (for reference), but implement NEON variants here
        functional_dict_warp, _ = self._construct_warp()
        functional_erode_warp = functional_dict_warp.get("functional_erode")
        functional_dilate_warp = functional_dict_warp.get("functional_dilate")
        functional_solid = functional_dict_warp.get("functional_solid")
        # We will not directly reuse functional_solid / functional_aabb from warp; we write NEON-specific ones.

        # We also need lattice info for neighbor iteration
        _c = self.velocity_set.c
        _q = self.velocity_set.q
        _opp_indices = self.velocity_set.opp_indices

        # Set local constants
        lattice_central_index = self.velocity_set.center_index

        # Main AABB close: sets bc_mask, missing_mask, distances based on solid_mask
        # bc_mask: wp.uint8, missing_mask: wp.uint8, distances: dtype from precision policy (float)
        @wp.func
        def mres_functional_aabb(
            index: Any,
            mesh_id: wp.uint64,
            id_number: wp.int32,
            distances_pn: Any,  # mPartition(dtype=distance type), cardinality=_q
            bc_mask_pn: Any,  # mPartition_uint8, cardinality=1
            missing_mask_pn: Any,  # mPartition_uint8, cardinality=_q
            solid_mask_pn: Any,  # mPartition_uint8, cardinality=1
            needs_mesh_distance: bool,
        ):
            # Cell center from bc_mask partition
            cell_center = self.helper_masker.index_to_position(bc_mask_pn, index)

            # If already solid or bc, mark solid
            solid_val = wp.neon_read(solid_mask_pn, index, 0)
            bc_val = wp.neon_read(bc_mask_pn, index, 0)
            if solid_val == wp.uint8(BC_SOLID) or bc_val == wp.uint8(BC_SOLID):
                wp.neon_write(bc_mask_pn, index, 0, wp.uint8(BC_SOLID))
                return

            # loop lattice directions
            for direction_idx in range(_q):
                # skip central if provided by velocity set
                if direction_idx == lattice_central_index:
                    continue

                # If neighbor index is valid at this resolution level
                ngh = wp.neon_ngh_idx(wp.int8(_c[0, direction_idx]), wp.int8(_c[1, direction_idx]), wp.int8(_c[2, direction_idx]))
                is_valid = wp.bool(False)
                nval = wp.neon_read_ngh(solid_mask_pn, index, ngh, 0, wp.uint8(0), is_valid)
                if is_valid:
                    if nval == wp.uint8(BC_SOLID):
                        # Found solid neighbor -> boundary cell
                        self.write_field(bc_mask_pn, index, 0, wp.uint8(id_number))
                        self.write_field(missing_mask_pn, index, _opp_indices[direction_idx], wp.uint8(True))

                        if not needs_mesh_distance:
                            # No distance needed; continue to next direction
                            continue

                        # Compute mesh distance along lattice direction
                        dir_vec = wp.vec3f(
                            wp.float32(_c[0, direction_idx]),
                            wp.float32(_c[1, direction_idx]),
                            wp.float32(_c[2, direction_idx]),
                        )
                        max_length = wp.length(dir_vec)
                        # Avoid division by zero for any pathological dir (shouldn't happen)
                        norm_dir = dir_vec / (max_length if max_length > 0.0 else 1.0)
                        query = wp.mesh_query_ray(mesh_id, cell_center, norm_dir, 1.5 * max_length)
                        if query.result:
                            pos_mesh = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
                            dist = wp.length(pos_mesh - cell_center) - 0.5 * max_length
                            weight = dist / (max_length if max_length > 0.0 else 1.0)
                            # distances has cardinality _q; store into this channel
                            self.write_field(distances_pn, index, direction_idx, self.store_dtype(weight))
                        else:
                            self.write_field(distances_pn, index, direction_idx, self.store_dtype(1.0))

        # Containers

        # Erode: f_field -> f_field_out
        @neon.Container.factory(name="Erode")
        def container_erode(f_field: wp.array3d(dtype=Any), f_field_out: wp.array3d(dtype=Any), level: int):
            def erode_launcher(loader: neon.Loader):
                loader.set_mres_grid(f_field.get_grid(), level)
                f_field_pn = loader.get_mres_read_handle(f_field)
                f_field_out_pn = loader.get_mres_write_handle(f_field_out)

                @wp.func
                def erode_kernel(index: Any):
                    functional_erode_warp(index, f_field_pn, f_field_out_pn)

                loader.declare_kernel(erode_kernel)

            return erode_launcher

        # Dilate: f_field -> f_field_out
        @neon.Container.factory(name="Dilate")
        def container_dilate(f_field: wp.array3d(dtype=Any), f_field_out: wp.array3d(dtype=Any), level: int):
            def dilate_launcher(loader: neon.Loader):
                loader.set_mres_grid(f_field.get_grid(), level)
                f_field_pn = loader.get_mres_read_handle(f_field)
                f_field_out_pn = loader.get_mres_write_handle(f_field_out)

                @wp.func
                def dilate_kernel(index: Any):
                    functional_dilate_warp(index, f_field_pn, f_field_out_pn)

                loader.declare_kernel(dilate_kernel)

            return dilate_launcher

        # Solid mask: voxelize mesh into solid_mask
        @neon.Container.factory(name="Solid")
        def container_solid(mesh_id: wp.uint64, solid_mask: wp.array3d(dtype=wp.uint8), level: int):
            def solid_launcher(loader: neon.Loader):
                loader.set_mres_grid(solid_mask.get_grid(), level)
                solid_mask_pn = loader.get_mres_write_handle(solid_mask)

                @wp.func
                def solid_kernel(index: Any):
                    # apply the functional
                    functional_solid(index, mesh_id, solid_mask_pn, wp.vec3f(0.0, 0.0, 0.0))

                loader.declare_kernel(solid_kernel)

            return solid_launcher

        # Main AABB container
        @neon.Container.factory(name="MeshMaskerAABBClose")
        def container(
            mesh_id: Any,
            id_number: Any,
            distances: Any,
            bc_mask: Any,
            missing_mask: Any,
            solid_mask: Any,
            needs_mesh_distance: Any,
            level: Any,
        ):
            def aabb_launcher(loader: neon.Loader):
                loader.set_mres_grid(bc_mask.get_grid(), level)
                distances_pn = loader.get_mres_write_handle(distances)
                bc_mask_pn = loader.get_mres_write_handle(bc_mask)
                missing_mask_pn = loader.get_mres_write_handle(missing_mask)
                solid_mask_pn = loader.get_mres_write_handle(solid_mask)

                @wp.func
                def aabb_kernel(index: Any):
                    mres_functional_aabb(
                        index,
                        mesh_id,
                        id_number,
                        distances_pn,
                        bc_mask_pn,
                        missing_mask_pn,
                        solid_mask_pn,
                        needs_mesh_distance,
                    )

                loader.declare_kernel(aabb_kernel)

            return aabb_launcher

        container_dict = {
            "container_erode": container_erode,
            "container_dilate": container_dilate,
            "container_solid": container_solid,
            "container_aabb": container,
        }

        # Expose NEON functionals too (in case callers want to reuse)
        functional_dict = {
            "mres_functional_aabb": mres_functional_aabb,
        }

        return functional_dict, container_dict

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(
        self,
        bc,
        distances,
        bc_mask,
        missing_mask,
        stream=0,
    ):
        # Prepare inputs
        mesh_id, bc_id = self._prepare_kernel_inputs(bc, bc_mask)

        grid = bc_mask.get_grid()
        # Create fields using new_field
        solid_mask = grid.new_field(cardinality=1, dtype=wp.uint8, memory_type=neon.MemoryType.device())
        solid_mask_out = grid.new_field(
            cardinality=1,
            dtype=wp.uint8,
            memory_type=neon.MemoryType.device(),
            # memory_type=neon.MemoryType.host_device()
        )

        for level in range(grid.num_levels):
            # Initialize to 0
            solid_mask.fill_run(level=level, value=wp.uint8(0), stream_idx=stream)
            solid_mask_out.fill_run(level=level, value=wp.uint8(0), stream_idx=stream)

            # Launch the neon containers
            container_solid = self.neon_container_dict["container_solid"](mesh_id, solid_mask, level)
            container_solid.run(0, container_runtime=neon.Container.ContainerRuntime.neon)

            for _ in range(self.close_voxels):
                container_dilate = self.neon_container_dict["container_dilate"](solid_mask, solid_mask_out, level)
                container_dilate.run(0, container_runtime=neon.Container.ContainerRuntime.neon)
                solid_mask, solid_mask_out = solid_mask_out, solid_mask

            if self.close_voxels % 2 > 0:
                solid_mask, solid_mask_out = solid_mask_out, solid_mask

            for _ in range(self.close_voxels):
                container_erode = self.neon_container_dict["container_erode"](solid_mask_out, solid_mask, level)
                container_erode.run(0, container_runtime=neon.Container.ContainerRuntime.neon)
                solid_mask, solid_mask_out = solid_mask_out, solid_mask

            if self.close_voxels % 2 > 0:
                solid_mask, solid_mask_out = solid_mask_out, solid_mask

            container_aabb = self.neon_container_dict["container_aabb"](
                mesh_id, bc_id, distances, bc_mask, missing_mask, solid_mask, wp.static(bc.needs_mesh_distance), level
            )
            container_aabb.run(0, container_runtime=neon.Container.ContainerRuntime.neon)

        return distances, bc_mask, missing_mask
