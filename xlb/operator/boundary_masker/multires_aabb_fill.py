import warp as wp
from typing import Any
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.boundary_masker import MeshMaskerAABB
from xlb.operator.operator import Operator
import neon


class MultiresMeshMaskerAABBFill(MeshMaskerAABBFill):
    """
    Operator for creating boundary missing_mask from mesh using Axis-Aligned Bounding Box (AABB) voxelization in multiresolution simulations.

    This implementation uses warp.mesh_query_aabb for efficient mesh-voxel intersection testing,
    providing approximate 1-voxel thick surface detection around the mesh geometry.
    Suitable for scenarios where fast, approximate boundary detection is sufficient.
    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
    ):
        # Call super
        super().__init__(velocity_set, precision_policy, compute_backend)
        if self.compute_backend in [ComputeBackend.JAX, ComputeBackend.WARP]:
            raise NotImplementedError(f"Operator {self.__class__.__name__} not supported in {self.compute_backend} backend.")
            

    def _construct_neon(self):
        # Use the warp functional for the NEON backend
        functional_dict, _ = self._construct_warp()
        functional_erode = functional_dict.get("functional_erode")
        functional_dilate = functional_dict.get("functional_dilate")
        functional_solid = functional_dict.get("functional_solid")
        functional_aabb = functional_dict.get("functional_aabb")

        # Erode the solid mask in f_field, removing a layer of outer solid voxels, storing output in f_field_out
        @neon.Container.factory(name="Erode")
        # TODO: check parameters
        def container_erode(f_field: wp.array3d(dtype=Any), f_field_out: wp.array3d(dtype=Any), level: int):
            def erode_launcher(loader: neon.Loader):
                loader.set_mres_grid(f_field.get_grid(), level)
                f_field_pn = loader.get_mres_read_handle(f_field)
                f_field_out_pn = loader.get_mres_write_handle(f_field_out)

                @wp.func
                def erode_kernel(index: Any):
                    # apply the functional
                    functional_erode(
                        index,
                        f_field_pn,
                        f_field_out_pn,
                    )

                loader.declare_kernel(erode_kernel)
            return erode_launcher

        # Dilate the solid mask in f_field, adding a layer of outer solid voxels, storing output in f_field_out
        @neon.Container.factory(name="Dilate")
        # TODO: check parameters
        def container_dilate(f_field: wp.array3d(dtype=Any), f_field_out: wp.array3d(dtype=Any), level: int):
            def dilate_launcher(loader: neon.Loader):
                loader.set_mres_grid(f_field.get_grid(), level)
                f_field_pn = loader.get_mres_read_handle(f_field)
                f_field_out_pn = loader.get_mres_write_handle(f_field_out)

                @wp.func
                def dilate_kernel(index: Any):
                    # apply the functional
                    functional_dilate(
                        index,
                        f_field_pn,
                        f_field_out_pn,
                    )

                loader.declare_kernel(dilate_kernel)
            return dilate_launcher

        # Construct the warp kernel
        # Find solid voxels that intersect the mesh
        @neon.Container.factory(name="Solid")
        # TODO: check parameters
        def container_solid(
            mesh_id: wp.uint64,
            solid_mask: wp.array3d(dtype=wp.int32),
            level: int
        ):
            def solid_launcher(loader: neon.Loader):
                loader.set_mres_grid(solid_mask.get_grid(), level)
                solid_mask_pn = loader.get_mres_write_handle(solid_mask)

                @wp.func
                def solid_kernel(index: Any):
                    # apply the functional
                    functional_solid(
                        index,
                        mesh_id,
                        solid_mask_pn
                    )

                loader.declare_kernel(solid_kernel)

            return solid_launcher

        @neon.Container.factory(name="MeshMaskerAABBFill")
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
                    # apply the functional
                    functional_aabb(
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
        # Is this right?
        solid_mask = grid.create_field(cardinality=1, dtype=Precision.UINT8, fill_value=0)
        solid_mask_out = grid.create_field(cardinality=1, dtype=Precision.UINT8, fill_value=0)
        for level in range(grid.num_levels):

            # TODO: Prepare kernel inputs?

            # Launch the neon container
            # c = self.neon_container(mesh_id, bc_id, distances, bc_mask, missing_mask, wp.static(bc.needs_mesh_distance), level)
            container_solid = self.neon_container_dict["container_solid"](mesh_id, solid_mask, level)
            container_solid.run(0, container_runtime=neon.Container.ContainerRuntime.neon)

            container_dilate = self.neon_container_dict["container_dilate"](solid_mask, solid_mask_out, level)
            container_dilate.run(0, container_runtime=neon.Container.ContainerRuntime.neon)

            container_erode = self.neon_container_dict["container_erode"](solid_mask_out, solid_mask, level)
            container_erode.run(0, container_runtime=neon.Container.ContainerRuntime.neon)

            container_aabb = self.neon_container_dict["container_aabb"](mesh_id, bc_id, distances, bc_mask, missing_mask, solid_mask, wp.static(bc.needs_mesh_distance), level)
            container_aabb.run(0, container_runtime=neon.Container.ContainerRuntime.neon)

        return distances, bc_mask, missing_mask
