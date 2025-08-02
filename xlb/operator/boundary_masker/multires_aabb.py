import warp as wp
from typing import Any
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.boundary_masker import MeshMaskerAABB
from xlb.operator.operator import Operator
import neon


class MultiresMeshMaskerAABB(MeshMaskerAABB):
    """
    Operator for creating boundary missing_mask from mesh using Axis-Aligned Bounding Box (AABB) voxelization in multiresolution simulations.

    This implementation uses warp.mesh_query_aabb for efficient mesh-voxel intersection testing,
    providing approximate 1-voxel thick surface detection around the mesh geometry.
    Suitable for scenarios where fast, approximate boundary detection is sufficient.

    Modified to always tag enclosed spaces as solid voxels.
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

    def _construct_warp(self):
        # Make constants for warp
        _c = self.velocity_set.c
        _q = self.velocity_set.q
        _opp_indices = self.velocity_set.opp_indices

        # Set local constants
        lattice_central_index = self.velocity_set.center_index

        @wp.func
        def is_point_inside_mesh(mesh_id: wp.uint64, pos: wp.vec3f, max_length: wp.float32):
            query = wp.mesh_query_point_sign_winding_number(mesh_id, pos, max_length)
            if query.result:
                return query.sign < 0  # Inside if sign is negative (adjust if needed based on convention)
            return False

        @wp.func
        def functional(
            index: Any,
            mesh_id: Any,
            id_number: Any,
            distances: Any,
            bc_mask: Any,
            missing_mask: Any,
            needs_mesh_distance: wp.bool,
            grid_shape: wp.vec3i,  # Grid dimensions for the current level
        ):
            # position of the point
            cell_center_pos = self.helper_masker.index_to_position(bc_mask, index)
            HALF_VOXEL = wp.vec3(0.5, 0.5, 0.5)

            # Compute max_length for winding number (diagonal of the domain)
            max_length_domain = wp.sqrt(
                wp.float32(grid_shape[0]) ** wp.float32(2.0)
                + wp.float32(grid_shape[1]) ** wp.float32(2.0)
                + wp.float32(grid_shape[2]) ** wp.float32(2.0)
            )

            # Check if this voxel is solid: already 255, intersects, or is inside the mesh
            if (
                self.read_field(bc_mask, index, 0) == wp.uint8(255)
                or self.mesh_voxel_intersect(mesh_id=mesh_id, low=cell_center_pos - HALF_VOXEL)
                or is_point_inside_mesh(mesh_id, cell_center_pos, max_length_domain)
            ):
                # Make solid voxel
                self.write_field(bc_mask, index, 0, wp.uint8(255))
            else:
                # Find the boundary voxels and their missing directions
                for direction_idx in range(_q):
                    if direction_idx == lattice_central_index:
                        # Skip the central index as it is not relevant for boundary masking
                        continue

                    # Get the lattice direction vector
                    direction_vec = wp.vec3f(wp.float32(_c[0, direction_idx]), wp.float32(_c[1, direction_idx]), wp.float32(_c[2, direction_idx]))

                    # Neighbor center position
                    neighbor_center_pos = cell_center_pos + direction_vec

                    # Check if neighbor is solid: intersects or is inside
                    if (
                        self.mesh_voxel_intersect(mesh_id=mesh_id, low=neighbor_center_pos - HALF_VOXEL)
                        or is_point_inside_mesh(mesh_id, neighbor_center_pos, max_length_domain)
                    ):
                        # We know we have a solid neighbor
                        # Set the boundary id and missing_mask
                        self.write_field(bc_mask, index, 0, wp.uint8(id_number))
                        self.write_field(missing_mask, index, _opp_indices[direction_idx], wp.uint8(True))

                        # If we don't need the mesh distance, we can continue
                        if not needs_mesh_distance:
                            continue

                        # Find the fractional distance to the mesh in each direction
                        # We increase max_length to find intersections in neighboring cells
                        max_length = wp.length(direction_vec)
                        query = wp.mesh_query_ray(mesh_id, cell_center_pos, direction_vec / max_length, 1.5 * max_length)
                        if query.result:
                            # get position of the mesh triangle that intersects with the ray
                            pos_mesh = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
                            # We reduce the distance to give some wall thickness
                            dist = wp.length(pos_mesh - cell_center_pos) - 0.5 * max_length
                            weight = dist / max_length
                            self.write_field(distances, index, direction_idx, self.store_dtype(weight))
                        else:
                            # Expected an intersection in this direction but none was found.
                            # Assume the solid extends one lattice unit beyond the BC voxel leading to a distance fraction of 1.
                            self.write_field(distances, index, direction_idx, self.store_dtype(1.0))

        @wp.kernel
        def kernel(
            mesh_id: wp.uint64,
            id_number: wp.int32,
            distances: wp.array4d(dtype=Any),
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.uint8),
            needs_mesh_distance: wp.bool,
            grid_shape: wp.vec3i,  # Grid dimensions for the current level
        ):
            # get index
            i, j, k = wp.tid()

            # Get local indices
            index = wp.vec3i(i, j, k)

            functional(
                index,
                mesh_id,
                id_number,
                distances,
                bc_mask,
                missing_mask,
                needs_mesh_distance,
                grid_shape,
            )

        return functional, kernel

    def _construct_neon(self):
        # Use the warp functional for the NEON backend (now modified for enclose)
        functional, _ = self._construct_warp()

        @neon.Container.factory(name="MeshMaskerAABB")
        def container(
            mesh_id: Any,
            id_number: Any,
            distances: Any,
            bc_mask: Any,
            missing_mask: Any,
            needs_mesh_distance: wp.bool,
            level: Any,
        ):
            def aabb_launcher(loader: neon.Loader):
                loader.set_mres_grid(bc_mask.get_grid(), level)
                distances_pn = loader.get_mres_write_handle(distances)
                bc_mask_pn = loader.get_mres_write_handle(bc_mask)
                missing_mask_pn = loader.get_mres_write_handle(missing_mask)
                # Compute grid shape for the current level (mimicking NeonMultiresGrid.level_to_shape)
                base_shape = bc_mask.get_grid().dim  # Access base dimensions (Index_3d)
                refinement_factor = 2  # Hardcoded as in NeonMultiresGrid
                grid_shape = wp.vec3i(
                    base_shape.x // (refinement_factor ** level),
                    base_shape.y // (refinement_factor ** level),
                    base_shape.z // (refinement_factor ** level)
                )

                @wp.func
                def aabb_kernel(index: Any):
                    # apply the functional
                    functional(
                        index,
                        mesh_id,
                        id_number,
                        distances_pn,
                        bc_mask_pn,
                        missing_mask_pn,
                        needs_mesh_distance,
                        grid_shape,
                    )

                loader.declare_kernel(aabb_kernel)

            return aabb_launcher

        return functional, container

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
        for level in range(grid.num_levels):
            # Launch the neon container
            c = self.neon_container(
                mesh_id,
                bc_id,
                distances,
                bc_mask,
                missing_mask,
                wp.static(bc.needs_mesh_distance),
                level,
            )
            c.run(stream, container_runtime=neon.Container.ContainerRuntime.neon)
        return distances, bc_mask, missing_mask