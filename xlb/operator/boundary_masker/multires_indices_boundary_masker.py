"""
Multi-resolution indices-based boundary masker for the Neon backend.

Creates boundary masks from explicit voxel indices on multi-resolution
grids, computing missing-population masks for each tagged voxel.
"""

from typing import Any
import copy
import numpy as np

import warp as wp

from xlb.operator.operator import Operator
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.boundary_masker import IndicesBoundaryMasker
from xlb.grid.multires_grid import NeonMultiresGrid


class MultiresIndicesBoundaryMasker(IndicesBoundaryMasker):
    """
    Operator for creating a boundary mask using indices of boundary conditions in a multi-resolution setting.
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
        import neon

        # Use the warp functional for the NEON backend
        functional_dict, _ = self._construct_warp()
        functional_domain_bounds = functional_dict.get("functional_domain_bounds")
        functional_interior_bc_mask = functional_dict.get("functional_interior_bc_mask")
        functional_interior_missing_mask = functional_dict.get("functional_interior_missing_mask")

        @neon.Container.factory(name="IndicesBoundaryMasker_DomainBounds")
        def container_domain_bounds(
            wp_bc_indices,
            wp_id_numbers,
            wp_is_interior,
            bc_mask,
            missing_mask,
            grid_shape,
            level,
        ):
            def domain_bounds_launcher(loader: neon.Loader):
                loader.set_mres_grid(bc_mask.get_grid(), level)
                bc_mask_pn = loader.get_mres_write_handle(bc_mask)
                missing_mask_pn = loader.get_mres_write_handle(missing_mask)

                @wp.func
                def domain_bounds_kernel(index: Any):
                    # apply the functional
                    functional_domain_bounds(
                        index,
                        wp_bc_indices,
                        wp_id_numbers,
                        wp_is_interior,
                        bc_mask_pn,
                        missing_mask_pn,
                        grid_shape,
                        level,
                    )

                loader.declare_kernel(domain_bounds_kernel)

            return domain_bounds_launcher

        @neon.Container.factory(name="IndicesBoundaryMasker_InteriorBcMask")
        def container_interior_bc_mask(
            wp_bc_indices,
            wp_id_numbers,
            bc_mask,
            level,
        ):
            def interior_bc_mask_launcher(loader: neon.Loader):
                loader.set_mres_grid(bc_mask.get_grid(), level)
                bc_mask_pn = loader.get_mres_write_handle(bc_mask)

                @wp.func
                def interior_bc_mask_kernel(index: Any):
                    # apply the functional
                    functional_interior_bc_mask(
                        index,
                        wp_bc_indices,
                        wp_id_numbers,
                        bc_mask_pn,
                    )

                loader.declare_kernel(interior_bc_mask_kernel)

            return interior_bc_mask_launcher

        @neon.Container.factory(name="IndicesBoundaryMasker_InteriorMissingMask")
        def container_interior_missing_mask(
            wp_bc_indices,
            bc_mask,
            missing_mask,
            grid_shape,
            level,
        ):
            def interior_bc_mask_launcher(loader: neon.Loader):
                loader.set_mres_grid(bc_mask.get_grid(), level)
                bc_mask_pn = loader.get_mres_write_handle(bc_mask)
                missing_mask_pn = loader.get_mres_write_handle(missing_mask)

                @wp.func
                def interior_missing_mask_kernel(index: Any):
                    # apply the functional
                    functional_interior_missing_mask(
                        index,
                        wp_bc_indices,
                        bc_mask_pn,
                        missing_mask_pn,
                        grid_shape,
                        level,
                    )

                loader.declare_kernel(interior_missing_mask_kernel)

            return interior_bc_mask_launcher

        container_dict = {
            "container_domain_bounds": container_domain_bounds,
            "container_interior_bc_mask": container_interior_bc_mask,
            "container_interior_missing_mask": container_interior_missing_mask,
        }

        return functional_dict, container_dict

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, bclist, bc_mask, missing_mask, start_index=None):
        import neon

        grid = bc_mask.get_grid()
        num_levels = grid.num_levels
        domain_min_finest, domain_max_finest = self._domain_bounds_from_neon_grid(grid)

        # For adaptive meshes grid_shape_finest (from get_grid_shape) may
        # include dyadic-alignment padding cells that have no active voxels.
        # Using the padded dimensions for interior classification and
        # is_in_bounds causes max-side boundary voxels to be misclassified
        # as interior and outward-facing pulls to appear "in bounds",
        # leaving missing_mask empty and destabilising the outlet.
        # Use the actual active-domain extent instead.
        effective_dims = tuple(int(domain_max_finest[i]) + 1 for i in range(3))
        effective_grid_shape = wp.vec3i(*effective_dims)

        for level in range(num_levels):
            origin_pt = grid.sparsity_pattern_origins[level]
            origin = np.array([origin_pt.x, origin_pt.y, origin_pt.z], dtype=np.int64).reshape(3, 1)
            # Create a copy of the boundary condition list for the current level if the indices at that level are not empty
            bclist_at_level = []
            for bc in bclist:
                if bc.indices is not None and bc.indices[level]:
                    bc_copy = copy.copy(bc)  # shallow copy of the whole object
                    indices = copy.deepcopy(bc.indices[level])  # deep copy only the modified part
                    indices = np.asarray(indices, dtype=np.int64)
                    finest_virtual = ((indices + origin) * (2**level)).astype(np.int32)
                    finest_indices = NeonMultiresGrid.virtual_finest_to_neon_global(
                        finest_virtual,
                        domain_min_finest,
                        domain_max_finest,
                        level=level,
                        num_levels=num_levels,
                    )
                    bc_copy.indices = tuple(finest_indices.tolist())  # convert to tuple
                    bclist_at_level.append(bc_copy)

            # If the boundary condition list is empty, skip to the next level
            if not bclist_at_level:
                continue

            # BC indices are stored in finest-lattice space via ``(local + origin) * 2**level``.
            # A level-L voxel on the max-side domain face has its finest starting
            # index at ``domain_max[i] // stride * stride``.
            # ``are_indices_in_interior`` treats an index as interior when
            # ``idx < shape - 1``, so we build a per-level "interior shape"
            # from the actual domain extent (not the padded grid_shape_finest).
            stride = 2**level
            interior_shape = tuple(int(domain_max_finest[i]) - stride + 2 for i in range(3))

            # find interior boundary conditions
            bc_interior = self._find_bclist_interior(bclist_at_level, interior_shape)

            # Prepare the first kernel inputs for all items in boundary condition list
            wp_bc_indices, wp_id_numbers, wp_is_interior = self._prepare_kernel_inputs(bclist_at_level, interior_shape)

            # Launch the first container
            container_domain_bounds = self.neon_container["container_domain_bounds"](
                wp_bc_indices,
                wp_id_numbers,
                wp_is_interior,
                bc_mask,
                missing_mask,
                effective_grid_shape,
                level,
            )
            container_domain_bounds.run(0, container_runtime=neon.Container.ContainerRuntime.neon)

            # If there are no interior boundary conditions, skip the rest of the processing for this level
            if not bc_interior:
                continue

            # Prepare the second and third kernel inputs for only a subset of boundary conditions associated with the interior
            # Note 1: launching order of the following kernels are important here!
            # Note 2: Due to race conditioning, the two kernels cannot be fused together.
            wp_bc_indices, wp_id_numbers, _ = self._prepare_kernel_inputs(bc_interior, interior_shape)
            container_interior_missing_mask = self.neon_container["container_interior_missing_mask"](
                wp_bc_indices,
                bc_mask,
                missing_mask,
                effective_grid_shape,
                level,
            )
            container_interior_missing_mask.run(0, container_runtime=neon.Container.ContainerRuntime.neon)

            # Launch the third container
            container_interior_bc_mask = self.neon_container["container_interior_bc_mask"](
                wp_bc_indices,
                wp_id_numbers,
                bc_mask,
                level,
            )
            container_interior_bc_mask.run(0, container_runtime=neon.Container.ContainerRuntime.neon)

        return bc_mask, missing_mask

    @staticmethod
    def _domain_bounds_from_neon_grid(grid, d: int = 3):
        """Compute finest domain bounds from a Neon ``mGrid`` sparsity data."""
        level_data = []
        for level in range(grid.num_levels):
            origin_pt = grid.sparsity_pattern_origins[level]
            origin = np.array([origin_pt.x, origin_pt.y, origin_pt.z], dtype=np.int64)
            if getattr(grid, "is_sparse", False):
                pattern = grid.active_voxels_list[level]
            else:
                pattern = grid.sparsity_pattern_list[level]
            level_data.append((pattern, None, origin, level))
        return NeonMultiresGrid._domain_finest_bounds(level_data, d)
