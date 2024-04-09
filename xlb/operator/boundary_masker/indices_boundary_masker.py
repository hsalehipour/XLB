# Base class for all equilibriums

from functools import partial
import numpy as np
import jax.numpy as jnp
from jax import jit, lax
import warp as wp
from typing import Tuple

from xlb.global_config import GlobalConfig
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.stream.stream import Stream


class IndicesBoundaryMasker(Operator):
    """
    Operator for creating a boundary mask
    """

    def __init__(
        self,
        velocity_set: VelocitySet,
        precision_policy: PrecisionPolicy,
        compute_backend: ComputeBackend.JAX,
    ):
        # Make stream operator
        self.stream = Stream(velocity_set, precision_policy, compute_backend)

        # Call super
        super().__init__(velocity_set, precision_policy, compute_backend)

    @staticmethod
    def _indices_to_tuple(indices):
        """
        Converts a tensor of indices to a tuple for indexing
        """
        return tuple(indices.T)

    @Operator.register_backend(ComputeBackend.JAX)
    def jax_implementation(
        self, indices, id_number, boundary_id, mask, start_index=(0, 0, 0)
    ):
        local_indices = indices - np.array(start_index)[np.newaxis, :]

        # Remove any indices that are out of bounds
        indices_mask_x = (local_indices[:, 0] >= 0) & (local_indices[:, 0] < mask.shape[1])
        indices_mask_y = (local_indices[:, 1] >= 0) & (local_indices[:, 1] < mask.shape[2])
        indices_mask_z = (local_indices[:, 2] >= 0) & (local_indices[:, 2] < mask.shape[3])
        indices_mask = indices_mask_x & indices_mask_y & indices_mask_z

        local_indices = self._indices_to_tuple(local_indices[indices_mask])

        @jit
        def compute_boundary_id_and_mask(boundary_id, mask):
            boundary_id = boundary_id.at[0, local_indices[0], local_indices[1], local_indices[2]].set(id_number)
            mask = mask.at[:, local_indices[0], local_indices[1], local_indices[2]].set(True)
            mask = self.stream(mask)
            return boundary_id, mask

        return compute_boundary_id_and_mask(boundary_id, mask)

    def _construct_warp(self):
        # Make constants for warp
        _c = self.velocity_set.wp_c
        _q = wp.constant(self.velocity_set.q)

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            indices: wp.array2d(dtype=wp.int32),
            id_number: wp.int32,
            boundary_id: wp.array4d(dtype=wp.uint8),
            mask: wp.array4d(dtype=wp.bool),
            start_index: wp.vec3i,
        ):
            # Get the index of indices
            ii = wp.tid()

            # Get local indices
            index = wp.vec3i()
            index[0] = indices[ii, 0] - start_index[0]
            index[1] = indices[ii, 1] - start_index[1]
            index[2] = indices[ii, 2] - start_index[2]

            # Check if in bounds
            if (
                index[0] >= 0
                and index[0] < mask.shape[1]
                and index[1] >= 0
                and index[1] < mask.shape[2]
                and index[2] >= 0
                and index[2] < mask.shape[3]
            ):
                # Stream indices
                for l in range(_q):
                    # Get the index of the streaming direction
                    push_index = wp.vec3i()
                    for d in range(self.velocity_set.d):
                        push_index[d] = index[d] + _c[d, l]

                    # Set the boundary id and mask
                    boundary_id[0, index[0], index[1], index[2]] = (
                        wp.uint8(id_number)
                    )
                    mask[l, push_index[0], push_index[1], push_index[2]] = True

        return None, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(
        self, indices, id_number, boundary_id, missing_mask, start_index=(0, 0, 0)
    ):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                indices,
                id_number,
                boundary_id,
                missing_mask,
                start_index,
            ],
            dim=indices.shape[0],
        )

        return boundary_id, missing_mask