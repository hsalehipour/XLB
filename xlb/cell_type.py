# Boundary-mask constants for the bc_mask field.
# Each voxel in the domain carries a uint8 tag in bc_mask that encodes its role:
#   BC_NONE  — regular fluid voxel (no boundary condition)
#   BC_SFV   — Simple Fluid Voxel: fluid cell not involved in any BC,
#              explosion, or coalescence (used for fast-path kernels)
#   BC_SOLID — solid / obstacle voxel (skipped by all LBM operators)
# Registered boundary conditions receive IDs in the range [1, 253].

BC_NONE = 0
BC_SFV = 254
BC_SOLID = 255
