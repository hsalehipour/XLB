import neon
import warp as wp
from xlb.operator.stepper import MultiresIncompressibleNavierStokesStepper
from xlb.operator.macroscopic import MultiresMacroscopic
from xlb.mres_perf_optimization_type import MresPerfOptimizationType


class MultiresSimulationManager(MultiresIncompressibleNavierStokesStepper):
    """
    A simulation manager for multiresolution simulations using the Neon backend in XLB.
    """

    def __init__(
        self,
        omega_finest,
        grid,
        boundary_conditions=[],
        collision_type="BGK",
        forcing_scheme="exact_difference",
        force_vector=None,
        initializer=None,
        mres_perf_opt: MresPerfOptimizationType = MresPerfOptimizationType.NAIVE_COLLIDE_STREAM,
    ):
        super().__init__(grid, boundary_conditions, collision_type, forcing_scheme, force_vector)

        self.initializer = initializer
        self.count_levels = grid.count_levels
        self.omega_list = [self.compute_omega(omega_finest, level) for level in range(self.count_levels)]
        self.mres_perf_opt = mres_perf_opt
        # Create fields
        self.rho = grid.create_field(cardinality=1, dtype=self.precision_policy.store_precision)
        self.u = grid.create_field(cardinality=3, dtype=self.precision_policy.store_precision)
        self.coalescence_factor = grid.create_field(cardinality=self.velocity_set.q, dtype=self.precision_policy.store_precision)

        for level in range(self.count_levels):
            self.u.fill_run(level, 0.0, 0)
            self.rho.fill_run(level, 1.0, 0)
            self.coalescence_factor.fill_run(level, 0.0, 0)

        # Prepare fields
        self.f_0, self.f_1, self.bc_mask, self.missing_mask = self.prepare_fields(self.rho, self.u, self.initializer)
        self.prepare_coalescence_count(coalescence_factor=self.coalescence_factor, bc_mask=self.bc_mask)

        self.iteration_idx = -1
        self.macro = MultiresMacroscopic(
            compute_backend=self.compute_backend,
            precision_policy=self.precision_policy,
            velocity_set=self.velocity_set,
        )

        # Construct the stepper skeleton
        self._construct_stepper_skeleton()

    def compute_omega(self, omega_finest, level):
        """
        Compute the relaxation parameter omega at a given grid level based on the finest level omega.
        We select a refinement ratio of 2 where a coarse cell at level L is uniformly divided into 2^d cells
        where d is the dimension. to arrive at level L - 1, or in other words ∆x_{L-1} = ∆x_L/2.
        For neighboring cells that interface two grid levels, a maximum jump in grid level of ∆L = 1 is
        allowed. Due to acoustic scaling which requires the speed of sound cs to remain constant across various grid levels,
        ∆tL ∝ ∆xL and hence ∆t_{L-1} = ∆t_{L}/2. In addition, the fluid viscosity \nu must also remain constant on each
        grid level which leads to the following relationship for the relaxation parameter omega at grid level L base
        on the finest grid level omega_finest.

        Args:
            omega_finest: Relaxation parameter at the finest grid level.
            level: Current grid level (0-indexed, with 0 being the finest level).

        Returns:
            Relaxation parameter omega at the specified grid level.
        """
        omega0 = omega_finest
        return 2 ** (level + 1) * omega0 / ((2**level - 1.0) * omega0 + 2.0)

    def export_macroscopic(self, fname_prefix):
        print(f"exporting macroscopic: #levels {self.count_levels}")
        self.macro(self.f_0, self.bc_mask, self.rho, self.u, streamId=0)

        wp.synchronize()
        self.u.update_host(0)
        wp.synchronize()
        self.u.export_vti(f"{fname_prefix}{self.iteration_idx}.vti", "u")
        print("DONE exporting macroscopic")

        return

    def step(self):
        self.iteration_idx = self.iteration_idx + 1
        self.sk.run()

    def _build_recursion(self, level, app, config):
        """Unified multi-resolution recursion builder.

        config keys:
            finest_ops:         list of (op_name, swap_fields, extra_kwargs) for level 0,
                                or None to treat level 0 like any coarse level.
            coarse_collide_ops: list of op_names for coarse collision.
            coarse_stream_ops:  list of (op_name, extra_kwargs) for coarse streaming.
            fuse_finest:        if True, recurse once (not twice) when child is at level 0.
        """
        if level < 0:
            return

        omega = self.omega_list[level]
        fields = dict(f_0_fd=self.f_0, f_1_fd=self.f_1, bc_mask_fd=self.bc_mask, missing_mask_fd=self.missing_mask)
        fields_swapped = dict(f_0_fd=self.f_1, f_1_fd=self.f_0, bc_mask_fd=self.bc_mask, missing_mask_fd=self.missing_mask)

        if level == 0 and config["finest_ops"] is not None:
            for op_name, swap, extra in config["finest_ops"]:
                base = fields_swapped if swap else fields
                self.add_to_app(app=app, op_name=op_name, level=level, **base, omega=omega, **extra)
            return

        for op_name in config["coarse_collide_ops"]:
            self.add_to_app(app=app, op_name=op_name, level=level, **fields, omega=omega, timestep=0)

        if config["fuse_finest"] and level - 1 == 0:
            self._build_recursion(level - 1, app, config)
        else:
            self._build_recursion(level - 1, app, config)
            self._build_recursion(level - 1, app, config)

        for op_name, extra in config["coarse_stream_ops"]:
            self.add_to_app(app=app, op_name=op_name, level=level, **fields_swapped, **extra)

    def _construct_stepper_skeleton(self):
        self.app = []

        stream_abc = {"omega": self.coalescence_factor, "timestep": 0}

        # Finest-level op descriptors: (op_name, swap_f0_f1, extra_kwargs)
        fused_pull_finest = [
            ("finest_fused_pull", False, {"timestep": 0, "is_f1_the_explosion_src_field": True}),
            ("finest_fused_pull", True, {"timestep": 0, "is_f1_the_explosion_src_field": False}),
        ]
        sfv_fused_pull_finest = [
            ("CFV_finest_fused_pull", False, {"timestep": 0, "is_f1_the_explosion_src_field": True}),
            ("SFV_finest_fused_pull", False, {}),
            ("CFV_finest_fused_pull", True, {"timestep": 0, "is_f1_the_explosion_src_field": False}),
            ("SFV_finest_fused_pull", True, {}),
        ]

        configs = {
            MresPerfOptimizationType.NAIVE_COLLIDE_STREAM: {
                "finest_ops": None,
                "coarse_collide_ops": ["collide_coarse"],
                "coarse_stream_ops": [("stream_coarse_step_ABC", stream_abc)],
                "fuse_finest": False,
            },
            MresPerfOptimizationType.FUSION_AT_FINEST: {
                "finest_ops": fused_pull_finest,
                "coarse_collide_ops": ["collide_coarse"],
                "coarse_stream_ops": [("stream_coarse_step_ABC", stream_abc)],
                "fuse_finest": True,
            },
            MresPerfOptimizationType.FUSION_AT_FINEST_SFV: {
                "finest_ops": sfv_fused_pull_finest,
                "coarse_collide_ops": ["collide_coarse"],
                "coarse_stream_ops": [("stream_coarse_step_ABC", stream_abc)],
                "fuse_finest": True,
            },
            MresPerfOptimizationType.FUSION_AT_FINEST_SFV_ALL: {
                "finest_ops": sfv_fused_pull_finest,
                "coarse_collide_ops": ["CFV_collide_coarse", "SFV_collide_coarse"],
                "coarse_stream_ops": [("SFV_stream_coarse_step_ABC", stream_abc), ("SFV_stream_coarse_step", {})],
                "fuse_finest": True,
            },
        }

        config = configs.get(self.mres_perf_opt)
        if config is None:
            raise ValueError(f"Unknown optimization level: {self.mres_perf_opt}")

        # Pre-recursion SFV mask setup
        if self.mres_perf_opt == MresPerfOptimizationType.FUSION_AT_FINEST_SFV:
            wp.synchronize()
            self.neon_container["SFV_reset_bc_mask"](0, self.f_0, self.f_1, self.bc_mask, self.bc_mask).run(0)
            wp.synchronize()
        elif self.mres_perf_opt == MresPerfOptimizationType.FUSION_AT_FINEST_SFV_ALL:
            wp.synchronize()
            for l in range(self.f_0.get_grid().num_levels):
                self.neon_container["SFV_reset_bc_mask"](l, self.f_0, self.f_1, self.bc_mask, self.bc_mask).run(0)
            wp.synchronize()

        self._build_recursion(self.count_levels - 1, self.app, config)

        bk = self.grid.get_neon_backend()
        self.sk = neon.Skeleton(backend=bk)
        self.sk.sequence("mres_nse_stepper", self.app)
