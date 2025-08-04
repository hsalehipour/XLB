import neon
import warp as wp
from xlb.operator.stepper import MultiresIncompressibleNavierStokesStepper
from xlb.operator.macroscopic import MultiresMacroscopic
from xlb.mres_perf_ptimization_type import MresPerfOptimizationType


class MultiresSimulationManager(MultiresIncompressibleNavierStokesStepper):
    """
    A simulation manager for multiresolution simulations using the Neon backend in XLB.
    """

    def __init__(
            self,
            omega,
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
        self.omega = omega
        self.count_levels = grid.count_levels
        self.mres_perf_opt = mres_perf_opt
        # Create fields
        self.rho = grid.create_field(cardinality=1, dtype=self.precision_policy.store_precision)
        self.u = grid.create_field(cardinality=3, dtype=self.precision_policy.store_precision)
        self.coalescence_factor = grid.create_field(cardinality=self.velocity_set.q,
                                                    dtype=self.precision_policy.store_precision)

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

    # Construct the stepper skeleton
    def _construct_stepper_skeleton(self):
        self.app = []

        def recursion_reference(level, app):
            if level < 0:
                return
            print(f"RECURSION down to level {level}")
            print(f"RECURSION Level {level}, COLLIDE")

            self.add_to_app(
                app=app,
                op_name="collide_coarse",
                mres_level=level,
                f_0=self.f_0,
                f_1=self.f_1,
                bc_mask=self.bc_mask,
                missing_mask=self.missing_mask,
                omega=self.omega,
                timestep=0,
            )

            recursion_reference(level - 1, app)
            recursion_reference(level - 1, app)

            # Important: swapping of f_0 and f_1 is done here
            print(f"RECURSION Level {level}, stream_coarse_step_ABC")
            self.add_to_app(
                app=app,
                op_name="stream_coarse_step_ABC",
                mres_level=level,
                f_0=self.f_1,
                f_1=self.f_0,
                bc_mask=self.bc_mask,
                missing_mask=self.missing_mask,
                omega=self.coalescence_factor,
                timestep=0,
            )

        def recursion_fused_finest(level,
                                   app,
                                   is_self_f1_the_explosion_src_field,
                                   is_self_f1_the_coalescence_dst_field):
            if level < 0:
                return

            if level == 0:
                print(f"RECURSION down to the finest level {level}")
                print(f"RECURSION Level {level}, Fused STREAM and COLLIDE")
                self.add_to_app(
                    app=app,
                    op_name="finest_fused_pull",
                    mres_level=level,
                    f_0=self.f_0,
                    f_1=self.f_1,
                    bc_mask=self.bc_mask,
                    missing_mask=self.missing_mask,
                    omega=self.omega,
                    timestep=0,
                    is_f1_the_explosion_src_field=is_self_f1_the_explosion_src_field,
                    is_f1_the_coalescence_dst_field=is_self_f1_the_coalescence_dst_field,
                )
                self.add_to_app(
                    app=app,
                    op_name="finest_fused_pull",
                    mres_level=level,
                    f_0=self.f_1,
                    f_1=self.f_0,
                    bc_mask=self.bc_mask,
                    missing_mask=self.missing_mask,
                    omega=self.omega,
                    timestep=0,
                    is_f1_the_explosion_src_field=not is_self_f1_the_explosion_src_field,
                    is_f1_the_coalescence_dst_field=not is_self_f1_the_coalescence_dst_field,
                )
                return

            print(f"RECURSION down to level {level}")
            print(f"RECURSION Level {level}, COLLIDE")

            self.add_to_app(
                app=app,
                op_name="collide_coarse",
                mres_level=level,
                f_0=self.f_0,
                f_1=self.f_1,
                bc_mask=self.bc_mask,
                missing_mask=self.missing_mask,
                omega=self.omega,
                timestep=0,
            )
            # 1. Accumulation is read from f_0 in the streaming step, where f_0=self.f_1.
            # so is_self_f1_the_coalescence_dst_field is True
            # 2. Explision data is the output from the corser collide, which is f_1=self.f_1.
            # so is_self_f1_the_explosion_src_field is True

            if level - 1 == 0:
                recursion_fused_finest(level - 1, app, is_self_f1_the_explosion_src_field=True,
                                       is_self_f1_the_coalescence_dst_field=True)
            else:
                recursion_fused_finest(level - 1, app, is_self_f1_the_explosion_src_field=None,
                                       is_self_f1_the_coalescence_dst_field=None)
                recursion_fused_finest(level - 1, app, is_self_f1_the_explosion_src_field=None,
                                       is_self_f1_the_coalescence_dst_field=None)
            # Important: swapping of f_0 and f_1 is done here
            print(f"RECURSION Level {level}, stream_coarse_step_ABC")
            self.add_to_app(
                app=app,
                op_name="stream_coarse_step_ABC",
                mres_level=level,
                f_0=self.f_1,
                f_1=self.f_0,
                bc_mask=self.bc_mask,
                missing_mask=self.missing_mask,
                omega=self.coalescence_factor,
                timestep=0,
            )

        if self.mres_perf_opt == MresPerfOptimizationType.NAIVE_COLLIDE_STREAM:
            recursion_reference(self.count_levels - 1, app=self.app)
        elif self.mres_perf_opt == MresPerfOptimizationType.FUSION_AT_FINEST:
            recursion_fused_finest(self.count_levels - 1,
                                   app=self.app,
                                   is_self_f1_the_coalescence_dst_field=None,
                                   is_self_f1_the_explosion_src_field=None)
        else:
            raise ValueError(f"Unknown optimization level: {self.opt_level}")

        bk = self.grid.get_neon_backend()
        self.sk = neon.Skeleton(backend=bk)
        self.sk.sequence("mres_nse_stepper", self.app)
