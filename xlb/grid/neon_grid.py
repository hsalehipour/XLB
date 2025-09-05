import warp as wp
import neon
from .grid import Grid
from xlb.precision_policy import Precision
from xlb.compute_backend import ComputeBackend
from typing import Literal, List
from xlb import DefaultConfig


class NeonGrid(Grid):
    def __init__(
        self,
        shape,  # bounding box of the domain
        velocity_set,  # velocity set for the grid
        backend_config=None,
    ):
        from .warp_grid import WarpGrid

        if backend_config is None:
            backend_config = {
                "device_list": [0],
                "skeleton_config": neon.SkeletonConfig.none(),
            }

        # check that the config dictionary has the required keys
        required_keys = ["device_list"]
        for key in required_keys:
            if key not in backend_config:
                raise ValueError(f"backend_config must contain a '{key}' key")

        # check that the device list is a list of integers
        if not isinstance(backend_config["device_list"], list):
            raise ValueError("backend_config['device_list'] must be a list of integers")
        for device in backend_config["device_list"]:
            if not isinstance(device, int):
                raise ValueError("backend_config['device_list'] must be a list of integers")

        self.config = backend_config
        self.bk = None
        self.dim = None
        self.grid = None
        self.velocity_set = velocity_set
        self.warp_grid = WarpGrid(shape)

        super().__init__(shape, ComputeBackend.NEON)

    def _get_velocity_set(self):
        return self.velocity_set

    def _initialize_backend(self):
        dev_idx_list = self.config["device_list"]

        if len(self.shape) == 2:
            import py_neon

            self.dim = py_neon.Index_3d(self.shape[0], 1, self.shape[1])
            self.neon_stencil = []
            for q in range(self.velocity_set.q):
                xval, yval = self.velocity_set._c[:, q]
                self.neon_stencil.append([xval, 0, yval])

        else:
            self.dim = neon.Index_3d(self.shape[0], self.shape[1], self.shape[2])

            self.neon_stencil = []
            for q in range(self.velocity_set.q):
                xval, yval, zval = self.velocity_set._c[:, q]
                self.neon_stencil.append([xval, yval, zval])

        self.bk = neon.Backend(runtime=neon.Backend.Runtime.stream, dev_idx_list=dev_idx_list)
        self.bk.info_print()
        self.grid = neon.dense.dGrid(backend=self.bk, dim=self.dim, sparsity=None, stencil=self.neon_stencil)
        pass

    def create_field(
        self,
        cardinality: int,
        dtype: Literal[Precision.FP32, Precision.FP64, Precision.FP16] = None,
        fill_value=None,
    ):
        dtype = dtype.wp_dtype if dtype else DefaultConfig.default_precision_policy.store_precision.wp_dtype
        field = self.grid.new_field(
            cardinality=cardinality,
            dtype=dtype,
        )

        if fill_value is None:
            field.zero_run(stream_idx=0)
        else:
            field.fill_run(value=fill_value, stream_idx=0)
        return field

    def _create_warp_field(
        self, cardinality: int, dtype: Literal[Precision.FP32, Precision.FP64, Precision.FP16] = None, fill_value=None, ne_field=None
    ):
        warp_field = self.warp_grid.create_field(cardinality, dtype, fill_value)
        if ne_field is None:
            return warp_field

        _d = self.velocity_set.d

        import typing

        @neon.Container.factory
        def container(src_field: typing.Any, dst_field: typing.Any, cardinality: wp.int32):
            def loading_step(loader: neon.Loader):
                loader.declare_execution_scope(self.grid)
                src_pn = loader.get_read_handel(src_field)

                @wp.func
                def cloning(gridIdx: typing.Any):
                    cIdx = wp.neon_global_idx(src_pn, gridIdx)
                    gx = wp.neon_get_x(cIdx)
                    gy = wp.neon_get_y(cIdx)
                    gz = wp.neon_get_z(cIdx)

                    # TODO@Max - XLB is flattening the z dimension in 3D, while neon uses the y dimension
                    if _d == 2:
                        gy, gz = gz, gy

                    for card in range(cardinality):
                        value = wp.neon_read(src_pn, gridIdx, card)
                        dst_field[card, gx, gy, gz] = value

                loader.declare_kernel(cloning)

            return loading_step

        c = container(src_field=ne_field, dst_field=warp_field, cardinality=cardinality)
        c.run(0)
        wp.synchronize()
        return warp_field

    def get_neon_backend(self):
        return self.bk
