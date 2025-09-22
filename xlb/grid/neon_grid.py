import neon
from .grid import Grid
from xlb.precision_policy import Precision
from xlb.compute_backend import ComputeBackend
from typing import Literal
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
                "skeleton_config": neon.SkeletonConfig.OCC.none(),
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

    def get_neon_backend(self):
        return self.bk
