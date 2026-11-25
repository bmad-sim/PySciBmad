"""GPUs. SciBmad tracks on a GPU when the coordinates are a GPU array; in Python::

    gpu = scibmad.device("cuda")            # or "metal", "amdgpu", "oneapi"; "auto" picks one
    v0 = gpu.asarray(np.random.rand(100_000, 6) * 1e-5)
    res = scibmad.track(fodo, v0=v0, n_turns=100)
    x = np.asarray(res.x(turn=100))         # copied to the host

The first ``device(...)`` of a kind installs its Julia package (CUDA.jl, Metal.jl, ...) if
needed, once.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys

import numpy as np

from . import _core
from ._core import wrap

# name -> (Julia package, array type)
BACKENDS = {
    "cuda": ("CUDA", "CuArray"),
    "metal": ("Metal", "MtlArray"),
    "amdgpu": ("AMDGPU", "ROCArray"),
    "rocm": ("AMDGPU", "ROCArray"),
    "oneapi": ("oneAPI", "oneArray"),
    "cpu": (None, "Array"),
}

_DEVICES: dict = {}


def _auto():
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "metal"
    if shutil.which("nvidia-smi") or os.path.exists("/proc/driver/nvidia"):
        return "cuda"
    if shutil.which("rocminfo") or os.path.exists("/opt/rocm"):
        return "amdgpu"
    raise RuntimeError("No GPU found: give the kind, device('cuda'), device('metal'), "
                       "device('amdgpu') or device('oneapi')")


class Device:
    """A kind of GPU (or the CPU, ``device("cpu")``, for code that works with either)."""

    def __init__(self, name):
        name = name.lower()
        if name not in BACKENDS:
            raise ValueError(f"Unknown device {name!r}: use one of {', '.join(BACKENDS)}")
        _core._init()
        self.name = name
        pkg, arr = BACKENDS[name]
        if pkg is None:
            self.module = None
            self.array_type = wrap(_core.jl.Base.Array)
        else:
            from ._extras import julia
            self.module = julia.package(pkg, install=True)
            self.array_type = getattr(self.module, arr)

    def asarray(self, x, dtype=None):
        """A copy of `x` (a NumPy array or anything NumPy can convert) on this device. Metal
        GPUs have no 64-bit floats: use ``dtype=numpy.float32``."""
        if isinstance(x, _core.JuliaArray):
            if dtype is None:
                return self.array_type(x)
            x = np.asarray(x)
        a = np.asarray(x, dtype=dtype)
        if self.name == "metal" and a.dtype in (np.float64, np.complex128):
            raise TypeError("Metal GPUs have no 64-bit floats: use dtype=numpy.float32 "
                            "(32-bit coordinates)")
        return self.array_type(a)

    def zeros(self, shape, dtype=np.float64):
        return self.asarray(np.zeros(shape, dtype=dtype))

    def ones(self, shape, dtype=np.float64):
        return self.asarray(np.ones(shape, dtype=dtype))

    def full(self, shape, value, dtype=np.float64):
        return self.asarray(np.full(shape, value, dtype=dtype))

    def synchronize(self):
        """Waits until the GPU has finished its work (for timing)."""
        if self.module is not None:
            self.module.synchronize()

    def __repr__(self):
        return f"scibmad.device({self.name!r})"


def device(name="auto"):
    """The GPU (or CPU) device `name`: "cuda", "metal", "amdgpu", "oneapi", "cpu", or "auto"."""
    if name == "auto":
        name = _auto()
    name = name.lower()
    d = _DEVICES.get(name)
    if d is None:
        d = Device(name)
        _DEVICES[name] = d
    return d
