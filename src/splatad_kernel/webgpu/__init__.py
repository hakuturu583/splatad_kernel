"""WebGPU backend for SplatAD's spherical LiDAR rasterizer.

A port of the CUDA forward pipeline to WGSL compute shaders, so the same
LiDAR render runs on anything with a WebGPU implementation -- a non-NVIDIA GPU,
a laptop's integrated one, a software adapter, or a browser -- instead of only
on CUDA.

    from splatad_kernel.webgpu import lidar_rasterization

    render, alphas, alpha_sum, meta = lidar_rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities,
        lidar_features=features, velocities=None, viewmats=viewmats,
        raster_pts=raster_pts, tile_elevation_boundaries=tile_bounds,
        n_elevation_channels=H, azimuth_resolution=360.0 / W,
    )
    distance = meta["median_depths"][0, ..., 0]

The signature, the output tuple and the ``meta`` keys match
``splatad_kernel.lidar_rasterization``; torch tensors in give torch tensors
out. The one structural difference is that this is **forward only** -- there is
no backward pass, so it is for inference, not training.

The WGSL in ``shaders/`` is plain WebGPU with no Python in it, so a JavaScript
or Rust host can drive the same pipeline; ``rasterizer.py`` is the reference
for the order the stages run in and what each buffer holds.
"""

from ._device import WebGPUUnavailable, device_summary, is_available

# The rasterizer pulls in `wgpu` and `numpy`; resolving it lazily means
# `is_available()` can answer "no" on a machine that has neither, instead of the
# import itself failing.
_LAZY = {
    "LidarRasterizer": "splatad_kernel.webgpu.rasterizer",
    "get_rasterizer": "splatad_kernel.webgpu.rendering",
    "lidar_rasterization": "splatad_kernel.webgpu.rendering",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "LidarRasterizer",
    "WebGPUUnavailable",
    "device_summary",
    "get_rasterizer",
    "is_available",
    "lidar_rasterization",
]
