"""SplatAD spherical LiDAR rasterizer, packaged on its own.

A CUDA rasterizer that projects 3D Gaussians onto a spinning LiDAR's spherical
sampling grid and returns a median (first-return) range image, rather than the
alpha-weighted expected depth a camera rasterizer produces. Expected depth is
pulled past the first surface and smears the thin rings a LiDAR actually
measures; the median return is as sharp as the sensor.

Extracted from the SplatAD gsplat fork (see NOTICE) and reduced to the LiDAR
path, so it coexists with an ordinary pip-installed ``gsplat`` for the camera
path instead of replacing it: different Python package, and a separately-named
CUDA extension (``splatad_kernel_cuda``) so the two never collide.

    from splatad_kernel import lidar_rasterization

The extension JIT-compiles on first use and needs a CUDA toolkit (``nvcc``) on
PATH. See ``cuda._backend`` for the pre-built ``.so`` path used when shipping
into an image that has no toolkit.
"""

from splatad_kernel.version import __version__

# The CUDA entry points are resolved on first attribute access rather than at
# import: importing them pulls in torch, and `splatad_kernel.webgpu` -- which
# needs neither torch nor CUDA -- would otherwise be unable to import on a
# machine that has no torch installed. Nothing else changes; a name below still
# resolves to exactly the object it used to.
_CUDA_EXPORTS = {
    "fully_fused_lidar_projection": "splatad_kernel.cuda._wrapper",
    "isect_lidar_tiles": "splatad_kernel.cuda._wrapper",
    "isect_offset_encode": "splatad_kernel.cuda._wrapper",
    "rasterize_to_points": "splatad_kernel.cuda._wrapper",
    "lidar_rasterization": "splatad_kernel.rendering",
}


def __getattr__(name):
    module = _CUDA_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_CUDA_EXPORTS))

__all__ = [
    "__version__",
    "fully_fused_lidar_projection",
    "isect_lidar_tiles",
    "isect_offset_encode",
    "lidar_rasterization",
    "rasterize_to_points",
]
