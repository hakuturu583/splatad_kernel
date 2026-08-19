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

from splatad_kernel.cuda._wrapper import (
    fully_fused_lidar_projection,
    isect_lidar_tiles,
    isect_offset_encode,
    rasterize_to_points,
)
from splatad_kernel.rendering import lidar_rasterization
from splatad_kernel.version import __version__

__all__ = [
    "__version__",
    "fully_fused_lidar_projection",
    "isect_lidar_tiles",
    "isect_offset_encode",
    "lidar_rasterization",
    "rasterize_to_points",
]
