"""``lidar_rasterization`` with the CUDA entry point's signature, on WebGPU.

Drop-in for ``splatad_kernel.lidar_rasterization`` in the forward direction:
same arguments, same output tuple, same ``meta`` keys. Inputs may be torch
tensors or numpy arrays; torch in gives torch out, so an existing call site
only changes which function it imports.

What it does not do is listed in ``_reject_unsupported`` -- each one raises
rather than silently rendering something else.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .rasterizer import LidarRasterizer

_shared: dict = {}


def get_rasterizer(
    device: Optional[Any] = None, sort_algorithm: str = "radix"
) -> LidarRasterizer:
    """A process-wide rasterizer, so pipelines and buffers survive between frames."""
    if device is not None:
        return LidarRasterizer(device, sort_algorithm=sort_algorithm)
    if sort_algorithm not in _shared:
        _shared[sort_algorithm] = LidarRasterizer(sort_algorithm=sort_algorithm)
    return _shared[sort_algorithm]


def _is_torch(x: Any) -> bool:
    return hasattr(x, "detach") and hasattr(x, "device")


def _reject_unsupported(packed, sparse_grad, absgrad, depth_lanes, requires_grad):
    if packed:
        raise NotImplementedError(
            "packed mode is not supported (the CUDA path does not support it either)"
        )
    if sparse_grad or absgrad or requires_grad:
        raise NotImplementedError(
            "the WebGPU backend is forward only: it has no backward pass, so it "
            "cannot be used for training. Use splatad_kernel.lidar_rasterization "
            "(CUDA) where gradients are needed."
        )
    if depth_lanes:
        raise NotImplementedError(
            "depth_lanes is a CUDA-specific scheduling variant and has no WebGPU "
            "equivalent; the result would be the same modulo float epsilon, so "
            "just leave it off"
        )


def lidar_rasterization(
    means,
    quats,
    scales,
    opacities,
    lidar_features,
    velocities,
    viewmats,
    raster_pts,
    tile_elevation_boundaries,
    linear_velocity=None,
    angular_velocity=None,
    rolling_shutter_time=None,
    min_azimuth: float = -180.0,
    max_azimuth: float = 180.0,
    min_elevation: float = -80.0,
    max_elevation: float = 80.0,
    n_elevation_channels: int = 32,
    azimuth_resolution: float = 0.1,
    tile_width: int = 32,
    tile_height: int = 8,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.017,
    compute_alpha_sum_until_points: bool = True,
    compute_alpha_sum_until_points_threshold: float = 0.2,
    row_elevations=None,
    packed: bool = False,
    sparse_grad: bool = False,
    absgrad: bool = False,
    tile_col_offset: int = 0,
    valid_mask=None,
    depth_lanes: bool = False,
    rasterize_mode: str = "classic",
    channel_chunk: int = 32,
    use_depth_compensation: bool = True,
    raydrop_sh_coeffs=None,
    raydrop_sh_degree: int = 0,
    raydrop_feature_index: int = 1,
    device: Optional[Any] = None,
    sort_algorithm: str = "radix",
) -> Tuple[Any, Any, Optional[Any], Dict[str, Any]]:
    """Rasterize 3D Gaussians to a batch of spherical LiDAR range images.

    See ``splatad_kernel.rendering.lidar_rasterization`` for what every argument
    means; this mirrors it. ``channel_chunk`` is accepted and ignored -- the
    CUDA kernel splits wide feature sets to stay inside its register budget,
    which is not a constraint here. ``sort_algorithm`` picks how the
    intersection list is ordered -- ``"radix"`` (the default, several times
    faster) or ``"bitonic"``; the rendered output is the same either way.

    Returns ``(render_lidar_features, render_alphas, alpha_sum_until_points,
    meta)``, with the rendered range as the last feature channel and the
    first-return range at ``meta["median_depths"]``.
    """
    requires_grad = any(
        getattr(t, "requires_grad", False)
        for t in (means, quats, scales, opacities, lidar_features)
        if t is not None
    )
    _reject_unsupported(packed, sparse_grad, absgrad, depth_lanes, requires_grad)
    if rasterize_mode not in ("classic", "antialiased"):
        raise ValueError(f"unknown rasterize_mode {rasterize_mode!r}")
    if raydrop_sh_degree > 4:
        raise ValueError(
            f"raydrop_sh_degree={raydrop_sh_degree} exceeds the shader maximum 4 "
            f"(the same cap sh.cu has)"
        )

    n_elev_tiles = math.ceil(n_elevation_channels / tile_height)
    boundaries = tile_elevation_boundaries
    if getattr(boundaries, "shape", (0,))[0] != n_elev_tiles + 1:
        raise ValueError(
            f"tile_elevation_boundaries must have "
            f"ceil({n_elevation_channels}/{tile_height}) + 1 = {n_elev_tiles + 1} "
            f"entries; got {getattr(boundaries, 'shape', None)}"
        )

    rasterizer = get_rasterizer(device, sort_algorithm)
    colors, alphas, alpha_sum, meta = rasterizer.render(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        lidar_features=lidar_features,
        velocities=velocities,
        viewmats=viewmats,
        raster_pts=raster_pts,
        tile_elevation_boundaries=tile_elevation_boundaries,
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
        rolling_shutter_time=rolling_shutter_time,
        min_azimuth=min_azimuth,
        max_azimuth=max_azimuth,
        min_elevation=min_elevation,
        max_elevation=max_elevation,
        n_elevation_channels=n_elevation_channels,
        azimuth_resolution=azimuth_resolution,
        tile_width=tile_width,
        tile_height=tile_height,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        eps2d=eps2d,
        compute_alpha_sum_until_points=compute_alpha_sum_until_points,
        compute_alpha_sum_until_points_threshold=compute_alpha_sum_until_points_threshold,
        row_elevations=row_elevations,
        tile_col_offset=tile_col_offset,
        valid_mask=valid_mask,
        rasterize_mode=rasterize_mode,
        use_depth_compensation=use_depth_compensation,
        raydrop_sh_coeffs=raydrop_sh_coeffs,
        raydrop_sh_degree=raydrop_sh_degree,
        raydrop_feature_index=raydrop_feature_index,
    )

    if _is_torch(means):
        import torch

        target = means.device
        # The rasterizer runs on the WebGPU adapter, not on `target`; the copy
        # back is what makes the call a drop-in for a CUDA-tensor call site.
        def to_torch(x):
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                return torch.from_numpy(np.ascontiguousarray(x)).to(target)
            return x

        colors = to_torch(colors)
        alphas = to_torch(alphas)
        alpha_sum = to_torch(alpha_sum)
        meta = {k: to_torch(v) if isinstance(v, np.ndarray) else v for k, v in meta.items()}

    return colors, alphas, alpha_sum, meta
