from typing import Callable, Optional, Tuple

import torch
from torch import Tensor


def _make_lazy_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from ._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    return call_cuda


def fully_fused_lidar_projection(
    means: Tensor,  # [N, 3]
    covars: Optional[Tensor],  # [N, 6] or None
    quats: Optional[Tensor],  # [N, 4] or None
    scales: Optional[Tensor],  # [N, 3] or None
    velocities: Optional[Tensor],  # [N, 3] or None
    viewmats: Tensor,  # [C, 4, 4]
    linear_velocity: Tensor,  # [C, 3]
    angular_velocity: Tensor,  # [C, 3]
    rolling_shutter_time: Tensor,  # [C]
    valid_mask: Optional[Tensor] = None,  # [N] bool
    min_elevation: float = -45,
    max_elevation: float = 45,
    min_azimuth: float = -180,
    max_azimuth: float = 180,
    eps2d: float = 0.01,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    packed: bool = False,
    sparse_grad: bool = False,
    calc_compensations: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Projects Gaussians to 2D.

    This function fuse the process of computing covariances
    (:func:`quat_scale_to_covar_preci()`), transforming to lidar space (:func:`world_to_cam()`),
    and spherical projection (:func:`lidar_proj()`).

    .. note::

        During projection, we ignore the Gaussians that are outside of the lidar frustum.
        This frustum is defined by the elevation angle (angle from horizontal xy-plane) and azimuth
        angle (angle around the z-axis, counter-clockwise from the x-axis).
        So not all the elements in the output tensors are valid. The output `radii` could serve as
        an indicator, in which zero radii means the corresponding elements are invalid in
        the output tensors and will be ignored in the next rasterization process. If `packed=True`,
        the output tensors will be packed into a flattened tensor, in which all elements are valid.
        In this case, a `camera_ids` tensor and `gaussian_ids` tensor will be returned to indicate the
        row (lidar) and column (Gaussian) indices of the packed flattened tensor, which is essentially
        following the COO sparse tensor format.

    .. note::

        This functions supports projecting Gaussians with either covariances or {quaternions, scales},
        which will be converted to covariances internally in a fused CUDA kernel. Either `covars` or
        {`quats`, `scales`} should be provided.

    Args:
        means: Gaussian means. [N, 3]
        covars: Gaussian covariances (flattened upper triangle). [N, 6] Optional.
        quats: Quaternions (No need to be normalized). [N, 4] Optional.
        scales: Scales. [N, 3] Optional.
        viewmats: Lidar-to-world matrices. [C, 4, 4]
        velocities: Gaussian velocities. [N, 3] Optional.
        linear_velocity: Linear velocity of the Lidar. [C, 3]
        angular_velocity: Angular velocity of the Lidar. [C, 3]
        rolling_shutter_time: Rolling shutter time of the Lidar. [C]
        valid_mask: Optional per-Gaussian keep mask (bool, [N]). Masked
          Gaussians are treated exactly like frustum-culled ones (radii = 0)
          without the caller having to compact the input arrays — the point is
          sector streaming, where a host-side azimuth cull would otherwise pay
          a nonzero()/index_select (and a device sync) per sector. Default: None.
        min_elevation: Minimum elevation angle in degrees. Default: -45.
        max_elevation: Maximum elevation angle in degrees. Default: 45.
        min_azimuth: Minimum azimuth angle in degrees. Default: -180.
        max_azimuth: Maximum azimuth angle in degrees. Default: 180.
        eps2d: A epsilon added to the 2D covariance for numerical stability. Default: 0.01.
        near_plane: Near plane distance. Default: 0.01.
        far_plane: Far plane distance. Default: 1e10.
        radius_clip: Gaussians with projected radii smaller than this value will be ignored. Default: 0.0.
        packed: If True, the output tensors will be packed into a flattened tensor. Default: False.
        sparse_grad: This is only effective when `packed` is True. If True, during backward the gradients
          of {`means`, `covars`, `quats`, `scales`} will be a sparse Tensor in COO layout. Default: False.
        calc_compensations: If True, a view-dependent opacity compensation factor will be computed, which
          is useful for anti-aliasing. Default: False.

    Returns:
        A tuple:

        If `packed` is True:

        - **camera_ids**. The row indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **gaussian_ids**. The column indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **radii**. The maximum radius of the projected Gaussians in spherical coords, degrees. Float32 tensor of shape [nnz, 2].
        - **means**. Projected Gaussian means in 2D. [nnz, 2]
        - **depths**. The Euclidian distance of the projected Gaussians. [nnz]
        - **conics**. Inverse of the projected covariances. Return the flattend upper triangle with [nnz, 3]
        - **compensations**. The view-dependent opacity compensation factor. [nnz]
        - **pix_vels**. The velocities of Gaussians in 2D coordiantes. [nnz, 2]
        - **depth_compensations**. The depth compensation factor, i.e., how the depth changes with angular coordinates due to Gaussian orientation. [nnz]

        If `packed` is False:

        - **radii**. The maximum radius of the projected Gaussians in spherical coords, degrees. Float32 tensor of shape [C, N, 2].
        - **means**. Projected Gaussian means in 2D. [C, N, 2]
        - **depths**. The Euclidian distance of the projected Gaussians. [C, N]
        - **conics**. Inverse of the projected covariances. Return the flattend upper triangle with [C, N, 3]
        - **compensations**. The view-dependent opacity compensation factor. [C, N]
        - **pix_vels**. The velocities of Gaussians in 2D coordiantes. [C, N, 2]
        - **depth_compensations**. The depth compensation factor, i.e., how the depth changes with angular coordinates due to Gaussian orientation. [C, N]
    """
    C = viewmats.size(0)
    N = means.size(0)
    assert means.size() == (N, 3), means.size()
    assert viewmats.size() == (C, 4, 4), viewmats.size()
    means = means.contiguous()
    if covars is not None:
        assert covars.size() == (N, 6), covars.size()
        covars = covars.contiguous()
    else:
        assert quats is not None, "covars or quats is required"
        assert scales is not None, "covars or scales is required"
        assert quats.size() == (N, 4), quats.size()
        assert scales.size() == (N, 3), scales.size()
        quats = quats.contiguous()
        scales = scales.contiguous()
    if velocities is not None:
        assert velocities.size() == (N, 3), velocities.size()
        velocities = velocities.contiguous()
    if valid_mask is not None:
        assert valid_mask.shape == (N,), valid_mask.shape
        assert valid_mask.dtype == torch.bool, valid_mask.dtype
        valid_mask = valid_mask.contiguous()
    if sparse_grad:
        assert packed, "sparse_grad is only supported when packed is True"

    viewmats = viewmats.contiguous()
    if packed:
        raise NotImplementedError(
            "FullyFusedLidarProjectionPacked is not implemented yet"
        )
    else:
        return _FullyFusedLidarProjection.apply(
            means,
            covars,
            quats,
            scales,
            velocities,
            valid_mask,
            viewmats,
            min_elevation,
            max_elevation,
            min_azimuth,
            max_azimuth,
            linear_velocity,
            angular_velocity,
            rolling_shutter_time,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
        )


@torch.no_grad()
def isect_lidar_tiles(
    means2d: Tensor,  # [C, N, 2] or [nnz, 2]
    radii: Tensor,  # [C, N, 2] or [nnz, 2]
    depths: Tensor,  # [C, N] or [nnz]
    elev_boundaries: Tensor,
    tile_azim_resolution: float,
    min_azim: float,
    sort: bool = True,
    packed: bool = False,
    n_cameras: Optional[int] = None,
    camera_ids: Optional[Tensor] = None,
    gaussian_ids: Optional[Tensor] = None,
    conics: Optional[Tensor] = None,
    opacities: Optional[Tensor] = None,
    row_elevations: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Maps projected Gaussians to intersecting tiles.

    Args:
        means2d: Projected Gaussian means. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        radii: Maximum radii of the projected Gaussians. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        depths: Euclidian distance of the projected Gaussians. [C, N] if packed is False, [nnz] if packed is True.
        elev_boundaries: Elevation boundaries, defining borders between lidar channels. [n_elev]
        tile_azim_resolution: Tile azimuth resolution.
        min_azim: Minimum azimuth angle in degrees.
        sort: If True, the returned intersections will be sorted by the intersection ids. Default: True.
        packed: If True, the input tensors are packed. Default: False.
        n_cameras: Number of lidars. Required if packed is True.
        camera_ids: The row indices of the projected Gaussians. Required if packed is True.
        gaussian_ids: The column indices of the projected Gaussians. Required if packed is True.
        conics: splatsim addition. When given together with ``opacities``, the
            azimuth span is recomputed per elevation row from the exact
            contribution test instead of applying the Gaussian's widest extent
            to every row it spans. Emits ~11% fewer (Gaussian, tile) pairs and
            cannot drop a pair a pixel could be hit by (the span is taken to the
            nearest elevation in each row). Omit both for the bbox binning.
        opacities: see ``conics``; sets the per-Gaussian alpha cutoff.
        row_elevations: the beam elevation of each tile row. Pass this only when
            a tile row IS one beam (tile height 1): the row then samples exactly
            that elevation, which tightens the span far more than the row's
            elevation band while staying exact.

    Returns:
        A tuple:

        - **Tiles per Gaussian**. The number of tiles intersected by each Gaussian.
          Int32 [C, N] if packed is False, Int32 [nnz] if packed is True.
        - **Intersection ids**. Each id is an 64-bit integer with the following
          information: camera_id (Xc bits) | tile_id (Xt bits) | depth (32 bits).
          Xc and Xt are the maximum number of bits required to represent the camera and
          tile ids, respectively. Int64 [n_isects]
        - **Flatten ids**. The global flatten indices in [C * N] or [nnz] (packed). [n_isects]
    """
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.size()
        assert radii.shape == (nnz, 2), radii.size()
        assert depths.shape == (nnz,), depths.size()
        assert camera_ids is not None, "camera_ids is required if packed is True"
        assert gaussian_ids is not None, "gaussian_ids is required if packed is True"
        assert n_cameras is not None, "n_cameras is required if packed is True"
        camera_ids = camera_ids.contiguous()
        gaussian_ids = gaussian_ids.contiguous()
        C = n_cameras

    else:
        C, N, _ = means2d.shape
        assert means2d.shape == (C, N, 2), means2d.size()
        assert radii.shape == (C, N, 2), radii.size()
        assert depths.shape == (C, N), depths.size()

    tiles_per_gauss, isect_ids, flatten_ids = _make_lazy_cuda_func("isect_lidar_tiles")(
        means2d.contiguous(),
        radii.contiguous(),
        depths.contiguous(),
        None if conics is None else conics.contiguous(),
        None if opacities is None else opacities.contiguous(),
        None if row_elevations is None else row_elevations.contiguous(),
        camera_ids,
        gaussian_ids,
        C,
        elev_boundaries.contiguous(),
        tile_azim_resolution,
        min_azim,
        sort,
        True,  # DoubleBuffer: memory efficient radixsort
    )
    return tiles_per_gauss, isect_ids, flatten_ids


@torch.no_grad()
def isect_offset_encode(
    isect_ids: Tensor, n_cameras: int, tile_width: int, tile_height: int
) -> Tensor:
    """Encodes intersection ids to offsets.

    Args:
        isect_ids: Intersection ids. [n_isects]
        n_cameras: Number of cameras.
        tile_width: Tile width.
        tile_height: Tile height.

    Returns:
        Offsets. [C, tile_height, tile_width]
    """
    return _make_lazy_cuda_func("isect_offset_encode")(
        isect_ids.contiguous(), n_cameras, tile_width, tile_height
    )


def rasterize_to_points(
    means2d: Tensor,  # [C, N, 2] or [nnz, 2]
    conics: Tensor,  # [C, N, 3] or [nnz, 3]
    lidar_features: Tensor,  # [C, N, channels] or [nnz, channels]
    opacities: Tensor,  # [C, N] or [nnz]
    pix_vels: Tensor,  # [C, N, 3] or [nnz, 3]
    depth_compensations: Tensor,  # [C, N, 2] or [nnz, 2]
    raster_pts: Tensor,  # [C, H, W, 3]
    image_width: int,
    image_height: int,
    tile_width: int,
    tile_height: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    compute_alpha_sum_until_points: bool,
    compute_alpha_sum_until_points_threshold: float,
    backgrounds: Optional[Tensor] = None,  # [C, channels]
    packed: bool = False,
    absgrad: bool = False,
    static_render: bool = False,
    tile_col_offset: int = 0,
    use_depth_comp: bool = True,
    depth_lanes: bool = False,
) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor, Tensor, Tensor]:
    """Rasterizes Gaussians to points.

    Args:
        means2d: Projected Gaussian means. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        conics: Inverse of the projected covariances with only upper triangle values. [C, N, 3] if packed is False, [nnz, 3] if packed is True.
        lidar_features: Gaussian ND features. [C, N, channels] if packed is False, [nnz, channels] if packed is True.
        opacities: Gaussian opacities that support per-view values. [C, N] if packed is False, [nnz] if packed is True.
        pix_vels: Spherical velocities. [C, N, 3] if packed is False, [nnz, 3] if packed is True.
        depth_compensations: Depth compensation factors. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        raster_pts: Spherical coordinates of the points to rasterize. [C, H, W, 3]
        image_width: Image width.
        image_height: Image height.
        tile_width: Tile width.
        tile_height: Tile height.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [C, tile_height, tile_width]
        flatten_ids: The global flatten indices in [C * N] or [nnz] from  `isect_tiles()`. [n_isects]
        compute_alpha_sum_until_points: Whether to compute the alpha sum until provided observed points.
        compute_alpha_sum_until_points_threshold: Alpha sum is computed up until we are threshold away from the point.
        backgrounds: Background colors. [C, channels]. Default: None.
        packed: If True, the input tensors are expected to be packed with shape [nnz, ...]. Default: False.
        absgrad: If True, the backward pass will compute a `.absgrad` attribute for `means2d`. Default: False.
        tile_col_offset: splatsim sector rendering — first tile-grid column the
            image covers. The tile grid (isect_offsets) always spans the full
            azimuth ring; a sector image rasterizes only its own tile columns,
            looked up at this offset. Forward-only (backward requires 0). Default: 0.
        depth_lanes: splatsim — give each pixel 16 threads along the DEPTH axis
            (associative alpha-blend composition; thresholds resolved by
            re-walking the crossing lane). Lifts rasterization parallelism from
            #pixels to 16x #pixels, which is what makes small azimuth-sector
            images fast on big GPUs. Output differs from the serial kernel at
            float-epsilon level (deterministic). Only lidar_features with 3
            channels and compute_alpha_sum_until_points=False take this path;
            anything else silently falls back to the serial kernel.
            Forward/inference only (backward refuses). Does not produce the
            training-side outputs: fr_depth comes back all-zero and median_ids
            all -1 on this path. Default: False.

    Returns:
        A tuple:

        - **Rendered colors**. [C, image_height, image_width, channels]
        - **Rendered alphas**. [C, image_height, image_width, 1]
        - **alpha_sum_until_points**. [C, image_height, image_width, 1].
        - **median_depths**. [C, image_height, image_width, 1].
        - **fr_depth**. Soft first-return depth. [C, image_height, image_width, 1].
        - **median_ids**. Global gaussian index at the 0.5 crossing, -1 if none.
          [C, image_height, image_width, 1].
    """

    C = isect_offsets.size(0)
    device = means2d.device
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert conics.shape == (nnz, 3), conics.shape
        assert lidar_features.shape[0] == nnz, lidar_features.shape
        assert opacities.shape == (nnz,), opacities.shape
        assert pix_vels.shape == (nnz, 3), pix_vels.shape
        assert depth_compensations.shape == (nnz, 2), depth_compensations.shape
    else:
        N = means2d.size(1)
        assert means2d.shape == (C, N, 2), means2d.shape
        assert conics.shape == (C, N, 3), conics.shape
        assert lidar_features.shape[:2] == (C, N), lidar_features.shape
        assert opacities.shape == (C, N), opacities.shape
        assert pix_vels.shape == (C, N, 3), pix_vels.shape
        assert depth_compensations.shape == (C, N, 2), depth_compensations.shape
    if backgrounds is not None:
        assert backgrounds.shape == (C, lidar_features.shape[-1]), backgrounds.shape
        backgrounds = backgrounds.contiguous()

    # Pad the channels to the nearest supported number if necessary
    channels = lidar_features.shape[-1]
    depth_channel_idx = channels - 1
    if channels > 513 or channels == 0:
        # TODO: maybe worth to support zero channels?
        raise ValueError(f"Unsupported number of lidar_features channels: {channels}")
    if channels not in (
        1,
        2,
        3,
        4,
        5,
        8,
        9,
        16,
        17,
        32,
        33,
        64,
        65,
        128,
        129,
        256,
        257,
        512,
        513,
    ):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        lidar_features = torch.cat(
            [
                lidar_features,
                torch.zeros(*lidar_features.shape[:-1], padded_channels, device=device),
            ],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(
                        *backgrounds.shape[:-1], padded_channels, device=device
                    ),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0

    tile_grid_height, tile_grid_width = isect_offsets.shape[1:3]
    assert tile_grid_height * tile_height >= image_height, (
        f"Assert Failed: {tile_grid_height} * {tile_height} >= {image_height}"
    )
    assert tile_grid_width * tile_width >= image_width, (
        f"Assert Failed: {tile_grid_width} * {tile_width} >= {image_width}"
    )
    assert raster_pts.shape == (
        C,
        image_height,
        image_width,
        4,
    ), "raster_pts does not have the correct shape"

    (
        render_lidar_features,
        render_alphas,
        alpha_sum_until_points,
        median_depths,
        fr_depth,
        median_ids,
    ) = (
        _RasterizeToPoints.apply(
            means2d.contiguous(),
            conics.contiguous(),
            lidar_features.contiguous(),
            opacities.contiguous(),
            pix_vels.contiguous(),
            depth_compensations.contiguous(),
            backgrounds,
            raster_pts.contiguous(),
            image_width,
            image_height,
            tile_width,
            tile_height,
            tile_col_offset,
            depth_lanes,
            isect_offsets.contiguous(),
            flatten_ids.contiguous(),
            compute_alpha_sum_until_points,
            compute_alpha_sum_until_points_threshold,
            absgrad,
            depth_channel_idx,
            static_render,
            use_depth_comp,
        )
    )

    if padded_channels > 0:
        render_lidar_features = render_lidar_features[..., :-padded_channels]
    return (
        render_lidar_features,
        render_alphas,
        alpha_sum_until_points,
        median_depths,
        fr_depth,
        median_ids,
    )


class _FullyFusedLidarProjection(torch.autograd.Function):
    """Projects Gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [N, 3]
        covars: Tensor,  # [N, 6] or None
        quats: Tensor,  # [N, 4] or None
        scales: Tensor,  # [N, 3] or None
        velocities: Tensor,  # [N, 3] or None
        valid_mask: Tensor,  # [N] bool or None
        viewmats: Tensor,  # [C, 4, 4]
        min_elevation: float,
        max_elevation: float,
        min_azimuth: float,
        max_azimuth: float,
        linear_velocity: Tensor,  # [C, 3]
        angular_velocity: Tensor,  # [C, 3]
        rolling_shutter_time: Tensor,  # [C]
        eps2d: float,
        near_plane: float,
        far_plane: float,
        radius_clip: float,
        calc_compensations: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # "covars" and {"quats", "scales"} are mutually exclusive
        radii, means2d, depths, conics, compensations, pix_vels, depth_compensation = (
            _make_lazy_cuda_func("fully_fused_lidar_projection_fwd")(
                means,
                covars,
                quats,
                scales,
                velocities,
                valid_mask,
                viewmats,
                min_elevation,
                max_elevation,
                min_azimuth,
                max_azimuth,
                linear_velocity,
                angular_velocity,
                rolling_shutter_time,
                eps2d,
                near_plane,
                far_plane,
                radius_clip,
                calc_compensations,
            )
        )
        if not calc_compensations:
            compensations = None
        ctx.save_for_backward(
            means,
            covars,
            quats,
            scales,
            velocities,
            viewmats,
            linear_velocity,
            angular_velocity,
            rolling_shutter_time,
            radii,
            conics,
            compensations,
        )
        ctx.min_elevation = min_elevation
        ctx.max_elevation = max_elevation
        ctx.min_azimuth = min_azimuth
        ctx.max_azimuth = max_azimuth
        ctx.eps2d = eps2d

        return (
            radii,
            means2d,
            depths,
            conics,
            compensations,
            pix_vels,
            depth_compensation,
        )

    @staticmethod
    def backward(
        ctx,
        v_radii,
        v_means2d,
        v_depths,
        v_conics,
        v_compensations,
        v_pix_vels,
        v_depth_compensations,
    ):
        (
            means,
            covars,
            quats,
            scales,
            velocities,
            viewmats,
            linear_velocity,
            angular_velocity,
            rolling_shutter_time,
            radii,
            conics,
            compensations,
        ) = ctx.saved_tensors
        min_elevation = ctx.min_elevation
        max_elevation = ctx.max_elevation
        min_azimuth = ctx.min_azimuth
        max_azimuth = ctx.max_azimuth
        eps2d = ctx.eps2d
        if v_compensations is not None:
            v_compensations = v_compensations.contiguous()
        v_means, v_covars, v_quats, v_scales, v_viewmats = _make_lazy_cuda_func(
            "fully_fused_lidar_projection_bwd"
        )(
            means,
            covars,
            quats,
            scales,
            velocities,
            viewmats,
            min_elevation,
            max_elevation,
            min_azimuth,
            max_azimuth,
            linear_velocity,
            angular_velocity,
            rolling_shutter_time,
            eps2d,
            radii,
            conics,
            compensations,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_conics.contiguous(),
            v_compensations,
            v_pix_vels.contiguous(),
            v_depth_compensations.contiguous(),
            ctx.needs_input_grad[6],  # viewmats_requires_grad
        )
        if not ctx.needs_input_grad[0]:
            v_means = None
        if not ctx.needs_input_grad[1]:
            v_covars = None
        if not ctx.needs_input_grad[2]:
            v_quats = None
        if not ctx.needs_input_grad[3]:
            v_scales = None
        if not ctx.needs_input_grad[6]:
            v_viewmats = None
        return (
            v_means,
            v_covars,
            v_quats,
            v_scales,
            None,
            None,  # valid_mask
            v_viewmats,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _RasterizeToPoints(torch.autograd.Function):
    """Rasterize gaussians"""

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,  # [C, N, 2]
        conics: Tensor,  # [C, N, 3]
        colors: Tensor,  # [C, N, D]
        opacities: Tensor,  # [C, N]
        pix_vels: Tensor,  # [C, N, 3]
        depth_compensations: Tensor,  # [C, N, 2]
        backgrounds: Tensor,  # [C, D], Optional
        raster_pts: Tensor,
        width: int,
        height: int,
        tile_width: int,
        tile_height: int,
        tile_col_offset: int,
        depth_lanes: bool,
        isect_offsets: Tensor,  # [C, tile_height, tile_width]
        flatten_ids: Tensor,  # [n_isects]
        compute_alpha_sum_until_points: bool,
        compute_alpha_sum_until_points_threshold: float,
        absgrad: bool,
        depth_channel_idx: int,
        static_render: bool = False,
        use_depth_comp: bool = True,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        (
            render_colors,
            render_alphas,
            last_ids,
            alpha_sum_until_points,
            median_depths,
            fr_depth,
            fr_weight,
            median_ids,
        ) = _make_lazy_cuda_func("rasterize_to_points_fwd")(
            means2d,
            conics,
            colors,
            opacities,
            pix_vels,
            depth_compensations,
            backgrounds,
            raster_pts,
            width,
            height,
            tile_width,
            tile_height,
            tile_col_offset,
            depth_lanes,
            compute_alpha_sum_until_points,
            compute_alpha_sum_until_points_threshold,
            isect_offsets,
            flatten_ids,
            depth_channel_idx,
            static_render,
            use_depth_comp,
        )

        ctx.save_for_backward(
            means2d,
            conics,
            colors,
            opacities,
            pix_vels,
            depth_compensations,
            backgrounds,
            raster_pts,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            fr_depth,
            fr_weight,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_width = tile_width
        ctx.tile_height = tile_height
        ctx.tile_col_offset = tile_col_offset
        ctx.depth_lanes = depth_lanes
        ctx.absgrad = absgrad
        ctx.compute_alpha_sum_until_points = compute_alpha_sum_until_points
        ctx.compute_alpha_sum_until_points_threshold = (
            compute_alpha_sum_until_points_threshold
        )
        ctx.depth_channel_idx = depth_channel_idx

        # double to float
        render_alphas = render_alphas.float()
        alpha_sum_until_points = (
            alpha_sum_until_points.float() if compute_alpha_sum_until_points else None
        )
        # fr_depth (soft-first-return) is DIFFERENTIABLE: its v_fr_depth backward propagates to
        # means/opacity/scale over the first-surface prefix. fr_weight is internal (saved for the
        # backward's 1/D normalization). median_depths carries no grad (hard selection; the
        # backward returns None for it) — deploy/BEV only. median_ids is the int crossing-gaussian
        # index for the autograd median-range GATHER (means[median_ids]) — non-diff itself.
        ctx.mark_non_differentiable(median_ids)
        return render_colors, render_alphas, alpha_sum_until_points, median_depths, fr_depth, median_ids

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,  # [C, H, W, 3]
        v_render_alphas: Tensor,  # [C, H, W, 1]
        v_alpha_sum_until_points: Tensor,  # [C, H, W, 1]
        v_median_depths: Tensor,  # [C, H, W, 1]
        v_fr_depth: Tensor,  # [C, H, W, 1] upstream grad of soft-first-return depth (-> means/opacity/scale)
        v_median_ids: Tensor,  # [C, H, W] None (median_ids is non-differentiable; gather grad flows via means)
    ):
        (
            means2d,
            conics,
            colors,
            opacities,
            pix_vels,
            depth_compensations,
            backgrounds,
            raster_pts,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            fr_depth,
            fr_weight,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        tile_width = ctx.tile_width
        tile_height = ctx.tile_height
        if ctx.tile_col_offset != 0:
            # The backward kernel has no sector support; sector rendering is an
            # inference-time feature (splatsim streams sectors under no_grad).
            raise NotImplementedError(
                "rasterize_to_points backward does not support tile_col_offset != 0"
            )
        if ctx.depth_lanes:
            # The backward kernel re-walks the blend serially and would see the
            # lane composition's float-epsilon drift; lanes are inference-only.
            raise NotImplementedError(
                "rasterize_to_points backward does not support depth_lanes"
            )
        absgrad = ctx.absgrad
        compute_alpha_sum_until_points = ctx.compute_alpha_sum_until_points
        compute_alpha_sum_until_points_threshold = (
            ctx.compute_alpha_sum_until_points_threshold
        )
        depth_channel_idx = ctx.depth_channel_idx

        (
            v_means2d_abs,
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_pix_vels,
            v_depth_compensations,
        ) = _make_lazy_cuda_func("rasterize_to_points_bwd")(
            means2d,
            conics,
            colors,
            opacities,
            pix_vels,
            depth_compensations,
            backgrounds,
            raster_pts,
            width,
            height,
            tile_width,
            tile_height,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            fr_depth,
            fr_weight,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            v_alpha_sum_until_points.contiguous()
            if compute_alpha_sum_until_points
            else torch.zeros_like(v_render_alphas),
            v_fr_depth.contiguous() if v_fr_depth is not None else None,
            absgrad,
            compute_alpha_sum_until_points,
            compute_alpha_sum_until_points_threshold,
            depth_channel_idx,
        )

        if absgrad:
            means2d.absgrad = v_means2d_abs

        if ctx.needs_input_grad[6]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
                dim=(1, 2)
            )
        else:
            v_backgrounds = None

        return (
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_pix_vels,
            v_depth_compensations,
            v_backgrounds,
            None,  # raster_pts
            None,  # width
            None,  # height
            None,  # tile_width
            None,  # tile_height
            None,  # tile_col_offset
            None,  # depth_lanes
            None,  # isect_offsets
            None,  # flatten_ids
            None,  # compute_alpha_sum_until_points
            None,  # compute_alpha_sum_until_points_threshold
            None,  # absgrad
            None,  # depth_channel_idx
            None,  # static_render
            None,  # use_depth_comp
        )
