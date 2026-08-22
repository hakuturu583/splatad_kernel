import math
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from splatad_kernel.cuda._wrapper import (
    fully_fused_lidar_projection,
    isect_lidar_tiles,
    isect_offset_encode,
    rasterize_to_points,
)


def lidar_rasterization(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    lidar_features: Tensor,  # [(C,) N, D]
    velocities: Optional[Tensor],  # [N, 3]
    viewmats: Tensor,  # [C, 4, 4]
    raster_pts: Tensor,  # [C, H, W, 4]
    tile_elevation_boundaries: Tensor,  # [n_elevation_channels//tile_height + 1]
    linear_velocity: Optional[Tensor] = None,  # [C, 3]
    angular_velocity: Optional[Tensor] = None,  # [C, 3]
    rolling_shutter_time: Optional[Tensor] = None,  # [C]
    min_azimuth: float = -180,
    max_azimuth: float = 180,
    min_elevation: float = -80,
    max_elevation: float = 80,
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
    row_elevations: Optional[Tensor] = None,  # [n_elevation_tiles], see below
    packed: bool = False,  # packed mode is not supported yet
    sparse_grad: bool = False,
    absgrad: bool = False,
    tile_col_offset: int = 0,  # sector rendering, see rasterize_to_points
    valid_mask: Optional[Tensor] = None,  # [N] bool, see fully_fused_lidar_projection
    depth_lanes: bool = False,  # 16 depth lanes/pixel, see rasterize_to_points
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
    use_depth_compensation: bool = True,
) -> Tuple[Tensor, Tensor, Union[Tensor, None], Dict]:
    """Rasterize a set of 3D Gaussians (N) to a batch of spherical lidar range images (C).

    This function provides a handful features for 3D Gaussian rasterization, which
    we detail in the following notes.

    .. note::
        **Batch Rasterization**: This function allows for rasterizing a set of 3D Gaussians
        to a batch of outputs in one go, by simplly providing the batched `viewmats` and `Ks`.

    .. note::
        **Memory-Speed Trade-off**: The `packed` argument provides a trade-off between
        memory footprint and runtime. If `packed` is True, the intermediate results are
        packed into sparse tensors, which is more memory efficient but might be slightly
        slower. This is especially helpful when the scene is large and each camera sees only
        a small portion of the scene. If `packed` is False, the intermediate results are
        with shape [C, N, ...], which is faster but might consume more memory. Not currently supported.

    .. note::
        **Sparse Gradients**: If `sparse_grad` is True, the gradients for {means, quats, scales}
        will be stored in a `COO sparse layout <https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html>`_.
        This can be helpful for saving memory
        for training when the scene is large and each iteration only activates a small portion
        of the Gaussians. Usually a sparse optimizer is required to work with sparse gradients,
        such as `torch.optim.SparseAdam <https://pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html#sparseadam>`_.
        This argument is only effective when `packed` is True. Not currently supported.

    .. note::
        **Speed-up for Large Scenes**: The `radius_clip` argument is extremely helpful for
        speeding up large scale scenes or scenes with large depth of fields. Gaussians with
        2D radius smaller or equal than this value (in degrees) will be skipped during rasterization.
        This will skip all the far-away Gaussians that are too small to be seen in the image.
        But be warned that if there are close-up Gaussians that are also below this threshold, they will
        also get skipped (which rarely happens in practice). This is by default disabled by setting
        `radius_clip` to 0.0.

    .. note::
        **Antialiased Rendering**: If `rasterize_mode` is "antialiased", the function will
        apply a view-dependent compensation factor
        :math:`\\rho=\\sqrt{\\frac{Det(\\Sigma)}{Det(\\Sigma+ \\epsilon I)}}` to Gaussian
        opacities, where :math:`\\Sigma` is the projected 2D covariance matrix and :math:`\\epsilon`
        is the `eps2d`. This will make the rendered output more antialiased, as proposed in
        the paper `Mip-Splatting: Alias-free 3D Gaussian Splatting <https://arxiv.org/pdf/2311.16493>`_.

    .. note::
        **AbsGrad**: If `absgrad` is True, the absolute gradients of the projected
        2D means will be computed during the backward pass, which could be accessed by
        `meta["means2d"].absgrad`. This is an implementation of the paper
        `AbsGS: Recovering Fine Details for 3D Gaussian Splatting <https://arxiv.org/abs/2404.10484>`_,
        which is shown to be more effective for splitting Gaussians during training.

    Args:
        means: The 3D centers of the Gaussians. [N, 3]
        quats: The quaternions of the Gaussians. It's not required to be normalized. [N, 4]
        scales: The scales of the Gaussians. [N, 3]
        opacities: The opacities of the Gaussians. [N]
        lidar_features: The features of the Gaussians. [(C,) N, D]
        velocities: The 3D velocities of the Gaussians. [N, 3]
        viewmats: The world-to-cam transformation of the lidars. [C, 4, 4]
        raster_pts: The rasterization points. This specficies the location of the points
            to rasterize in terms of spherical coordinates (3D), and their rolling shutter
            times. [C, H, W, 4]
        linear_velocity: The linear velocities of the lidars in world frame. [C, 3]
        angular_velocity: The angular velocities of the lidars in their own frames. [C, 3]
        rolling_shutter_time: The rolling shutter duration of the lidars. [C]
        min_azimuth: The minimum azimuth angle. Default is -180.0.
        max_azimuth: The maximum azimuth angle. Default is 180.0.
        min_elevation: The minimum elevation angle. Default is -80.0.
        max_elevation: The maximum elevation angle. Default is 80.0.
        n_elevation_channels: The number of elevation channels. Default is 32.
        azimuth_resolution: The azimuth angle between beams. Default is 0.1.
        tile_width: The width of the tiles for rasterization. Default is 1.
        tile_height: The height of the tiles for rasterization. Default is 1.
        tile_elevation_boundaries: The elevation boundaries of the tiles. [n_elevation_channels//tile_height + 1]
        near_plane: The near plane for clipping. Default is 0.01.
        far_plane: The far plane for clipping. Default is 1e10.
        radius_clip: Gaussians with 2D radius smaller or equal than this value will be
            skipped. This is extremely helpful for speeding up large scale scenes.
            Default is 0.0.
        eps2d: An epsilon added to the egienvalues of projected 2D covariance matrices.
            This will prevents the projected GS to be too small. Default is 0.017.
        compute_alpha_sum_until_points: Whether to compute the sum of alpha values until the depth specified
            by raster_pts. Default is True.
        compute_alpha_sum_until_points_threshold: Alpha sum is calculated until raster_pts depth - threshold.
            Default is 0.2.
        packed: Whether to use packed mode which is more memory efficient but might or
            might not be as fast. Default is False. Currently not supported.
        sparse_grad: If true, the gradients for {means, quats, scales} will be stored in
            a COO sparse layout. This can be helpful for saving memory. Default is False. Currently not supported.
        absgrad: If true, the absolute gradients of the projected 2D means
            will be computed during the backward pass, which could be accessed by
            `meta["means2d"].absgrad`. Default is False.
        rasterize_mode: The rasterization mode. Supported modes are "classic" and
            "antialiased". Default is "classic".
        channel_chunk: The number of channels to render in one go. Default is 32.
            If the required rendering channels are larger than this value, the rendering
            will be done looply in chunks.
        use_depth_compensation: Whether to use depth compensation, i.e., calculate the change in depth
            due to orientation of the Gaussian and the distance between the ray and the Gaussian center. Default is True.

    Returns:
        A tuple:

        **render_lidar_features**: The rendered features+expected distance. Expected distance is the last channel. [C, height, width, D+1].

        **render_alphas**: The rendered alphas. [C, height, width, 1].

        **alpha_sum_until_points**: The sum of alpha values until the depth specified by raster_pts minus threshold. [C, height, width, 1].

        **meta**: A dictionary of intermediate results of the rasterization. Contains median depths.

    Examples:

    .. code-block:: python

        >>> # define Gaussians
        >>> means = torch.randn((100, 3), device=device)
        >>> quats = torch.randn((100, 4), device=device)
        >>> scales = torch.rand((100, 3), device=device) * 0.1
        >>> opacities = torch.rand((100,), device=device)
        >>> lidar_feats = torch.rand((100, 16), device=device)
        >>> velocities = torch.randn((100, 3), device=device)
        >>> # define lidars
        >>> viewmats = torch.eye(4, device=device)[None, :, :]
        >>> width, height = 300, 200
        >>> raster_pts = torch.rand((1, height, width, 4), device=device)
        >>> tile_elevation_boundaries = torch.linspace(min_elevation, max_elevation, n_elevation_channels//tile_height + 1, device=device)
        >>> # render
        >>> lidar_features, alphas, meta = lidar_rasterization(
        >>>    means, quats, scales, opacities, lidar_feats, velocities, viewmats, raster_pts, tile_elevation_boundaries, width, height
        >>> )
        >>> print (lidar_features.shape, alphas.shape)
        torch.Size([1, 200, 300, 17]) torch.Size([1, 200, 300, 1])
        >>> print (meta.keys())
        dict_keys(['camera_ids', 'gaussian_ids', 'radii', 'means2d', 'depths', 'conics',
        'opacities', 'pix_vels', 'tile_grid_width', 'tile_grid_height', 'tiles_per_gauss', 'isect_ids',
        'flatten_ids', 'isect_offsets', 'width', 'height', 'tile_width', 'tile_width', 'n_cameras', 'median_depths'])

    """

    N = means.shape[0]  # number of Gaussians
    C = viewmats.shape[0]  # number of lidars
    D = lidar_features.shape[-1]  # feature dimension
    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert opacities.shape == (N,), opacities.shape
    if velocities is not None:
        assert velocities.shape == (N, 3), velocities.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert lidar_features.shape == (C, N, D), lidar_features.shape
    assert min_azimuth < max_azimuth, (min_azimuth, max_azimuth)
    assert min_elevation < max_elevation, (min_elevation, max_elevation)
    assert min_azimuth >= -180.0 and max_azimuth <= 180.0, (min_azimuth, max_azimuth)
    assert min_elevation >= -85.0 and max_elevation <= 85.0, (
        min_elevation,
        max_elevation,
    )  # beyond this range, the function is not numerically stable
    assert tile_width * tile_height <= 256, (
        tile_width,
        tile_height,
    )  # tile size is limited to 256
    assert n_elevation_channels > 0, n_elevation_channels
    # assert n_elevation_channels % tile_height == 0, (n_elevation_channels, tile_height)
    assert tile_elevation_boundaries.shape == (
        math.ceil(n_elevation_channels / tile_height) + 1,
    ), tile_elevation_boundaries.shape
    assert azimuth_resolution > 0.0, azimuth_resolution
    # splatsim static fast path: with no per-Gaussian velocity and no camera
    # linear/angular velocity, the projected pix_vels are zero, so the
    # rolling-shutter term (roll_time * pix_vel) vanishes regardless of roll_time;
    # combined with depth compensation off (depth_compensations forced to 0
    # below) the rasterizer's velocity / depth-comp terms are all zero and can be
    # skipped for a big occupancy win. Any velocity input falls back to the full
    # path, so rolling shutter stays fully supported.
    static_render = (
        velocities is None
        and linear_velocity is None
        and angular_velocity is None
        and not use_depth_compensation
    )
    if linear_velocity is not None:
        assert linear_velocity.shape == (C, 3), linear_velocity.shape
    else:
        linear_velocity = torch.zeros(C, 3, device=means.device)
    if angular_velocity is not None:
        assert angular_velocity.shape == (C, 3), angular_velocity.shape
    else:
        angular_velocity = torch.zeros(C, 3, device=means.device)
    if rolling_shutter_time is not None:
        assert rolling_shutter_time.shape == (C,), rolling_shutter_time.shape
    else:
        rolling_shutter_time = torch.zeros(C, device=means.device)
    assert raster_pts.shape == (C, *raster_pts.shape[1:]), raster_pts.shape
    assert raster_pts.shape[-1] == 4, raster_pts.shape

    # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
    proj_results = fully_fused_lidar_projection(
        means,
        None,  # covars,
        quats,
        scales,
        velocities,
        viewmats,
        linear_velocity,
        angular_velocity,
        rolling_shutter_time,
        valid_mask=valid_mask,
        min_elevation=min_elevation,
        max_elevation=max_elevation,
        min_azimuth=min_azimuth,
        max_azimuth=max_azimuth,
        eps2d=eps2d,
        packed=packed,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        sparse_grad=sparse_grad,
        calc_compensations=(rasterize_mode == "antialiased"),
    )

    if packed:
        # The results are packed into shape [nnz, ...]. All elements are valid.
        # TODO: Implement packed mode for lidar_rasterization
        raise NotImplementedError(
            "Packed mode is not supported for lidar_rasterization"
        )
        (
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            conics,
            compensations,
        ) = proj_results
        opacities = opacities[gaussian_ids]  # [nnz]
    else:
        # The results are with shape [C, N, ...]. Only the elements with radii > 0 are valid.
        radii, means2d, depths, conics, compensations, pix_vels, depth_compensations = (
            proj_results
        )
        # [C, N]; for the single-lidar case an unsqueeze view is contiguous
        # already, so skip the repeat's full copy of the opacity buffer.
        opacities = opacities.unsqueeze(0) if C == 1 else opacities.repeat(C, 1)
        camera_ids, gaussian_ids = None, None

    if not use_depth_compensation and not static_render:
        # Zeroing matters only when the kernel actually reads the depth-comp
        # batch (non-static path); the STATIC kernel never touches it, so the
        # [C, N, 2] zeroing launch would be pure overhead there.
        depth_compensations = depth_compensations * 0

    if compensations is not None:
        opacities = opacities * compensations

    # Identify intersecting tiles
    n_elevation_tiles = math.ceil(n_elevation_channels / tile_height)
    tile_azimuth_resolution = azimuth_resolution * tile_width
    n_azimuth_tiles = math.ceil((max_azimuth - min_azimuth) / tile_azimuth_resolution)
    tiles_per_gauss, isect_ids, flatten_ids = isect_lidar_tiles(
        means2d,
        radii,
        depths,
        elev_boundaries=tile_elevation_boundaries,
        tile_azim_resolution=tile_azimuth_resolution,
        min_azim=min_azimuth,
        packed=packed,
        n_cameras=C,
        camera_ids=camera_ids,
        gaussian_ids=gaussian_ids,
        # Exact per-row azimuth spans (see isect_lidar_tiles): fewer pairs,
        # still a superset of what any pixel in the row can be hit by.
        conics=conics,
        opacities=opacities,
        row_elevations=row_elevations,
    )
    isect_offsets = isect_offset_encode(
        isect_ids, C, n_azimuth_tiles, n_elevation_tiles
    )

    image_width = raster_pts.shape[-2]
    image_height = raster_pts.shape[-3]

    if (lidar_features.shape[-1] + 1) > channel_chunk:
        # slice into chunks
        n_chunks = (lidar_features.shape[-1] + channel_chunk) // channel_chunk
        render_lidar_features, render_alphas, alpha_sum_until_points, median_depths, fr_depth, median_ids = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for i in range(n_chunks):
            lidar_features_chunk = lidar_features[
                ..., i * (channel_chunk - 1) : (i + 1) * (channel_chunk - 1)
            ]

            (
                render_lidar_features_,
                render_alphas_,
                alpha_sum_until_points_,
                median_depths_,
                fr_depth_,
                median_ids_,
            ) = rasterize_to_points(
                means2d,
                conics,
                torch.cat([lidar_features_chunk, depths[..., None]], dim=-1),
                opacities,
                pix_vels,
                depth_compensations,
                raster_pts,
                image_width,
                image_height,
                tile_width,
                tile_height,
                isect_offsets,
                flatten_ids,
                compute_alpha_sum_until_points=compute_alpha_sum_until_points,
                compute_alpha_sum_until_points_threshold=compute_alpha_sum_until_points_threshold,
                packed=packed,
                absgrad=absgrad,
                static_render=static_render,
                tile_col_offset=tile_col_offset,
                use_depth_comp=use_depth_compensation,
                depth_lanes=depth_lanes,
            )
            if i == (n_chunks - 1):
                render_lidar_features.append(render_lidar_features_)
            else:
                render_lidar_features.append(render_lidar_features_[..., :-1])
            render_alphas.append(render_alphas_)
            alpha_sum_until_points.append(alpha_sum_until_points_)
            median_depths.append(median_depths_)
            fr_depth.append(fr_depth_)
            median_ids.append(median_ids_)
        render_lidar_features = torch.cat(render_lidar_features, dim=-1)
        median_depths = torch.cat(median_depths, dim=-1)
        fr_depth = torch.cat(fr_depth, dim=-1)
        median_ids = median_ids[0]  # crossing index is chunk-invariant (same geometry)
        render_alphas = render_alphas[0]  # discard the rest
        alpha_sum_until_points = alpha_sum_until_points[0]  # same alphas for all chunks
    else:
        render_lidar_features, render_alphas, alpha_sum_until_points, median_depths, fr_depth, median_ids = (
            rasterize_to_points(
                means2d,
                conics,
                torch.cat([lidar_features, depths[..., None]], dim=-1),
                opacities,
                pix_vels,
                depth_compensations,
                raster_pts,
                image_width,
                image_height,
                tile_width,
                tile_height,
                isect_offsets,
                flatten_ids,
                compute_alpha_sum_until_points=compute_alpha_sum_until_points,
                compute_alpha_sum_until_points_threshold=compute_alpha_sum_until_points_threshold,
                packed=packed,
                absgrad=absgrad,
                static_render=static_render,
                tile_col_offset=tile_col_offset,
                use_depth_comp=use_depth_compensation,
                depth_lanes=depth_lanes,
            )
        )

    meta = {
        "camera_ids": camera_ids,
        "gaussian_ids": gaussian_ids,
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "conics": conics,
        "opacities": opacities,
        "pix_vels": pix_vels,
        "tile_grid_width": n_azimuth_tiles,
        "tile_grid_height": n_elevation_tiles,
        "tiles_per_gauss": tiles_per_gauss,
        "isect_ids": isect_ids,
        "flatten_ids": flatten_ids,
        "isect_offsets": isect_offsets,
        "width": image_width,
        "height": image_height,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "n_cameras": C,
        "median_depths": median_depths,
        "fr_depth": fr_depth,
        "median_ids": median_ids,
    }
    return render_lidar_features, render_alphas, alpha_sum_until_points, meta
