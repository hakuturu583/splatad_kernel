"""The WebGPU LiDAR rasterizer: the compute stages and the host that drives them.

Stage for stage this is the CUDA pipeline in ``lidar_rasterization``:

    project  -> isect count -> prefix sum -> isect encode -> sort
             -> tile offsets -> rasterize

Forward only. The CUDA path also carries the backward pass, which needs
per-Gaussian float atomics and the saved-tensor bookkeeping of an autograd
Function; nothing here is differentiable and the module is for inference.

One host round trip sits in the middle, exactly where the CUDA path has one:
the intersection count is a device-side prefix sum whose total decides how big
the intersection buffers have to be.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ._device import get_device
from ._pipeline import (
    FLAG_CALC_COMPENSATIONS,
    FLAG_COMPUTE_ALPHA_SUM,
    FLAG_EXACT_ROW_SPANS,
    FLAG_HAS_ROW_ELEVATIONS,
    FLAG_HAS_VALID_MASK,
    FLAG_HAS_VELOCITIES,
    PARAMS_SIZE,
    UNIFORM_USAGE,
    Resources,
    pack_params,
)

# Mirrors the constants at the top of shaders/common.wgsl.
PROJ_STRIDE = 16
GAUSS_STRIDE = 16
SENSOR_STRIDE = 32
AUX_STRIDE = 8
AUX_ALPHA, AUX_ALPHA_SUM, AUX_MEDIAN_DEPTH = 0, 1, 2
AUX_FR_DEPTH, AUX_FR_WEIGHT, AUX_MEDIAN_ID, AUX_LAST_ID = 3, 4, 5, 6

# shaders/sort.wgsl LOCAL_ELEMS: the chunk a workgroup sorts in place.
SORT_LOCAL_ELEMS = 512
# shaders/radix.wgsl: WG * PER_THREAD elements per workgroup, 4 bits per pass.
RADIX_WG = 64
RADIX_CHUNK = 1024
RADIX_BINS = 16
ISECT_STRIDE_BYTES = 16  # vec4<u32>: camera|tile, depth bits, flatten id, unused
LINEAR_WG = 256
SCAN_WG = 256
SORT_WG = 256
OFFSET_WG = 64
UNIFORM_SLOT = 256  # max of the min-uniform-offset-alignment limits in the wild


def _as_f32(array: Any, name: str) -> np.ndarray:
    """Accept torch tensors, numpy arrays or anything array-like."""
    if array is None:
        return None
    if hasattr(array, "detach"):  # torch.Tensor
        array = array.detach().cpu().numpy()
    out = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains NaN or inf")
    return out


def _next_pow2(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


def _radix_passes(max_hi: int):
    """The (word, shift) of each 4-bit radix pass, least significant first.

    The depth word is sorted through all 32 bits; the (camera | tile) word only
    through the bits the tile grid can reach, which for a typical panorama is
    two passes instead of eight.
    """
    passes = [(0, shift) for shift in range(0, 32, 4)]
    hi_bits = max(int(max_hi).bit_length(), 1)
    passes += [(1, shift) for shift in range(0, hi_bits, 4)]
    return passes


def _sort_schedule(n_pad: int):
    """The (kind, k, j) bitonic steps for a padded length of ``n_pad``.

    ``sort_local_init`` covers stages up to SORT_LOCAL_ELEMS in one dispatch;
    beyond that each stage runs its wide strides globally and finishes the
    tail (every stride that fits one workgroup's chunk) in ``step_local``.
    """
    steps = [("init", 0, 0)]
    k = SORT_LOCAL_ELEMS * 2
    while k <= n_pad:
        j = k >> 1
        while j >= SORT_LOCAL_ELEMS:
            steps.append(("global", k, j))
            j >>= 1
        steps.append(("local", k, SORT_LOCAL_ELEMS >> 1))
        k <<= 1
    return steps


class LidarRasterizer:
    """A reusable WebGPU LiDAR rasterizer.

    Holds the device, the compiled pipelines and the GPU buffers, so rendering
    frame after frame of the same scene re-uploads only what changed shape.
    Construct one and call it repeatedly; a throwaway instance per frame
    recompiles every shader.
    """

    def __init__(self, device: Optional[Any] = None, sort_algorithm: str = "radix"):
        if sort_algorithm not in ("radix", "bitonic"):
            raise ValueError(f"unknown sort_algorithm {sort_algorithm!r}")
        self.sort_algorithm = sort_algorithm
        self.device = get_device(device)
        self.res = Resources(self.device)
        limits = self.device.limits
        self.max_workgroups = int(limits.get("max-compute-workgroups-per-dimension", 65535))
        self.max_wg_storage = int(limits.get("max-compute-workgroup-storage-size", 16384))
        self.max_binding = int(limits.get("max-storage-buffer-binding-size", 128 << 20))

    # ------------------------------------------------------------------
    def _dispatch(self, cpass, pipeline, bind_group, n_threads, wg_size, offsets=None):
        n_wg = (int(n_threads) + wg_size - 1) // wg_size
        if n_wg == 0:
            return
        if n_wg > self.max_workgroups:
            raise ValueError(
                f"{n_threads} threads need {n_wg} workgroups, over this device's "
                f"limit of {self.max_workgroups}. Split the render (fewer Gaussians "
                f"or fewer sensors per call)."
            )
        cpass.set_pipeline(pipeline)
        if offsets:
            cpass.set_bind_group(0, bind_group, offsets, 0, len(offsets))
        else:
            cpass.set_bind_group(0, bind_group)
        cpass.dispatch_workgroups(n_wg)

    def _scan_inclusive(self, cpass, src, n, level=0, prefix="scan"):
        """Inclusive prefix sum of ``src[:n]``; returns the buffer holding it.

        ``prefix`` names the scratch buffers, so two scans that have to coexist
        within a frame (the tile counts and the radix histograms) do not share
        them.
        """
        n_blocks = (n + SCAN_WG - 1) // SCAN_WG
        dst = self.res.buffer(f"{prefix}_dst_{level}", n * 4)
        sums = self.res.buffer(f"{prefix}_sums_{level}", n_blocks * 4)
        scan_block = self.res.pipeline("scan", "scan.wgsl", "scan_block")
        bg = self.res.bind_group(
            "scan", [(src, n * 4), (dst, n * 4), (sums, n_blocks * 4)]
        )
        self._dispatch(cpass, scan_block, bg, n, SCAN_WG)
        if n_blocks > 1:
            scanned = self._scan_inclusive(cpass, sums, n_blocks, level + 1, prefix)
            add = self.res.pipeline("scan", "scan.wgsl", "add_offsets")
            bg2 = self.res.bind_group(
                "scan",
                [(scanned, n_blocks * 4), (dst, n * 4), (sums, n_blocks * 4)],
            )
            self._dispatch(cpass, add, bg2, n, SCAN_WG)
        return dst

    # ------------------------------------------------------------------
    def _bitonic_sort(self, cpass, data, n_pad):
        """Sort ``data[:n_pad]`` in place; returns the buffer holding the result."""
        res = self.res
        steps = _sort_schedule(n_pad)
        step_data = np.zeros((len(steps) * UNIFORM_SLOT) // 4, np.uint32)
        for i, (_, k, j) in enumerate(steps):
            base = i * UNIFORM_SLOT // 4
            step_data[base : base + 4] = (k, j, n_pad, 0)
        step_buf = res.buffer("sort_steps", max(step_data.nbytes, 256), UNIFORM_USAGE)
        self.device.queue.write_buffer(step_buf, 0, step_data)

        nbytes = n_pad * ISECT_STRIDE_BYTES
        kinds = {
            "init": (res.pipeline("sort", "sort.wgsl", "sort_local_init"), SORT_LOCAL_ELEMS),
            "local": (res.pipeline("sort", "sort.wgsl", "step_local"), SORT_LOCAL_ELEMS),
            "global": (res.pipeline("sort", "sort.wgsl", "step_global"), SORT_WG),
        }
        for i, (kind, _, _) in enumerate(steps):
            pipeline, per_group = kinds[kind]
            bg = res.bind_group(
                "sort", [(step_buf, 16), (data, nbytes)], offsets=[i * UNIFORM_SLOT, 0]
            )
            self._dispatch(cpass, pipeline, bg, n_pad, per_group)
        return data

    def _radix_sort(self, cpass, data, n_pad, n_cam, tile_n_bits, n_tiles):
        """Stable LSD radix sort; returns whichever ping-pong buffer ends up sorted."""
        res = self.res
        num_chunks = n_pad // RADIX_CHUNK
        hist_len = RADIX_BINS * num_chunks
        max_hi = ((n_cam - 1) << tile_n_bits) | max(n_tiles - 1, 0)
        passes = _radix_passes(max_hi)

        pass_data = np.zeros((len(passes) * UNIFORM_SLOT) // 4, np.uint32)
        for i, (word, shift) in enumerate(passes):
            base = i * UNIFORM_SLOT // 4
            pass_data[base : base + 4] = (n_pad, shift, word, num_chunks)
        pass_buf = res.buffer("radix_params", pass_data.nbytes, UNIFORM_USAGE)
        self.device.queue.write_buffer(pass_buf, 0, pass_data)

        nbytes = n_pad * ISECT_STRIDE_BYTES
        alt = res.buffer("isects_alt", nbytes)
        hist = res.buffer("radix_hist", hist_len * 4)
        histogram = res.pipeline("radix", "radix.wgsl", "histogram")
        scatter = res.pipeline("radix", "radix.wgsl", "scatter")

        src, dst = data, alt
        for i in range(len(passes)):
            # The histogram dispatch does not read hist_incl, but the binding
            # has to exist and has to be a different buffer from `hist`.
            placeholder = res.buffer("radix_scan_dst_0", hist_len * 4)
            self._dispatch(
                cpass,
                histogram,
                res.bind_group(
                    "radix",
                    [
                        (pass_buf, 16),
                        (src, nbytes),
                        (dst, nbytes),
                        (hist, hist_len * 4),
                        (placeholder, hist_len * 4),
                    ],
                    offsets=[i * UNIFORM_SLOT, 0, 0, 0, 0],
                ),
                n_pad // RADIX_CHUNK * RADIX_WG,
                RADIX_WG,
            )
            incl = self._scan_inclusive(cpass, hist, hist_len, prefix="radix_scan")
            self._dispatch(
                cpass,
                scatter,
                res.bind_group(
                    "radix",
                    [
                        (pass_buf, 16),
                        (src, nbytes),
                        (dst, nbytes),
                        (hist, hist_len * 4),
                        (incl, hist_len * 4),
                    ],
                    offsets=[i * UNIFORM_SLOT, 0, 0, 0, 0],
                ),
                n_pad // RADIX_CHUNK * RADIX_WG,
                RADIX_WG,
            )
            src, dst = dst, src
        return src

    # ------------------------------------------------------------------
    def render(
        self,
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
        tile_col_offset: int = 0,
        valid_mask=None,
        rasterize_mode: str = "classic",
        use_depth_compensation: bool = True,
        raydrop_sh_coeffs=None,
        raydrop_sh_degree: int = 0,
        raydrop_feature_index: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        means = _as_f32(means, "means")
        quats = _as_f32(quats, "quats")
        scales = _as_f32(scales, "scales")
        opacities = _as_f32(opacities, "opacities")
        lidar_features = _as_f32(lidar_features, "lidar_features")
        velocities = _as_f32(velocities, "velocities")
        viewmats = _as_f32(viewmats, "viewmats")
        raster_pts = _as_f32(raster_pts, "raster_pts")
        elev_boundaries = _as_f32(tile_elevation_boundaries, "tile_elevation_boundaries")
        row_elevations = _as_f32(row_elevations, "row_elevations")
        raydrop_sh_coeffs = _as_f32(raydrop_sh_coeffs, "raydrop_sh_coeffs")

        n_gauss = means.shape[0]
        n_cam = viewmats.shape[0]
        n_feat = lidar_features.shape[-1]
        height, width = raster_pts.shape[1], raster_pts.shape[2]

        if lidar_features.shape != (n_cam, n_gauss, n_feat):
            raise ValueError(
                f"lidar_features must be [C, N, D]; got {lidar_features.shape} "
                f"for C={n_cam}, N={n_gauss}"
            )
        if raster_pts.shape != (n_cam, height, width, 4):
            raise ValueError(f"raster_pts must be [C, H, W, 4]; got {raster_pts.shape}")
        if tile_width * tile_height > 256:
            raise ValueError("tile_width * tile_height must be <= 256")
        if not min_azimuth < max_azimuth or not min_elevation < max_elevation:
            raise ValueError("the azimuth and elevation ranges must be non-empty")

        n_elev_tiles = math.ceil(n_elevation_channels / tile_height)
        if elev_boundaries.shape != (n_elev_tiles + 1,):
            raise ValueError(
                f"tile_elevation_boundaries must have {n_elev_tiles + 1} entries; "
                f"got {elev_boundaries.shape}"
            )
        tile_azim_res = azimuth_resolution * tile_width
        n_azim_tiles = math.ceil((max_azimuth - min_azimuth) / tile_azim_res)
        n_tiles = n_azim_tiles * n_elev_tiles
        tile_n_bits = int(math.floor(math.log2(n_tiles))) + 1

        n_pix_tiles_x = (width + tile_width - 1) // tile_width
        if tile_col_offset + n_pix_tiles_x > n_azim_tiles:
            raise ValueError(
                f"tile_col_offset ({tile_col_offset}) + image tile columns "
                f"({n_pix_tiles_x}) exceeds the tile grid width ({n_azim_tiles})"
            )

        # Same rule as rendering.py: with no velocity anywhere and depth
        # compensation off, every rolling-shutter and depth-tilt term is zero
        # and the rasterizer drops to the narrower staged record.
        static_render = (
            velocities is None
            and linear_velocity is None
            and angular_velocity is None
            and not use_depth_compensation
        )

        lin_vel = (
            np.zeros((n_cam, 3), np.float32)
            if linear_velocity is None
            else _as_f32(linear_velocity, "linear_velocity")
        )
        ang_vel = (
            np.zeros((n_cam, 3), np.float32)
            if angular_velocity is None
            else _as_f32(angular_velocity, "angular_velocity")
        )
        rs_time = (
            np.zeros((n_cam,), np.float32)
            if rolling_shutter_time is None
            else _as_f32(rolling_shutter_time, "rolling_shutter_time")
        )

        flags = 0
        if velocities is not None:
            flags |= FLAG_HAS_VELOCITIES
        if valid_mask is not None:
            flags |= FLAG_HAS_VALID_MASK
        if rasterize_mode == "antialiased":
            flags |= FLAG_CALC_COMPENSATIONS
        # The exact per-row azimuth span needs the conic and the opacity, both
        # of which the projection record always carries, so it is always on --
        # as rendering.py has it.
        flags |= FLAG_EXACT_ROW_SPANS
        if row_elevations is not None:
            flags |= FLAG_HAS_ROW_ELEVATIONS
        if compute_alpha_sum_until_points:
            flags |= FLAG_COMPUTE_ALPHA_SUM

        # ---- host-side packing -----------------------------------------
        gauss = np.zeros((n_gauss, GAUSS_STRIDE), np.float32)
        gauss[:, 0:3] = means
        gauss[:, 3] = opacities
        gauss[:, 4:8] = quats
        gauss[:, 8:11] = scales
        gauss[:, 11] = 1.0 if valid_mask is None else np.asarray(valid_mask).astype(np.float32)
        if velocities is not None:
            gauss[:, 12:15] = velocities

        sensors = np.zeros((n_cam, SENSOR_STRIDE), np.float32)
        sensors[:, 0:16] = viewmats.reshape(n_cam, 16)
        sensors[:, 16:19] = lin_vel
        sensors[:, 19] = rs_time
        sensors[:, 20:23] = ang_vel
        # Sensor origin in world coordinates: the translation of inv(viewmat).
        rot = viewmats[:, :3, :3]
        trans = viewmats[:, :3, 3]
        sensors[:, 24:27] = -np.einsum("cji,cj->ci", rot, trans)

        elev_base = n_cam * SENSOR_STRIDE
        row_elev_base = elev_base + n_elev_tiles + 1
        sensor_blob = np.concatenate(
            [
                sensors.ravel(),
                elev_boundaries.ravel(),
                (
                    row_elevations.ravel()
                    if row_elevations is not None
                    else np.zeros(n_elev_tiles, np.float32)
                ),
            ]
        ).astype(np.float32)

        params = dict(
            n_gaussians=n_gauss,
            n_cameras=n_cam,
            n_features=n_feat,
            flags=flags,
            min_azimuth=min_azimuth,
            max_azimuth=max_azimuth,
            min_elevation=min_elevation,
            max_elevation=max_elevation,
            eps2d=eps2d,
            near_plane=near_plane,
            far_plane=far_plane,
            radius_clip=radius_clip,
            n_tiles_azim=n_azim_tiles,
            n_tiles_elev=n_elev_tiles,
            tile_azim_resolution=tile_azim_res,
            tile_n_bits=tile_n_bits,
            image_width=width,
            image_height=height,
            tile_width=tile_width,
            tile_height=tile_height,
            tile_col_offset=tile_col_offset,
            alpha_sum_threshold=compute_alpha_sum_until_points_threshold,
            n_isects=0,
            sh_degree=raydrop_sh_degree if raydrop_sh_coeffs is not None else 0,
            raydrop_feature_index=(
                raydrop_feature_index if raydrop_feature_index >= 0
                else n_feat + raydrop_feature_index
            ),
            elev_base=elev_base,
            row_elev_base=row_elev_base,
            n_isects_padded=0,
        )

        # ---- buffers ----------------------------------------------------
        res = self.res
        queue = self.device.queue
        cn = n_cam * n_gauss
        pix = n_cam * height * width

        proj_bytes = cn * PROJ_STRIDE * 4
        if proj_bytes > self.max_binding:
            raise ValueError(
                f"the projection buffer needs {proj_bytes / 2**20:.0f} MiB but this "
                f"device caps a storage binding at {self.max_binding / 2**20:.0f} MiB; "
                f"render fewer Gaussians or fewer sensors per call"
            )

        p_buf = res.buffer("params", PARAMS_SIZE, UNIFORM_USAGE)
        g_buf = res.buffer("gaussians", gauss.nbytes)
        s_buf = res.buffer("sensors", sensor_blob.nbytes)
        proj_buf = res.buffer("proj", proj_bytes)
        feat_buf = res.buffer("features", lidar_features.nbytes)
        pts_buf = res.buffer("raster_pts", raster_pts.nbytes)
        counts_buf = res.buffer("counts", cn * 4)

        queue.write_buffer(p_buf, 0, pack_params(params))
        queue.write_buffer(g_buf, 0, gauss)
        queue.write_buffer(s_buf, 0, sensor_blob)
        queue.write_buffer(feat_buf, 0, np.ascontiguousarray(lidar_features))
        queue.write_buffer(pts_buf, 0, raster_pts)

        # ---- pass 1: project, optional SH raydrop, count, prefix sum -----
        encoder = self.device.create_command_encoder()
        cpass = encoder.begin_compute_pass()

        project = res.pipeline("project", "project.wgsl", "main")
        bg = res.bind_group(
            "project",
            [
                (p_buf, PARAMS_SIZE),
                (g_buf, gauss.nbytes),
                (s_buf, sensor_blob.nbytes),
                (proj_buf, proj_bytes),
            ],
        )
        self._dispatch(cpass, project, bg, cn, LINEAR_WG)

        if params["sh_degree"] > 0:
            n_bands = (raydrop_sh_degree + 1) ** 2 - 1
            if raydrop_sh_coeffs.shape != (n_gauss, n_bands):
                raise ValueError(
                    f"raydrop_sh_coeffs must be [N, {n_bands}] for degree "
                    f"{raydrop_sh_degree}; got {raydrop_sh_coeffs.shape}"
                )
            if raydrop_sh_degree > 4:
                raise ValueError(
                    f"raydrop_sh_degree={raydrop_sh_degree} exceeds the shader maximum 4"
                )
            sh_buf = res.buffer("sh_coeffs", raydrop_sh_coeffs.nbytes)
            queue.write_buffer(sh_buf, 0, raydrop_sh_coeffs)
            sh = res.pipeline("sh", "sh_raydrop.wgsl", "main")
            bg_sh = res.bind_group(
                "sh",
                [
                    (p_buf, PARAMS_SIZE),
                    (proj_buf, proj_bytes),
                    (g_buf, gauss.nbytes),
                    (s_buf, sensor_blob.nbytes),
                    (sh_buf, raydrop_sh_coeffs.nbytes),
                    (feat_buf, lidar_features.nbytes),
                ],
            )
            self._dispatch(cpass, sh, bg_sh, cn, LINEAR_WG)

        count = res.pipeline("isect", "isect.wgsl", "count")
        # The count pass reads neither the prefix sums nor the intersection
        # records -- the latter are not even sized yet -- but both still have to
        # be bound, and not to `counts`: a buffer cannot be both the read-write
        # and the read-only binding of one dispatch.
        dummy_ro = res.buffer("dummy_ro", 4)
        dummy_rw = res.buffer("dummy_rw", ISECT_STRIDE_BYTES)
        bg_count = res.bind_group(
            "isect",
            [
                (p_buf, PARAMS_SIZE),
                (proj_buf, proj_bytes),
                (s_buf, sensor_blob.nbytes),
                (counts_buf, cn * 4),
                (dummy_ro, 4),
                (dummy_rw, ISECT_STRIDE_BYTES),
            ],
        )
        self._dispatch(cpass, count, bg_count, cn, LINEAR_WG)

        cum_buf = self._scan_inclusive(cpass, counts_buf, cn)
        cpass.end()
        queue.submit([encoder.finish()])

        # The one host round trip: the total intersection count sizes the sort.
        n_isects = int(
            np.frombuffer(queue.read_buffer(cum_buf, (cn - 1) * 4, 4), np.uint32)[0]
        )

        # ---- pass 2: encode, sort, offsets, rasterize --------------------
        use_radix = self.sort_algorithm == "radix"
        if use_radix:
            # The radix sort wants whole chunks; the bitonic one wants a power
            # of two. Either way the tail is filled with sentinel keys.
            n_pad = max(RADIX_CHUNK, -(-n_isects // RADIX_CHUNK) * RADIX_CHUNK)
        else:
            n_pad = max(SORT_LOCAL_ELEMS, _next_pow2(n_isects))
        params["n_isects"] = n_isects
        params["n_isects_padded"] = n_pad
        queue.write_buffer(p_buf, 0, pack_params(params))

        isect_bytes = n_pad * ISECT_STRIDE_BYTES
        isects = res.buffer("isects", isect_bytes)
        offsets_buf = res.buffer("tile_offsets", (n_cam * n_tiles + 1) * 4)
        colors_buf = res.buffer("out_colors", pix * (n_feat + 1) * 4)
        aux_buf = res.buffer("out_aux", pix * AUX_STRIDE * 4)

        encoder = self.device.create_command_encoder()
        cpass = encoder.begin_compute_pass()

        encode = res.pipeline("isect", "isect.wgsl", "encode")
        bg_encode = res.bind_group(
            "isect",
            [
                (p_buf, PARAMS_SIZE),
                (proj_buf, proj_bytes),
                (s_buf, sensor_blob.nbytes),
                (counts_buf, cn * 4),
                (cum_buf, cn * 4),
                (isects, isect_bytes),
            ],
        )
        self._dispatch(cpass, encode, bg_encode, cn, LINEAR_WG)

        # pad_tail rides the bitonic sort's uniform, whichever sort follows.
        pad_data = np.zeros(UNIFORM_SLOT // 4, np.uint32)
        pad_data[0:4] = (n_pad, n_isects, n_pad, 0)  # fills [j, n) = [n_isects, n_pad)
        pad_buf = res.buffer("pad_step", pad_data.nbytes, UNIFORM_USAGE)
        queue.write_buffer(pad_buf, 0, pad_data)
        pad = res.pipeline("sort", "sort.wgsl", "pad_tail")
        self._dispatch(
            cpass,
            pad,
            res.bind_group("sort", [(pad_buf, 16), (isects, isect_bytes)]),
            n_pad - n_isects,
            SORT_WG,
        )

        if use_radix:
            sorted_buf = self._radix_sort(cpass, isects, n_pad, n_cam, tile_n_bits, n_tiles)
        else:
            sorted_buf = self._bitonic_sort(cpass, isects, n_pad)

        offsets_pipe = res.pipeline("offsets", "offsets.wgsl", "main")
        bg_off = res.bind_group(
            "offsets",
            [
                (p_buf, PARAMS_SIZE),
                (sorted_buf, isect_bytes),
                (offsets_buf, (n_cam * n_tiles + 1) * 4),
            ],
        )
        self._dispatch(cpass, offsets_pipe, bg_off, n_cam * n_tiles + 1, OFFSET_WG)

        block_size = tile_width * tile_height
        vec_per = 2 if static_render else 3
        # done_count + done_flag take 8 bytes of the workgroup allowance.
        per_gauss = vec_per * 16
        batch_mult = max(1, min(16, (self.max_wg_storage - 16) // (block_size * per_gauss)))
        raster = res.pipeline(
            "raster",
            "rasterize.wgsl",
            "main",
            defines={
                "STATIC": static_render,
                "DEPTH_COMP": use_depth_compensation and not static_render,
            },
            template=dict(
                COLOR_DIM=n_feat + 1,
                N_FEATURES=n_feat,
                BLOCK_SIZE=block_size,
                BATCH_MULT=batch_mult,
                VEC_PER=vec_per,
                STAGE_VEC4=block_size * batch_mult * vec_per,
                TILE_W=tile_width,
                TILE_H=tile_height,
            ),
        )
        bg_raster = res.bind_group(
            "raster",
            [
                (p_buf, PARAMS_SIZE),
                (proj_buf, proj_bytes),
                (feat_buf, lidar_features.nbytes),
                (pts_buf, raster_pts.nbytes),
                (offsets_buf, (n_cam * n_tiles + 1) * 4),
                (sorted_buf, isect_bytes),
                (colors_buf, pix * (n_feat + 1) * 4),
                (aux_buf, pix * AUX_STRIDE * 4),
            ],
        )
        cpass.set_pipeline(raster)
        cpass.set_bind_group(0, bg_raster)
        cpass.dispatch_workgroups(n_pix_tiles_x, n_elev_tiles, n_cam)
        cpass.end()
        queue.submit([encoder.finish()])

        # ---- readback ----------------------------------------------------
        colors = np.frombuffer(
            queue.read_buffer(colors_buf, 0, pix * (n_feat + 1) * 4), np.float32
        ).reshape(n_cam, height, width, n_feat + 1)
        aux = np.frombuffer(
            queue.read_buffer(aux_buf, 0, pix * AUX_STRIDE * 4), np.float32
        ).reshape(n_cam, height, width, AUX_STRIDE)

        alphas = aux[..., AUX_ALPHA : AUX_ALPHA + 1].copy()
        alpha_sum = (
            aux[..., AUX_ALPHA_SUM : AUX_ALPHA_SUM + 1].copy()
            if compute_alpha_sum_until_points
            else None
        )
        median_depths = aux[..., AUX_MEDIAN_DEPTH : AUX_MEDIAN_DEPTH + 1].copy()
        fr_depth = aux[..., AUX_FR_DEPTH : AUX_FR_DEPTH + 1].copy()
        median_ids = aux[..., AUX_MEDIAN_ID].view(np.int32).copy()

        proj_flat = np.frombuffer(
            queue.read_buffer(proj_buf, 0, proj_bytes), np.float32
        ).reshape(n_cam, n_gauss, PROJ_STRIDE)
        meta = {
            "camera_ids": None,
            "gaussian_ids": None,
            "radii": proj_flat[..., 7:9].copy(),
            "means2d": proj_flat[..., 0:2].copy(),
            "depths": proj_flat[..., 6].copy(),
            "conics": proj_flat[..., 3:6].copy(),
            "opacities": proj_flat[..., 2].copy(),
            "pix_vels": proj_flat[..., 9:12].copy(),
            "depth_compensations": proj_flat[..., 12:14].copy(),
            "tile_grid_width": n_azim_tiles,
            "tile_grid_height": n_elev_tiles,
            "tiles_per_gauss": np.frombuffer(
                queue.read_buffer(counts_buf, 0, cn * 4), np.uint32
            ).reshape(n_cam, n_gauss).astype(np.int32),
            "isect_offsets": np.frombuffer(
                queue.read_buffer(offsets_buf, 0, n_cam * n_tiles * 4), np.int32
            ).reshape(n_cam, n_elev_tiles, n_azim_tiles).copy(),
            "flatten_ids": np.frombuffer(
                queue.read_buffer(sorted_buf, 0, max(n_isects, 1) * ISECT_STRIDE_BYTES),
                np.uint32,
            ).reshape(-1, 4)[:n_isects, 2].astype(np.int32),
            "n_isects": n_isects,
            "width": width,
            "height": height,
            "tile_width": tile_width,
            "tile_height": tile_height,
            "n_cameras": n_cam,
            "median_depths": median_depths,
            "fr_depth": fr_depth,
            "fr_weight": aux[..., AUX_FR_WEIGHT : AUX_FR_WEIGHT + 1].copy(),
            "median_ids": median_ids,
            "last_ids": aux[..., AUX_LAST_ID].view(np.int32).copy(),
        }
        return colors.copy(), alphas, alpha_sum, meta
