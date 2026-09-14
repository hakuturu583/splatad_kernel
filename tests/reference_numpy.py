"""A NumPy transcription of the CUDA LiDAR forward pass, used as ground truth.

This is deliberately a separate, straightforward reading of the CUDA sources
(cuda/csrc/projection.cu, rasterization.cu, utils.cuh, helpers.cuh) rather than
a refactoring of the WebGPU host: the point is for the two to be able to
disagree. Everything is row-major numpy with explicit loops where the kernel
loops, so it is slow and only useful on small scenes.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

RAD_TO_DEG = np.float32(57.2957795131)
F32 = np.float32


def quat_to_rotmat(quats: np.ndarray) -> np.ndarray:
    """(N, 4) wxyz -> (N, 3, 3), normalising as gsplat's quat_to_rotmat does."""
    q = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=-2,
    ).astype(F32)


def quat_scale_to_covar(quats: np.ndarray, scales: np.ndarray) -> np.ndarray:
    rot = quat_to_rotmat(quats)
    m = rot * scales[:, None, :]  # R @ diag(scale)
    return (m @ np.swapaxes(m, -1, -2)).astype(F32)


def spherical_jacobian(p: np.ndarray) -> np.ndarray:
    """(N, 3) sensor-frame points -> (N, 3, 3) d(azimuth_deg, elevation_deg, range)/dp."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    x2, y2, z2 = x * x, y * y, z * z
    r2 = x2 + y2 + z2
    rinv = 1.0 / np.sqrt(r2)
    sxy = np.sqrt(x2 + y2)
    r2sxy_inv = 1.0 / (r2 * sxy)
    zero = np.zeros_like(x)
    return np.stack(
        [
            np.stack([-y / (x2 + y2), x / (x2 + y2), zero], -1) * RAD_TO_DEG,
            np.stack([-x * z * r2sxy_inv, -y * z * r2sxy_inv, sxy / r2], -1) * RAD_TO_DEG,
            np.stack([x * rinv, y * rinv, z * rinv], -1),
        ],
        axis=-2,
    ).astype(F32)


def project(
    means, quats, scales, opacities, velocities, viewmats,
    lin_vel, ang_vel, rs_time, valid_mask,
    min_azimuth, max_azimuth, min_elevation, max_elevation,
    eps2d, near_plane, far_plane, radius_clip, calc_compensations,
) -> Dict[str, np.ndarray]:
    """fully_fused_lidar_projection_fwd_kernel, per (camera, Gaussian)."""
    n_cam = viewmats.shape[0]
    n = means.shape[0]
    out = {
        "means2d": np.zeros((n_cam, n, 2), F32),
        "radii": np.zeros((n_cam, n, 2), F32),
        "depths": np.zeros((n_cam, n), F32),
        "conics": np.zeros((n_cam, n, 3), F32),
        "opacities": np.zeros((n_cam, n), F32),
        "pix_vels": np.zeros((n_cam, n, 3), F32),
        "depth_compensations": np.zeros((n_cam, n, 2), F32),
    }
    covar_w = quat_scale_to_covar(quats, scales)

    for c in range(n_cam):
        rot = viewmats[c, :3, :3]
        trans = viewmats[c, :3, 3]
        mean_c = means @ rot.T + trans
        dist = np.linalg.norm(mean_c, axis=-1)
        covar_c = rot @ covar_w @ rot.T

        jac = spherical_jacobian(mean_c)
        sph = jac @ covar_c @ np.swapaxes(jac, -1, -2)
        cov2d = sph[:, :2, :2].copy()

        det_orig = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
        cov2d[:, 0, 0] += eps2d
        cov2d[:, 1, 1] += eps2d
        det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            compensation = np.sqrt(np.maximum(0.0, det_orig / det))

        azim = np.arctan2(mean_c[:, 1], mean_c[:, 0]) * RAD_TO_DEG
        elev = np.arcsin(np.clip(mean_c[:, 2] / dist, -1.0, 1.0)) * RAD_TO_DEG

        # depth_compensation_from_cov3d on the spherical covariance
        a1 = sph[:, 0, 0] + eps2d
        a2 = sph[:, 0, 1]
        b1 = sph[:, 1, 0]
        b2 = sph[:, 1, 1] + eps2d
        c1 = sph[:, 2, 0]
        c2 = sph[:, 2, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_d = 1.0 / (a1 * b2 - a2 * b1)
        dcomp = np.stack([(b1 * c2 - b2 * c1) * inv_d, (a2 * c1 - a1 * c2) * inv_d], -1)

        ext_az = 3.0 * np.sqrt(np.maximum(0.0, cov2d[:, 0, 0]))
        ext_el = 3.0 * np.sqrt(np.maximum(0.0, cov2d[:, 1, 1]))

        pix_vel = np.zeros((n, 3), F32)
        if rs_time[c] > 0:
            vel_c = np.zeros((n, 3), F32) if velocities is None else velocities @ rot.T
            total = lin_vel[c] + np.cross(np.broadcast_to(ang_vel[c], (n, 3)), mean_c) - vel_c
            pix_vel = -np.einsum("nij,nj->ni", jac, total).astype(F32)
            ext_az = ext_az + np.abs(pix_vel[:, 0]) * 0.5 * rs_time[c]
            ext_el = ext_el + np.abs(pix_vel[:, 1]) * 0.5 * rs_time[c]

        keep = (dist >= near_plane) & (dist <= far_plane) & (det > 0)
        keep &= ~((ext_az <= radius_clip) & (ext_el <= radius_clip))
        keep &= ~(
            (elev + ext_el <= min_elevation)
            | (elev - ext_el >= max_elevation)
            | (azim + ext_az <= min_azimuth)
            | (azim - ext_az >= max_azimuth)
        )
        if valid_mask is not None:
            keep &= valid_mask.astype(bool)

        inv_det = 1.0 / det
        conics = np.stack(
            [cov2d[:, 1, 1] * inv_det, -cov2d[:, 0, 1] * inv_det, cov2d[:, 0, 0] * inv_det], -1
        )
        opac = opacities * compensation if calc_compensations else opacities

        idx = np.nonzero(keep)[0]
        out["means2d"][c, idx] = np.stack([azim, elev], -1)[idx]
        out["radii"][c, idx] = np.stack([ext_az, ext_el], -1)[idx]
        out["depths"][c, idx] = dist[idx]
        out["conics"][c, idx] = conics[idx]
        out["opacities"][c, idx] = opac[idx]
        out["pix_vels"][c, idx] = pix_vel[idx]
        out["depth_compensations"][c, idx] = dcomp[idx]
    return out


def _row_span(
    proj, c, g, row, exact_rows, row_elevations, elev_boundaries,
    min_azimuth, n_tiles_azim, tile_azim_res,
):
    azim = proj["means2d"][c, g, 0]
    elev = proj["means2d"][c, g, 1]
    half = proj["radii"][c, g, 0]
    centre = azim
    cx, cy, cz = proj["conics"][c, g]
    if exact_rows and cx > 0:
        smax = math.log(max(float(proj["opacities"][c, g]), 1e-6) * 255.0)
        if row_elevations is not None:
            dy = row_elevations[row] - elev
        else:
            dy = min(max(elev, elev_boundaries[row]), elev_boundaries[row + 1]) - elev
        qa = 0.5 * cx
        qb = cy * dy
        qc = 0.5 * cz * dy * dy
        disc = qb * qb - 4.0 * qa * (qc - smax)
        if disc < 0:
            return None
        sq = math.sqrt(disc)
        inv2a = 0.5 / qa
        half = sq * inv2a
        centre = azim - qb * inv2a

    azim_max = n_tiles_azim * tile_azim_res
    a_lo = centre - half - min_azimuth
    a_hi = centre + half - min_azimuth
    if a_lo >= 0:
        t_lo = a_lo / tile_azim_res
    else:
        t_lo = (math.fmod(a_lo + 360.0, 360.0) - azim_max) / tile_azim_res
    if a_hi <= 360.0:
        t_hi = a_hi / tile_azim_res
    else:
        t_hi = n_tiles_azim + math.fmod(a_hi + 360.0, 360.0) / tile_azim_res
    lo, hi = int(math.floor(t_lo)), int(math.ceil(t_hi))
    return (lo, hi) if hi > lo else None


def _elev_range(proj, c, g, elev_boundaries, n_tiles_elev):
    elev = proj["means2d"][c, g, 1]
    ext = proj["radii"][c, g, 1]
    low, high = elev - ext, elev + ext
    i = 0
    while i <= n_tiles_elev and elev_boundaries[i] < low:
        i += 1
    lo = max(i - 1, 0)
    while i <= n_tiles_elev and elev_boundaries[i] < high:
        i += 1
    return lo, min(i, n_tiles_elev)


def isect(
    proj, elev_boundaries, row_elevations, min_azimuth,
    n_tiles_azim, n_tiles_elev, tile_azim_res, exact_rows=True,
):
    """isect_lidar_tiles + the sort: returns (tiles_per_gauss, sorted flatten ids, keys)."""
    n_cam, n = proj["depths"].shape
    counts = np.zeros((n_cam, n), np.int64)
    keys, flat = [], []
    for c in range(n_cam):
        for g in range(n):
            if proj["radii"][c, g, 0] <= 0:
                continue
            lo_row, hi_row = _elev_range(proj, c, g, elev_boundaries, n_tiles_elev)
            depth_bits = int(np.float32(proj["depths"][c, g]).view(np.uint32))
            for row in range(lo_row, hi_row):
                span = _row_span(
                    proj, c, g, row, exact_rows, row_elevations, elev_boundaries,
                    min_azimuth, n_tiles_azim, tile_azim_res,
                )
                if span is None:
                    continue
                lo, hi = span
                counts[c, g] += hi - lo
                for j in range(lo, hi):
                    wrapped = (j + n_tiles_azim) % n_tiles_azim
                    tile = row * n_tiles_azim + wrapped
                    keys.append((c * n_tiles_azim * n_tiles_elev + tile, depth_bits))
                    flat.append(c * n + g)
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    return (
        counts,
        np.array([flat[i] for i in order], np.int64),
        np.array([keys[i] for i in order], np.int64).reshape(-1, 2),
    )


def rasterize(
    proj, features, raster_pts, flatten_ids, keys,
    n_tiles_azim, n_tiles_elev, tile_width, tile_height,
    static_render, use_depth_comp, compute_alpha_sum, alpha_sum_threshold,
    tile_col_offset=0,
):
    """rasterize_to_points_fwd_kernel, one Python loop per pixel."""
    n_cam, height, width, _ = raster_pts.shape
    n = proj["depths"].shape[1]
    n_feat = features.shape[-1]
    n_tiles = n_tiles_azim * n_tiles_elev

    colors = np.zeros((n_cam, height, width, n_feat + 1), F32)
    alphas = np.zeros((n_cam, height, width, 1), F32)
    alpha_sums = np.zeros((n_cam, height, width, 1), F32)
    median_depths = np.zeros((n_cam, height, width, 1), F32)
    fr_depth = np.zeros((n_cam, height, width, 1), F32)
    median_ids = np.full((n_cam, height, width), -1, np.int32)

    # Tile list boundaries from the sorted keys.
    starts = np.zeros(n_cam * n_tiles + 1, np.int64)
    linear = keys[:, 0] if len(keys) else np.zeros(0, np.int64)
    for t in range(n_cam * n_tiles):
        starts[t] = int(np.searchsorted(linear, t, side="left"))
    starts[-1] = len(linear)

    means2d = proj["means2d"]
    conics = proj["conics"]
    opac_all = proj["opacities"]
    pix_vels = proj["pix_vels"]
    dcomp = proj["depth_compensations"]
    depths = proj["depths"]

    for c in range(n_cam):
        for row in range(height):
            for col in range(width):
                px, py, pz, roll = raster_pts[c, row, col]
                if pz <= 0:
                    continue
                tile = (row // tile_height) * n_tiles_azim + col // tile_width + tile_col_offset
                lo = starts[c * n_tiles + tile]
                hi = starts[c * n_tiles + tile + 1]

                T = np.float32(1.0)
                acc = np.zeros(n_feat + 1, F32)
                fr_num = fr_den = np.float32(0.0)
                asum = np.float32(0.0)
                for k in range(lo, hi):
                    gid = int(flatten_ids[k])
                    ci, gi = gid // n, gid % n
                    mx, my = means2d[ci, gi]
                    cx, cy, cz = conics[ci, gi]
                    opac = opac_all[ci, gi]
                    depth_extra = np.float32(0.0)
                    if static_render:
                        dx = _angle_diff(mx, px)
                        dy = my - py
                    else:
                        vx, vy, vz = pix_vels[ci, gi]
                        dx = _angle_diff(mx + roll * vx, px)
                        dy = (my + roll * vy) - py
                        depth_extra = vz * roll
                        if use_depth_comp:
                            depth_extra = depth_extra + dcomp[ci, gi, 0] * dx + dcomp[ci, gi, 1] * dy
                    sigma = 0.5 * (cx * dx * dx + cz * dy * dy) + cy * dx * dy
                    if sigma < 0:
                        continue
                    alpha = min(np.float32(0.999), np.float32(opac * math.exp(-sigma)))
                    if alpha < 1.0 / 255.0:
                        continue
                    next_T = np.float32(T * (1.0 - alpha))
                    if next_T <= 1e-4:
                        break
                    vis = np.float32(alpha * T)
                    acc[:n_feat] += features[ci, gi] * vis
                    rng = np.float32(depths[ci, gi] + depth_extra)
                    acc[n_feat] += rng * vis
                    if T > 0.5:
                        fr_num = fr_num + rng * vis
                        fr_den = fr_den + vis
                        if next_T <= 0.5:
                            median_depths[c, row, col, 0] = rng
                            median_ids[c, row, col] = gid
                    if compute_alpha_sum and depths[ci, gi] < (pz - alpha_sum_threshold):
                        asum = asum + alpha
                    T = next_T

                colors[c, row, col] = acc
                alphas[c, row, col, 0] = 1.0 - T
                alpha_sums[c, row, col, 0] = asum
                fr_depth[c, row, col, 0] = fr_num / fr_den if fr_den > 1e-6 else 0.0
    return dict(
        colors=colors, alphas=alphas, alpha_sums=alpha_sums,
        median_depths=median_depths, fr_depth=fr_depth, median_ids=median_ids,
    )


def _angle_diff(a: float, b: float) -> float:
    diff = a - b
    if diff > 180.0:
        diff -= 360.0
    if diff < -180.0:
        diff += 360.0
    return diff


def render(
    means, quats, scales, opacities, lidar_features, velocities, viewmats, raster_pts,
    tile_elevation_boundaries, linear_velocity=None, angular_velocity=None,
    rolling_shutter_time=None, min_azimuth=-180.0, max_azimuth=180.0,
    min_elevation=-80.0, max_elevation=80.0, n_elevation_channels=32,
    azimuth_resolution=0.1, tile_width=32, tile_height=8, near_plane=0.01,
    far_plane=1e10, radius_clip=0.0, eps2d=0.017, compute_alpha_sum_until_points=True,
    compute_alpha_sum_until_points_threshold=0.2, row_elevations=None,
    tile_col_offset=0, valid_mask=None, rasterize_mode="classic",
    use_depth_compensation=True,
):
    """End-to-end reference matching webgpu.lidar_rasterization's outputs."""
    n_cam = viewmats.shape[0]
    lin = np.zeros((n_cam, 3), F32) if linear_velocity is None else linear_velocity
    ang = np.zeros((n_cam, 3), F32) if angular_velocity is None else angular_velocity
    rst = np.zeros((n_cam,), F32) if rolling_shutter_time is None else rolling_shutter_time
    static_render = (
        velocities is None and linear_velocity is None
        and angular_velocity is None and not use_depth_compensation
    )

    proj = project(
        means, quats, scales, opacities, velocities, viewmats, lin, ang, rst, valid_mask,
        min_azimuth, max_azimuth, min_elevation, max_elevation,
        eps2d, near_plane, far_plane, radius_clip, rasterize_mode == "antialiased",
    )
    n_elev_tiles = math.ceil(n_elevation_channels / tile_height)
    tile_azim_res = azimuth_resolution * tile_width
    n_azim_tiles = math.ceil((max_azimuth - min_azimuth) / tile_azim_res)
    counts, flat, keys = isect(
        proj, tile_elevation_boundaries, row_elevations, min_azimuth,
        n_azim_tiles, n_elev_tiles, tile_azim_res,
    )
    out = rasterize(
        proj, lidar_features, raster_pts, flat, keys,
        n_azim_tiles, n_elev_tiles, tile_width, tile_height,
        static_render, use_depth_compensation, compute_alpha_sum_until_points,
        compute_alpha_sum_until_points_threshold, tile_col_offset,
    )
    out["proj"] = proj
    out["tiles_per_gauss"] = counts
    out["flatten_ids"] = flat
    return out


def sh_residual(degree: int, dirs: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """The l >= 1 part of sh_coeffs_to_color_fast (cuda/csrc/sh.cu), scalar.

    ``coeffs`` is [N, (degree+1)^2 - 1]: band (l, m) at index l*l + m - 1, with
    no DC term, which is what the ray-drop path carries per Gaussian.
    """
    n = dirs.shape[0]
    if degree < 1:
        return np.zeros(n, F32)
    inorm = 1.0 / np.linalg.norm(dirs, axis=-1)
    x, y, z = (dirs[:, 0] * inorm, dirs[:, 1] * inorm, dirs[:, 2] * inorm)
    c = coeffs
    result = 0.48860251190292 * (-y * c[:, 0] + z * c[:, 1] - x * c[:, 2])
    if degree >= 2:
        z2 = z * z
        f_tmp0b = -1.092548430592079 * z
        fc1 = x * x - y * y
        fs1 = 2.0 * x * y
        p6 = 0.9461746957575601 * z2 - 0.3153915652525201
        p7 = f_tmp0b * x
        p5 = f_tmp0b * y
        p8 = 0.5462742152960395 * fc1
        p4 = 0.5462742152960395 * fs1
        result = result + (
            p4 * c[:, 3] + p5 * c[:, 4] + p6 * c[:, 5] + p7 * c[:, 6] + p8 * c[:, 7]
        )
        if degree >= 3:
            f_tmp0c = -2.285228997322329 * z2 + 0.4570457994644658
            f_tmp1b = 1.445305721320277 * z
            fc2 = x * fc1 - y * fs1
            fs2 = x * fs1 + y * fc1
            p12 = z * (1.865881662950577 * z2 - 1.119528997770346)
            p13 = f_tmp0c * x
            p11 = f_tmp0c * y
            p14 = f_tmp1b * fc1
            p10 = f_tmp1b * fs1
            p15 = -0.5900435899266435 * fc2
            p9 = -0.5900435899266435 * fs2
            result = result + (
                p9 * c[:, 8] + p10 * c[:, 9] + p11 * c[:, 10] + p12 * c[:, 11]
                + p13 * c[:, 12] + p14 * c[:, 13] + p15 * c[:, 14]
            )
            if degree >= 4:
                f_tmp0d = z * (-4.683325804901025 * z2 + 2.007139630671868)
                f_tmp1c = 3.31161143515146 * z2 - 0.47308734787878
                f_tmp2b = -1.770130769779931 * z
                fc3 = x * fc2 - y * fs2
                fs3 = x * fs2 + y * fc2
                p20 = 1.984313483298443 * z * p12 - 1.006230589874905 * p6
                p21 = f_tmp0d * x
                p19 = f_tmp0d * y
                p22 = f_tmp1c * fc1
                p18 = f_tmp1c * fs1
                p23 = f_tmp2b * fc2
                p17 = f_tmp2b * fs2
                p24 = 0.6258357354491763 * fc3
                p16 = 0.6258357354491763 * fs3
                result = result + (
                    p16 * c[:, 15] + p17 * c[:, 16] + p18 * c[:, 17] + p19 * c[:, 18]
                    + p20 * c[:, 19] + p21 * c[:, 20] + p22 * c[:, 21] + p23 * c[:, 22]
                    + p24 * c[:, 23]
                )
    return result.astype(F32)
