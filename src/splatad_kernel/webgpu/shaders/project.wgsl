//#include "common.wgsl"
//
// Port of fully_fused_lidar_projection_fwd_kernel (cuda/csrc/projection.cu).
// One invocation per (camera, Gaussian) pair, exactly as the CUDA grid.

@group(0) @binding(1) var<storage, read> gaussians: array<f32>;
@group(0) @binding(2) var<storage, read> sensors: array<f32>;
@group(0) @binding(3) var<storage, read_write> proj: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= P.n_cameras * P.n_gaussians) {
        return;
    }
    let cid = idx / P.n_gaussians;
    let g = idx % P.n_gaussians;
    let o = idx * PROJ_STRIDE;

    // The CUDA kernel signals "culled" by writing radii.x = 0 and returning
    // without touching the other outputs, so every early exit below goes
    // through this one store.
    proj[o + PROJ_RADII] = 0.0;

    let gb = g * GAUSS_STRIDE;
    if (has_flag(FLAG_HAS_VALID_MASK) && gaussians[gb + GAUSS_VALID] == 0.0) {
        return;
    }

    let sb = cid * SENSOR_STRIDE;
    // The viewmat arrives row-major; a column-major mat3x3 of its rotation is
    // built column by column, as the CUDA kernel does with glm.
    let R = mat3x3<f32>(
        vec3<f32>(sensors[sb + 0u], sensors[sb + 4u], sensors[sb + 8u]),
        vec3<f32>(sensors[sb + 1u], sensors[sb + 5u], sensors[sb + 9u]),
        vec3<f32>(sensors[sb + 2u], sensors[sb + 6u], sensors[sb + 10u]),
    );
    let t = vec3<f32>(sensors[sb + 3u], sensors[sb + 7u], sensors[sb + 11u]);

    let mean_w = vec3<f32>(gaussians[gb + 0u], gaussians[gb + 1u], gaussians[gb + 2u]);
    let mean_c = R * mean_w + t;
    let distance = length(mean_c);
    if (distance < P.near_plane || distance > P.far_plane) {
        return;
    }

    let quat = vec4<f32>(
        gaussians[gb + GAUSS_QUAT + 0u], gaussians[gb + GAUSS_QUAT + 1u],
        gaussians[gb + GAUSS_QUAT + 2u], gaussians[gb + GAUSS_QUAT + 3u],
    );
    let scale = vec3<f32>(
        gaussians[gb + GAUSS_SCALE + 0u], gaussians[gb + GAUSS_SCALE + 1u],
        gaussians[gb + GAUSS_SCALE + 2u],
    );
    let covar_w = quat_scale_to_covar(quat, scale);
    let covar_c = R * covar_w * transpose(R);

    let lp = lidar_proj(mean_c, covar_c, P.eps2d);
    let blur = add_blur(P.eps2d, lp.cov2d);
    if (blur.det <= 0.0) {
        return;
    }

    // 3 sigma of each marginal, in degrees. Not the ellipse's true bounding
    // box, but what the CUDA kernel uses, so the tile binning agrees.
    var extent_azimuth = 3.0 * sqrt(max(0.0, blur.cov2d[0][0]));
    var extent_elevation = 3.0 * sqrt(max(0.0, blur.cov2d[1][1]));
    if (extent_azimuth <= P.radius_clip && extent_elevation <= P.radius_clip) {
        return;
    }

    // Rolling shutter: grow the extents by half the sweep the Gaussian makes
    // over the shutter interval, so the tile binning still covers every column
    // the displaced Gaussian can reach.
    var pix_vel = vec3<f32>(0.0, 0.0, 0.0);
    let rs_time = sensors[sb + SENSOR_RSTIME];
    if (rs_time > 0.0) {
        var vel_c = vec3<f32>(0.0, 0.0, 0.0);
        if (has_flag(FLAG_HAS_VELOCITIES)) {
            let vel_w = vec3<f32>(
                gaussians[gb + GAUSS_VEL + 0u], gaussians[gb + GAUSS_VEL + 1u],
                gaussians[gb + GAUSS_VEL + 2u],
            );
            vel_c = R * vel_w;
        }
        let lin_vel = vec3<f32>(
            sensors[sb + SENSOR_LINVEL + 0u], sensors[sb + SENSOR_LINVEL + 1u],
            sensors[sb + SENSOR_LINVEL + 2u],
        );
        let ang_vel = vec3<f32>(
            sensors[sb + SENSOR_ANGVEL + 0u], sensors[sb + SENSOR_ANGVEL + 1u],
            sensors[sb + SENSOR_ANGVEL + 2u],
        );
        pix_vel = compute_lidar_velocity(mean_c, lin_vel, ang_vel, vel_c, lp.jacobian);
        extent_azimuth = extent_azimuth + abs(pix_vel.x) * 0.5 * rs_time;
        extent_elevation = extent_elevation + abs(pix_vel.y) * 0.5 * rs_time;
    }

    if (lp.mean2d.y + extent_elevation <= P.min_elevation
        || lp.mean2d.y - extent_elevation >= P.max_elevation
        || lp.mean2d.x + extent_azimuth <= P.min_azimuth
        || lp.mean2d.x - extent_azimuth >= P.max_azimuth) {
        return;
    }

    // det > 0 was established by add_blur, so the 2x2 inverse is well defined.
    let inv_det = 1.0 / blur.det;
    let conic = vec3<f32>(
        blur.cov2d[1][1] * inv_det,
        -blur.cov2d[0][1] * inv_det,
        blur.cov2d[0][0] * inv_det,
    );

    // rendering.py folds the antialias compensation into the opacity before
    // binning and rasterizing; doing it here keeps opacity a single per-(C, N)
    // value the later stages can read straight out of this record.
    var opacity = gaussians[gb + GAUSS_OPAC];
    if (has_flag(FLAG_CALC_COMPENSATIONS)) {
        opacity = opacity * blur.compensation;
    }

    proj[o + PROJ_MEAN2D + 0u] = lp.mean2d.x;
    proj[o + PROJ_MEAN2D + 1u] = lp.mean2d.y;
    proj[o + PROJ_OPAC] = opacity;
    proj[o + PROJ_CONIC + 0u] = conic.x;
    proj[o + PROJ_CONIC + 1u] = conic.y;
    proj[o + PROJ_CONIC + 2u] = conic.z;
    proj[o + PROJ_DEPTH] = distance;
    proj[o + PROJ_RADII + 0u] = extent_azimuth;
    proj[o + PROJ_RADII + 1u] = extent_elevation;
    proj[o + PROJ_PIXVEL + 0u] = pix_vel.x;
    proj[o + PROJ_PIXVEL + 1u] = pix_vel.y;
    proj[o + PROJ_PIXVEL + 2u] = pix_vel.z;
    proj[o + PROJ_DCOMP + 0u] = lp.depth_comp.x;
    proj[o + PROJ_DCOMP + 1u] = lp.depth_comp.y;
}
