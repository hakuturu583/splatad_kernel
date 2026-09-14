// Shared declarations for the WebGPU port of SplatAD's spherical LiDAR
// rasterizer. Textually prepended to every other shader by _shaders.py.
//
// The device math here is transcribed from the CUDA kernel it has to match:
// cuda/csrc/utils.cuh (lidar_proj, depth_compensation_from_cov3d,
// angle_difference), cuda/csrc/helpers.cuh (compute_lidar_velocity) and
// gsplat 1.5.3's Utils.cuh (quat_to_rotmat, quat_scale_to_covar_preci, posW2C,
// covarW2C, add_blur). glm and WGSL matrices are both column-major with the
// same multiplication order, so the transcription is line for line.

const RAD_TO_DEG: f32 = 57.2957795131;

// Per-(camera, Gaussian) projection record. One 64-byte struct instead of the
// CUDA path's seven separate [C, N, ...] tensors: the rasterizer reads almost
// all of these for the same Gaussian, and WebGPU's default limit of 8 storage
// buffers per shader stage does not stretch to one binding each.
const PROJ_STRIDE: u32 = 16u;
const PROJ_MEAN2D: u32 = 0u;  // azimuth, elevation (degrees)
const PROJ_OPAC: u32 = 2u;    // opacity, already multiplied by the antialias compensation
const PROJ_CONIC: u32 = 3u;   // 3 floats: inverse of the blurred 2D covariance
const PROJ_DEPTH: u32 = 6u;   // range to the Gaussian centre (metres)
const PROJ_RADII: u32 = 7u;   // azimuth extent, elevation extent (degrees); x <= 0 means culled
const PROJ_PIXVEL: u32 = 9u;  // 3 floats: d(azimuth, elevation, range)/dt
const PROJ_DCOMP: u32 = 12u;  // 2 floats: depth tilt across the Gaussian

// Per-Gaussian input record, 64 bytes.
const GAUSS_STRIDE: u32 = 16u;
const GAUSS_MEAN: u32 = 0u;
const GAUSS_OPAC: u32 = 3u;
const GAUSS_QUAT: u32 = 4u;   // w, x, y, z
const GAUSS_SCALE: u32 = 8u;
const GAUSS_VALID: u32 = 11u; // 0.0 = masked out (sector streaming), else kept
const GAUSS_VEL: u32 = 12u;

// Per-sensor record, 128 bytes, followed by the elevation tables (see
// Params.elev_base / Params.row_elev_base).
const SENSOR_STRIDE: u32 = 32u;
const SENSOR_VIEWMAT: u32 = 0u; // 4x4, row-major
const SENSOR_LINVEL: u32 = 16u;
const SENSOR_RSTIME: u32 = 19u;
const SENSOR_ANGVEL: u32 = 20u;

// Per-pixel auxiliary outputs, kept interleaved for the same binding-budget
// reason as PROJ_*.
const AUX_STRIDE: u32 = 8u;
const AUX_ALPHA: u32 = 0u;
const AUX_ALPHA_SUM: u32 = 1u;
const AUX_MEDIAN_DEPTH: u32 = 2u;
const AUX_FR_DEPTH: u32 = 3u;
const AUX_FR_WEIGHT: u32 = 4u;
const AUX_MEDIAN_ID: u32 = 5u; // i32, bitcast
const AUX_LAST_ID: u32 = 6u;   // i32, bitcast

const FLAG_HAS_VELOCITIES: u32 = 1u;
const FLAG_HAS_VALID_MASK: u32 = 2u;
const FLAG_CALC_COMPENSATIONS: u32 = 4u;
const FLAG_EXACT_ROW_SPANS: u32 = 8u;
const FLAG_HAS_ROW_ELEVATIONS: u32 = 16u;
const FLAG_COMPUTE_ALPHA_SUM: u32 = 32u;

struct Params {
    n_gaussians: u32,
    n_cameras: u32,
    n_features: u32, // D, the feature channels; the depth channel is appended as channel D
    flags: u32,

    min_azimuth: f32,
    max_azimuth: f32,
    min_elevation: f32,
    max_elevation: f32,

    eps2d: f32,
    near_plane: f32,
    far_plane: f32,
    radius_clip: f32,

    n_tiles_azim: u32,
    n_tiles_elev: u32,
    tile_azim_resolution: f32,
    tile_n_bits: u32,

    image_width: u32,
    image_height: u32,
    tile_width: u32,
    tile_height: u32,

    tile_col_offset: u32,
    alpha_sum_threshold: f32,
    n_isects: u32,
    sh_degree: u32,

    raydrop_feature_index: u32,
    elev_base: u32,     // index into `sensors` of elev_boundaries[n_tiles_elev + 1]
    row_elev_base: u32, // index into `sensors` of row_elevations[n_tiles_elev]
    n_isects_padded: u32,
}

// Bound by every shader that includes this file.
@group(0) @binding(0) var<uniform> P: Params;

fn has_flag(f: u32) -> bool {
    return (P.flags & f) != 0u;
}

// gsplat Utils.cuh quat_to_rotmat. quat is (w, x, y, z) and need not be normalized.
fn quat_to_rotmat(quat: vec4<f32>) -> mat3x3<f32> {
    var w = quat.x;
    var x = quat.y;
    var y = quat.z;
    var z = quat.w;
    let inv_norm = inverseSqrt(x * x + y * y + z * z + w * w);
    x = x * inv_norm;
    y = y * inv_norm;
    z = z * inv_norm;
    w = w * inv_norm;
    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;
    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (y2 + z2), 2.0 * (xy + wz), 2.0 * (xz - wy)),
        vec3<f32>(2.0 * (xy - wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz + wx)),
        vec3<f32>(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (x2 + y2)),
    );
}

// gsplat quat_scale_to_covar_preci, covariance branch only: C = (R S)(R S)^T.
fn quat_scale_to_covar(quat: vec4<f32>, scale: vec3<f32>) -> mat3x3<f32> {
    let R = quat_to_rotmat(quat);
    let M = mat3x3<f32>(R[0] * scale.x, R[1] * scale.y, R[2] * scale.z);
    return M * transpose(M);
}

// utils.cuh depth_compensation_from_cov3d. cov3d is the spherical-space
// covariance; the result is the (d range / d azimuth, d range / d elevation)
// tilt of the Gaussian, i.e. rows 2,0 and 2,1 of its inverse over element 2,2.
fn depth_compensation_from_cov3d(cov3d: mat3x3<f32>, eps2d: f32) -> vec2<f32> {
    let a1 = cov3d[0][0] + eps2d;
    let a2 = cov3d[1][0];
    let b1 = cov3d[0][1];
    let b2 = cov3d[1][1] + eps2d;
    let c1 = cov3d[0][2];
    let c2 = cov3d[1][2];
    let inv_d = 1.0 / (a1 * b2 - a2 * b1);
    return vec2<f32>((b1 * c2 - b2 * c1) * inv_d, (a2 * c1 - a1 * c2) * inv_d);
}

struct LidarProj {
    mean2d: vec2<f32>,
    cov2d: mat2x2<f32>,
    depth_comp: vec2<f32>,
    jacobian: mat3x3<f32>,
}

// utils.cuh lidar_proj: the spherical projection's Jacobian, the covariance
// pushed through it, and the (azimuth, elevation) of the centre in degrees.
fn lidar_proj(mean3d: vec3<f32>, cov3d: mat3x3<f32>, eps2d: f32) -> LidarProj {
    let x2 = mean3d.x * mean3d.x;
    let y2 = mean3d.y * mean3d.y;
    let z2 = mean3d.z * mean3d.z;
    let xz = mean3d.x * mean3d.z;
    let yz = mean3d.y * mean3d.z;
    let r2 = x2 + y2 + z2;
    // 1/sqrt(x) rather than inverseSqrt(): WGSL leaves inverseSqrt's accuracy to
    // the implementation, and a hardware rsqrt approximation costs ~12 bits --
    // which lands as a hundredth of a degree of elevation error, far above what
    // the rest of the projection contributes.
    let rinv = 1.0 / sqrt(r2);
    let sqrtx2y2 = sqrt(x2 + y2);
    let sqrtx2y2_inv = 1.0 / sqrtx2y2;
    let r2sqrtx2y2_inv = (1.0 / r2) * sqrtx2y2_inv;

    let J = mat3x3<f32>(
        vec3<f32>(-mean3d.y / (x2 + y2) * RAD_TO_DEG, -xz * r2sqrtx2y2_inv * RAD_TO_DEG, mean3d.x * rinv),
        vec3<f32>(mean3d.x / (x2 + y2) * RAD_TO_DEG, -yz * r2sqrtx2y2_inv * RAD_TO_DEG, mean3d.y * rinv),
        vec3<f32>(0.0, sqrtx2y2 / r2 * RAD_TO_DEG, mean3d.z * rinv),
    );

    let spherical = J * cov3d * transpose(J);

    var out: LidarProj;
    out.jacobian = J;
    out.cov2d = mat2x2<f32>(
        vec2<f32>(spherical[0][0], spherical[0][1]),
        vec2<f32>(spherical[1][0], spherical[1][1]),
    );
    // The CUDA kernel writes elevation as asin(z / r). atan2(z, hypot(x, y)) is
    // the same angle, and WGSL only pins down the accuracy of its transcendental
    // functions loosely: measured against a float64 reference, asin here drifts
    // by ~0.02 degrees while atan2 is exact to the last f32 bit.
    out.mean2d = vec2<f32>(
        atan2(mean3d.y, mean3d.x) * RAD_TO_DEG,
        atan2(mean3d.z, sqrtx2y2) * RAD_TO_DEG,
    );
    out.depth_comp = depth_compensation_from_cov3d(spherical, eps2d);
    return out;
}

struct Blur {
    cov2d: mat2x2<f32>,
    compensation: f32,
    det: f32,
}

// gsplat add_blur: dilate the projected covariance by eps2d and report the
// Mip-Splatting opacity compensation sqrt(det(orig) / det(blurred)).
fn add_blur(eps2d: f32, covar: mat2x2<f32>) -> Blur {
    let det_orig = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    var c = covar;
    c[0][0] = c[0][0] + eps2d;
    c[1][1] = c[1][1] + eps2d;
    let det_blur = c[0][0] * c[1][1] - c[0][1] * c[1][0];
    var out: Blur;
    out.cov2d = c;
    out.compensation = sqrt(max(0.0, det_orig / det_blur));
    out.det = det_blur;
    return out;
}

// helpers.cuh compute_lidar_velocity: the sensor-frame velocity of the
// Gaussian pushed through the spherical Jacobian, in deg/s and m/s. The
// negative sign moves points opposite to the sensor.
fn compute_lidar_velocity(
    p_view: vec3<f32>,
    lin_vel: vec3<f32>,
    ang_vel: vec3<f32>,
    vel_view: vec3<f32>,
    J: mat3x3<f32>,
) -> vec3<f32> {
    let total_vel = lin_vel + cross(ang_vel, p_view) - vel_view;
    return -(J * total_vel);
}

// utils.cuh angle_difference: both angles are panorama coordinates in
// [-180, 180], so two branchless offsets wrap the difference to (-180, 180].
fn angle_difference(angle1: f32, angle2: f32) -> f32 {
    var diff = angle1 - angle2;
    diff = diff - 360.0 * f32(diff > 180.0);
    diff = diff + 360.0 * f32(diff < -180.0);
    return diff;
}
