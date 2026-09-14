//#include "common.wgsl"
//
// View-dependent ray-drop, matching the SH prepass rendering.py runs through
// gsplat's compute_sh before rasterizing (cuda/csrc/sh.cu
// sh_coeffs_to_color_fast, Sloan's JCGT 2013 recurrences).
//
// Only the l >= 1 bands are carried per Gaussian; the DC term stays in the
// feature channel, so what this adds is exactly the direction-dependent
// residual. Evaluated only for Gaussians this sensor can see (radii > 0), which
// is what the Python path does too.

@group(0) @binding(1) var<storage, read> proj: array<f32>;
@group(0) @binding(2) var<storage, read> gaussians: array<f32>;
@group(0) @binding(3) var<storage, read> sensors: array<f32>;
@group(0) @binding(4) var<storage, read> sh_coeffs: array<f32>; // [N, K - 1]
@group(0) @binding(5) var<storage, read_write> features: array<f32>; // [C, N, D]

const SENSOR_CAMPOS: u32 = 24u;

// coeffs are the bands l >= 1 only, so band l,m sits at index (l*l + m) - 1.
fn sh_residual(degree: u32, dir: vec3<f32>, base: u32) -> f32 {
    if (degree < 1u) {
        return 0.0;
    }
    let inorm = 1.0 / sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    let x = dir.x * inorm;
    let y = dir.y * inorm;
    let z = dir.z * inorm;

    var result = 0.48860251190292 * (
        -y * sh_coeffs[base + 0u] + z * sh_coeffs[base + 1u] - x * sh_coeffs[base + 2u]
    );
    if (degree < 2u) {
        return result;
    }

    let z2 = z * z;
    let f_tmp0b = -1.092548430592079 * z;
    let fc1 = x * x - y * y;
    let fs1 = 2.0 * x * y;
    let p_sh6 = 0.9461746957575601 * z2 - 0.3153915652525201;
    let p_sh7 = f_tmp0b * x;
    let p_sh5 = f_tmp0b * y;
    let p_sh8 = 0.5462742152960395 * fc1;
    let p_sh4 = 0.5462742152960395 * fs1;
    result = result + p_sh4 * sh_coeffs[base + 3u] + p_sh5 * sh_coeffs[base + 4u]
        + p_sh6 * sh_coeffs[base + 5u] + p_sh7 * sh_coeffs[base + 6u]
        + p_sh8 * sh_coeffs[base + 7u];
    if (degree < 3u) {
        return result;
    }

    let f_tmp0c = -2.285228997322329 * z2 + 0.4570457994644658;
    let f_tmp1b = 1.445305721320277 * z;
    let fc2 = x * fc1 - y * fs1;
    let fs2 = x * fs1 + y * fc1;
    let p_sh12 = z * (1.865881662950577 * z2 - 1.119528997770346);
    let p_sh13 = f_tmp0c * x;
    let p_sh11 = f_tmp0c * y;
    let p_sh14 = f_tmp1b * fc1;
    let p_sh10 = f_tmp1b * fs1;
    let p_sh15 = -0.5900435899266435 * fc2;
    let p_sh9 = -0.5900435899266435 * fs2;
    result = result + p_sh9 * sh_coeffs[base + 8u] + p_sh10 * sh_coeffs[base + 9u]
        + p_sh11 * sh_coeffs[base + 10u] + p_sh12 * sh_coeffs[base + 11u]
        + p_sh13 * sh_coeffs[base + 12u] + p_sh14 * sh_coeffs[base + 13u]
        + p_sh15 * sh_coeffs[base + 14u];
    if (degree < 4u) {
        return result;
    }

    let f_tmp0d = z * (-4.683325804901025 * z2 + 2.007139630671868);
    let f_tmp1c = 3.31161143515146 * z2 - 0.47308734787878;
    let f_tmp2b = -1.770130769779931 * z;
    let fc3 = x * fc2 - y * fs2;
    let fs3 = x * fs2 + y * fc2;
    let p_sh20 = 1.984313483298443 * z * p_sh12 - 1.006230589874905 * p_sh6;
    let p_sh21 = f_tmp0d * x;
    let p_sh19 = f_tmp0d * y;
    let p_sh22 = f_tmp1c * fc1;
    let p_sh18 = f_tmp1c * fs1;
    let p_sh23 = f_tmp2b * fc2;
    let p_sh17 = f_tmp2b * fs2;
    let p_sh24 = 0.6258357354491763 * fc3;
    let p_sh16 = 0.6258357354491763 * fs3;
    result = result + p_sh16 * sh_coeffs[base + 15u] + p_sh17 * sh_coeffs[base + 16u]
        + p_sh18 * sh_coeffs[base + 17u] + p_sh19 * sh_coeffs[base + 18u]
        + p_sh20 * sh_coeffs[base + 19u] + p_sh21 * sh_coeffs[base + 20u]
        + p_sh22 * sh_coeffs[base + 21u] + p_sh23 * sh_coeffs[base + 22u]
        + p_sh24 * sh_coeffs[base + 23u];
    return result;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= P.n_cameras * P.n_gaussians) {
        return;
    }
    if (proj[idx * PROJ_STRIDE + PROJ_RADII] <= 0.0) {
        return;
    }
    let cid = idx / P.n_gaussians;
    let g = idx % P.n_gaussians;

    let sb = cid * SENSOR_STRIDE + SENSOR_CAMPOS;
    let cam_pos = vec3<f32>(sensors[sb + 0u], sensors[sb + 1u], sensors[sb + 2u]);
    let gb = g * GAUSS_STRIDE;
    let dir = vec3<f32>(gaussians[gb + 0u], gaussians[gb + 1u], gaussians[gb + 2u]) - cam_pos;

    let n_bands = (P.sh_degree + 1u) * (P.sh_degree + 1u) - 1u;
    let resid = sh_residual(P.sh_degree, dir, g * n_bands);

    let fi = idx * P.n_features + P.raydrop_feature_index;
    features[fi] = features[fi] + resid;
}
