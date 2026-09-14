//#include "common.wgsl"
//
// Port of rasterize_to_points_fwd_kernel (cuda/csrc/rasterization.cu).
//
// One workgroup per tile, one invocation per pixel, with the whole workgroup
// cooperatively staging STAGE_SIZE Gaussians of the tile's depth-sorted list
// into workgroup memory per round and then blending them front to back.
//
// Specialised by the host on COLOR_DIM / tile shape / STATIC / DEPTH_COMP, the
// same axes the CUDA kernel templates on. The ROW_TILE refactor is not ported:
// it only reassociates the sigma arithmetic for one-beam rows, and leaving it
// out keeps every tile shape on one code path.

@group(0) @binding(1) var<storage, read> proj: array<f32>;
@group(0) @binding(2) var<storage, read> features: array<f32>;
@group(0) @binding(3) var<storage, read> raster_pts: array<f32>;
@group(0) @binding(4) var<storage, read> tile_offsets: array<i32>;
@group(0) @binding(5) var<storage, read> isects: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read_write> out_colors: array<f32>;
@group(0) @binding(7) var<storage, read_write> out_aux: array<f32>;

const COLOR_DIM: u32 = ${COLOR_DIM}u;   // N_FEATURES + 1; the depth channel is last
const N_FEATURES: u32 = ${N_FEATURES}u;
const BLOCK_SIZE: u32 = ${BLOCK_SIZE}u;
const BATCH_MULT: u32 = ${BATCH_MULT}u;
const STAGE_SIZE: u32 = BLOCK_SIZE * BATCH_MULT;
const VEC_PER: u32 = ${VEC_PER}u;

// (mean.x, mean.y, opacity, conic.x) | (conic.y, conic.z, id, sigma_max or
// pix_vel.z) | (pix_vel.x, pix_vel.y, depth_comp.x, depth_comp.y)
var<workgroup> stage: array<vec4<f32>, ${STAGE_VEC4}>;
var<workgroup> done_count: atomic<u32>;
var<workgroup> done_flag: u32;

@compute @workgroup_size(${TILE_W}, ${TILE_H})
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) lindex: u32,
) {
    let camera_id = wid.z;
    // The tile grid always spans the full azimuth ring; a sector image draws
    // its own columns but looks them up at tile_col_offset (splatsim sector
    // rendering).
    let tile_id = wid.y * P.n_tiles_azim + wid.x + P.tile_col_offset;
    let row = wid.y * ${TILE_H}u + lid.y;
    let col = wid.x * ${TILE_W}u + lid.x;

    let n_tiles = P.n_tiles_azim * P.n_tiles_elev;
    let camera_offset = camera_id * P.image_height * P.image_width;
    let pix_id = row * P.image_width + col;
    let in_image = row < P.image_height && col < P.image_width;

    var px = 0.0;       // azimuth of this beam, degrees
    var py = 0.0;       // elevation, degrees
    var pz = 0.0;       // measured range, metres
    var roll_time = 0.0;
    var inside = in_image;
    if (in_image) {
        let rb = (camera_offset + pix_id) * 4u;
        px = raster_pts[rb + 0u];
        py = raster_pts[rb + 1u];
        pz = raster_pts[rb + 2u];
        roll_time = raster_pts[rb + 3u];
        if (pz <= 0.0) {
            inside = false;
        }
    }
    // Threads with no ray still stage Gaussians for the rest of the workgroup.
    var done = !inside;

    let base = camera_id * n_tiles + tile_id;
    let range_start = u32(tile_offsets[base]);
    let range_end = u32(tile_offsets[base + 1u]);
    var num_batches = 0u;
    if (range_end > range_start) {
        num_batches = (range_end - range_start + STAGE_SIZE - 1u) / STAGE_SIZE;
    }

    var T = 1.0;
    var cur_idx = 0u;
    var pix_out: array<f32, ${COLOR_DIM}>;
    for (var k = 0u; k < COLOR_DIM; k = k + 1u) {
        pix_out[k] = 0.0;
    }
    var fr_num = 0.0; // soft first return: sum of vis * range over the T > 0.5 prefix
    var fr_den = 0.0;
    var median_depth = 0.0;
    var median_id = -1;
    var alpha_sum = 0.0;
    let want_alpha_sum = has_flag(FLAG_COMPUTE_ALPHA_SUM);

    for (var b = 0u; b < num_batches; b = b + 1u) {
        // __syncthreads_count(done) >= block_size: leave once no thread in the
        // workgroup has anything left to blend. Counted through an atomic and
        // republished as a uniform value so the barriers below stay in uniform
        // control flow.
        if (lindex == 0u) {
            atomicStore(&done_count, 0u);
        }
        workgroupBarrier();
        if (done) {
            atomicAdd(&done_count, 1u);
        }
        workgroupBarrier();
        if (lindex == 0u) {
            done_flag = atomicLoad(&done_count);
        }
        let all_done = workgroupUniformLoad(&done_flag);
        if (all_done >= BLOCK_SIZE) {
            break;
        }

        let batch_start = range_start + STAGE_SIZE * b;
        for (var q = 0u; q < BATCH_MULT; q = q + 1u) {
            let slot = q * BLOCK_SIZE + lindex;
            let isect = batch_start + slot;
            if (isect >= range_end) {
                break;
            }
            let g = isects[isect].z;
            let po = g * PROJ_STRIDE;
            let mean_x = proj[po + PROJ_MEAN2D + 0u];
            let mean_y = proj[po + PROJ_MEAN2D + 1u];
            let opac = proj[po + PROJ_OPAC];
            let cx = proj[po + PROJ_CONIC + 0u];
            let cy = proj[po + PROJ_CONIC + 1u];
            let cz = proj[po + PROJ_CONIC + 2u];
            let sv = slot * VEC_PER;
            stage[sv + 0u] = vec4<f32>(mean_x, mean_y, opac, cx);
//#if STATIC
            // A per-Gaussian sigma cutoff in place of the pix_vel slot: the
            // contribution test alpha = opac * exp(-sigma) >= 1/255 is
            // sigma <= ln(255 * opac), so most of the tile list can be rejected
            // without evaluating the exponential.
            stage[sv + 1u] = vec4<f32>(cy, cz, bitcast<f32>(g), log(opac * 255.0));
//#else
            stage[sv + 1u] = vec4<f32>(cy, cz, bitcast<f32>(g), proj[po + PROJ_PIXVEL + 2u]);
            stage[sv + 2u] = vec4<f32>(
                proj[po + PROJ_PIXVEL + 0u], proj[po + PROJ_PIXVEL + 1u],
                proj[po + PROJ_DCOMP + 0u], proj[po + PROJ_DCOMP + 1u],
            );
//#endif
        }
        workgroupBarrier();

        let batch_size = min(STAGE_SIZE, range_end - batch_start);
        for (var t = 0u; t < batch_size && !done; t = t + 1u) {
            let sv = t * VEC_PER;
            let A = stage[sv + 0u];
            let B = stage[sv + 1u];
            let opac = A.z;
            let g = bitcast<u32>(B.z);
            let conic = vec3<f32>(A.w, B.x, B.y);

            // Azimuth wraps at +/-180 so it needs the wrapping difference;
            // elevation is bounded to [-85, 85] and never wraps.
            var delta: vec2<f32>;
            var depth_extra = 0.0;
//#if STATIC
            delta = vec2<f32>(angle_difference(A.x, px), A.y - py);
//#else
            let Cv = stage[sv + 2u];
            delta = vec2<f32>(
                angle_difference(A.x + roll_time * Cv.x, px),
                (A.y + roll_time * Cv.y) - py,
            );
            depth_extra = B.w * roll_time;
//#if DEPTH_COMP
            depth_extra = depth_extra + Cv.z * delta.x + Cv.w * delta.y;
//#endif
//#endif
            let sigma = 0.5 * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y)
                + conic.y * delta.x * delta.y;
//#if STATIC
            if (sigma < 0.0 || sigma > B.w) {
                continue;
            }
//#endif
            let alpha = min(0.999, opac * exp(-sigma));
            if (sigma < 0.0 || alpha < 1.0 / 255.0) {
                continue;
            }

            let next_T = T * (1.0 - alpha);
            if (next_T <= 1e-4) {
                // This pixel is saturated; the crossing Gaussian is excluded.
                done = true;
                break;
            }

            let vis = alpha * T;
            let fb = g * N_FEATURES;
            for (var k = 0u; k < N_FEATURES; k = k + 1u) {
                pix_out[k] = pix_out[k] + features[fb + k] * vis;
            }
            let g_depth = proj[g * PROJ_STRIDE + PROJ_DEPTH];
            let rng = g_depth + depth_extra;
            pix_out[N_FEATURES] = pix_out[N_FEATURES] + rng * vis;

            if (T > 0.5) {
                fr_num = fr_num + rng * vis;
                fr_den = fr_den + vis;
                if (next_T <= 0.5) {
                    median_depth = rng;
                    median_id = i32(g);
                }
            }
            if (want_alpha_sum && g_depth < (pz - P.alpha_sum_threshold)) {
                alpha_sum = alpha_sum + alpha;
            }
            cur_idx = batch_start + t;
            T = next_T;
        }
    }

    // Pixels whose ray was rejected (pz <= 0) fall through with the same values
    // the CUDA path leaves in its zero-initialised output tensors, so they are
    // written unconditionally for every pixel inside the image.
    if (in_image) {
        let cb = (camera_offset + pix_id) * COLOR_DIM;
        for (var k = 0u; k < COLOR_DIM; k = k + 1u) {
            out_colors[cb + k] = pix_out[k];
        }
        let ab = (camera_offset + pix_id) * AUX_STRIDE;
        out_aux[ab + AUX_ALPHA] = 1.0 - T;
        out_aux[ab + AUX_ALPHA_SUM] = alpha_sum;
        out_aux[ab + AUX_MEDIAN_DEPTH] = median_depth;
        out_aux[ab + AUX_FR_DEPTH] = select(0.0, fr_num / max(fr_den, 1e-30), fr_den > 1e-6);
        out_aux[ab + AUX_FR_WEIGHT] = fr_den;
        out_aux[ab + AUX_MEDIAN_ID] = bitcast<f32>(median_id);
        out_aux[ab + AUX_LAST_ID] = bitcast<f32>(i32(cur_idx));
    }
}
