//#include "common.wgsl"
//
// Port of isect_lidar_tiles (cuda/csrc/rasterization.cu). Two entry points
// over the same bindings and the same span arithmetic: `count` fills
// tiles_per_gauss, and after the host has prefix-summed it, `encode` writes the
// (tile, depth) sort keys and the flatten ids.

@group(0) @binding(1) var<storage, read> proj: array<f32>;
@group(0) @binding(2) var<storage, read> sensors: array<f32>;
@group(0) @binding(3) var<storage, read_write> counts: array<u32>;
@group(0) @binding(4) var<storage, read> cum: array<u32>;
// One interleaved record per intersection: (camera | tile, depth bits,
// flatten id, unused). The sort moves whole records, so keeping the key words
// and the payload together halves the bindings and the memory traffic.
@group(0) @binding(5) var<storage, read_write> isects: array<vec4<u32>>;

struct Span {
    lo: i32,
    hi: i32,
    hit: bool,
}

// The azimuth tile interval a Gaussian can reach in one elevation row.
//
// The bounding box applies the Gaussian's widest azimuth extent to every row it
// spans, but the reachable interval narrows towards the poles of the ellipse.
// With the conic and the opacity available the interval is solved exactly from
// the same contribution test the rasterizer applies:
//   0.5*cx*dx^2 + cy*dy*dx + 0.5*cz*dy^2 <= ln(255*opacity)
// taking dy to the nearest elevation the row samples, which keeps the result a
// superset of what any pixel in the row can be hit by.
fn row_span(o: u32, row: i32, exact_rows: bool) -> Span {
    let g_azim = proj[o + PROJ_MEAN2D + 0u];
    let g_elev = proj[o + PROJ_MEAN2D + 1u];
    let cx = proj[o + PROJ_CONIC + 0u];

    var half = proj[o + PROJ_RADII + 0u];
    var centre = g_azim;
    if (exact_rows && cx > 0.0) {
        let cy = proj[o + PROJ_CONIC + 1u];
        let cz = proj[o + PROJ_CONIC + 2u];
        let smax = log(max(proj[o + PROJ_OPAC], 1e-6) * 255.0);
        var dy: f32;
        if (has_flag(FLAG_HAS_ROW_ELEVATIONS)) {
            // One beam per row: its elevation is the only one sampled.
            dy = sensors[P.row_elev_base + u32(row)] - g_elev;
        } else {
            let e0 = sensors[P.elev_base + u32(row)];
            let e1 = sensors[P.elev_base + u32(row) + 1u];
            dy = clamp(g_elev, e0, e1) - g_elev;
        }
        let qa = 0.5 * cx;
        let qb = cy * dy;
        let qc = 0.5 * cz * dy * dy;
        let disc = qb * qb - 4.0 * qa * (qc - smax);
        if (disc < 0.0) {
            return Span(0, 0, false);
        }
        let sq = sqrt(disc);
        let inv2a = 0.5 / qa;
        half = sq * inv2a;
        centre = g_azim - qb * inv2a;
    }

    let a_lo = centre - half - P.min_azimuth;
    let a_hi = centre + half - P.min_azimuth;
    let azim_max = f32(P.n_tiles_azim) * P.tile_azim_resolution;
    var t_lo: f32;
    var t_hi: f32;
    if (a_lo >= 0.0) {
        t_lo = a_lo / P.tile_azim_resolution;
    } else {
        t_lo = (((a_lo + 360.0) % 360.0) - azim_max) / P.tile_azim_resolution;
    }
    if (a_hi <= 360.0) {
        t_hi = a_hi / P.tile_azim_resolution;
    } else {
        t_hi = f32(P.n_tiles_azim) + ((a_hi + 360.0) % 360.0) / P.tile_azim_resolution;
    }
    let lo = i32(floor(t_lo));
    let hi = i32(ceil(t_hi));
    return Span(lo, hi, hi > lo);
}

struct ElevRange {
    lo: i32,
    hi: i32,
}

// Walk the (monotone, possibly non-uniform) elevation boundaries for the tile
// rows the Gaussian's elevation extent overlaps.
fn elev_range(o: u32) -> ElevRange {
    let elev = proj[o + PROJ_MEAN2D + 1u];
    let ext = proj[o + PROJ_RADII + 1u];
    let low = elev - ext;
    let high = elev + ext;
    let n = i32(P.n_tiles_elev);

    var i = 0;
    loop {
        if (!(i <= n && sensors[P.elev_base + u32(i)] < low)) { break; }
        i = i + 1;
    }
    let lo = max(i - 1, 0);
    loop {
        if (!(i <= n && sensors[P.elev_base + u32(i)] < high)) { break; }
        i = i + 1;
    }
    return ElevRange(lo, min(i, n));
}

@compute @workgroup_size(256)
fn count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= P.n_cameras * P.n_gaussians) {
        return;
    }
    let o = idx * PROJ_STRIDE;
    if (proj[o + PROJ_RADII] <= 0.0) {
        counts[idx] = 0u;
        return;
    }
    let exact_rows = has_flag(FLAG_EXACT_ROW_SPANS);
    let rows = elev_range(o);
    var n = 0u;
    for (var row = rows.lo; row < rows.hi; row = row + 1) {
        let s = row_span(o, row, exact_rows);
        if (s.hit) {
            n = n + u32(s.hi - s.lo);
        }
    }
    counts[idx] = n;
}

@compute @workgroup_size(256)
fn encode(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= P.n_cameras * P.n_gaussians) {
        return;
    }
    let o = idx * PROJ_STRIDE;
    if (proj[o + PROJ_RADII] <= 0.0) {
        return;
    }
    let exact_rows = has_flag(FLAG_EXACT_ROW_SPANS);
    let rows = elev_range(o);

    let cid = idx / P.n_gaussians;
    // The CUDA key is camera | tile | depth packed into 64 bits. WGSL has no
    // u64, so it is carried as two u32 sorted lexicographically: the high word
    // is camera << tile_n_bits | tile, the low word the depth's bit pattern
    // (depth > 0 here, so its IEEE bits order the same way the float does).
    let hi_base = cid << P.tile_n_bits;
    let depth_bits = bitcast<u32>(proj[o + PROJ_DEPTH]);

    // select() would evaluate both arms, so the idx == 0 case is an if.
    var cur = 0u;
    if (idx > 0u) {
        cur = cum[idx - 1u];
    }
    for (var row = rows.lo; row < rows.hi; row = row + 1) {
        let s = row_span(o, row, exact_rows);
        if (!s.hit) {
            continue;
        }
        for (var j = s.lo; j < s.hi; j = j + 1) {
            // Floored modulo, not C's truncated one. The CUDA kernel writes
            // `(j + n_tiles_azim) % n_tiles_azim`, which is only a wrap while
            // j >= -n_tiles_azim; a Gaussian whose azimuth extent exceeds the
            // full ring (rolling shutter at very short range) reaches j below
            // that and lands on a negative tile id, which then corrupts the
            // camera field of the 64-bit sort key. Wrapping properly keeps
            // every pair on a real tile and is identical wherever the CUDA
            // expression is a wrap at all.
            var w = j % i32(P.n_tiles_azim);
            if (w < 0) {
                w = w + i32(P.n_tiles_azim);
            }
            let tile_id = u32(row) * P.n_tiles_azim + u32(w);
            isects[cur] = vec4<u32>(hi_base | tile_id, depth_bits, idx, 0u);
            cur = cur + 1u;
        }
    }
}
