//#include "common.wgsl"
//
// Port of isect_offset_encode (cuda/csrc/rasterization.cu), turned inside out:
// rather than have each intersection write the run of offsets that ends at it,
// each tile binary-searches the sorted key array for its own first entry. Same
// result, no unbounded per-thread loops, and one invocation per tile instead of
// one per intersection.
//
// One extra entry is appended past the grid holding n_isects, so the rasterizer
// can read tile+1 for the range end without a special case on the last tile.

@group(0) @binding(1) var<storage, read> isects: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> tile_offsets: array<i32>;

// The sort key packs the tile into tile_n_bits, which is generally wider than
// the tile count; unpack to the dense (camera, tile) index the grid uses. The
// mapping is monotone, so the binary search stays valid.
fn linear_tile(h: u32, n_tiles: u32) -> u32 {
    let mask = (1u << P.tile_n_bits) - 1u;
    return (h >> P.tile_n_bits) * n_tiles + (h & mask);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_tiles = P.n_tiles_azim * P.n_tiles_elev;
    let total = P.n_cameras * n_tiles;
    let t = gid.x;
    if (t > total) {
        return;
    }
    if (t == total) {
        tile_offsets[t] = i32(P.n_isects);
        return;
    }
    var lo = 0u;
    var hi = P.n_isects;
    loop {
        if (lo >= hi) { break; }
        let mid = lo + (hi - lo) / 2u;
        if (linear_tile(isects[mid].x, n_tiles) < t) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    tile_offsets[t] = i32(lo);
}
