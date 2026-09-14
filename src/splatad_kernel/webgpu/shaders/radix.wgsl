// Stable LSD radix sort of the intersection list, 4 bits per pass.
//
// This is what the pipeline sorts with; sort.wgsl's bitonic is the fallback.
// Bitonic moves the whole array through global memory once per compare-exchange
// step -- 78 of them for a million intersections -- where radix touches it
// twice per 4-bit digit and skips the leading digits of the (camera | tile)
// word that the tile grid cannot reach.
//
// Elements are vec4<u32>: .x = camera | tile, .y = the depth's bit pattern,
// .z = the flatten id. The sort is over (.x, .y) lexicographically, so .y is
// sorted first and .x last, which an LSD sort gives exactly as long as every
// pass is stable.
//
// Stability comes from deterministic ranking rather than atomics: each
// invocation owns PER_THREAD *contiguous* elements, and a bin-major scan of the
// (bin, invocation) count matrix gives every invocation the offset of its own
// run within each bin. An invocation then walks its elements in order. No two
// elements can land on the same slot and the original order within a bin is
// preserved, run to run and device to device.

const WG: u32 = 64u;
const PER_THREAD: u32 = 16u;
const CHUNK: u32 = WG * PER_THREAD;
const RADIX: u32 = 16u;
const DIGIT_MASK: u32 = RADIX - 1u;

struct RadixParams {
    n: u32,          // padded element count, a multiple of CHUNK
    shift: u32,      // bit offset of this pass's digit within the word
    word: u32,       // 0 = depth (.y), 1 = camera | tile (.x)
    num_chunks: u32, // n / CHUNK
}

@group(0) @binding(0) var<uniform> R: RadixParams;
@group(0) @binding(1) var<storage, read> src: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> hist: array<u32>;      // [RADIX][num_chunks]
@group(0) @binding(4) var<storage, read> hist_incl: array<u32>;       // inclusive scan of hist

// Counts per (bin, invocation) for this workgroup's chunk, bin-major so a plain
// scan of the flat array yields both the bin bases and the within-bin ranks.
var<workgroup> mat: array<u32, RADIX * WG>;
var<workgroup> run_totals: array<u32, WG>;

fn digit_of(e: vec4<u32>) -> u32 {
    var word = e.y;
    if (R.word == 1u) {
        word = e.x;
    }
    return (word >> R.shift) & DIGIT_MASK;
}

@compute @workgroup_size(64)
fn histogram(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) t: u32,
) {
    let base = wid.x * CHUNK + t * PER_THREAD;
    var counts: array<u32, RADIX>;
    for (var d = 0u; d < RADIX; d = d + 1u) {
        counts[d] = 0u;
    }
    for (var r = 0u; r < PER_THREAD; r = r + 1u) {
        let d = digit_of(src[base + r]);
        counts[d] = counts[d] + 1u;
    }
    for (var d = 0u; d < RADIX; d = d + 1u) {
        mat[d * WG + t] = counts[d];
    }
    workgroupBarrier();

    // One invocation per bin folds that bin's row into the global histogram.
    if (t < RADIX) {
        var total = 0u;
        for (var tt = 0u; tt < WG; tt = tt + 1u) {
            total = total + mat[t * WG + tt];
        }
        hist[t * R.num_chunks + wid.x] = total;
    }
}

// Exclusive scan of `mat` in place, over the flat bin-major array.
fn scan_mat(t: u32) {
    let span = (RADIX * WG) / WG; // contiguous entries per invocation
    var sum = 0u;
    for (var r = 0u; r < span; r = r + 1u) {
        sum = sum + mat[t * span + r];
    }
    run_totals[t] = sum;
    workgroupBarrier();

    for (var offset = 1u; offset < WG; offset = offset << 1u) {
        var add = 0u;
        if (t >= offset) {
            add = run_totals[t - offset];
        }
        workgroupBarrier();
        run_totals[t] = run_totals[t] + add;
        workgroupBarrier();
    }

    var running = 0u;
    if (t > 0u) {
        running = run_totals[t - 1u];
    }
    for (var r = 0u; r < span; r = r + 1u) {
        let v = mat[t * span + r];
        mat[t * span + r] = running;
        running = running + v;
    }
    workgroupBarrier();
}

@compute @workgroup_size(64)
fn scatter(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) t: u32,
) {
    let base = wid.x * CHUNK + t * PER_THREAD;
    var digits: array<u32, PER_THREAD>;
    var counts: array<u32, RADIX>;
    for (var d = 0u; d < RADIX; d = d + 1u) {
        counts[d] = 0u;
    }
    for (var r = 0u; r < PER_THREAD; r = r + 1u) {
        let d = digit_of(src[base + r]);
        digits[r] = d;
        counts[d] = counts[d] + 1u;
    }
    for (var d = 0u; d < RADIX; d = d + 1u) {
        mat[d * WG + t] = counts[d];
    }
    workgroupBarrier();
    scan_mat(t);

    // mat[d * WG + t] is now "everything before bin d, plus this bin's entries
    // from lower invocations"; subtracting mat[d * WG] leaves the within-bin
    // rank, which slots into this chunk's share of bin d.
    var cursor: array<u32, RADIX>;
    for (var d = 0u; d < RADIX; d = d + 1u) {
        let slot = d * R.num_chunks + wid.x;
        let chunk_base = hist_incl[slot] - hist[slot]; // exclusive from inclusive
        cursor[d] = chunk_base + (mat[d * WG + t] - mat[d * WG]);
    }
    for (var r = 0u; r < PER_THREAD; r = r + 1u) {
        let d = digits[r];
        dst[cursor[d]] = src[base + r];
        cursor[d] = cursor[d] + 1u;
    }
}
