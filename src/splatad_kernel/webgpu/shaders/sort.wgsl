// Bitonic sort of the intersection list by the 64-bit (camera | tile | depth)
// key the CUDA path hands to cub::DeviceRadixSort.
//
// Kept as the fallback and as a cross-check for radix.wgsl, which is what the
// pipeline uses by default: bitonic is O(n log^2 n) and moves the whole array
// through global memory once per step, which on a million intersections is
// several times the traffic of the radix sort.
//
// The key is carried as two u32 words compared lexicographically (element .x
// and .y; .z is the payload). The array is padded to a power of two with
// 0xFFFFFFFF keys, which sort to the tail and are ignored by everything
// downstream -- the host only ever reads the first n_isects entries.

const LOCAL_ELEMS: u32 = 512u;
const WG: u32 = 256u;
const PER_THREAD: u32 = LOCAL_ELEMS / WG;

struct SortStep {
    k: u32, // bitonic stage: the length of the monotone runs being merged
    j: u32, // compare-exchange stride for this dispatch
    n: u32, // padded element count
    pad: u32,
}

@group(0) @binding(0) var<uniform> S: SortStep;
@group(0) @binding(1) var<storage, read_write> data: array<vec4<u32>>;

var<workgroup> s_data: array<vec4<u32>, LOCAL_ELEMS>;

fn gt_global(a: u32, b: u32) -> bool {
    let ea = data[a];
    let eb = data[b];
    if (ea.x != eb.x) {
        return ea.x > eb.x;
    }
    return ea.y > eb.y;
}

fn gt_local(a: u32, b: u32) -> bool {
    let ea = s_data[a];
    let eb = s_data[b];
    if (ea.x != eb.x) {
        return ea.x > eb.x;
    }
    return ea.y > eb.y;
}

// One compare-exchange step with a stride too large to keep in workgroup memory.
@compute @workgroup_size(256)
fn step_global(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= S.n) {
        return;
    }
    let l = i ^ S.j;
    if (l <= i) {
        return; // the partner invocation owns this pair
    }
    let ascending = (i & S.k) == 0u;
    if (gt_global(i, l) == ascending) {
        let t = data[i];
        data[i] = data[l];
        data[l] = t;
    }
}

// Every compare-exchange step of stage `k` from stride `j_start` down to 1.
// All partners of a stride below LOCAL_ELEMS live in the same aligned chunk.
fn descend(base: u32, t: u32, k: u32, j_start: u32) {
    var j = j_start;
    loop {
        if (j < 1u) { break; }
        for (var r = 0u; r < PER_THREAD; r = r + 1u) {
            let li = r * WG + t;
            let ll = li ^ j;
            if (ll > li) {
                let ascending = ((base + li) & k) == 0u;
                if (gt_local(li, ll) == ascending) {
                    let tmp = s_data[li];
                    s_data[li] = s_data[ll];
                    s_data[ll] = tmp;
                }
            }
        }
        workgroupBarrier();
        if (j == 1u) { break; }
        j = j >> 1u;
    }
}

// Tail of a stage whose larger strides were already done globally.
@compute @workgroup_size(256)
fn step_local(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let base = wid.x * LOCAL_ELEMS;
    for (var r = 0u; r < PER_THREAD; r = r + 1u) {
        s_data[r * WG + lid.x] = data[base + r * WG + lid.x];
    }
    workgroupBarrier();
    descend(base, lid.x, S.k, S.j);
    for (var r = 0u; r < PER_THREAD; r = r + 1u) {
        data[base + r * WG + lid.x] = s_data[r * WG + lid.x];
    }
}

// Stages 2 .. LOCAL_ELEMS in one dispatch: sorts each chunk into the
// alternating ascending/descending runs the later global stages expect.
@compute @workgroup_size(256)
fn sort_local_init(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let base = wid.x * LOCAL_ELEMS;
    for (var r = 0u; r < PER_THREAD; r = r + 1u) {
        s_data[r * WG + lid.x] = data[base + r * WG + lid.x];
    }
    workgroupBarrier();
    var k = 2u;
    loop {
        if (k > LOCAL_ELEMS) { break; }
        descend(base, lid.x, k, k >> 1u);
        k = k << 1u;
    }
    for (var r = 0u; r < PER_THREAD; r = r + 1u) {
        data[base + r * WG + lid.x] = s_data[r * WG + lid.x];
    }
}

// Fill [S.j, S.n) with the sentinel key so the padded tail sorts past every
// real intersection. Reuses the SortStep uniform rather than carrying a second
// bind group layout for three lines of work.
@compute @workgroup_size(256)
fn pad_tail(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = S.j + gid.x;
    if (i >= S.n) {
        return;
    }
    data[i] = vec4<u32>(0xFFFFFFFFu, 0xFFFFFFFFu, 0u, 0u);
}
