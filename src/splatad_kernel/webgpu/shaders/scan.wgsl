// Exclusive-scan building blocks for the (Gaussian -> first output slot) offset
// table. The CUDA path gets this from cub::DeviceScan; here it is the textbook
// three-kernel scan, recursed on the block sums by the host.
//
// No uniforms: each level binds buffers sized exactly to its element count, so
// arrayLength() carries the length.

const WG: u32 = 256u;

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;

var<workgroup> shared_scan: array<u32, WG>;

// Inclusive Hillis-Steele scan inside each block; the block total goes to
// block_sums[block] for the next level up.
@compute @workgroup_size(256)
fn scan_block(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let n = arrayLength(&src);
    let i = gid.x;
    let t = lid.x;

    var v = 0u;
    if (i < n) {
        v = src[i];
    }
    shared_scan[t] = v;
    workgroupBarrier();

    for (var offset = 1u; offset < WG; offset = offset << 1u) {
        var add = 0u;
        if (t >= offset) {
            add = shared_scan[t - offset];
        }
        workgroupBarrier();
        shared_scan[t] = shared_scan[t] + add;
        workgroupBarrier();
    }

    if (i < n) {
        dst[i] = shared_scan[t];
    }
    if (t == WG - 1u) {
        block_sums[wid.x] = shared_scan[t];
    }
}

// Add each block's exclusive prefix (from the scanned block sums bound as
// `src`) to every element of that block.
@compute @workgroup_size(256)
fn add_offsets(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let n = arrayLength(&dst);
    let i = gid.x;
    if (i >= n || wid.x == 0u) {
        return;
    }
    dst[i] = dst[i] + src[wid.x - 1u];
}
