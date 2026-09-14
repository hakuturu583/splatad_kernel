# splatad_kernel

SplatAD's spherical **LiDAR** rasterizer, packaged on its own.

A camera rasterizer returns alpha-weighted *expected* depth, which sits past the
first surface a ray meets and smears the thin rings a spinning LiDAR actually
measures. This returns the **median (first) return** instead — as sharp as the
sensor — on the sensor's own spherical sampling grid, with per-Gaussian
intensity / ray-drop features carried through the same pass.

Extracted from the [SplatAD](https://github.com/carlinds/splatad) gsplat fork
and reduced to the LiDAR path, so it installs *next to* an ordinary `gsplat`
rather than replacing it: separate Python package, separately-named CUDA
extension (`splatad_kernel_cuda`). See [NOTICE](NOTICE) for attribution and the
full list of changes.

## Install

```bash
pip install git+https://github.com/hakuturu583/splatad_kernel.git
```

## Use

```python
import torch
from splatad_kernel import lidar_rasterization

render, alphas, alpha_sum, meta = lidar_rasterization(
    means=means,                  # (N, 3) world
    quats=quats,                  # (N, 4) wxyz
    scales=scales,                # (N, 3) post-exp
    opacities=opacities,          # (N,)
    lidar_features=features,      # (C, N, D) e.g. [intensity, raydrop_logit]
    velocities=None,
    viewmats=viewmats,            # (C, 4, 4) world -> sensor
    raster_pts=raster_pts,        # (C, H, W, 4) [azimuth°, elevation°, range, t]
    tile_elevation_boundaries=tile_bounds,
    n_elevation_channels=H,
    azimuth_resolution=360.0 / W,
)
distance = meta["median_depths"][0, ..., 0]   # (H, W) first-return range
intensity = render[0, ..., 0]
```

`raster_pts` is where the sensor model lives: each cell carries the azimuth and
elevation of the beam that samples it, so non-uniform beam tables work directly,
and a per-column time offset, which is what makes rolling shutter possible —
give the rasterizer the sensor's linear/angular velocity and each column is
displaced by its own scan time.

## CUDA

The extension is JIT-compiled by `torch.utils.cpp_extension` on first import, so
a CUDA toolkit (`nvcc`) must be on `PATH` and `TORCH_CUDA_ARCH_LIST` should name
your architectures. Nothing else to install: the shared device math and the GLM
headers are read out of the installed `gsplat` (known good against 1.5.3), which
you need for the camera path anyway. Only SplatAD's own spherical LiDAR math
lives here.

```bash
export TORCH_CUDA_ARCH_LIST="8.6"     # your GPU(s)
export MAX_JOBS=4                     # nvcc is memory-hungry; cap it
python -c "from splatad_kernel.cuda._backend import _C; print(_C)"
```

To ship into a runtime image without a toolkit, compile once in the builder
stage with `TORCH_EXTENSIONS_DIR` pointed somewhere you can copy, then set the
same variable at runtime — the loader picks up the pre-built `.so` and never
invokes `nvcc`.

Tuning knobs are compile-time defines, set through `extra_cuda_cflags`:

| Define | Default | Effect |
|---|---|---|
| `LIDAR_BATCH_MULT` | 16 | Gaussians each thread stages into shared memory per round |

## WebGPU

`splatad_kernel.webgpu` is the same forward pipeline in WGSL compute shaders,
for running the LiDAR render where CUDA is not an option — a non-NVIDIA GPU, a
laptop's integrated one, a software adapter in CI, or (the shaders are plain
WebGPU) a browser.

```bash
pip install "splatad-kernel[webgpu] @ git+https://github.com/hakuturu583/splatad_kernel.git"
```

```python
from splatad_kernel.webgpu import lidar_rasterization   # same call as the CUDA one

render, alphas, alpha_sum, meta = lidar_rasterization(
    means=means, quats=quats, scales=scales, opacities=opacities,
    lidar_features=features, velocities=None, viewmats=viewmats,
    raster_pts=raster_pts, tile_elevation_boundaries=tile_bounds,
    n_elevation_channels=H, azimuth_resolution=360.0 / W,
)
distance = meta["median_depths"][0, ..., 0]
```

Same arguments, same output tuple, same `meta` keys. Inputs may be torch
tensors or numpy arrays and torch in gives torch out, so an existing call site
changes only which function it imports. Nothing here needs torch or a CUDA
toolkit; `wgpu` and `numpy` are the whole dependency list, and
`splatad_kernel.webgpu.is_available()` says whether this machine has an adapter.

Keep one `LidarRasterizer` and call it per frame rather than going through the
module function with a fresh device — pipelines and buffers live on the
instance, and the module function already holds a shared one for you.

### What is different

**Forward only.** There is no backward pass: this is for inference, and
anything that would need gradients (`sparse_grad`, `absgrad`, a `requires_grad`
input) raises rather than quietly returning something undifferentiable. Train
with the CUDA path, deploy with either.

Also missing, all of them rejected explicitly rather than ignored: `packed`
mode (the CUDA path has not implemented it either) and `depth_lanes` (a CUDA
scheduling variant whose output only differs at float epsilon). `channel_chunk`
is accepted and ignored — it exists to keep the CUDA kernel inside its register
budget, which is not a constraint here.

One deliberate divergence in the tile binning: where a Gaussian's azimuth extent
exceeds the full ring — very short range under rolling shutter — the CUDA
kernel's `(j + n_tiles_azim) % n_tiles_azim` stops being a wrap and lands on a
negative tile id, which then corrupts the camera field of its 64-bit sort key.
The WGSL wraps properly. Everywhere the CUDA expression is a wrap at all the two
agree exactly.

### Accuracy

`tests/test_webgpu_parity.py` checks the WebGPU output against
`tests/reference_numpy.py`, an independent NumPy transcription of the CUDA
forward pass, stage by stage and end to end: projection, tile counts, the sort,
the blended features, alphas, median depths, first-return depths and the median
Gaussian ids. Agreement is to ~3e-5 relative, which is where two f32
implementations of `exp` put it. The tests need a WebGPU adapter; mesa's
lavapipe is enough and is how they run without a GPU.

That validates the WGSL against a second reading of the CUDA source, not
against the CUDA kernel itself — this repository's tests have never had a GPU to
run it on. Two caveats follow from that. First, the CUDA build may be compiled
with `--use_fast_math`, so its own `exp`/`rsqrt` are approximations and the
WebGPU output is if anything the more accurate of the two. Second, the CUDA
kernel's one-beam-row specialisation (`ROW_TILE`) reassociates the sigma
arithmetic; it is not ported, so a one-beam-row render differs from CUDA at the
same float-epsilon level as everything else.

The elevation angle is the one place the port does not transcribe the CUDA
expression literally: `asin(z / r)` became `atan2(z, hypot(x, y))`, the same
angle through a function whose accuracy WGSL pins down better. With `asin` the
elevation drifted by up to 0.02° on a software adapter, which is enough to move
a Gaussian between beams.

### Performance

The pipeline is the CUDA one stage for stage — project, count intersections,
prefix sum, encode, sort, tile offsets, rasterize — with one host round trip in
the middle where the CUDA path has one, to size the intersection buffers from
the prefix sum's total.

The sort is the interesting stage. WebGPU has no `cub::DeviceRadixSort` and no
64-bit integers, so `radix.wgsl` is a stable LSD radix sort over the key as two
u32 words, ranked deterministically (no atomics) so its output does not vary
run to run or device to device. It skips the leading digits of the
camera-and-tile word that the tile grid cannot reach. `sort.wgsl`'s bitonic sort
is kept as a fallback and as a cross-check — both produce bit-identical renders,
which is worth more as a test than as an option.

On llvmpipe (a software CPU adapter — a real GPU is a different scale
entirely), one 32x1800 panorama:

| Gaussians | intersections | radix | bitonic |
|---|---|---|---|
| 5 000 | 31 k | 21 ms | 26 ms |
| 20 000 | 152 k | 53 ms | 158 ms |
| 100 000 | 981 k | 222 ms | 548 ms |

Sizing limits come from the adapter: `maxStorageBufferBindingSize` caps
Gaussians per call (the projection record is 64 bytes per Gaussian per sensor)
and `maxComputeWorkgroupsPerDimension` caps `n_cameras * n_gaussians` at about
16.7M. Both raise a clear error rather than rendering something wrong.

### Reusing the shaders

`src/splatad_kernel/webgpu/shaders/*.wgsl` is plain WebGPU with no Python in it
beyond a `//#include` and a handful of `${}` template variables that specialise
the rasterizer the way the CUDA kernel is templated (feature count, tile shape,
static/depth-compensation). A JavaScript or Rust host can drive the same
pipeline; `rasterizer.py` is the reference for the order the stages run in, and
the comments at the top of `common.wgsl` for what each buffer holds.

## Status

Used in production by [splatsim](https://github.com/hakuturu583/splatsim) for
multi-LiDAR sensor simulation. The rasterizer has had substantial inference-side
performance work; `git log` records what each change was measured to be worth,
and which plausible-looking ones turned out not to pay.
