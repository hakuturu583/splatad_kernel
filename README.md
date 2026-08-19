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
your architectures. GLM is vendored (headers only) — nothing else to install.

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

## Status

Used in production by [splatsim](https://github.com/hakuturu583/splatsim) for
multi-LiDAR sensor simulation. The rasterizer has had substantial inference-side
performance work; `git log` records what each change was measured to be worth,
and which plausible-looking ones turned out not to pay.
