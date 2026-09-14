"""Parity tests for the WebGPU backend against the NumPy reference.

The reference in ``reference_numpy.py`` is an independent transcription of the
CUDA forward pass, so agreement here means the WGSL and the CUDA source say the
same thing. Both are f32 and the shading languages approximate transcendentals
differently, so the comparisons are to a relative tolerance rather than exact.

These need a working WebGPU adapter. On a machine with no GPU, a software
Vulkan driver (mesa's lavapipe) is enough and is what CI uses.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

wgpu = pytest.importorskip("wgpu")

from splatad_kernel.webgpu import LidarRasterizer, is_available, lidar_rasterization  # noqa: E402
from splatad_kernel.webgpu.rasterizer import (  # noqa: E402
    ISECT_STRIDE_BYTES,
    RADIX_CHUNK,
    SORT_LOCAL_ELEMS,
    _next_pow2,
)

import reference_numpy as ref  # noqa: E402

pytestmark = pytest.mark.skipif(
    not is_available(), reason="no WebGPU adapter on this machine"
)

# Relative to the largest magnitude in the reference array. The dominant term is
# the difference between two exp() implementations inside the alpha blend.
RTOL = 2e-4


def assert_close(got, want, rtol=RTOL, name=""):
    got = np.asarray(got, np.float64)
    want = np.asarray(want, np.float64)
    assert got.shape == want.shape, f"{name}: {got.shape} != {want.shape}"
    scale = max(float(np.abs(want).max()), 1e-6)
    err = float(np.abs(got - want).max()) / scale
    assert err <= rtol, f"{name}: relative error {err:.3e} > {rtol:.1e}"


def make_scene(
    seed=0, n_gauss=96, n_cam=1, n_feat=2, height=8, width=64,
    tile_height=2, tile_width=8, spread=3.0,
):
    """A small random scene plus the sensor grid that samples it."""
    rng = np.random.default_rng(seed)
    means = rng.normal(0, spread, (n_gauss, 3)).astype(np.float32)
    means[:, 2] *= 0.4
    scene = dict(
        means=means,
        quats=rng.normal(0, 1, (n_gauss, 4)).astype(np.float32),
        scales=(rng.random((n_gauss, 3)).astype(np.float32) * 0.35 + 0.05),
        opacities=(rng.random(n_gauss).astype(np.float32) * 0.8 + 0.2),
        lidar_features=rng.random((n_cam, n_gauss, n_feat)).astype(np.float32),
    )
    viewmats = np.tile(np.eye(4, dtype=np.float32), (n_cam, 1, 1))
    for c in range(n_cam):
        viewmats[c, :3, 3] = rng.normal(0, 0.4, 3)
    min_el, max_el = -20.0, 20.0
    n_elev_tiles = math.ceil(height / tile_height)
    raster_pts = np.zeros((n_cam, height, width, 4), np.float32)
    raster_pts[..., 0] = np.linspace(-180, 180, width, endpoint=False, dtype=np.float32)
    raster_pts[..., 1] = np.linspace(min_el, max_el, height, dtype=np.float32)[:, None]
    raster_pts[..., 2] = 8.0
    scene.update(
        viewmats=viewmats,
        raster_pts=raster_pts,
        tile_elevation_boundaries=np.linspace(
            min_el, max_el, n_elev_tiles + 1
        ).astype(np.float32),
        min_elevation=min_el,
        max_elevation=max_el,
        n_elevation_channels=height,
        azimuth_resolution=360.0 / width,
        tile_width=tile_width,
        tile_height=tile_height,
    )
    return scene, rng


def compare_render(scene, **kwargs):
    """Run both implementations on the same inputs and assert every output agrees."""
    kwargs.setdefault("velocities", None)
    got = lidar_rasterization(**scene, **kwargs)
    want = ref.render(**scene, **kwargs)
    colors, alphas, alpha_sum, meta = got

    # Culled Gaussians keep whatever the projection buffer held before, the way
    # the CUDA kernel leaves its output tensors untouched past radii = 0. Only
    # the visible entries carry meaning, so compare the visibility itself and
    # then the projections behind it.
    visible = meta["radii"][..., 0] > 0
    np.testing.assert_array_equal(
        visible, want["proj"]["radii"][..., 0] > 0, "visibility"
    )
    assert visible.any(), "the scene projects nothing; the test would be vacuous"
    assert_close(meta["means2d"][visible], want["proj"]["means2d"][visible], 1e-5, "means2d")
    assert_close(meta["conics"][visible], want["proj"]["conics"][visible], 1e-4, "conics")
    assert_close(meta["depths"][visible], want["proj"]["depths"][visible], 1e-5, "depths")
    assert_close(meta["radii"][visible], want["proj"]["radii"][visible], 1e-5, "radii")
    np.testing.assert_array_equal(
        meta["tiles_per_gauss"], want["tiles_per_gauss"], "tiles_per_gauss"
    )
    assert_close(colors, want["colors"], name="render_lidar_features")
    assert_close(alphas, want["alphas"], name="render_alphas")
    assert_close(meta["median_depths"], want["median_depths"], name="median_depths")
    assert_close(meta["fr_depth"], want["fr_depth"], name="fr_depth")
    np.testing.assert_array_equal(meta["median_ids"], want["median_ids"], "median_ids")
    if alpha_sum is not None:
        assert_close(alpha_sum, want["alpha_sums"], name="alpha_sum_until_points")
    return got, want


# ---------------------------------------------------------------- stages

@pytest.mark.parametrize("algorithm", ["radix", "bitonic"])
@pytest.mark.parametrize("n_isects", [1, 300, 512, 513, 4096, 5000, 70000])
def test_sort_orders_the_intersection_list(algorithm, n_isects):
    """Both sorts must reproduce a lexicographic sort on (tile, depth)."""
    rng = np.random.default_rng(n_isects)
    if algorithm == "radix":
        n_pad = max(RADIX_CHUNK, -(-n_isects // RADIX_CHUNK) * RADIX_CHUNK)
    else:
        n_pad = max(SORT_LOCAL_ELEMS, _next_pow2(n_isects))

    n_tiles = 64
    keys_hi = rng.integers(0, n_tiles, n_isects, dtype=np.uint32)
    keys_lo = np.float32(rng.random(n_isects) * 100 + 0.1).view(np.uint32)
    values = np.arange(n_isects, dtype=np.uint32)

    records = np.full((n_pad, 4), 0xFFFFFFFF, np.uint32)
    records[n_isects:, 2:] = 0
    records[:n_isects, 0] = keys_hi
    records[:n_isects, 1] = keys_lo
    records[:n_isects, 2] = values
    records[:n_isects, 3] = 0

    rasterizer = LidarRasterizer(sort_algorithm=algorithm)
    device = rasterizer.device
    buf = rasterizer.res.buffer("test_isects", n_pad * ISECT_STRIDE_BYTES)
    device.queue.write_buffer(buf, 0, records)

    encoder = device.create_command_encoder()
    cpass = encoder.begin_compute_pass()
    if algorithm == "radix":
        out_buf = rasterizer._radix_sort(cpass, buf, n_pad, 1, 6, n_tiles)
    else:
        out_buf = rasterizer._bitonic_sort(cpass, buf, n_pad)
    cpass.end()
    device.queue.submit([encoder.finish()])

    got = np.frombuffer(
        device.queue.read_buffer(out_buf, 0, n_pad * ISECT_STRIDE_BYTES), np.uint32
    ).reshape(n_pad, 4)

    order = np.lexsort((keys_lo, keys_hi))
    np.testing.assert_array_equal(got[:n_isects, 0], keys_hi[order])
    np.testing.assert_array_equal(got[:n_isects, 1], keys_lo[order])
    # Bitonic is not a stable sort, so entries sharing a full (tile, depth) key
    # may come back in either order. What must hold is that the payloads are the
    # same multiset, still attached to their own keys.
    triples = np.rec.fromarrays(
        [got[:n_isects, 0], got[:n_isects, 1], got[:n_isects, 2]]
    )
    want_triples = np.rec.fromarrays([keys_hi, keys_lo, values])
    np.testing.assert_array_equal(np.sort(triples), np.sort(want_triples))
    # The padding must all have sorted past the real entries.
    assert (got[n_isects:, 0] == 0xFFFFFFFF).all()


def test_radix_sort_is_stable():
    """Equal keys must come back in their original order, run after run."""
    n_isects = 8192
    n_pad = n_isects
    rng = np.random.default_rng(5)
    keys_hi = rng.integers(0, 4, n_isects, dtype=np.uint32)
    keys_lo = rng.integers(0, 8, n_isects, dtype=np.uint32)  # many exact ties
    values = np.arange(n_isects, dtype=np.uint32)
    records = np.zeros((n_pad, 4), np.uint32)
    records[:, 0] = keys_hi
    records[:, 1] = keys_lo
    records[:, 2] = values

    rasterizer = LidarRasterizer(sort_algorithm="radix")
    device = rasterizer.device
    buf = rasterizer.res.buffer("test_isects", n_pad * ISECT_STRIDE_BYTES)
    device.queue.write_buffer(buf, 0, records)
    encoder = device.create_command_encoder()
    cpass = encoder.begin_compute_pass()
    out_buf = rasterizer._radix_sort(cpass, buf, n_pad, 1, 6, 4)
    cpass.end()
    device.queue.submit([encoder.finish()])
    got = np.frombuffer(
        device.queue.read_buffer(out_buf, 0, n_pad * ISECT_STRIDE_BYTES), np.uint32
    ).reshape(n_pad, 4)
    order = np.lexsort((keys_lo, keys_hi))  # stable
    np.testing.assert_array_equal(got[:, 2], values[order])


@pytest.mark.parametrize("n", [1, 255, 256, 257, 1000, 100_000])
def test_scan_matches_cumsum(n):
    """The multi-level prefix sum must match numpy's inclusive cumsum."""
    rng = np.random.default_rng(n)
    data = rng.integers(0, 7, n, dtype=np.uint32)

    rasterizer = LidarRasterizer()
    res, device = rasterizer.res, rasterizer.device
    src = res.buffer("test_scan_src", n * 4)
    device.queue.write_buffer(src, 0, data)

    encoder = device.create_command_encoder()
    cpass = encoder.begin_compute_pass()
    dst = rasterizer._scan_inclusive(cpass, src, n)
    cpass.end()
    device.queue.submit([encoder.finish()])

    got = np.frombuffer(device.queue.read_buffer(dst, 0, n * 4), np.uint32)
    np.testing.assert_array_equal(got, np.cumsum(data.astype(np.uint64)).astype(np.uint32))


# ---------------------------------------------------------------- end to end

def test_static_render_matches_reference():
    scene, _ = make_scene()
    compare_render(scene, use_depth_compensation=False)


def test_depth_compensation_matches_reference():
    scene, _ = make_scene()
    compare_render(scene, use_depth_compensation=True)


def test_rolling_shutter_matches_reference():
    scene, rng = make_scene(seed=3)
    n_gauss = scene["means"].shape[0]
    n_cam = scene["viewmats"].shape[0]
    width = scene["raster_pts"].shape[2]
    scene["raster_pts"][..., 3] = np.linspace(0, 0.1, width, dtype=np.float32)
    compare_render(
        scene,
        velocities=rng.normal(0, 0.5, (n_gauss, 3)).astype(np.float32),
        linear_velocity=rng.normal(0, 1.0, (n_cam, 3)).astype(np.float32),
        angular_velocity=rng.normal(0, 0.2, (n_cam, 3)).astype(np.float32),
        rolling_shutter_time=np.full(n_cam, 0.1, np.float32),
    )


def test_multiple_sensors_match_reference():
    scene, _ = make_scene(seed=5, n_cam=3)
    compare_render(scene, use_depth_compensation=False)


def test_antialiased_mode_matches_reference():
    scene, _ = make_scene(seed=7)
    compare_render(scene, use_depth_compensation=False, rasterize_mode="antialiased")


@pytest.mark.parametrize("n_feat", [1, 3, 8])
def test_feature_counts_match_reference(n_feat):
    scene, _ = make_scene(seed=11, n_feat=n_feat)
    compare_render(scene, use_depth_compensation=False)


@pytest.mark.parametrize("tile", [(1, 32), (4, 16), (8, 8), (2, 128)])
def test_tile_shapes_match_reference(tile):
    tile_height, tile_width = tile
    scene, _ = make_scene(
        seed=13, height=8, width=256, tile_height=tile_height, tile_width=tile_width
    )
    compare_render(scene, use_depth_compensation=False)


def test_one_beam_rows_use_row_elevations():
    """tile_height == 1 lets the binner use each row's exact beam elevation."""
    scene, _ = make_scene(seed=17, height=8, tile_height=1)
    row_elevations = scene["raster_pts"][0, :, 0, 1].copy()
    compare_render(scene, use_depth_compensation=False, row_elevations=row_elevations)


def test_valid_mask_culls_like_the_reference():
    scene, rng = make_scene(seed=19)
    mask = rng.random(scene["means"].shape[0]) > 0.4
    compare_render(scene, use_depth_compensation=False, valid_mask=mask)


def test_larger_scene_exercises_the_global_sort_steps():
    """Enough intersections that the sort leaves workgroup-local territory."""
    scene, _ = make_scene(
        seed=23, n_gauss=600, height=16, width=256, tile_height=4, tile_width=16, spread=6.0
    )
    (_, _, _, meta), _ = compare_render(scene, use_depth_compensation=False)
    assert meta["n_isects"] > SORT_LOCAL_ELEMS, "scene too small to reach the global steps"


def test_both_sorts_render_identically():
    """Radix and bitonic are independent sorts; the render must not be able to tell."""
    scene, _ = make_scene(
        seed=47, n_gauss=400, height=16, width=256, tile_height=4, tile_width=16, spread=5.0
    )
    radix = lidar_rasterization(
        velocities=None, use_depth_compensation=False, sort_algorithm="radix", **scene
    )
    bitonic = lidar_rasterization(
        velocities=None, use_depth_compensation=False, sort_algorithm="bitonic", **scene
    )
    assert radix[3]["n_isects"] > SORT_LOCAL_ELEMS
    np.testing.assert_array_equal(radix[0], bitonic[0])
    np.testing.assert_array_equal(radix[1], bitonic[1])
    np.testing.assert_array_equal(radix[3]["median_ids"], bitonic[3]["median_ids"])


def test_empty_scene_renders_zeros():
    """Everything culled: zero intersections must not trip the sort or the offsets."""
    scene, _ = make_scene(seed=29, n_gauss=8)
    scene["means"] = scene["means"] * 0.0 + 1e9  # all beyond the far plane
    colors, alphas, alpha_sum, meta = lidar_rasterization(
        velocities=None, use_depth_compensation=False, far_plane=1e4, **scene
    )
    assert meta["n_isects"] == 0
    assert not colors.any()
    assert not alphas.any()
    assert (meta["median_ids"] == -1).all()


def test_sector_render_matches_the_full_panorama():
    """A tile-column slice must render the same pixels as the full frame."""
    scene, _ = make_scene(seed=31, height=8, width=256, tile_height=2, tile_width=16)
    full, _, _, _ = lidar_rasterization(velocities=None, use_depth_compensation=False, **scene)

    tile_width = scene["tile_width"]
    offset_tiles = 4
    start = offset_tiles * tile_width
    sector = dict(scene)
    sector["raster_pts"] = scene["raster_pts"][:, :, start : start + 4 * tile_width].copy()
    part, _, _, _ = lidar_rasterization(
        velocities=None,
        use_depth_compensation=False,
        tile_col_offset=offset_tiles,
        **sector,
    )
    assert_close(part, full[:, :, start : start + 4 * tile_width], 1e-6, "sector")


def test_raydrop_sh_matches_reference():
    """The SH prepass must fold the same residual into the ray-drop channel."""
    degree = 3
    scene, rng = make_scene(seed=37, n_feat=2)
    n_gauss = scene["means"].shape[0]
    coeffs = rng.normal(0, 0.4, (n_gauss, (degree + 1) ** 2 - 1)).astype(np.float32)

    got = lidar_rasterization(
        velocities=None,
        use_depth_compensation=False,
        raydrop_sh_coeffs=coeffs,
        raydrop_sh_degree=degree,
        raydrop_feature_index=1,
        **scene,
    )
    # Fold the residual into the features by hand and render without the prepass.
    expected_scene = dict(scene)
    features = scene["lidar_features"].copy()
    plain = lidar_rasterization(velocities=None, use_depth_compensation=False, **scene)
    visible = plain[3]["radii"][..., 0] > 0
    cam_pos = -np.einsum(
        "cji,cj->ci", scene["viewmats"][:, :3, :3], scene["viewmats"][:, :3, 3]
    )
    for c in range(scene["viewmats"].shape[0]):
        sel = visible[c]
        dirs = scene["means"][sel] - cam_pos[c]
        features[c, sel, 1] += ref.sh_residual(degree, dirs, coeffs[sel])
    expected_scene["lidar_features"] = features
    want = ref.render(velocities=None, use_depth_compensation=False, **expected_scene)

    assert_close(got[0], want["colors"], name="render with SH raydrop")
    assert visible.any(), "no visible Gaussians, the SH prepass was never exercised"


def test_backward_is_refused():
    torch = pytest.importorskip("torch")
    scene, _ = make_scene(seed=41, n_gauss=8)
    means = torch.tensor(scene["means"], requires_grad=True)
    scene = dict(scene, means=means)
    with pytest.raises(NotImplementedError, match="forward only"):
        lidar_rasterization(velocities=None, **scene)


def test_torch_tensors_round_trip():
    torch = pytest.importorskip("torch")
    scene, _ = make_scene(seed=43, n_gauss=32)
    numpy_out = lidar_rasterization(velocities=None, use_depth_compensation=False, **scene)
    torch_scene = {
        k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v)
        for k, v in scene.items()
    }
    torch_out = lidar_rasterization(
        velocities=None, use_depth_compensation=False, **torch_scene
    )
    assert isinstance(torch_out[0], torch.Tensor)
    np.testing.assert_allclose(torch_out[0].numpy(), numpy_out[0], rtol=0, atol=0)
