#include "bindings.h"
#include "camera_bindings.h"
#include <torch/extension.h>

// Every entry point runs under gil_scoped_release: these functions only touch
// the torch C++ API, and several of them block on the GPU mid-body
// (.item() syncs in isect_lidar_tiles / rasterize_to_points_fwd). Holding the
// GIL across those waits starves every other Python thread in the host
// process — e.g. splatsim's gRPC pose-ingestion thread — for tens of ms per
// frame.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fully_fused_lidar_projection_fwd", &fully_fused_lidar_projection_fwd_tensor,
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("fully_fused_lidar_projection_bwd", &fully_fused_lidar_projection_bwd_tensor,
          pybind11::call_guard<pybind11::gil_scoped_release>());

    m.def("isect_lidar_tiles", &isect_lidar_tiles_tensor,
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("isect_offset_encode", &isect_offset_encode_tensor,
          pybind11::call_guard<pybind11::gil_scoped_release>());

    m.def("rasterize_to_points_fwd", &rasterize_to_points_fwd_tensor,
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("rasterize_to_points_bwd", &rasterize_to_points_bwd_tensor,
          pybind11::call_guard<pybind11::gil_scoped_release>());

    // ── SplatAD camera path (ported from the original camera+lidar fork; used by
    // splatsim + gaussian_factory unified_kernel=2 for full train/deploy parity) ──
    // Same gil_scoped_release policy as the LiDAR entry points above: these are
    // CUDA ops that sync on the GPU, so holding the GIL across them starves other
    // Python threads (splatsim's gRPC pose thread) for tens of ms per frame.
    m.def("compute_sh_fwd", &compute_sh_fwd_tensor,
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("compute_sh_bwd", &compute_sh_bwd_tensor,
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("quat_scale_to_covar_preci_fwd", &quat_scale_to_covar_preci_fwd_tensor,
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("quat_scale_to_covar_preci_bwd", &quat_scale_to_covar_preci_bwd_tensor,
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("persp_proj_fwd", &persp_proj_fwd_tensor,
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("persp_proj_bwd", &persp_proj_bwd_tensor,
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("world_to_cam_fwd", &world_to_cam_fwd_tensor,
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("world_to_cam_bwd", &world_to_cam_bwd_tensor,
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("compute_pix_velocity_fwd", &compute_pix_velocity_fwd_tensor,
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("compute_pix_velocity_bwd", &compute_pix_velocity_bwd_tensor,
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("fully_fused_projection_fwd", &fully_fused_projection_fwd_tensor,
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("fully_fused_projection_bwd", &fully_fused_projection_bwd_tensor,
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("fully_fused_projection_packed_fwd", &fully_fused_projection_packed_fwd_tensor,
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("fully_fused_projection_packed_bwd", &fully_fused_projection_packed_bwd_tensor,
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("isect_tiles", &isect_tiles_tensor,
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("rasterize_to_pixels_fwd", &rasterize_to_pixels_fwd_tensor,
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("rasterize_to_pixels_bwd", &rasterize_to_pixels_bwd_tensor,
          pybind11::call_guard<pybind11::gil_scoped_release>());
}
