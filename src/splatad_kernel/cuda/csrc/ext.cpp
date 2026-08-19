#include "bindings.h"
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
}
