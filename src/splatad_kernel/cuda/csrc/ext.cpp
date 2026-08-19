#include "bindings.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fully_fused_lidar_projection_fwd", &fully_fused_lidar_projection_fwd_tensor);
    m.def("fully_fused_lidar_projection_bwd", &fully_fused_lidar_projection_bwd_tensor);

    m.def("isect_lidar_tiles", &isect_lidar_tiles_tensor);
    m.def("isect_offset_encode", &isect_offset_encode_tensor);

    m.def("rasterize_to_points_fwd", &rasterize_to_points_fwd_tensor);
    m.def("rasterize_to_points_bwd", &rasterize_to_points_bwd_tensor);
}
