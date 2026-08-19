// SplatAD LiDAR rasterizer: host-side declarations.
//
// The shared macros (CHECK_*, DEVICE_GUARD, CUB_WRAPPER) and the glm typedefs
// come from gsplat 1.5.3's Common.h rather than being copied here; see NOTICE.
#include "Common.h"
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <tuple>

#define N_THREADS 256

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fully_fused_lidar_projection_fwd_tensor(
    const torch::Tensor &means,                // [N, 3]
    const at::optional<torch::Tensor> &covars, // [N, 6] optional
    const at::optional<torch::Tensor> &quats,  // [N, 4] optional
    const at::optional<torch::Tensor> &scales, // [N, 3] optional
    const at::optional<torch::Tensor> &velocities, // [N, 3]
    const torch::Tensor &viewmats,             // [C, 4, 4]
    const float min_elevation,
    const float max_elevation,
    const float min_azimuth,
    const float max_azimuth,
    const torch::Tensor &linear_velocity,      // [C, 3]
    const torch::Tensor &angular_velocity,     // [C, 3]
    const torch::Tensor &rolling_shutter_time, // [C]
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool calc_compensations);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fully_fused_lidar_projection_bwd_tensor(
    // fwd inputs
    const torch::Tensor &means,                         // [N, 3]
    const at::optional<torch::Tensor> &covars,          // [N, 6] optional
    const at::optional<torch::Tensor> &quats,           // [N, 4] optional
    const at::optional<torch::Tensor> &scales,          // [N, 3] optional
    const at::optional<torch::Tensor> &velocities, // [N, 3]
    const torch::Tensor &viewmats,                      // [C, 4, 4]
    const float min_elevation,
    const float max_elevation,
    const float min_azimuth,
    const float max_azimuth,
    const torch::Tensor &linear_velocity,               // [C, 3]
    const torch::Tensor &angular_velocity,              // [C, 3]
    const torch::Tensor &rolling_shutter_time,          // [C]
    const float eps2d,
    // fwd outputs
    const torch::Tensor &radii,                         // [C, N, 2]
    const torch::Tensor &conics,                        // [C, N, 3]
    const at::optional<torch::Tensor> &compensations,   // [C, N] optional
    // grad outputs
    const torch::Tensor &v_means2d,                     // [C, N, 2]
    const torch::Tensor &v_depths,                      // [C, N]
    const torch::Tensor &v_conics,                      // [C, N, 3]
    const at::optional<torch::Tensor> &v_compensations, // [C, N] optional
    const torch::Tensor &v_pix_vels,                    // [C, N, 2]
    const torch::Tensor &v_depth_compensations,         // [C, N, 2]
    const bool viewmats_requires_grad);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
isect_lidar_tiles_tensor(const torch::Tensor &means2d, // [C, N, 2] or [nnz, 2]
                   const torch::Tensor &radii,   // [C, N, 2] or [nnz, 2]
                   const torch::Tensor &depths,  // [C, N] or [nnz]
                   const at::optional<torch::Tensor> &conics,
                   const at::optional<torch::Tensor> &opacities,
                   const at::optional<torch::Tensor> &row_elevations,
                   const at::optional<torch::Tensor> &camera_ids,   // [nnz]
                   const at::optional<torch::Tensor> &gaussian_ids, // [nnz]
                   const uint32_t C,
                   const torch::Tensor &elev_boundaries,
                   const float tile_azim_resolution,
                   const float min_azim,
                   const bool sort, const bool double_buffer);

torch::Tensor isect_offset_encode_tensor(const torch::Tensor &isect_ids, // [n_isects]
                                         const uint32_t C, const uint32_t tile_width,
                                         const uint32_t tile_height);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize_to_points_fwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2]
    const torch::Tensor &conics,                    // [C, N, 3]
    const torch::Tensor &colors,                    // [C, N, D]
    const torch::Tensor &opacities,                 // [N]
    const torch::Tensor &pix_vels,                  // [C, N, 3]
    const torch::Tensor &depth_compensations,     // [C, N, 2]
    const at::optional<torch::Tensor> &backgrounds, // [C, D]
    // Points to rasterize
    const torch::Tensor &raster_pts, // [C, image_height, image_width, 4]
    // image size
    const uint32_t image_width, const uint32_t image_height,
    const uint32_t tile_width, const uint32_t tile_height,
    // compute alphas until point
    const bool compute_alpha_sum_until_points,
    const float compute_alpha_sum_until_points_threshold,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,   // [n_isects]
    // depth channel index
    const uint32_t depth_channel_idx,
    // splatsim: skip zero rolling-shutter / depth-comp terms (static fast path)
    const bool static_render
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_to_points_bwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2]
    const torch::Tensor &conics,                    // [C, N, 3]
    const torch::Tensor &colors,                    // [C, N, 3]
    const torch::Tensor &opacities,                 // [N]
    const torch::Tensor &pix_vels,                  // [C, N, 3]
    const torch::Tensor &depth_compensations,       // [C, N, 2]
    const at::optional<torch::Tensor> &backgrounds, // [C, 3]
    // Points to rasterize
    const torch::Tensor &raster_pts, // [C, image_height, image_width, 4]
    // image size
    const uint32_t image_width, const uint32_t image_height,
    const uint32_t tile_width, const uint32_t tile_height,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,  // [n_isects]
    // forward outputs
    const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
    const torch::Tensor &last_ids,      // [C, image_height, image_width]
    // gradients of outputs
    const torch::Tensor &v_render_colors, // [C, image_height, image_width, 3]
    const torch::Tensor &v_render_alphas, // [C, image_height, image_width, 1]
    const torch::Tensor &v_alpha_sum_until_points, // [C, image_height, image_width, 1])
    // options
    bool absgrad,
    const bool compute_alpha_sum_until_points,
    const float compute_alpha_sum_until_points_threshold,
    const uint32_t depth_channel_idx
);
