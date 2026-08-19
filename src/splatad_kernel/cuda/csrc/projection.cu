#include "bindings.h"
#include "helpers.cuh"
#include "utils.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;



__global__ void
fully_fused_lidar_projection_fwd_kernel(const uint32_t C, const uint32_t N,
                                  const float *__restrict__ means,                // [N, 3]
                                  const float *__restrict__ covars,               // [N, 6] optional
                                  const float *__restrict__ quats,                // [N, 4] optional
                                  const float *__restrict__ scales,               // [N, 3] optional
                                  const float *__restrict__ velocities,           // [N, 3]
                                  const float *__restrict__ viewmats,             // [C, 4, 4]
                                  const float min_elevation,
                                  const float max_elevation,
                                  const float min_azimuth,
                                  const float max_azimuth,
                                  const float *__restrict__ lin_vel,              // [C, 3]
                                  const float *__restrict__ ang_vel,              // [C, 3]
                                  const float *__restrict__ rolling_shutter_time, // [C]
                                  const float eps2d,
                                  const float near_plane,
                                  const float far_plane,
                                  const float radius_clip,
                                  // outputs
                                  float *__restrict__ radii,                      // [C, N]
                                  float *__restrict__ means2d,                    // [C, N, 2]
                                  float *__restrict__ depths,                     // [C, N]
                                  float *__restrict__ conics,                     // [C, N, 3]
                                  float *__restrict__ compensations,              // [C, N] optional
                                  float *__restrict__ pix_vels,                    // [C, N, 3]
                                  float *__restrict__ depth_compensations        // [C, N, 2]
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const uint32_t cid = idx / N; // lidar id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current lidar and gaussian
    means += gid * 3;
    viewmats += cid * 16;
    lin_vel += cid * 3;
    ang_vel += cid * 3;
    const float rs_time = rolling_shutter_time[cid];

    // glm is column-major but input is row-major
    glm::mat3 R = glm::mat3(viewmats[0], viewmats[4], viewmats[8], // 1st column
                            viewmats[1], viewmats[5], viewmats[9], // 2nd column
                            viewmats[2], viewmats[6], viewmats[10] // 3rd column
    );
    glm::vec3 t = glm::vec3(viewmats[3], viewmats[7], viewmats[11]);

    // transform Gaussian center to lidar space
    glm::vec3 mean_c;
    posW2C(R, t, glm::make_vec3(means), mean_c);
    float distance = norm3df(mean_c.x , mean_c.y, mean_c.z);
    if (distance < near_plane || distance > far_plane) {
        radii[idx * 2] = 0.f;
        return;
    }

    // get covariance, either directly from input or compute from quaternions and scales
    glm::mat3 covar;
    if (covars != nullptr) {
        covars += gid * 6;
        covar = glm::mat3(covars[0], covars[1], covars[2], // 1st column
                          covars[1], covars[3], covars[4], // 2nd column
                          covars[2], covars[4], covars[5]  // 3rd column
        );
    } else {
        // compute from quaternions and scales
        quats += gid * 4;
        scales += gid * 3;
        quat_scale_to_covar_preci(glm::make_vec4(quats), glm::make_vec3(scales), &covar,
                                  nullptr);
    }
    glm::mat3 covar_c;
    covarW2C(R, covar, covar_c);

    // project to spherical lidar
    glm::mat2 covar2d;
    glm::vec2 mean2d;
    glm::vec2 depth_compensation;
    glm::mat3 jacobian;
    lidar_proj(mean_c, covar_c, eps2d, mean2d, covar2d, depth_compensation, jacobian);

    float compensation;
    float det = add_blur(eps2d, covar2d, compensation);
    if (det <= 0.f) {
        radii[idx * 2] = 0.f;
        return;
    }


    // take 3 sigma as the radius (non differentiable)
    // float b = 0.5f * (covar2d[0][0] + covar2d[1][1]);
    // float v1 = b + sqrt(max(1e-6, b * b - det));
    // float radius = 3.f * sqrt(v1);
    // float v2 = b - sqrt(max(0.1f, b * b - det));
    // float radius = ceil(3.f * sqrt(max(v1, v2)));
    float extent_azimuth = 3.f * sqrt(max(0.f, covar2d[0][0]));
    float extent_elevation = 3.f * sqrt(max(0.f, covar2d[1][1]));

    if (extent_azimuth <= radius_clip && extent_elevation <= radius_clip) {
        radii[idx * 2] = 0.f;
        return;
    }

    // increase radius to compensate for rolling shutter
    glm::vec3 pix_vel = { 0.f, 0.f, 0.f };
    if (rs_time > 0) {
        // move velocities to lidar space
        glm::vec3 vel_c(0.f);
        if (velocities != nullptr) {
            glm::vec3 vel_w = glm::make_vec3(velocities + gid * 3);
            vel_world_to_cam(R, vel_w, vel_c);
        }
        compute_lidar_velocity(mean_c, glm::make_vec3(lin_vel), glm::make_vec3(ang_vel), vel_c, jacobian, pix_vel);
        extent_azimuth += fabs(pix_vel.x) * 0.5f * rs_time;
        extent_elevation += fabs(pix_vel.y) * 0.5f * rs_time;
    }

    if (mean2d.y + extent_elevation <= min_elevation
        || mean2d.y - extent_elevation >= max_elevation
        || mean2d.x + extent_azimuth <= min_azimuth
        || mean2d.x - extent_azimuth >= max_azimuth) {
        radii[idx * 2] = 0.f;
        return;
    }

    // compute the inverse of the 2d covariance
    glm::mat2 covar2d_inv;
    inverse(covar2d, covar2d_inv);

    // write to outputs
    radii[idx * 2] = extent_azimuth;
    radii[idx * 2 + 1] = extent_elevation;
    means2d[idx * 2] = mean2d.x;
    means2d[idx * 2 + 1] = mean2d.y;
    depths[idx] = distance;
    conics[idx * 3] = covar2d_inv[0][0];
    conics[idx * 3 + 1] = covar2d_inv[0][1];
    conics[idx * 3 + 2] = covar2d_inv[1][1];
    if (compensations != nullptr) {
        compensations[idx] = compensation;
    }
    pix_vels[idx * 3] = pix_vel.x;
    pix_vels[idx * 3 + 1] = pix_vel.y;
    pix_vels[idx * 3 + 2] = pix_vel.z;
    depth_compensations[idx * 2] = depth_compensation.x;
    depth_compensations[idx * 2 + 1] = depth_compensation.y;
}

__global__ void fully_fused_lidar_projection_bwd_kernel(
    // fwd inputs
    const uint32_t C, const uint32_t N,
    const float *__restrict__ means,                // [N, 3]
    const float *__restrict__ covars,               // [N, 6] optional
    const float *__restrict__ quats,                // [N, 4] optional
    const float *__restrict__ scales,               // [N, 3] optional
    const float *__restrict__ velocities,           // [N, 3] optional
    const float *__restrict__ viewmats,             // [C, 4, 4]
    const float min_elevation,
    const float max_elevation,
    const float min_azimuth,
    const float max_azimuth,
    const float *__restrict__ lin_vel,              // [C, 3]
    const float *__restrict__ ang_vel,              // [C, 3]
    const float *__restrict__ rolling_shutter_time, // [C]
    const float eps2d,
    // fwd outputs
    const float *__restrict__ radii,                // [C, N, 2]
    const float *__restrict__ conics,               // [C, N, 3]
    const float *__restrict__ compensations,        // [C, N] optional
    // grad outputs
    const float *__restrict__ v_means2d,            // [C, N, 2]
    const float *__restrict__ v_depths,             // [C, N]
    const float *__restrict__ v_conics,             // [C, N, 3]
    const float *__restrict__ v_compensations,      // [C, N] optional
    const float *__restrict__ v_pix_vels,           // [C, N, 3]
    const float *__restrict__ v_depth_compensations, // [C, N, 2]
    // grad inputs
    float *__restrict__ v_means,                    // [N, 3]
    float *__restrict__ v_covars,                   // [N, 6] optional
    float *__restrict__ v_quats,                    // [N, 4] optional
    float *__restrict__ v_scales,                   // [N, 3] optional
    float *__restrict__ v_viewmats                  // [C, 4, 4] optional
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N || radii[idx * 2] <= 0.f) {
        return;
    }
    const uint32_t cid = idx / N; // lidar id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current lidar and gaussian
    means += gid * 3;
    viewmats += cid * 16;
    lin_vel += cid * 3;
    ang_vel += cid * 3;
    rolling_shutter_time += cid;

    conics += idx * 3;

    v_means2d += idx * 2;
    v_depths += idx;
    v_conics += idx * 3;
    v_pix_vels += idx * 3;
    v_depth_compensations += idx * 2;

    // vjp: compute the inverse of the 2d covariance
    glm::mat2 covar2d_inv = glm::mat2(conics[0], conics[1], conics[1], conics[2]);
    glm::mat2 v_covar2d_inv =
        glm::mat2(v_conics[0], v_conics[1] * .5f, v_conics[1] * .5f, v_conics[2]);
    glm::mat2 v_covar2d(0.f);
    inverse_vjp(covar2d_inv, v_covar2d_inv, v_covar2d);

    if (v_compensations != nullptr) {
        // vjp: compensation term
        const float compensation = compensations[idx];
        const float v_compensation = v_compensations[idx];
        add_blur_vjp(eps2d, covar2d_inv, compensation, v_compensation, v_covar2d);
    }

    // transform Gaussian to lidar space
    glm::mat3 R = glm::mat3(viewmats[0], viewmats[4], viewmats[8], // 1st column
                            viewmats[1], viewmats[5], viewmats[9], // 2nd column
                            viewmats[2], viewmats[6], viewmats[10] // 3rd column
    );
    glm::vec3 t = glm::vec3(viewmats[3], viewmats[7], viewmats[11]);

    glm::mat3 covar;
    glm::vec4 quat;
    glm::vec3 scale;
    if (covars != nullptr) {
        covars += gid * 6;
        covar = glm::mat3(covars[0], covars[1], covars[2], // 1st column
                          covars[1], covars[3], covars[4], // 2nd column
                          covars[2], covars[4], covars[5]  // 3rd column
        );
    } else {
        // compute from quaternions and scales
        quat = glm::make_vec4(quats + gid * 4);
        scale = glm::make_vec3(scales + gid * 3);
        quat_scale_to_covar_preci(quat, scale, &covar, nullptr);
    }
    glm::vec3 mean_c;
    posW2C(R, t, glm::make_vec3(means), mean_c);
    glm::mat3 covar_c;
    covarW2C(R, covar, covar_c);

    // vjp: vel world to cam

    glm::vec3 v_p_view_pix_vel(0.f);
    glm::mat3 v_R(0.f);
    if (rolling_shutter_time[0] > 0 ) {
        glm::vec3 vel_c(0.f);
        glm::vec3 vel_w(0.f);
        glm::vec3 v_vel_c(0.f);
        if (velocities != nullptr) {
            vel_w = glm::make_vec3(velocities + gid * 3);
            vel_world_to_cam(R, vel_w, vel_c);
        }
        compute_and_sum_lidar_velocity_vjp(
            mean_c,
            glm::make_vec3(lin_vel),
            glm::make_vec3(ang_vel),
            vel_c,
            glm::make_vec3(v_pix_vels),
            v_p_view_pix_vel,
            v_vel_c);
        glm::vec3 v_vel_w(0.f);
        if (velocities != nullptr) {
            vel_world_to_cam_vjp(R, vel_w, v_vel_c, v_R, v_vel_w);
        }
    }

    // vjp: lidar projection
    glm::mat3 v_covar_c(0.f);
    glm::vec3 v_mean_c(0.f);
    lidar_proj_vjp(mean_c, covar_c, eps2d, glm::make_vec2(v_means2d), v_covar2d, glm::make_vec2(v_depth_compensations), v_mean_c, v_covar_c);

    // add contribution from v_depths
    const float disparity = rnorm3df(mean_c.x, mean_c.y, mean_c.z);
    v_mean_c.x += mean_c.x * disparity * v_depths[0];
    v_mean_c.y += mean_c.y * disparity * v_depths[0];
    v_mean_c.z += mean_c.z * disparity * v_depths[0];

    // add contribution from pix velocities
    v_mean_c.x += v_p_view_pix_vel.x;
    v_mean_c.y += v_p_view_pix_vel.y;
    v_mean_c.z += v_p_view_pix_vel.z;

    // vjp: transform Gaussian covariance to lidar space
    glm::vec3 v_mean(0.f);
    glm::mat3 v_covar(0.f);
    glm::vec3 v_t(0.f);
    posW2C_VJP(R, t, glm::make_vec3(means), v_mean_c, v_R, v_t, v_mean);
    covarW2C_VJP(R, covar, v_covar_c, v_R, v_covar);

    // #if __CUDA_ARCH__ >= 700
    // write out results with warp-level reduction
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto warp_group_g = cg::labeled_partition(warp, gid);
    if (v_means != nullptr) {
        warpSum(v_mean, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_means += gid * 3;
            PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) {
                gpuAtomicAdd(v_means + i, v_mean[i]);
            }
        }
    }
    if (v_covars != nullptr) {
        // Output gradients w.r.t. the covariance matrix
        warpSum(v_covar, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_covars += gid * 6;
            gpuAtomicAdd(v_covars, v_covar[0][0]);
            gpuAtomicAdd(v_covars + 1, v_covar[0][1] + v_covar[1][0]);
            gpuAtomicAdd(v_covars + 2, v_covar[0][2] + v_covar[2][0]);
            gpuAtomicAdd(v_covars + 3, v_covar[1][1]);
            gpuAtomicAdd(v_covars + 4, v_covar[1][2] + v_covar[2][1]);
            gpuAtomicAdd(v_covars + 5, v_covar[2][2]);
        }
    } else {
        // Directly output gradients w.r.t. the quaternion and scale
        glm::mat3 rotmat = quat_to_rotmat(quat);
        glm::vec4 v_quat(0.f);
        glm::vec3 v_scale(0.f);
        quat_scale_to_covar_vjp(quat, scale, rotmat, v_covar, v_quat, v_scale);
        warpSum(v_quat, warp_group_g);
        warpSum(v_scale, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_quats += gid * 4;
            v_scales += gid * 3;
            gpuAtomicAdd(v_quats, v_quat[0]);
            gpuAtomicAdd(v_quats + 1, v_quat[1]);
            gpuAtomicAdd(v_quats + 2, v_quat[2]);
            gpuAtomicAdd(v_quats + 3, v_quat[3]);
            gpuAtomicAdd(v_scales, v_scale[0]);
            gpuAtomicAdd(v_scales + 1, v_scale[1]);
            gpuAtomicAdd(v_scales + 2, v_scale[2]);
        }
    }
    if (v_viewmats != nullptr) {
        auto warp_group_c = cg::labeled_partition(warp, cid);
        warpSum(v_R, warp_group_c);
        warpSum(v_t, warp_group_c);
        if (warp_group_c.thread_rank() == 0) {
            v_viewmats += cid * 16;
            PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) { // rows
                PRAGMA_UNROLL
                for (uint32_t j = 0; j < 3; j++) { // cols
                    gpuAtomicAdd(v_viewmats + i * 4 + j, v_R[j][i]);
                }
                gpuAtomicAdd(v_viewmats + i * 4 + 3, v_t[i]);
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fully_fused_lidar_projection_fwd_tensor(
    const torch::Tensor &means,                // [N, 3]
    const at::optional<torch::Tensor> &covars, // [N, 6] optional
    const at::optional<torch::Tensor> &quats,  // [N, 4] optional
    const at::optional<torch::Tensor> &scales, // [N, 3] optional
    const at::optional<torch::Tensor> &velocities, // [N, 3] optional
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
    const bool calc_compensations) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    if (covars.has_value()) {
        CHECK_INPUT(covars.value());
    } else {
        assert(quats.has_value() && scales.has_value());
        CHECK_INPUT(quats.value());
        CHECK_INPUT(scales.value());
    }
    if (velocities.has_value()) {
        CHECK_INPUT(velocities.value());
    }
    CHECK_INPUT(viewmats);
    CHECK_INPUT(linear_velocity);
    CHECK_INPUT(angular_velocity);
    CHECK_INPUT(rolling_shutter_time);

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of lidars
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor radii = torch::empty({C, N, 2}, means.options());
    torch::Tensor means2d = torch::empty({C, N, 2}, means.options());
    torch::Tensor depths = torch::empty({C, N}, means.options());
    torch::Tensor conics = torch::empty({C, N, 3}, means.options());
    torch::Tensor compensations;
    if (calc_compensations) {
        // we dont want NaN to appear in this tensor, so we zero intialize it
        compensations = torch::zeros({C, N}, means.options());
    }
    torch::Tensor pix_vels = torch::empty({C, N, 3}, means.options());
    torch::Tensor depth_compensations = torch::empty({C, N, 2}, means.options());
    if (C && N) {
        fully_fused_lidar_projection_fwd_kernel<<<(C * N + N_THREADS - 1) / N_THREADS,
                                            N_THREADS, 0, stream>>>(
            C, N, means.data_ptr<float>(),
            covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
            quats.has_value() ? quats.value().data_ptr<float>() : nullptr,
            scales.has_value() ? scales.value().data_ptr<float>() : nullptr,
            velocities.has_value() ? velocities.value().data_ptr<float>() : nullptr,
            viewmats.data_ptr<float>(),
            min_elevation,
            max_elevation,
            min_azimuth,
            max_azimuth,
            linear_velocity.data_ptr<float>(),
            angular_velocity.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            radii.data_ptr<float>(),
            means2d.data_ptr<float>(),
            depths.data_ptr<float>(),
            conics.data_ptr<float>(),
            calc_compensations ? compensations.data_ptr<float>() : nullptr,
            pix_vels.data_ptr<float>(),
            depth_compensations.data_ptr<float>()
            );
    }
    return std::make_tuple(radii, means2d, depths, conics, compensations, pix_vels, depth_compensations);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fully_fused_lidar_projection_bwd_tensor(
    // fwd inputs
    const torch::Tensor &means,                         // [N, 3]
    const at::optional<torch::Tensor> &covars,          // [N, 6] optional
    const at::optional<torch::Tensor> &quats,           // [N, 4] optional
    const at::optional<torch::Tensor> &scales,          // [N, 3] optional
    const at::optional<torch::Tensor> &velocities,      // [N, 3] optional
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
    const torch::Tensor &v_pix_vels,                    // [C, N, 3]
    const torch::Tensor &v_depth_compensations,         // [C, N, 2]
    const bool viewmats_requires_grad) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    if (covars.has_value()) {
        CHECK_INPUT(covars.value());
    } else {
        assert(quats.has_value() && scales.has_value());
        CHECK_INPUT(quats.value());
        CHECK_INPUT(scales.value());
    }
    if (velocities.has_value()) {
        CHECK_INPUT(velocities.value());
    }
    CHECK_INPUT(viewmats);
    CHECK_INPUT(radii);
    CHECK_INPUT(conics);
    CHECK_INPUT(linear_velocity);
    CHECK_INPUT(angular_velocity);
    CHECK_INPUT(rolling_shutter_time);
    CHECK_INPUT(v_means2d);
    CHECK_INPUT(v_depths);
    CHECK_INPUT(v_conics);
    if (compensations.has_value()) {
        CHECK_INPUT(compensations.value());
    }
    if (v_compensations.has_value()) {
        CHECK_INPUT(v_compensations.value());
        assert(compensations.has_value());
    }
    CHECK_INPUT(v_pix_vels);
    CHECK_INPUT(v_depth_compensations);

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of lidars
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor v_means = torch::zeros_like(means);
    torch::Tensor v_covars, v_quats, v_scales; // optional
    if (covars.has_value()) {
        v_covars = torch::zeros_like(covars.value());
    } else {
        v_quats = torch::zeros_like(quats.value());
        v_scales = torch::zeros_like(scales.value());
    }
    torch::Tensor v_viewmats;
    if (viewmats_requires_grad) {
        v_viewmats = torch::zeros_like(viewmats);
    }
    if (C && N) {
        fully_fused_lidar_projection_bwd_kernel<<<(C * N + N_THREADS - 1) / N_THREADS,
                                            N_THREADS, 0, stream>>>(
            C, N, means.data_ptr<float>(),
            covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
            covars.has_value() ? nullptr : quats.value().data_ptr<float>(),
            covars.has_value() ? nullptr : scales.value().data_ptr<float>(),
            velocities.has_value() ? velocities.value().data_ptr<float>() : nullptr,
            viewmats.data_ptr<float>(),
            min_elevation,
            max_elevation,
            min_azimuth,
            max_azimuth,
            linear_velocity.data_ptr<float>(),
            angular_velocity.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            eps2d,
            radii.data_ptr<float>(),
            conics.data_ptr<float>(),
            compensations.has_value() ? compensations.value().data_ptr<float>()
                                      : nullptr,
            v_means2d.data_ptr<float>(),
            v_depths.data_ptr<float>(),
            v_conics.data_ptr<float>(),
            v_compensations.has_value() ? v_compensations.value().data_ptr<float>()
                                        : nullptr,
            v_pix_vels.data_ptr<float>(),
            v_depth_compensations.data_ptr<float>(),
            v_means.data_ptr<float>(),
            covars.has_value() ? v_covars.data_ptr<float>() : nullptr,
            covars.has_value() ? nullptr : v_quats.data_ptr<float>(),
            covars.has_value() ? nullptr : v_scales.data_ptr<float>(),
            viewmats_requires_grad ? v_viewmats.data_ptr<float>() : nullptr);
    }
    return std::make_tuple(v_means, v_covars, v_quats, v_scales, v_viewmats);
}
