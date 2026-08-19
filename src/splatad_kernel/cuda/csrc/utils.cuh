#ifndef SPLATAD_CUDA_UTILS_H
#define SPLATAD_CUDA_UTILS_H

// Only the device math that is specific to SplatAD's spherical LiDAR model
// lives here. Everything the camera path shares -- quat_to_rotmat,
// quat_scale_to_covar_preci, posW2C, covarW2C, add_blur, inverse_vjp and their
// VJPs -- comes from gsplat 1.5.3's Utils.cuh (pulled in via helpers.cuh)
// rather than being copied. See NOTICE.
//
// gsplat 1.5.3 has no 2x2 `inverse`, so that one stays below.

#include "helpers.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
inline __device__ void depth_compensation_from_cov3d(const glm::mat3 cov3d, const float eps2d, glm::vec2 &depth_compensation) {
    // cov3d is a 3x3 matrix [a1, a2, a3; b1, b2, b3; c1, c2, c3]
    // but glm::mat3 is column major
    // depth_compensation is a 2x1 vector [dc1; dc2]
    // in rowmajor terms: [invcov3d[2,0]/invcov3d[2,2]; invcov3d[2,1]/invcov3d[2,2]]
    // we can compute this by inverting the 3x3 matrix and taking the 3rd column
    // hence, we don't have to compute the full inverse
    float a1 = cov3d[0][0] + eps2d;
    float a2 = cov3d[1][0];
    float b1 = cov3d[0][1];
    float b2 = cov3d[1][1] + eps2d;
    float c1 = cov3d[0][2];
    float c2 = cov3d[1][2];

    float invD = 1/(a1*b2-a2*b1);
    depth_compensation = glm::vec2(
        (b1*c2-b2*c1) * invD,
        (a2*c1-a1*c2) * invD
    );
}

inline __device__ void depth_compensation_from_cov3d_vjp(const glm::mat3 cov3d, const float eps2d, const glm::vec2 v_depth_compensation, glm::mat3 &v_cov3d) {
    float a1 = cov3d[0][0] + eps2d;
    float a2 = cov3d[1][0];
    float b1 = cov3d[0][1];
    float b2 = cov3d[1][1] + eps2d;
    float c1 = cov3d[0][2];
    float c2 = cov3d[1][2];

    float invD = 1/(a1*b2-a2*b1);

    float dc1 = (b1*c2-b2*c1) * invD;
    float dc2 = (a2*c1-a1*c2) * invD;

    v_cov3d[0][0] += (v_depth_compensation[0] * (-dc1*b2) + v_depth_compensation[1] * (-c2-dc2*b2))*invD;
    v_cov3d[1][0] += (v_depth_compensation[0] * (dc1*b1) + v_depth_compensation[1] * (c1+dc2*b1))*invD;
    //v_cov3d[2][0] += 0.f;
    v_cov3d[0][1] += (v_depth_compensation[0] * (c2 + dc1*a2) + v_depth_compensation[1] * (dc2*a2))*invD;
    v_cov3d[1][1] += (v_depth_compensation[0] * (-c1 - dc1*a1) + v_depth_compensation[1] * (-dc2*a1))*invD;
    //v_cov3d[2][1] += 0.f;
    v_cov3d[0][2] += (v_depth_compensation[0] * (-b2) + v_depth_compensation[1] * (a2))*invD;
    v_cov3d[1][2] += (v_depth_compensation[0] * (b1) + v_depth_compensation[1] * (-a1))*invD;
    //v_cov3d[2][2] += 0.f;
}



inline __device__ void lidar_proj(
    const glm::vec3& __restrict__ mean3d,
    const glm::mat3& __restrict__ cov3d,
    const float eps2d,
    glm::vec2 &mean2d,
    glm::mat2 &cov2d,
    glm::vec2 &depth_compensation,
    glm::mat3 &jacobian
) {
    const float x2 = mean3d.x * mean3d.x;
    const float y2 = mean3d.y * mean3d.y;
    const float z2 = mean3d.z * mean3d.z;
    const float xz = mean3d.x * mean3d.z;
    const float yz = mean3d.y * mean3d.z;
    const float r2 = x2 + y2 + z2;
    const float rinv = rnorm3df(mean3d.x, mean3d.y, mean3d.z);
    const float sqrtx2y2 = hypotf(mean3d.x, mean3d.y);
    const float sqrtx2y2_inv = rhypotf(mean3d.x, mean3d.y);
    const float r2sqrtx2y2_inv = 1.f / (r2) * sqrtx2y2_inv;
    // column major
    // we only care about the top 2x2 submatrix
    jacobian = glm::mat3(
        -mean3d.y / (x2 + y2) * RAD_TO_DEG, -xz * r2sqrtx2y2_inv * RAD_TO_DEG, mean3d.x * rinv, // 1st column
        mean3d.x / (x2 + y2)  * RAD_TO_DEG, -yz * r2sqrtx2y2_inv * RAD_TO_DEG, mean3d.y * rinv, // 2nd column
        0.f                               , sqrtx2y2 / r2  * RAD_TO_DEG      , mean3d.z * rinv  // 3rd column
    );

    glm::mat3 cov3d_spherical = jacobian * cov3d * glm::transpose(jacobian);

    cov2d = glm::mat2(cov3d_spherical[0][0], cov3d_spherical[0][1],
                      cov3d_spherical[1][0], cov3d_spherical[1][1]);

    mean2d = glm::vec2(
        glm::atan(mean3d.y, mean3d.x) * RAD_TO_DEG, glm::asin(mean3d.z * rinv) * RAD_TO_DEG
    );

    depth_compensation_from_cov3d(cov3d_spherical, eps2d, depth_compensation);
}

inline __device__ void lidar_proj_vjp(
    // fwd inputs
    const glm::vec3 mean3d, const glm::mat3 cov3d, const float eps2d,
    // grad outputs
    const glm::vec2 v_mean2d, const glm::mat2 v_cov2d, const glm::vec2 v_depth_compensation,
    // grad inputs
    glm::vec3 &v_mean3d, glm::mat3 &v_cov3d) {

    const float x2 = mean3d.x * mean3d.x;
    const float y2 = mean3d.y * mean3d.y;
    const float z2 = mean3d.z * mean3d.z;
    const float r2 = x2 + y2 + z2;
    const float rinv = rnorm3df(mean3d.x, mean3d.y, mean3d.z);
    const float sqrtx2y2 = hypotf(mean3d.x, mean3d.y);
    const float sqrtx2y2_inv = rhypotf(mean3d.x, mean3d.y);
    const float xy = mean3d.x * mean3d.y;
    const float xz = mean3d.x * mean3d.z;
    const float yz = mean3d.y * mean3d.z;
    const float r2sqrtx2y2_inv = 1 / (r2) * sqrtx2y2_inv;

    // column major, mat3x2 is 3 columns x 2 rows.
    glm::mat3x2 J = glm::mat3x2(
        -mean3d.y / (x2 + y2), -xz * r2sqrtx2y2_inv, // 1st column
        mean3d.x / (x2 + y2) , -yz * r2sqrtx2y2_inv, // 2nd column
        0.f                  , sqrtx2y2 / r2     // 3rd column
    ) * RAD_TO_DEG;

    glm::mat3 J3x3 = glm::mat3(
        J[0][0], J[0][1], mean3d.x * rinv,
        J[1][0], J[1][1], mean3d.y * rinv,
        J[2][0], J[2][1], mean3d.z * rinv
    );

    glm::mat3 cov3d_spherical = J3x3 * cov3d * glm::transpose(J3x3);
    glm::mat3 v_cov3d_spherical(0.f);
    depth_compensation_from_cov3d_vjp(cov3d_spherical, eps2d, v_depth_compensation, v_cov3d_spherical);
    v_cov3d += glm::transpose(J3x3) * v_cov3d_spherical * J3x3;

    // cov = J * V * Jt; G = df/dcov = v_cov
    // -> df/dV = Jt * G * J
    // -> df/dJ = G * J * Vt + Gt * J * V
    v_cov3d += glm::transpose(J) * v_cov2d * J;

    v_mean3d += v_mean2d * J;
    // glm::vec3(
    //     -mean3d.y/(x2y2)*v_mean2d.x - mean3d.x*mean3d.z/r2sqrtx2y2*v_mean2d.y,
    //     mean3d.x/x2y2 * v_mean2d.x - mean3d.y*mean3d.z/r2sqrtx2y2*v_mean2d.y,
    //     sqrtx2y2/r2*v_mean2d.y
    // );

    const float x2plusy2 = x2 + y2;
    const float x2plusy2squared = x2plusy2 * x2plusy2;
    const float x2plusy2sqrt3andahalf = sqrtx2y2 * x2plusy2;
    const float x4 = x2 * x2;
    const float y4 = y2 * y2;
    const float r4 = r2 * r2;
    const float xyz = mean3d.x * mean3d.y * mean3d.z;
    const float denom1 = 1.f / (x2plusy2sqrt3andahalf * r4);
    const float denom2 = 1.f / r4 * sqrtx2y2_inv;
    const float r3_inv = 1.f / r2 * rinv;

    glm::mat3x2 v_J =
        v_cov2d * J * glm::transpose(cov3d) + glm::transpose(v_cov2d) * J * cov3d;

    glm::mat3 v_J3x3 = v_cov3d_spherical * J3x3 * glm::transpose(cov3d) + glm::transpose(v_cov3d_spherical) * J3x3 * cov3d;
    v_mean3d.x += (
        2.f * mean3d.x * mean3d.y / x2plusy2squared * (v_J[0][0] + v_J3x3[0][0])
        + (y2 - x2) / x2plusy2squared * (v_J[1][0] + v_J3x3[1][0])
        + 0.f * (v_J[2][0] + v_J3x3[2][0])
        + mean3d.z * (2.f*x4 + x2*y2 - y2*(y2 + z2)) * denom1 * (v_J[0][1] + v_J3x3[0][1])
        + xyz * (3.f*x2 + 3.f*y2 + z2) * denom1 * (v_J[1][1] + v_J3x3[1][1])
        - mean3d.x * (x2 + y2 - z2) * denom2 * (v_J[2][1]+v_J3x3[2][1])
        ) * RAD_TO_DEG;
        v_mean3d.x += (
            (y2 + z2) * r3_inv * v_J3x3[0][2]
            - xy * r3_inv * v_J3x3[1][2]
            - xz * r3_inv * v_J3x3[2][2]);

    v_mean3d.y += (
        (y2 - x2) / x2plusy2squared * (v_J[0][0] + v_J3x3[0][0])
        - 2.f * mean3d.x * mean3d.y / x2plusy2squared * (v_J[1][0] + v_J3x3[1][0])
        + 0.f * (v_J[2][0] + v_J3x3[2][0])
        + xyz * (3.f*x2 + 3.f*y2 + z2) * denom1 * (v_J[0][1] + v_J3x3[0][1])
        - mean3d.z * (x4 + x2*(z2-y2)-2.f*y4) * denom1 * (v_J[1][1] + v_J3x3[1][1])
        - mean3d.y * (x2 + y2 - z2) * denom2 * (v_J[2][1] + v_J3x3[2][1])
        ) * RAD_TO_DEG;
    v_mean3d.y += (
        - xy * r3_inv * v_J3x3[0][2]
        + (x2 + z2) * r3_inv * v_J3x3[1][2]
        - yz * r3_inv * v_J3x3[2][2]
    );

    v_mean3d.z += (
        0.f * (v_J[0][0] + v_J3x3[0][0])
        + 0.f * (v_J[1][0] + v_J3x3[1][0])
        + 0.f * (v_J[2][0] + v_J3x3[2][0])
        - mean3d.x * (x2 + y2 -z2) * denom2 * (v_J[0][1] + v_J3x3[0][1])
        - mean3d.y * (x2 + y2 -z2) * denom2 * (v_J[1][1] + v_J3x3[1][1])
        - 2.f*mean3d.z * sqrtx2y2 / r4 * (v_J[2][1] + v_J3x3[2][1])
        ) * RAD_TO_DEG;
    v_mean3d.z += (
        - xz * r3_inv * v_J3x3[0][2]
        - yz * r3_inv * v_J3x3[1][2]
        + (x2 + y2) * r3_inv * v_J3x3[2][2]
    );

}

inline __device__ void vel_world_to_cam(
    // [R] is the world-to-camera rotation
    const glm::mat3 R, const glm::vec3 vel, glm::vec3 &vel_c) {
    vel_c = R * vel;
}

inline __device__ void vel_world_to_cam_vjp(
    // fwd inputs
    const glm::mat3 R, const glm::vec3 vel,
    // grad outputs
    const glm::vec3 v_vel_c,
    // grad inputs
    glm::mat3 &v_R, glm::vec3 &v_vel) {
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    v_R += glm::outerProduct(v_vel_c, vel);
    v_vel += glm::transpose(R) * v_vel_c;
}


inline __device__ float inverse(const glm::mat2 M, glm::mat2 &Minv) {
    float det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    if (det <= 0.f) {
        return det;
    }
    float invDet = 1.f / det;
    Minv[0][0] = M[1][1] * invDet;
    Minv[0][1] = -M[0][1] * invDet;
    Minv[1][0] = Minv[0][1];
    Minv[1][1] = M[0][0] * invDet;
    return det;
}


inline __device__ float angle_difference(const float angle1, const float angle2) {
    // splatsim: both angles are panorama coordinates in [-180, 180] deg, so their
    // difference is in [-360, 360]; wrap to (-180, 180] with two branchless
    // offsets instead of the expensive fmodf. This is the LiDAR raster inner-loop
    // hot path (called twice per Gaussian per ray).
    float diff = angle1 - angle2;
    diff -= 360.f * static_cast<float>(diff > 180.f);
    diff += 360.f * static_cast<float>(diff < -180.f);
    return diff;
}


#endif // SPLATAD_CUDA_UTILS_H
