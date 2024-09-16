#ifndef RANSAC_HH
#define RANSAC_HH

#include <nova/vec.h>

#include <vector>


struct RANSAC_diffs {
    std::size_t num_inliers;
    std::vector<float> distances;
    std::vector<bool> is_inliers;
};


/*
 * @brief   Calculate the cylinder-point differences
 *
 * TODO: generalize
 */
[[nodiscard]] auto calculate_RANSAC_diffs(const auto& points, const nova::Vec4f& cylinder, float threshold)
        -> RANSAC_diffs
{
    const nova::Vec3f S0 { cylinder.x(), cylinder.y(), cylinder.z() };
    const float r = cylinder.w();

    std::size_t num_inliers = 0;
    std::vector<float> distances;
    std::vector<bool> is_inliers;

    for (const auto& point : points) {
        const auto tmp_v3 = point - S0;
        const auto tmp_v2 = nova::Vec2f { tmp_v3.x(), tmp_v3.y() };
        const float dist = std::abs(tmp_v2.length() - r);

        distances.push_back(dist);
        is_inliers.push_back(dist < threshold);
        num_inliers += dist < threshold ? 1 : 0;
    }

    return {
        .num_inliers = num_inliers,
        .distances = distances,
        .is_inliers = is_inliers,
    };
}

struct RANSAC_diffs_cuda {
    std::size_t num_inliers;
    std::vector<float> distances;
    std::vector<char> is_inliers;
};

// Declaration of the CUDA wrapper function
auto calculate_RANSAC_diffs_cuda(
    const nova::Vec3f* points,
    std::size_t points_size,
    const nova::Vec4f& cylinder,
    float threshold
)
        -> RANSAC_diffs_cuda;

#endif // RANSAC_HH
