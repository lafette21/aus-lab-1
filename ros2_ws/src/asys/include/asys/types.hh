#ifndef TYPES_HH
#define TYPES_HH

#include <Eigen/Core>

#include <vector>


struct trafo_2d {
    Eigen::Matrix2f R;
    Eigen::Vector2f t;
};

struct RANSAC_diffs {
    std::size_t num_inliers;
    std::vector<float> distances;
    std::vector<bool> is_inliers;
};

#endif // TYPES_HH
