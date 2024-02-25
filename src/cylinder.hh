#ifndef CYLINDER_HH
#define CYLINDER_HH

#include "ransac.hh"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <fmt/core.h>
#include <nova/vec.h>
#include <nova/random.h>
#include <nova/types.h>

#include <cmath>
#include <ranges>
#include <iostream>


[[nodiscard]] auto estimate_cylinder_min(const auto& points)
        -> nova::Vec4f
{
    assert(points.size() == 3);

    std::array<nova::Vec3f, 3> midP;
    std::array<nova::Vec3f, 3> n;
    Eigen::MatrixXf nM(3, 2);
    Eigen::MatrixXf d(3, 1);
    Eigen::MatrixXf S0(2, 1);

    for (std::size_t i = 0; i < 3; ++i) {
        midP[i] = (points[i] + points[(i + 1) % 3]) / 2;
        n[i] = points[i] - midP[i];
        // std::cout << "pre " << n[i].x() << " " << n[i].y() << std::endl;
        // FIXME: This is probably just a workaround
        if (nova::Vec2f { n[i].x(), n[i].y() }.length() != 0) {
            n[i] /= nova::Vec2f { n[i].x(), n[i].y() }.length();
        }
        // std::cout << "post " << n[i].x() << " " << n[i].y() << std::endl;
        d(static_cast<int>(i) ,0) = n[i].x() * midP[i].x() + n[i].y() * midP[i].y();
        nM(static_cast<int>(i), 0) = n[i].x();
        nM(static_cast<int>(i), 1) = n[i].y();
    }

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(nM, Eigen::ComputeThinU | Eigen::ComputeThinV);
    S0 = svd.solve(d);

    const auto r = nova::Vec2f { S0(0, 0) - points[0].x(), S0(1, 0) - points[0].y() };

    return {
        S0(0, 0),
        S0(1, 0),
        points[0].z(),
        r.length(),
    };
}

[[nodiscard]] auto estimate_cylinder_lsq(const auto& points, const nova::Vec4f& params_init)
        -> nova::Vec4f
{
    constexpr auto Epsilon = 1e-6f;
    constexpr auto MaxIter = 1000;
    const auto points_size = points.size();
    const auto sum = std::accumulate(std::begin(points), std::end(points), nova::Vec3f(0, 0, 0), std::plus<nova::Vec3f>());
    const auto mean = sum / static_cast<float>(points_size);
    const auto pp = points
                  | std::views::transform([mean](const auto& elem) { return elem - mean; });
    auto S0 = nova::Vec2f(params_init.x() - mean.x(), params_init.y() - mean.y());
    float r = params_init.w();
    auto diff = nova::Vec2f { 1, 1 };
    std::size_t it = 0;

    // TODO: Here I changed the equation from `diff != Point2f{0,0}`
    while (diff.length() < Epsilon && it < MaxIter) {
        float r_avg = 0;
        auto dir_avg = nova::Vec2f { 0, 0 };
        auto S0_tmp = S0;

        for (const auto& point : pp) {
            auto dir_i = S0 - nova::Vec2f(point.x(), point.y());
            float r_i = dir_i.length();
            dir_avg += dir_i / r_i;
            r_avg += r_i;
        }

        S0 = (dir_avg * r_avg) / static_cast<float>(std::pow(points_size, 2));
        r = r_avg / static_cast<float>(points_size);
        diff = S0 - S0_tmp;
        ++it;
    }

    return nova::Vec4f(
        S0.x() + mean.x(),
        S0.y() + mean.y(),
        mean.z(),
        r
    );
};

[[nodiscard]] auto estimate_cylinder_RANSAC(const auto& points, float threshold, std::size_t iter)
        -> nova::Vec4f
{
    constexpr auto RMax = 1.0f;
    constexpr auto RMin = 0.1f;
    const auto points_size = points.size();
    std::size_t best_sample_inlier_num = 0;
    nova::Vec4f best_cylinder;

    fmt::println("RANDOM ENGINE SEED: {}", nova::random().seed());

    for (std::size_t i = 0; i < iter; ++i) {
        const auto p1 = nova::random().choice(points);
        auto p2 = nova::random().choice(points);
        while (p1 == p2) { p2 = nova::random().choice(points); }
        auto p3 = nova::random().choice(points);
        while (p1 == p3) { p3 = nova::random().choice(points); }

        const auto min_sample = std::vector<nova::Vec3f> { p1, p2, p3 };
        const auto sample_cylinder = estimate_cylinder_min(min_sample);
        const auto sample_result = calculate_RANSAC_diffs(points, sample_cylinder, threshold);

        if (sample_result.num_inliers > best_sample_inlier_num
            && sample_cylinder.w() < RMax
            && sample_cylinder.w() < RMin
        ) {
            best_sample_inlier_num = sample_result.num_inliers;
            best_cylinder = sample_cylinder;
            fmt::println("Inlier num. update: {}\t\tradius: {}", best_sample_inlier_num, best_cylinder.w());
        }
    }

    // Finally, the cylinder is refitted from the best consensus set
    const auto best_result = calculate_RANSAC_diffs(points, best_cylinder, threshold);

    std::vector<nova::Vec3f> inliers;

    for (std::size_t i = 0; i < points_size; ++i) {
        if (best_result.is_inliers[i]) {
            inliers.push_back(points[i]);
        }
    }

    return estimate_cylinder_lsq(inliers, best_cylinder);
}

#endif // CYLINDER_HH
