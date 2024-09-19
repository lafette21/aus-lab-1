#ifndef UTILS_HH
#define UTILS_HH

#include "types.hh"

#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <nova/utils.h>
#include <nova/vec.h>
#include <range/v3/range/conversion.hpp>

#include <random>
#include <ranges>


// TODO(refact): Nova to handle normal_distribution beside uniform
[[nodiscard]] inline auto random_noise(float sigma) {
    std::random_device rd{};
    std::mt19937 gen{ rd() };

    // values near the mean are the most likely
    // standard deviation affects the dispersion of generated values from the mean
    std::normal_distribution distribution{ 0.f, sigma };

    return distribution(gen);
}

template <std::floating_point T = float>
auto interpolate_and_differentiate(const std::vector<T>& x, const std::vector<T>& y, std::size_t num_points = 100)
        -> std::array<std::vector<T>, 6>
{
    assert(x.size() == y.size());

    std::vector<double> _x;
    std::vector<double> _y;

    if constexpr (std::is_same_v<T, double>) {
        _x = x;
        _y = y;
    } else {
        _x = x
           | ranges::to<std::vector<double>>();
        _y = y
           | ranges::to<std::vector<double>>();
    }

    gsl_interp_accel* acc = gsl_interp_accel_alloc();
    gsl_spline* spline_x = gsl_spline_alloc(gsl_interp_cspline, _x.size());
    gsl_spline* spline_y = gsl_spline_alloc(gsl_interp_cspline, _y.size());

    const auto t = nova::linspace<double>(nova::range<double>{ 0, static_cast<double>(_x.size()) }, _x.size(), false);

    gsl_spline_init(spline_x, t.data(), _x.data(), t.size());
    gsl_spline_init(spline_y, t.data(), _y.data(), t.size());

    std::vector<T> ret_x;
    std::vector<T> ret_y;
    std::vector<T> ret_dx;
    std::vector<T> ret_dy;
    std::vector<T> ret_ddx;
    std::vector<T> ret_ddy;

    ret_x.reserve(_x.size());
    ret_y.reserve(_y.size());
    ret_dx.reserve(_x.size());
    ret_dy.reserve(_y.size());
    ret_ddx.reserve(_x.size());
    ret_ddy.reserve(_y.size());

    double t_max = t.back();
    double step = t_max / (static_cast<double>(num_points) - 1.0);

    for (std::size_t i = 0; i < num_points; ++i) {
        double ti = step * static_cast<double>(i);
        ret_x.push_back(static_cast<T>(gsl_spline_eval(spline_x, ti, acc)));
        ret_y.push_back(static_cast<T>(gsl_spline_eval(spline_y, ti, acc)));
        ret_dx.push_back(static_cast<T>(gsl_spline_eval_deriv(spline_x, ti, acc)));
        ret_dy.push_back(static_cast<T>(gsl_spline_eval_deriv(spline_y, ti, acc)));
        ret_ddx.push_back(static_cast<T>(gsl_spline_eval_deriv2(spline_x, ti, acc)));
        ret_ddy.push_back(static_cast<T>(gsl_spline_eval_deriv2(spline_y, ti, acc)));
    }

    gsl_spline_free(spline_x);
    gsl_spline_free(spline_y);
    gsl_interp_accel_free(acc);

    return {
        ret_x,
        ret_y,
        ret_dx,
        ret_dy,
        ret_ddx,
        ret_ddy
    };
}

float deg2rad(float degrees) {
    return degrees * (std::numbers::pi_v<float> / 180.0f);
}

auto calc_normal_vec(const std::vector<nova::Vec3f>& points)
        -> nova::Vec3f
{
    const auto point1 = points[0];
    const auto point2 = points[1];
    const auto point3 = points[2];

    // Calculate two vectors lying on the plane
    const auto vec1 = point2 - point1;
    const auto vec2 = point3 - point1;

    // Calculate the cross product of the two vectors to find the normal vector
    auto normal_vec = nova::cross(vec1, vec2);

    // Normalize the normal vector
    normal_vec /= normal_vec.length();

    return normal_vec;
}

auto closest_to(const nova::Vec3f& point, const std::vector<nova::Vec3f>& points)
        -> nova::Vec3f
{
    nova::Vec3f closest;
    auto min_dist = std::numeric_limits<float>::max();

    for (const auto& p : points) {
        if (const auto dist = (point - p).length(); dist < min_dist) {
            min_dist = dist;
            closest = p;
        }
    }

    return closest;
}

std::optional<hit_record> hit(const cylinder& cyl, const ray& r) {
    // Translate the line and cylinder center to the origin
    const auto point_cyl = r.origin - cyl.center;

    // Calculate the direction vector of the line in the cylinder coordinate system
    const auto vec_cyl = r.direction - cyl.axis * nova::dot(r.direction, cyl.axis);

    // Calculate the coefficients for the quadratic equation
    const auto a = vec_cyl.x() * vec_cyl.x() + vec_cyl.y() * vec_cyl.y();
    const auto b = 2 * (point_cyl.x() * vec_cyl.x() + point_cyl.y() * vec_cyl.y());
    const auto c = point_cyl.x() * point_cyl.x() + point_cyl.y() * point_cyl.y() - cyl.radius * cyl.radius;

    // Calculate the discriminant
    const auto discriminant = b * b - 4 * a * c;

    // Check if there are any real roots (intersection points)
    if (discriminant < 0) {
        return std::nullopt;  // No intersection
    }

    // Two intersection points
    // const auto t1 = (-b + std::sqrt(discriminant)) / (2 * a);
    const auto t2 = (-b - std::sqrt(discriminant)) / (2 * a);

    hit_record ret;

    // ret.t = r.at(t1).length() < r.at(t2).length() ? t1 : t2;
    ret.t = t2;
    ret.point = r.at(ret.t);
    ret.normal = (ret.point - cyl.center) / cyl.radius;

    return ret;
}

std::optional<hit_record> hit(const plane& plane, const ray& r) {
    const auto points = std::vector<nova::Vec3f>{ plane.p0, plane.p1, plane.p2, plane.p3 };
    const auto max_x = std::ranges::max(points, [](const auto& lhs, const auto& rhs) { return lhs.x() < rhs.x(); }).x();
    const auto max_y = std::ranges::max(points, [](const auto& lhs, const auto& rhs) { return lhs.y() < rhs.y(); }).y();
    const auto max_z = std::ranges::max(points, [](const auto& lhs, const auto& rhs) { return lhs.z() < rhs.z(); }).z();
    const auto min_x = std::ranges::min(points, [](const auto& lhs, const auto& rhs) { return lhs.x() < rhs.x(); }).x();
    const auto min_y = std::ranges::min(points, [](const auto& lhs, const auto& rhs) { return lhs.y() < rhs.y(); }).y();
    const auto min_z = std::ranges::min(points, [](const auto& lhs, const auto& rhs) { return lhs.z() < rhs.z(); }).z();
    const auto plane_normal = calc_normal_vec(points);
    const auto plane_point = plane.p3;

    // Calculate the dot product of the plane normal and the line direction vector
    const auto dot_prod = nova::dot(plane_normal, r.direction);

    // Check if the line is parallel to the plane
    if (std::abs(dot_prod) < 1e-6f) {
        return {};
    }

    // Calculate the vector from a point on the plane to the line's point of origin
    const auto plane_to_line = r.origin - plane_point;

    // Calculate the distance along the line to the intersection point
    const auto t = -nova::dot(plane_to_line, plane_normal) / dot_prod;

    // Calculate the intersection point
    const auto intersection = r.at(t);

    // Check if the intersection point lies outside the bounds of the finite plane
    if (not (min_x <= intersection.x() and intersection.x() <= max_x
        and min_y <= intersection.y() and intersection.y() <= max_y
        and min_z <= intersection.z() and intersection.z() <= max_z)) {
        return std::nullopt;
    }

    hit_record ret;

    ret.t = t;
    ret.point = intersection;
    // ret.normal

    return ret;
}

std::optional<hit_record> hit(const sphere& sphere, const ray& r) {
    const auto x = r.origin - sphere.center;
    const auto a = nova::dot(r.direction, r.direction);
    const auto b = nova::dot(x, r.direction) * 2.f;
    const auto c = nova::dot(x, x) - sphere.radius * sphere.radius;
    const auto discriminant = b * b - 4 * a * c;

    if (discriminant < 0) {
        return std::nullopt;
    }

    hit_record ret;
    ret.t = (-b - std::sqrt(discriminant)) / (2.f * a);
    ret.point = r.at(ret.t);
    ret.normal = (ret.point - sphere.center) / sphere.radius;

    return ret;
}

auto distort(const nova::Vec3f& point, const ray& r, float sigma)
        -> nova::Vec3f
{
    const auto dist = (point - r.origin).length();
    const auto noise = random_noise(sigma);
    const auto distorted_dist = dist + noise;

    return r.origin + r.direction * distorted_dist;
}

auto ray_cast(const ray& r, const std::vector<primitive>& primitives, float sigma)
        -> std::vector<nova::Vec3f>
{
    std::vector<nova::Vec3f> ret;
    ret.reserve(primitives.size());

    for (const auto& elem : primitives) {
        const std::optional<hit_record> hit_rec = std::visit(
            lambdas{
                [&r](const sphere& p)   { return hit(p, r); },
                [&r](const cylinder& p) { return hit(p, r); },
                [&r](const plane& p)    { return hit(p, r); },
            },
            elem
        );
        if (hit_rec.has_value()) {
            const auto& hit_point = hit_rec->point;
            // Filter out points behind the lidar
            if (nova::dot(hit_point - r.origin, r.direction) >= 0) {
                ret.push_back(distort(hit_point, r, sigma));
            }
        }
    }

    auto filtered = ret
                  | std::views::filter([point = r.origin](const auto& x) { return (point - x).length() <= 100; })
                  | ranges::to<std::vector>();

    return filtered;
}

template <>
struct fmt::formatter<nova::Vec3f> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template <typename ParseContext>
    auto format(const nova::Vec3f& obj, ParseContext& ctx) const {
        return fmt::format_to(ctx.out(), "{{ {}, {}, {} }}", obj.x(), obj.y(), obj.z());
    }
};

#endif // UTILS_HH
