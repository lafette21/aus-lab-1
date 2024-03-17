#include "types.hh"

#include <fmt/chrono.h>
#include <fmt/format.h>
#include <nova/io.h>
#include <nova/json.h>
#include <nova/vec.h>
#include <nova/utils.h>
#include <spdlog/spdlog.h>

#include <fstream>
#include <numbers>
#include <ranges>
#include <span>
#include <string>
#include <vector>

using json = nova::json;


inline auto& init(const std::string& name) {
    spdlog::set_default_logger(spdlog::create<spdlog::sinks::ansicolor_stdout_sink_mt>(name));

    auto& logger = *spdlog::get(name);
    logger.set_pattern("[%Y-%m-%d %H:%M:%S.%f %z] [%n @%t] %^[%l]%$ %v");

    return logger;
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

auto line_plane_intersection(
    const nova::Vec3f& point,
    const nova::Vec3f& vec,
    const nova::Vec3f& plane_normal,
    const nova::Vec3f& plane_point,
    const nova::Vec3f& min,
    const nova::Vec3f& max
)
        -> std::vector<nova::Vec3f>
{
    // Calculate the dot product of the plane normal and the line direction vector
    const auto dot_prod = nova::dot(plane_normal, vec);

    // Check if the line is parallel to the plane
    if (std::abs(dot_prod) < 1e-6f) {
        return {};
    }

    // Calculate the vector from a point on the plane to the line's point of origin
    const auto plane_to_line = point - plane_point;

    // Calculate the distance along the line to the intersection point
    const auto t = -nova::dot(plane_to_line, plane_normal) / dot_prod;

    // Calculate the intersection point
    const auto intersection = point + vec * t;

    // Check if the intersection point lies outside the bounds of the finite plane
    if (not (min.x() <= intersection.x() and intersection.x() <= max.x()
        and min.y() <= intersection.y() and intersection.y() <= max.y()
        and min.z() <= intersection.z() and intersection.z() <= max.z())) {
        return {};
    }

    return { intersection };
}

auto line_cylinder_intersection(
    const nova::Vec3f& point,
    const nova::Vec3f& vec,
    const nova::Vec3f& center,
    const nova::Vec3f& axis,
    float radius
)
        -> std::vector<nova::Vec3f>
{
    // Translate the line and cylinder center to the origin
    const auto point_cyl = point - center;

    // Calculate the direction vector of the line in the cylinder coordinate system
    const auto vec_cyl = vec - axis * nova::dot(vec, axis);

    // Calculate the coefficients for the quadratic equation
    const auto a = vec_cyl.x() * vec_cyl.x() + vec_cyl.y() * vec_cyl.y();
    const auto b = 2 * (point_cyl.x() * vec_cyl.x() + point_cyl.y() * vec_cyl.y());
    const auto c = point_cyl.x() * point_cyl.x() + point_cyl.y() * point_cyl.y() - radius * radius;

    // Calculate the discriminant
    const auto discriminant = b * b - 4 * a * c;

    // Check if there are any real roots (intersection points)
    if (discriminant < 0) {
        return {};  // No intersection
    } else if (discriminant == 0) {
        // One intersection point
        const auto t = -b / (2 * a);

        return {
            point + vec * t
        };
    }

    // Two intersection points
    const auto t1 = (-b + std::sqrt(discriminant)) / (2 * a);
    const auto t2 = (-b - std::sqrt(discriminant)) / (2 * a);

    return {
        point + vec * t1,
        point + vec * t2
    };
}

auto intersections(const nova::Vec3f& point, const nova::Vec3f& vec, const std::pair<object_type, std::vector<nova::Vec3f>>& obj)
        -> std::vector<nova::Vec3f>
{
    std::vector<nova::Vec3f> result;

    if (obj.first == object_type::plane) {
        const auto points = obj.second;
        // TODO: This shouldn't be in a hot-loop
        const auto max_x = std::ranges::max(points, [](const auto& lhs, const auto& rhs) { return lhs.x() < rhs.x(); }).x();
        const auto max_y = std::ranges::max(points, [](const auto& lhs, const auto& rhs) { return lhs.y() < rhs.y(); }).y();
        const auto max_z = std::ranges::max(points, [](const auto& lhs, const auto& rhs) { return lhs.z() < rhs.z(); }).z();
        const auto min_x = std::ranges::min(points, [](const auto& lhs, const auto& rhs) { return lhs.x() < rhs.x(); }).x();
        const auto min_y = std::ranges::min(points, [](const auto& lhs, const auto& rhs) { return lhs.y() < rhs.y(); }).y();
        const auto min_z = std::ranges::min(points, [](const auto& lhs, const auto& rhs) { return lhs.z() < rhs.z(); }).z();
        result = line_plane_intersection(point, vec, calc_normal_vec(points), points[3], { min_x, min_y, min_z }, { max_x, max_y, max_z });
    } else if (obj.first == object_type::cylinder) {
        result = line_cylinder_intersection(point, vec, obj.second[0], { 0, 0, 1 }, 0.35f);
    }
    auto filtered = result
                  | std::views::filter([point](const auto& x) { return (point - x).length() <= 100; });

    return std::vector(std::begin(filtered), std::end(filtered));
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

class lidar {
public:
    lidar(const json& config, const auto& objects)
        : m_config(config), m_objects(objects)
    {
        m_ang_res_h = m_config.lookup<float>("rpm.value") / 60 * 360 * m_config.lookup<float>("firing_cycle");
        m_angles_hor = nova::linspace(nova::range{ 0.f, m_config.lookup<float>("fov_h") }, static_cast<std::size_t>(m_config.lookup<float>("fov_h") / m_ang_res_h), false);
        m_angles_ver = nova::linspace(nova::range{ -m_config.lookup<float>("fov_v") / 2, m_config.lookup<float>("fov_v") / 2 }, m_config.lookup<std::size_t>("channels"), true);
    }

    auto string() {
        return fmt::format("lidar(m_config={}, m_ang_res_h={})", m_config.dump(), m_ang_res_h);
    }

    auto data() { return m_data; }

    void start() {
        m_running = true;

        while (m_running) {
            m_data = rotation();

            break;
        }
    }

    void stop() {
        m_running = false;
    }

private:
    json m_config;
    float m_ang_res_h;
    std::vector<float> m_angles_hor;
    std::vector<float> m_angles_ver;
    std::vector<nova::Vec3f> m_data;
    std::vector<std::pair<object_type, std::vector<nova::Vec3f>>> m_objects;
    nova::Vec3f m_origin = { 0, 0, 1.5 };
    bool m_running = false;

    std::vector<nova::Vec3f> rotation() {
        std::vector<nova::Vec3f> result;
        result.reserve(m_angles_ver.size() * m_angles_hor.size());

        const auto start = nova::now();

        for (auto angle_h : m_angles_hor) {
            angle_h = deg2rad(angle_h);
            for (auto angle_v : m_angles_ver) {
                angle_v = deg2rad(angle_v);

                // Define the point on the line (Origin)
                const auto p0 = m_origin;

                const auto v = nova::Vec3f{
                    std::cos(angle_v) * std::sin(angle_h),
                    std::cos(angle_v) * std::cos(angle_h),
                    std::sin(angle_v)
                };

                for (const auto& obj : m_objects) {
                    const auto points = intersections(p0, v, obj);
                    if (points.size() > 0) {
                        result.push_back(closest_to(p0, points));
                    }
                }
            }
        }

        const auto end = nova::now();
        const auto took = end - start;
        fmt::println("rotation took: {}", took);

        return result;
    }
};

int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[]) {
    [[maybe_unused]] auto& logger = init("logger");

    const auto args = std::span<char*>(argv, static_cast<std::size_t>(argc))
                    | std::views::transform([](const auto& arg) { return std::string_view{ arg }; });

    const auto objects = nova::read_file<map_parser>(std::filesystem::path(args[2]).string()).value();

    // for (const auto& [type, data] : objects) {
        // fmt::println("{}", std::to_underlying<object_type>(type));
    // }

    json config(nova::read_file(std::filesystem::path(args[1]).string()).value());
    lidar lidar { config.at("vlp_16"), objects };

    spdlog::info(lidar.string());

    lidar.start();

    std::ofstream oF("./l_out.xyz");

    for (const auto& d : lidar.data()) {
        oF << fmt::format("{} {} {} 0 0 0\n", d.x(), d.y(), d.z());
    }

    return EXIT_SUCCESS;
}
