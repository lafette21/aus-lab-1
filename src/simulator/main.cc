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
    const auto t1 = (-b + std::sqrt(discriminant)) / (2 * a);
    const auto t2 = (-b - std::sqrt(discriminant)) / (2 * a);

    hit_record ret;

    ret.t = r.at(t1).length() < r.at(t2).length() ? t1 : t2;
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

auto ray_cast(const ray& r, const std::vector<primitive>& primitives)
        -> std::vector<nova::Vec3f>
{
    std::vector<nova::Vec3f> ret;
    ret.reserve(primitives.size());

    for (const auto& elem : primitives) {
        const std::optional<hit_record> hit_rec = std::visit(lambdas{
                [&r](const cylinder& p) { return hit(p, r); },
                [&r](const plane& p)    { return hit(p, r); },
            },
            elem
        );
        if (hit_rec.has_value()) {
            const auto& hit_point = hit_rec->point;
            // Filter out points behind the lidar
            if (nova::dot(hit_point - r.origin, r.direction) >= 0) {
                ret.push_back(hit_point);
            }
        }
    }

    auto filtered = ret
                  | std::views::filter([point = r.origin](const auto& x) { return (point - x).length() <= 100; });

    return std::vector(std::begin(filtered), std::end(filtered));
}

class lidar {
public:
    lidar(const json& config)
        : m_config(config)
    {
        m_ang_res_h = m_config.lookup<float>("rpm.value") / 60 * 360 * m_config.lookup<float>("firing_cycle");
        m_angles_hor = nova::linspace(nova::range{ 0.f, m_config.lookup<float>("fov_h") }, static_cast<std::size_t>(m_config.lookup<float>("fov_h") / m_ang_res_h), false);
        m_angles_ver = nova::linspace(nova::range{ -m_config.lookup<float>("fov_v") / 2, m_config.lookup<float>("fov_v") / 2 }, m_config.lookup<std::size_t>("channels"), true);
    }

    auto string() {
        return fmt::format("lidar(m_config={}, m_ang_res_h={})", m_config.dump(), m_ang_res_h);
    }

    auto data() { return m_data; }

    auto scan(const auto& objects) {
        std::vector<nova::Vec3f> result;
        result.reserve(m_angles_ver.size() * m_angles_hor.size());

        const auto start = nova::now();

        for (auto angle_h : m_angles_hor) {
            angle_h = deg2rad(angle_h);
            for (auto angle_v : m_angles_ver) {
                angle_v = deg2rad(angle_v);

                const auto direction = nova::Vec3f{
                    std::cos(angle_v) * std::sin(angle_h),
                    std::cos(angle_v) * std::cos(angle_h),
                    std::sin(angle_v)
                };

                result.push_back(closest_to(m_origin, ray_cast(ray{ m_origin, direction }, objects)));
            }
        }

        const auto end = nova::now();
        const auto took = end - start;
        fmt::println("rotation took: {}", took);

        return result;
    }

    void replace(const nova::Vec3f& pos) {
        m_origin = pos;
    }

    void shift(const nova::Vec3f& t) {
        m_origin += t;
    }

private:
    json m_config;
    float m_ang_res_h;
    std::vector<float> m_angles_hor;
    std::vector<float> m_angles_ver;
    std::vector<nova::Vec3f> m_data;
    nova::Vec3f m_origin = { 0, 0, 1.5 };
};

void print(const std::string& path, const auto& data) {
    std::ofstream oF(path);

    for (const auto& d : data) {
        oF << fmt::format("{} {} {} 0 0 0\n", d.x(), d.y(), d.z());
    }
}

template <typename Lidar>
class vehicle {
public:
    vehicle(const json& config, const auto& objects)
        : m_config(config)
        , m_objects(objects)
        , m_lidar(m_config.at("lidar.vlp_16"))
    {
        spdlog::info(m_lidar.string());
    }

    auto func() {
        m_lidar.replace({ -5, 0, 1.5 });

        for (std::size_t i = 0; i < 10; ++i) {
            print(fmt::format("./out/l_out_{}.xyz", i + 1), m_lidar.scan(m_objects));
            m_lidar.shift({ 1, 0, 0 });
        }
    }

private:
    json m_config;
    std::vector<primitive> m_objects;
    Lidar m_lidar;
};

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    [[maybe_unused]] auto& logger = init("simulator");

    const auto args = std::span<char*>(argv, static_cast<std::size_t>(argc))
                    | std::views::transform([](const auto& arg) { return std::string_view{ arg }; });

    const auto objects = nova::read_file<map_parser>(std::filesystem::path(args[2]).string()).value();
    const json config(nova::read_file(std::filesystem::path(args[1]).string()).value());

    vehicle<lidar> car(config, objects);

    car.func();

    return EXIT_SUCCESS;
}
