#include "types.hh"

#include <fmt/chrono.h>
#include <fmt/format.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <matplot/matplot.h>
#include <nova/io.h>
#include <nova/json.h>
#include <nova/vec.h>
#include <nova/utils.h>
#include <range/v3/range/conversion.hpp>
#include <spdlog/spdlog.h>

#include <array>
#include <fstream>
#include <future>
#include <numbers>
#include <ranges>
#include <span>
#include <string>
#include <vector>

#include <cmath>

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
        spdlog::info("[LIDAR] rotation took: {}", took);

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

struct pose {
    nova::Vec3f position;
    nova::Vec3f orientation;
};

class gyroscope {
public:
    gyroscope(const pose& pose)
        : m_curr_pose(pose)
        , m_prev_pose(m_curr_pose)
    {}

    // TODO: unit tests needed
    // TODO: currently only the orientation along the Z axis is used
    auto measure()
            -> nova::Vec3f
    {
        const auto angle_curr = m_curr_pose.orientation.z();
        const auto angle_prev = m_prev_pose.orientation.z();
        float angle_diff;

        spdlog::info("[IMU] angle_curr: {}\tangle_prev:{}", angle_curr, angle_prev);

        if (std::signbit(angle_curr) == std::signbit(angle_prev)
            or (angle_curr == 0 and angle_prev != 0)
            or (angle_curr != 0 and angle_prev == 0)
        ) {
            angle_diff = angle_curr - angle_prev;
        } else {
            const auto dominant = (std::numbers::pi_v<float> - std::abs(angle_curr)) > (std::numbers::pi_v<float> - std::abs(angle_prev)) ? angle_curr : angle_prev;
            angle_diff = std::abs(std::abs(angle_curr) - std::abs(angle_prev));
            angle_diff *= std::signbit(dominant) ? -1.f : 1.f;
        }

        m_prev_pose = m_curr_pose;

        return {
            0,
            0,
            angle_diff
        };
    }

private:
    const pose& m_curr_pose;
    pose m_prev_pose;
};

class speedometer {
public:
    speedometer() {}

    auto measure() {
        return 1.f;
    }
};

template <>
struct fmt::formatter<nova::Vec3f> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template <typename ParseContext>
    auto format(const nova::Vec3f& obj, ParseContext& ctx) {
        return fmt::format_to(ctx.out(), "{{ {}, {}, {} }}", obj.x(), obj.y(), obj.z());
    }
};

class vehicle {
public:
    vehicle(const json& config)
        : m_config(config)
        , m_lidar(m_config.at("lidar.vlp_16"))
        , m_gyroscope(m_pose)
    {
        spdlog::info(m_lidar.string());
    }

    // auto func() {
        // m_lidar.replace({ -5, 0, 1.5 });

        // for (std::size_t i = 0; i < 10; ++i) {
            // print(fmt::format("./out/l_out_{}.xyz", i + 1), m_lidar.scan(m_objects));
            // m_lidar.shift({ 1, 0, 0 });
        // }
    // }

    auto move(const nova::Vec3f& t, const nova::Vec3f& ang_vel) {
        m_pose.position += t;
        m_pose.orientation += ang_vel;
    }

    auto func(const auto& objects) {
        const auto start = nova::now();

        auto future_lidar = std::async([&objects, this] { return m_lidar.scan(objects); });
        auto future_velocity = std::async([this] { return m_speedometer.measure(); });
        auto future_ang_vel = std::async([this] { return m_gyroscope.measure(); });

        const auto lidar_meas = future_lidar.get();
        const auto velocity = future_velocity.get();
        const auto ang_vel = future_ang_vel.get();

        spdlog::info("[VEHICLE] velocity: {}\tang_vel: {}\tcloud.size: {}", velocity, ang_vel, lidar_meas.size());

        const auto end = nova::now();
        const auto took = end - start;
        spdlog::info("[VEHICLE] measurements took: {}", took);
    }

private:
    json m_config;
    pose m_pose{};
    lidar m_lidar;
    gyroscope m_gyroscope;
    speedometer m_speedometer{};

    // TODO: What if the sampling rate differs between the two
    pose motion_model(float velocity, const nova::Vec3f& ang_vel, float side_slipping = 0) {
        const auto x = m_pose.position.x();
        const auto y = m_pose.position.y();
        const auto theta = m_pose.orientation.z();

        const auto new_x = x + velocity * std::cos(theta + ang_vel.z() / 2 + side_slipping);
        const auto new_y = y + velocity * std::sin(theta + ang_vel.z() / 2 + side_slipping);
        const auto new_theta = theta + ang_vel.z();

        return {
            { new_x, new_y, 0 },
            { 0, 0, new_theta }
        };
    }
};

template <std::floating_point T = float>
auto interpolate(const std::vector<T>& x, const std::vector<T>& y, std::size_t num_points = 100)
        -> std::array<std::vector<T>, 2>
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

    ret_x.reserve(_x.size());
    ret_y.reserve(_y.size());

    double t_max = t.back();
    double step = t_max / (static_cast<double>(num_points) - 1.0);

    for (std::size_t i = 0; i < num_points; ++i) {
        double ti = step * static_cast<double>(i);
        ret_x.push_back(static_cast<T>(gsl_spline_eval(spline_x, ti, acc)));
        ret_y.push_back(static_cast<T>(gsl_spline_eval(spline_y, ti, acc)));
    }

    gsl_spline_free(spline_x);
    gsl_spline_free(spline_y);
    gsl_interp_accel_free(acc);

    return {
        ret_x,
        ret_y
    };
}

class simulation {
public:
    simulation(const json& config, const auto& objects, const auto& path)
        : m_config(config)
        , m_objects(objects)
        , m_path(path)
        , m_vehicle(config)
    {}

    auto start() {
        setup();

        for (const auto& [i, p] : std::views::enumerate(m_path_interp)) {
            m_vehicle.func(m_objects);
            m_vehicle.move(p, { 0, 0, 0.1f * (i + 1) });
        }
    }

private:
    json m_config;
    std::vector<primitive> m_objects;
    std::vector<nova::Vec3f> m_path;
    std::vector<nova::Vec3f> m_path_interp;
    vehicle m_vehicle;

    void setup() {
        const auto xs = m_path
                      | std::views::transform([](const auto& elem) { return elem.x(); })
                      | ranges::to<std::vector>();

        const auto ys = m_path
                      | std::views::transform([](const auto& elem) { return elem.y(); })
                      | ranges::to<std::vector>();

        for (const auto& [x, y] : std::views::zip(xs, ys)) {
            spdlog::debug("x={}\ty={}", x, y);
        }

        const auto [xs_interp, ys_interp] = interpolate(xs, ys, m_config.lookup<std::size_t>("simulation.path.num_interp_points"));

        // matplot::plot(xs_interp, ys_interp, "o");
        // matplot::show();

        const auto path = std::views::zip(xs_interp, ys_interp)
                        | std::views::transform([](const auto& elem) { const auto& [x, y] = elem; return nova::Vec3f{ x, y, 0 }; })
                        | ranges::to<std::vector>();

        for (const auto& p : path) {
            spdlog::debug("{}", p);
        }

        m_path_interp = path;
    }
};

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    [[maybe_unused]] auto& logger = init("simulator");

    const auto args = std::span<char*>(argv, static_cast<std::size_t>(argc))
                    | std::views::transform([](const auto& arg) { return std::string_view{ arg }; });

    const auto objects = nova::read_file<map_parser>(std::filesystem::path(args[2]).string()).value();
    const auto path = nova::read_file<xyz_parser>(std::filesystem::path(args[3]).string()).value();
    const json config(nova::read_file(std::filesystem::path(args[1]).string()).value());

    simulation simulation(config, objects, path);

    simulation.start();

    return EXIT_SUCCESS;
}
