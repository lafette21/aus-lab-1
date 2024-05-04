#include "types.hh"

#include <fmt/chrono.h>
#include <fmt/format.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
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
#include <random>
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

// TODO(refact): Nova to handle normal_distribution beside uniform
auto random_noise(float sigma) {
    std::random_device rd{};
    std::mt19937 gen{ rd() };

    // values near the mean are the most likely
    // standard deviation affects the dispersion of generated values from the mean
    std::normal_distribution distribution{ 0.f, sigma };

    return distribution(gen);
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

    auto pos() {
        return m_origin;
    }

    auto scan(const auto& objects, float orientation) {
        std::vector<nova::Vec3f> result;
        result.reserve(m_angles_ver.size() * m_angles_hor.size());

        const auto start = nova::now();

        if (orientation < 0) {
            orientation += 2.f * std::numbers::pi_v<float>;
        }

        const auto sigma = m_config.lookup<float>("accuracy.value");

        for (auto angle_h : m_angles_hor) {
            angle_h = deg2rad(angle_h);
            angle_h += orientation;
            if (angle_h > 2.f * std::numbers::pi_v<float>) {
                angle_h -= 2.f * std::numbers::pi_v<float>;
            } else if (angle_h < -2.f * std::numbers::pi_v<float>) {
                angle_h += 2.f * std::numbers::pi_v<float>;
            }
            for (auto angle_v : m_angles_ver) {
                angle_v = deg2rad(angle_v);

                const auto direction = nova::Vec3f{
                    std::cos(angle_v) * std::sin(angle_h),
                    std::cos(angle_v) * std::cos(angle_h),
                    std::sin(angle_v)
                };

                result.push_back(closest_to(m_origin, ray_cast(ray{ m_origin, direction }, objects, sigma)));
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

struct motion {
    float velocity;
    nova::Vec3f ang_vel;
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

class simulation {
public:
    simulation(const json& config, const auto& objects, const auto& path)
        : m_config(config)
        , m_objects(objects)
        , m_path(path)
        , m_lidar(m_config.at("lidar.vlp_16"))
    {}

    auto start() {
        setup();

        const auto origin = m_path[0];

        for (auto& p : m_path) {
            p -= origin;
        }

        for (auto& o : m_objects) {
            std::visit(
                lambdas{
                    [origin](sphere& p)   { p.center -= origin; },
                    [origin](cylinder& p) { p.center -= origin; },
                    [origin](plane& p)    { p.p0 -= origin; p.p1 -= origin; p.p2 -= origin; p.p3 -= origin; },
                },
                o
            );
        }

        std::vector<float> orientations;
        orientations.reserve(m_path.size());
        orientations.push_back(0);

        for (std::size_t i = 1; i < m_path.size() - 1; ++i) {
            const auto angle1 = std::atan2(m_path[i].x() - m_path[i - 1].x(), m_path[i].y() - m_path[i - 1].y());
            const auto angle2 = std::atan2(m_path[i + 1].x() - m_path[i].x(), m_path[i + 1].y() - m_path[i].y());

            float angle;

            if (std::signbit(angle1) == std::signbit(angle2)) {
                angle = angle1 + angle2;
            } else {
                const auto dominant = (std::numbers::pi_v<float> - std::abs(angle1)) > (std::numbers::pi_v<float> - std::abs(angle2)) ? angle1 : angle2;

                angle = std::abs(angle1) + std::abs(angle2);
                angle *= std::signbit(dominant) ? -1.f : 1.f;
            }

            orientations.push_back(angle / 2.f);
        }

        orientations.push_back(orientations.back());

        // matplot::axis({ -matplot::inf, matplot::inf, -1, +1});
        // matplot::plot(orientations, ".");
        // matplot::show();

        const auto poses = std::views::zip(m_path, orientations)
                         | std::views::transform([](const auto& elem) { const auto& [pos, theta] = elem; return pose{ pos, nova::Vec3f{ 0, 0, theta } }; })
                         | ranges::to<std::vector>();

        const auto max_velocity = m_config.lookup<float>("simulation.velocity.max");
        const auto velocities = normalize_data_into_range(m_curvature, 0.f, 0.7f)
                              | std::views::transform([max_velocity](const auto& elem) { return (1.f - elem) * max_velocity; })
                              | ranges::to<std::vector>();

        const auto motion_sampling = m_config.lookup<float>("motion.sampling");
        const auto distances = velocities
                             | std::views::transform([motion_sampling](const auto& elem) { return elem * (1.f / motion_sampling); })
                             | ranges::to<std::vector>();

        // matplot::plot(velocities, ".");
        // matplot::show();

        const auto [sparse_path, indices] = select_data(m_path, distances);

        const auto xs = sparse_path
                      | std::views::transform([](const auto& elem) { return elem.x(); })
                      | ranges::to<std::vector>();

        const auto ys = sparse_path
                      | std::views::transform([](const auto& elem) { return elem.y(); })
                      | ranges::to<std::vector>();

        // matplot::axis({ -20, 20, -20, 20 });
        // matplot::plot(xs, ys, ".r");
        // matplot::show();

        const auto sparse_poses = [&poses, &indices] () -> std::vector<pose> {
            std::vector<pose> ret;
            ret.reserve(indices.size());

            for (const auto& i : indices) {
                ret.push_back(poses[i]);
            }

            return ret;
        }();

        for (const auto& [i, pose] : std::views::enumerate(sparse_poses)) {
            m_lidar.replace({ pose.position.x(), pose.position.y(), m_lidar.pos().z() });
            const auto data = m_lidar.scan(m_objects, pose.orientation.z())
                            | std::views::transform([pose](const auto& elem) { return nova::Vec3f{ elem.x() - pose.position.x(), elem.y() - pose.position.y(), elem.z() }; })
                            | ranges::to<std::vector>();
            print(fmt::format("./out/l_out_{}.xyz", i + 1), data);
        }
    }

private:
    json m_config;
    std::vector<primitive> m_objects;
    std::vector<nova::Vec3f> m_path;
    std::vector<float> m_curvature;
    lidar m_lidar;

    void setup() {
        const auto _xs = m_path
                       | std::views::transform([](const auto& elem) { return elem.x(); })
                       | ranges::to<std::vector>();

        const auto _ys = m_path
                       | std::views::transform([](const auto& elem) { return elem.y(); })
                       | ranges::to<std::vector>();

        for (const auto& [x, y] : std::views::zip(_xs, _ys)) {
            spdlog::debug("x={}\ty={}", x, y);
        }

        const auto [xs, ys, dxs, dys, ddxs, ddys] = interpolate_and_differentiate(_xs, _ys, m_config.lookup<std::size_t>("simulation.path.num_interp_points"));

        // matplot::plot(xs, ys, ".b");
        // matplot::show();

        const auto path = std::views::zip(xs, ys)
                        | std::views::transform([](const auto& elem) { const auto& [x, y] = elem; return nova::Vec3f{ x, y, 0 }; })
                        | ranges::to<std::vector>();

        for (const auto& p : path) {
            spdlog::debug("{}", p);
        }

        m_path = path;

        assert(dxs.size() == dys.size() && dys.size() == ddxs.size() && ddxs.size() == ddys.size());

        for (std::size_t i = 0; i < dxs.size(); ++i) {
            m_curvature.push_back(std::abs(dxs[i] * ddys[i] - dys[i] * ddxs[i]) / std::pow(dxs[i] * dxs[i] + dys[i] * dys[i], 1.5f));
        }
    }

    template <std::floating_point T = float>
    auto normalize_data_into_range(const std::vector<T>& data, T min, T max)
            -> std::vector<T>
    {
        std::vector<T> ret;
        ret.reserve(data.size());

        for (std::size_t i = 0; i < data.size(); ++i) {
            const T div = (data[i] - std::ranges::min(data)) / (std::ranges::max(data) - std::ranges::min(data));
            const T tmp = std::isnan(div) ? 0 : div * (max - min) + min;
            ret.push_back(tmp);
        }

        return ret;
    }

    auto select_data(const std::vector<nova::Vec3f>& data, const std::vector<float>& distances)
            -> std::pair<std::vector<nova::Vec3f>, std::vector<std::size_t>>
    {
        std::vector<nova::Vec3f> ret { data[0] };
        std::vector<std::size_t> indices { 0 };
        std::vector<float> _distances{ distances };
        std::size_t i = 1;
        float distance = 0;

        while (i < data.size()) {
            if (distance < _distances[indices.back()]) {
                distance += (data[i - 1] - data[i]).length();
                i++;
            } else {
                const auto dist_before = distance - (data[i] - data[i - 1]).length();
                std::size_t idx;

                if (not std::ranges::contains(ret, data[i - 1])) {
                    _distances[indices.back()] = _distances[indices.back()] - dist_before < distance - _distances[indices.back()] ? dist_before : distance;
                    ret.push_back(_distances[indices.back()] - dist_before <= distance - _distances[indices.back()] ? data[i - 1] : data[i]);
                    idx = _distances[indices.back()] - dist_before <= distance - _distances[indices.back()] ? i - 1 : i;
                } else {
                    _distances[indices.back()] = distance;
                    ret.push_back(data[i]);
                    idx = i;
                }

                indices.push_back(idx);

                if (indices.back() < i) {
                    i--;
                }

                distance = 0;
            }
        }

        return {
            ret,
            indices
        };
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
