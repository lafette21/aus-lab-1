#ifndef SIMULATION_HH
#define SIMULATION_HH

#include "lidar.hh"
#include "logging.hh"
#include "types.hh"
#include "utils.hh"

#include <nova/json.h>

using json = nova::json;


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
            print(fmt::format("./out/test_fn{}.xyz", i + 1), data);
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
            logging::debug("x={}\ty={}", x, y);
        }

        const auto [xs, ys, dxs, dys, ddxs, ddys] = interpolate_and_differentiate(_xs, _ys, m_config.lookup<std::size_t>("simulation.path.num_interp_points"));

        // matplot::plot(xs, ys, ".b");
        // matplot::show();

        const auto path = std::views::zip(xs, ys)
                        | std::views::transform([](const auto& elem) { const auto& [x, y] = elem; return nova::Vec3f{ x, y, 0 }; })
                        | ranges::to<std::vector>();

        for (const auto& p : path) {
            logging::debug("{}", p);
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

    void print(const std::string& path, const auto& data) {
        std::ofstream oF(path);

        for (const auto& d : data) {
            oF << fmt::format("{} {} {} 0 0 0\n", d.x(), d.y(), d.z());
        }
    }
};

#endif // SIMULATION_HH
