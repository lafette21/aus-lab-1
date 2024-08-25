#ifndef LIDAR_HH
#define LIDAR_HH

#include "logging.hh"
#include "utils.hh"

#include <fmt/chrono.h>
#include <nova/json.h>
#include <nova/utils.h>
#include <nova/vec.h>

using json = nova::json;


class lidar {
public:
    lidar(const json& config)
        : m_config(config)
        , m_ang_res_h(m_config.lookup<float>("rpm.value") / 60 * 360 * m_config.lookup<float>("firing_cycle"))
        , m_angles_hor(nova::linspace(nova::range{ 0.f, m_config.lookup<float>("fov_h") }, static_cast<std::size_t>(m_config.lookup<float>("fov_h") / m_ang_res_h), false))
        , m_angles_ver(nova::linspace(nova::range{ -m_config.lookup<float>("fov_v") / 2, m_config.lookup<float>("fov_v") / 2 }, m_config.lookup<std::size_t>("channels"), true))
    {}

    auto string() const {
        return fmt::format("lidar(m_config={}, m_ang_res_h={})", m_config.dump(), m_ang_res_h);
    }

    auto pos() const {
        return m_origin;
    }

    // TODO(refact): Use radians everywhere
    auto scan(const std::vector<primitive>& primitives, float orientation) const {
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

                result.push_back(closest_to(m_origin, ray_cast(ray{ m_origin, direction }, primitives, sigma)));
            }
        }

        const auto end = nova::now();
        const auto took = end - start;
        logging::debug("[LIDAR] rotation took: {}", took);

        return result;
    }

    void move(const nova::Vec3f& pos) {
        m_origin = pos;
    }

private:
    json m_config;
    float m_ang_res_h;
    std::vector<float> m_angles_hor;
    std::vector<float> m_angles_ver;
    std::vector<nova::Vec3f> m_data;
    nova::Vec3f m_origin = { 0, 0, 1.5 };
};

#endif // LIDAR_HH
