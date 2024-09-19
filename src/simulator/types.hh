#ifndef TYPES_HH
#define TYPES_HH

#include <nova/vec.h>
#include <fmt/format.h>

#include <fstream>
#include <sstream>
#include <string>
#include <variant>
#include <vector>


template <typename... Ts>
struct lambdas : Ts... { using Ts::operator()...; };

template <typename... Ts>
lambdas(Ts...) -> lambdas<Ts...>;

struct pose {
    nova::Vec3f position;
    nova::Vec3f orientation;
};

struct motion {
    float velocity;
    nova::Vec3f ang_vel;
};

struct cylinder {
    nova::Vec3f center;
    nova::Vec3f axis;
    float radius;
    float height;
};

struct plane {
    nova::Vec3f p0, p1, p2, p3;
};

struct sphere {
    nova::Vec3f center;
    float radius;
};

using primitive = std::variant<cylinder, plane, sphere>;

struct hit_record {
    nova::Vec3f point;
    nova::Vec3f normal;
    float t;
};

struct ray {
    nova::Vec3f origin;
    nova::Vec3f direction;

    constexpr nova::Vec3f at(float distance) const {
        return origin + direction * distance;
    }
};

std::istream& operator>>(std::istream& is, nova::Vec3f& vec) {
    is >> vec.x() >> vec.y() >> vec.z();
    return is;
}

struct map_parser {
    [[nodiscard]] std::vector<primitive> operator()(std::istream& iF) {
        using namespace std::string_literals;
        std::vector<primitive> ret;
        for (std::string line; std::getline(iF, line); ) {
            std::stringstream ss;
            ss << line << '\n';
            std::string obj_type;
            ss >> obj_type;
            if (obj_type == "plane"s) {
                plane plane;
                ss >> plane.p0 >> plane.p1 >> plane.p2 >> plane.p3;
                ret.push_back(plane);
            } else if (obj_type == "cylinder"s) {
                cylinder cyl;
                ss >> cyl.center >> cyl.axis >> cyl.radius >> cyl.height;
                ret.push_back(cyl);
            } else if (obj_type == "sphere"s) {
                sphere sphere;
                ss >> sphere.center >> sphere.radius;
                ret.push_back(sphere);
            }
        }
        return ret;
    }
};

struct xyz_parser {
    [[nodiscard]] std::vector<nova::Vec3f> operator()(std::istream& iF) {
        using namespace std::string_literals;
        std::vector<nova::Vec3f> ret;
        for (std::string line; std::getline(iF, line); ) {
            std::stringstream ss;
            ss << line << '\n';
            nova::Vec3f vec;
            ss >> vec;
            ret.push_back(vec);
        }
        return ret;
    }
};

#endif // TYPES_HH
