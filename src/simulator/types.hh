#ifndef TYPES_HH
#define TYPES_HH

#include <nova/vec.h>
#include <fmt/format.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>


enum class object_type {
    cylinder,
    plane,
};

struct error {
    std::string msg;
    operator std::string() { return msg; }
};

struct def_parser {
    [[nodiscard]] std::string operator()(std::ifstream&& iF) {
        std::stringstream ss;
        for (std::string line; std::getline(iF, line); ) {
            ss << line << '\n';
        }
        return ss.str();
    }
};

struct map_parser {
    [[nodiscard]] std::vector<std::pair<object_type, std::vector<nova::Vec3f>>> operator()(std::ifstream&& iF) {
        using namespace std::string_literals;
        std::vector<std::pair<object_type, std::vector<nova::Vec3f>>> result;
        for (std::string line; std::getline(iF, line); ) {
            std::stringstream ss;
            ss << line << '\n';
            std::string obj_type;
            nova::Vec3f point;
            ss >> obj_type;
            if (obj_type == "plane"s) {
                std::vector<nova::Vec3f> vec;
                for (std::size_t i = 0; i < 4; ++i) {
                    ss >> point.x() >> point.y() >> point.z();
                    vec.push_back(point);
                }
                result.emplace_back(object_type::plane, vec);
            } else if (obj_type == "cylinder"s) {
                std::vector<nova::Vec3f> vec;
                for (std::size_t i = 0; i < 1; ++i) {
                    ss >> point.x() >> point.y() >> point.z();
                    vec.push_back(point);
                }
                result.emplace_back(object_type::cylinder, vec);
            }
        }
        return result;
    }
};

#endif // TYPES_HH
