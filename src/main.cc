#include "utils.hh"

#include <fmt/core.h>
#include <fmt/format.h>

#include <cstdlib>
#include <span>
#include <ranges>


int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[]) {
    const auto args = std::span<char*>(argv, static_cast<std::size_t>(argc))
                    | std::views::transform([](const auto& arg) { return std::string_view {arg}; });

    constexpr std::size_t From = 165;
    constexpr std::size_t To = 166;

    for (std::size_t i = From; i < To; ++i) {
        const auto cloud = read_file<lidar_data_parser>(std::filesystem::path(args[1]).string()).value();
        fmt::println("cloud size: {}", cloud.size());
    }

    return EXIT_SUCCESS;
}
