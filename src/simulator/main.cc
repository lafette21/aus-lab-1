#include "simulation.hh"

#include <nova/io.h>
#include <nova/json.h>
#include <nova/utils.h>

#include <ranges>
#include <span>
#include <string>

using json = nova::json;


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
