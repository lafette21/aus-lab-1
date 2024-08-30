#include "simulation.hh"

#include <boost/program_options.hpp>
#include <nova/io.h>
#include <nova/json.h>
#include <nova/utils.h>

#include <iostream>
#include <ranges>
#include <span>
#include <string>

namespace po = boost::program_options;

using json = nova::json;


void validate_format(const std::string& format) {
#ifdef ROS2_BUILD
    const auto valid_formats = std::vector<std::string>{ "xyz", "rosbag2" };
#else
    const auto valid_formats = std::vector<std::string>{ "xyz" };
#endif

    if (std::find(valid_formats.begin(), valid_formats.end(), format) == valid_formats.end()) {
        throw po::validation_error(po::validation_error::invalid_option_value, "format", format);
    }
}

auto parse_args(int argc, char* argv[]) -> std::optional<po::variables_map> {
    auto arg_parser = po::options_description("Simulator");
    arg_parser.add_options()
        ("config,c", po::value<std::string>()->required()->value_name("FILE"), "Config file")
        ("map,m", po::value<std::string>()->required()->value_name("FILE"), "Map file")
        ("path,p", po::value<std::string>()->required()->value_name("FILE"), "Path file")
#ifdef ROS2_BUILD
        ("format,f", po::value<std::string>()->required()->value_name("xyz|rosbag2")->notifier(validate_format)->default_value("xyz"), "Output format")
#else
        ("format,f", po::value<std::string>()->required()->value_name("xyz")->notifier(validate_format)->default_value("xyz"), "Output format")
#endif
        ("outdir,o", po::value<std::string>()->required()->value_name("DIR")->default_value("out"), "Output directory")
        ("help,h", "Show this help");

    po::variables_map args;
    po::store(po::parse_command_line(argc, argv, arg_parser), args);

    if (args.contains("help")) {
        std::cerr << arg_parser << std::endl;
        return std::nullopt;
    }

    args.notify();

    return args;
}


int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    [[maybe_unused]] auto& logger = init("simulator");

    try {
        const auto args = parse_args(argc, argv);

        if (not args.has_value()) {
            return EXIT_SUCCESS;
        }

        const auto out_dir = (*args)["outdir"].as<std::string>();

        if (not std::filesystem::exists(out_dir)) {
            if (std::filesystem::create_directory(out_dir)) {
                logging::info("Directory created: {}", out_dir);
            } else {
                logging::error("Failed to create directory: {}", out_dir);
            }
        }

        const json config(nova::read_file(std::filesystem::path((*args)["config"].as<std::string>()).string()).value());
        const auto objects = nova::read_file<map_parser>(std::filesystem::path((*args)["map"].as<std::string>()).string()).value();
        const auto path = nova::read_file<xyz_parser>(std::filesystem::path((*args)["path"].as<std::string>()).string()).value();

        simulation simulation(config, objects, path, out_dir);

        simulation.start();

        return EXIT_SUCCESS;

    } catch (const std::exception& ex) {
        logging::error("Fatal error happened: {}", ex.what());
        return EXIT_FAILURE;
    }
}
