#include "simulator/logging.hh"
#include "utils.hh"
#include "types.hh"

#include <fmt/format.h>
#include <nova/io.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <numbers>
#include <ranges>


int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[]) {
    [[maybe_unused]] auto& logger = init("logger");

    const auto args = std::span<char*>(argv, static_cast<std::size_t>(argc))
                    | std::views::transform([](const auto& arg) { return std::string_view{ arg }; });

    std::size_t cloud_id;

    std::from_chars(args[2].begin(), args[2].begin() + args[2].size(), cloud_id);

    logging::info("Reading cloud");

    const auto cloud_tmp = nova::read_file<lidar_data_parser>(
        (std::filesystem::path(args[1]) / fmt::format("test_fn{}.xyz", cloud_id)).string()
    ).value();

    logging::info("Processing cloud(s)");

    const auto downsampled = downsample(cloud_tmp);
    const auto filtered = filter_planes(downsampled);
    const auto clusters = cluster(filtered);

    logging::info("Clusters found: {}", clusters.size());

    for (const auto& cl : clusters) {
        logging::info("Cluster size: {}", cl.indices.size());
    }

    pcl::io::savePLYFile("./downsampled.ply", downsampled);
    pcl::io::savePLYFile("./filtered.ply", filtered);
}
