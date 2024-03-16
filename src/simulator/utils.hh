#ifndef UTILS_HH
#define UTILS_HH

#include "types.hh"

#include <fmt/format.h>

#include <expected>
#include <filesystem>
#include <fstream>
#include <string_view>


template <typename Parser = def_parser>
[[nodiscard]] auto read_file(std::string_view path, Parser&& parser = {})
        -> std::expected<std::remove_cvref_t<std::invoke_result_t<Parser, std::ifstream>>, error>
{
    const auto fs = std::filesystem::path(path);
    if (not std::filesystem::is_regular_file(fs)) {
        return std::unexpected<error>(fmt::format("{} is not a regular file!", std::filesystem::absolute(fs).string()));
    }

    return parser(std::ifstream(fs));
}

#endif // UTILS_HH
