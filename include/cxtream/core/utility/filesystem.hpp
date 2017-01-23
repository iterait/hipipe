/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/
/// \defgroup Filesystem Filesystem utilities.

#ifndef CXTREAM_CORE_UTILITY_FILESYSTEM_HPP
#define CXTREAM_CORE_UTILITY_FILESYSTEM_HPP

#include <exception>
#include <experimental/filesystem>
#include <random>
#include <string>

namespace cxtream::utility {

namespace detail {

    // Generate random path inside the temp directory. This path is not
    // checked for existence.
    std::experimental::filesystem::path generate_temp_path(std::string pattern)
    {
        constexpr const char* hexchars = "0123456789abcdef";
        static thread_local std::mt19937 gen{std::random_device{}()};
        std::uniform_int_distribution<> dis{0, 15};
        for (char& ch: pattern) {
            if (ch == '%') ch = hexchars[dis(gen)];
        }
        namespace fs = std::experimental::filesystem;
        return fs::temp_directory_path() / pattern;
    }

}  // namespace detail

/// \ingroup Filesystem
/// \brief Create a temporary directory.
///
/// \param pattern Directory name pattern. All '\%' symbols in the pattern are
///                replaced by a random character from [0-9a-f].
std::experimental::filesystem::path create_temp_directory(const std::string &pattern)
{
    namespace fs = std::experimental::filesystem;
    int max_retries = 100;
    while (max_retries--) {
        auto temp_path = detail::generate_temp_path(pattern);
        bool success = fs::create_directory(temp_path);
        if (success) return temp_path;
    }
    throw std::runtime_error(std::string{"Cannot create temporary directory ["} + pattern + "]");
}

}  // namespace cxtream
#endif
