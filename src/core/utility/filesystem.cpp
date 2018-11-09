/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/
/// \defgroup Filesystem Filesystem utilities.

#include <hipipe/core/utility/filesystem.hpp>

#include <exception>
#include <random>
#include <string>

namespace hipipe::utility {


// Generate random path inside the temp directory. This path is not
// checked for existence.
static std::experimental::filesystem::path generate_temp_path(std::string pattern)
{
    const char* hexchars = "0123456789abcdef";
    static thread_local std::mt19937 gen{std::random_device{}()};
    std::uniform_int_distribution<> dis{0, 15};
    for (char& ch: pattern) {
        if (ch == '%') ch = hexchars[dis(gen)];
    }
    namespace fs = std::experimental::filesystem;
    return fs::temp_directory_path() / pattern;
}


std::experimental::filesystem::path create_temp_directory(const std::string &pattern)
{
    namespace fs = std::experimental::filesystem;
    int max_retries = 100;
    while (max_retries--) {
        auto temp_path = generate_temp_path(pattern);
        bool success = fs::create_directory(temp_path);
        if (success) return temp_path;
    }
    throw std::runtime_error(std::string{"Cannot create temporary directory ["} + pattern + "]");
}

}  // namespace hipipe::utility
