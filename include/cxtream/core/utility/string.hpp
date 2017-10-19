/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/
/// \defgroup String String utilities.

#ifndef CXTREAM_CORE_UTILITY_STRING_HPP
#define CXTREAM_CORE_UTILITY_STRING_HPP

#include <boost/lexical_cast.hpp>
#include <range/v3/view/transform.hpp>

#include <algorithm>
#include <locale>
#include <experimental/filesystem>
#include <sstream>
#include <string>

namespace cxtream::utility {

/// \ingroup String
/// \brief Convert std::string to the given type.
///
/// This function is either specialized for the given type or internally uses boost::lexical_cast.
///
/// \throws std::ios_base::failure If the conversion fails.
template<typename T>
T string_to(const std::string& str)
{
    try {
        return boost::lexical_cast<T>(str);
    } catch(const boost::bad_lexical_cast &) {
        throw std::ios_base::failure{std::string{"Failed to read type <"} + typeid(T).name() +
                                     "> from string \"" + str + "\"."};
    }
}

template<>
std::string string_to<std::string>(const std::string& str)
{
    return str;
}

template<>
std::experimental::filesystem::path
string_to<std::experimental::filesystem::path>(const std::string& str)
{
    return str;
}

/// \ingroup String
/// \brief Convert the given type to std::string.
///
/// This function is either overloaded for the given type or internally uses boost::lexical_cast.
///
/// \throws std::ios_base::failure If the conversion fails.
template<typename T>
std::string to_string(const T& value)
{
    try {
        return boost::lexical_cast<std::string>(value);
    } catch(const boost::bad_lexical_cast &) {
        throw std::ios_base::failure{std::string{"Failed to read string from type <"}
                                     + typeid(T).name() + ">."};
    }
}

std::string to_string(const std::experimental::filesystem::path& path)
{
    return path.string();
}

std::string to_string(const std::string& str)
{
    return str;
}

std::string to_string(const char* const& str)
{
    return str;
}

std::string to_string(const bool& b)
{
    return b ? "true" : "false";
}

}  // namespace cxtream
#endif
