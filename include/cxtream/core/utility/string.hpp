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
#include <set>
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

/// Specialization of string_to() for std::string.
template<>
std::string string_to<std::string>(const std::string& str)
{
    return str;
}

/// Specialization of string_to() for std::experimental::filesystem::path.
template<>
std::experimental::filesystem::path
string_to<std::experimental::filesystem::path>(const std::string& str)
{
    return str;
}

namespace detail
{
    /// List of recognized boolean strings for value `true`.
    const std::set<std::string> true_set =
      {"true ", "True ", "TRUE ", "1", "y", "Y", "yes", "Yes", "YES", "on ", "On ", "ON"};
    /// List of recognized boolean strings for value `false`.
    const std::set<std::string> false_set =
      {"false", "False", "FALSE", "0", "n", "N", "no ", "No ", "NO ", "off", "Off", "OFF"};
}

/// Specialization of string_to() for bool.
///
/// The list of recognized terms:
/// \code
///     true |True |TRUE |1|y|Y|yes|Yes|YES|on |On |ON
///     false|False|FALSE|0|n|N|no |No |NO |off|Off|OFF
/// \endcode
/// \throws std::ios_base::failure If an unrecognizable string is provided.
template<>
bool string_to<bool>(const std::string& str)
{
    if (detail::true_set.count(str)) return true;
    if (detail::false_set.count(str)) return false;
    throw std::ios_base::failure{"Failed to convert string \"" + str + "\" to bool."};
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

/// Specialization of to_string() for std::experimental::filesystem::path.
std::string to_string(const std::experimental::filesystem::path& path)
{
    return path.string();
}

/// Specialization of to_string() for std::string.
std::string to_string(const std::string& str)
{
    return str;
}

/// Specialization of to_string() for const char *.
std::string to_string(const char* const& str)
{
    return str;
}

/// Specialization of to_string() for bool.
///
/// The generated string is either "true" or "false".
std::string to_string(const bool& b)
{
    return b ? "true" : "false";
}

}  // namespace cxtream
#endif
