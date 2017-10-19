/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE utility_string_test

#include <cxtream/core/utility/string.hpp>

#include <boost/test/unit_test.hpp>

#include <experimental/filesystem>
#include <string>
#include <type_traits>
#include <vector>

using namespace cxtream::utility;

BOOST_AUTO_TEST_CASE(test_string_to__string)
{
    std::string str1 = "test";
    auto str2 = string_to<std::string>(str1);
    static_assert(std::is_same<std::string, decltype(str2)>{});
    BOOST_TEST(str2 == "test");
    str1[0] = 'b';
    BOOST_TEST(str2 == "test");
}

BOOST_AUTO_TEST_CASE(test_string_to__path)
{
    namespace fs = std::experimental::filesystem;
    auto p1 = fs::path{"ro ot"} / "this is folder" / "my file.txt";
    std::string str1 = p1.string();
    auto p2 = string_to<fs::path>(str1);
    static_assert(std::is_same<fs::path, decltype(p2)>{});
    BOOST_TEST(p2 == p1);
}

BOOST_AUTO_TEST_CASE(test_string_to__float)
{
    std::string str = "0.25";
    auto flt = string_to<float>(str);
    static_assert(std::is_same<float, decltype(flt)>{});
    BOOST_TEST(flt == 0.25);
}

BOOST_AUTO_TEST_CASE(test_string_to__float_exc)
{
    std::string str = "0,25";
    BOOST_CHECK_THROW(string_to<float>(str), std::ios_base::failure);
}

BOOST_AUTO_TEST_CASE(test_string__to_string)
{
    std::string str1 = "test";
    auto str2 = to_string(str1);
    static_assert(std::is_same<std::string, decltype(str2)>{});
    BOOST_TEST(str2 == "test");
    str1[0] = 'b';
    BOOST_TEST(str2 == "test");
}

BOOST_AUTO_TEST_CASE(test_path__to_string)
{
    namespace fs = std::experimental::filesystem;
    auto p1 = fs::path{"rooty root"} / "this is folder" / "nice file .csv";
    auto str = to_string(std::move(p1));
    static_assert(std::is_same<std::string, decltype(str)>{});
    BOOST_TEST(fs::path{str} == p1);
}

BOOST_AUTO_TEST_CASE(test_float__to_string)
{
    float flt = 0.25;
    auto str = to_string(flt);
    static_assert(std::is_same<std::string, decltype(str)>{});
    BOOST_TEST(std::stof(str) == 0.25);
}
