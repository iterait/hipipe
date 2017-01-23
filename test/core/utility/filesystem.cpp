/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE utility_filesystem_test

#include <cxtream/core/utility/filesystem.hpp>

#include <boost/test/unit_test.hpp>

#include <set>
#include <string>

using namespace cxtream::utility;
namespace fs = std::experimental::filesystem;
const char* hexchars = "0123456789abcdef";
const char* pattern = "my_dir_%%%";

void check_correct(const fs::path& temp_dir_full)
{
    BOOST_CHECK(fs::exists(temp_dir_full));
    BOOST_CHECK(fs::is_directory(temp_dir_full));
    BOOST_CHECK(fs::is_empty(temp_dir_full));

    std::string temp_dir_fname = temp_dir_full.filename();
    BOOST_TEST(temp_dir_fname.size() == 10UL);
    BOOST_TEST(temp_dir_fname.substr(0, 6) == "my_dir");
    BOOST_TEST(temp_dir_fname.substr(7, 3).find_first_not_of(hexchars) == std::string::npos);

    BOOST_TEST(fs::temp_directory_path() / temp_dir_fname == temp_dir_full);
}

BOOST_AUTO_TEST_CASE(test_create_temp_directory)
{
    fs::path temp_dir = create_temp_directory(pattern);
    check_correct(temp_dir);
    fs::remove(temp_dir);
}

BOOST_AUTO_TEST_CASE(test_create_many_temp_directories)
{
    std::set<fs::path> paths;
    for (int i = 0; i < 2000; ++i) {
        fs::path temp_dir = create_temp_directory(pattern);
        check_correct(temp_dir);
        BOOST_TEST(paths.count(temp_dir) == 0);
        paths.insert(temp_dir);
    }
    for (const fs::path& p : paths) fs::remove(p);
}

BOOST_AUTO_TEST_CASE(test_throws_when_pattern_unsatisfiable)
{
    fs::path temp_dir = create_temp_directory(pattern);
    BOOST_CHECK_THROW(create_temp_directory(temp_dir.filename()), std::runtime_error);
    fs::remove(temp_dir);
}
