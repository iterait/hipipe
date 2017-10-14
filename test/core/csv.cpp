/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE csv_test

#include "common.hpp"

#include <cxtream/core/csv.hpp>

#include <boost/test/unit_test.hpp>
#include <range/v3/algorithm/find_first_of.hpp>
#include <range/v3/view/slice.hpp>

#include <experimental/filesystem>
#include <iostream>
#include <sstream>
#include <vector>

using namespace cxtream;
namespace fs = std::experimental::filesystem;

// 2d container transposition //

template<typename Container2d>
Container2d transpose(const Container2d& data)
{
    auto width = ranges::size(data);
    auto height = ranges::size(ranges::at(data, 0));
    Container2d res(height, ranges::range_value_type_t<Container2d>(width));
    for (std::size_t i = 0; i < height; ++i) {
        for (std::size_t j = 0; j < width; ++j) {
            res[i][j] = data[j][i];
        }
    }
    return res;
}

// simple csv example //

const std::string simple_csv{
  "Id,  A,   B \n"
  " 1, a1, 1.1 \n"
  " 2, a2, 1.2 \n"
  " 3, a3, 1.3 \n"
};
const std::vector<std::vector<std::string>> simple_csv_rows{
  std::vector<std::string>{"Id", "A", "B"},
  std::vector<std::string>{"1", "a1", "1.1"},
  std::vector<std::string>{"2", "a2", "1.2"},
  std::vector<std::string>{"3", "a3", "1.3"}
};
const auto simple_csv_cols = transpose(simple_csv_rows);

// complex quoted csv example //

const std::string quoted_csv{
  "  *Column| 1*| \t *Column| 2*  | * Column +*3+* *\n"
  "Field 1| *Field|\n 2*  | * Field 3 *    \n"
  "*Field\n1*|   *Field| 2 * |   * Field 3 *    "
};
const std::vector<std::vector<std::string>> quoted_csv_rows{
  std::vector<std::string>{"Column| 1", "Column| 2", " Column *3* "},
  std::vector<std::string>{"Field 1", "Field|\n 2", " Field 3 "},
  std::vector<std::string>{"Field\n1", "Field| 2 ", " Field 3 "}
};

// invalid csv examples //

const std::vector<std::string> invalid_csvs{
  {
    "Id,  A,   B \n"
    " 1, a1      \n"
    " 3, a3, 1.3 \n"
  },
  {
    "Id,  A,   B \n"
    " 1, \"Field 1, Field 2\n"
  },
  {
    "Id,  A,   B \n"
    " 1, a1, 1.2 \n"
    " 3, a3 \n"
  }
};

// tests //

BOOST_AUTO_TEST_CASE(test_csv_istream_range_simple_csv)
{
    std::istringstream simple_csv_ss{simple_csv};
    auto csv_rows = csv_istream_range{simple_csv_ss};
    test_ranges_equal(csv_rows, simple_csv_rows);
}

BOOST_AUTO_TEST_CASE(test_csv_istream_range_quoted_csv)
{
    std::istringstream quoted_csv_ss{quoted_csv};
    auto csv_rows = csv_istream_range{quoted_csv_ss, '|', '*', '+'};
    test_ranges_equal(csv_rows, quoted_csv_rows);
}

BOOST_AUTO_TEST_CASE(test_read_csv_from_istream)
{
    std::istringstream simple_csv_ss{simple_csv};
    const dataframe<> df = read_csv(simple_csv_ss);
    BOOST_TEST(df.n_cols() == 3);
    BOOST_TEST(df.n_rows() == 3);
    test_ranges_equal(df.header(), simple_csv_rows[0]);
    test_ranges_equal(df.raw_cols()[0], simple_csv_cols[0] | ranges::view::slice(1, ranges::end));
    test_ranges_equal(df.raw_cols()[1], simple_csv_cols[1] | ranges::view::slice(1, ranges::end));
    test_ranges_equal(df.raw_cols()[2], simple_csv_cols[2] | ranges::view::slice(1, ranges::end));
}

BOOST_AUTO_TEST_CASE(test_read_csv_from_no_file)
{
    BOOST_CHECK_THROW(read_csv("no_file.csv"), std::ios_base::failure);
}

BOOST_AUTO_TEST_CASE(test_read_csv_from_istream_no_header)
{
    std::istringstream simple_csv_ss{simple_csv};
    const dataframe<> df = read_csv(simple_csv_ss, 1, false);
    BOOST_TEST(df.n_cols() == 3);
    BOOST_TEST(df.n_rows() == 3);
    BOOST_TEST(df.header().empty());
    test_ranges_equal(df.raw_cols()[0], simple_csv_cols[0] | ranges::view::slice(1, ranges::end));
    test_ranges_equal(df.raw_cols()[1], simple_csv_cols[1] | ranges::view::slice(1, ranges::end));
    test_ranges_equal(df.raw_cols()[2], simple_csv_cols[2] | ranges::view::slice(1, ranges::end));
}

BOOST_AUTO_TEST_CASE(test_read_quoted_csv_from_istream)
{
    std::istringstream quoted_csv_ss{quoted_csv};
    const dataframe<> df = read_csv(quoted_csv_ss, 0, true, '|', '*', '+');
    BOOST_TEST(df.n_cols() == 3);
    BOOST_TEST(df.n_rows() == 2);
    test_ranges_equal(df.header(), quoted_csv_rows[0]);
    test_ranges_equal(df.raw_rows()[0], quoted_csv_rows[1]);
    test_ranges_equal(df.raw_rows()[1], quoted_csv_rows[2]);
}

BOOST_AUTO_TEST_CASE(test_write_quoted_to_ostream)
{
    std::istringstream quoted_csv_ss{quoted_csv};
    const dataframe<> df = read_csv(quoted_csv_ss, 0, true, '|', '*', '+');
  
    std::ostringstream oss;
    write_csv(oss, df, '|', '*', '+');
    BOOST_TEST(oss.str() == 
      "*Column| 1*|*Column| 2*|* Column +*3+* *\n"
      "Field 1|*Field|\n 2*|* Field 3 *\n"
      "*Field\n1*|*Field| 2 *|* Field 3 *\n"
    );
}

BOOST_AUTO_TEST_CASE(test_compare_after_write_and_read)
{
    std::istringstream quoted_csv_ss{quoted_csv};
    const dataframe<> df1 = read_csv(quoted_csv_ss);
  
    std::stringstream oss;
    write_csv(oss, df1, '|', '*', '+');
    const dataframe<> df2 = read_csv(oss, 0, true, '|', '*', '+');
    test_ranges_equal(df1.header(), df2.header());
    std::vector<std::vector<std::string>> df1_cols = df1.raw_cols();
    std::vector<std::vector<std::string>> df2_cols = df2.raw_cols();
    BOOST_CHECK(df1_cols == df2_cols);
}

BOOST_AUTO_TEST_CASE(test_write_to_no_file)
{
    namespace fs = std::experimental::filesystem;
    std::istringstream simple_csv_ss{simple_csv};
    const dataframe<> df = read_csv(simple_csv_ss);
    fs::path dir{"dummy_directory"};
    fs::create_directory(dir);
    BOOST_CHECK_THROW(write_csv(dir, df), std::ios_base::failure);
}

BOOST_AUTO_TEST_CASE(test_file_write_and_read)
{
    std::istringstream quoted_csv_ss{quoted_csv};
    const dataframe<> df1 = read_csv(quoted_csv_ss, 0, true, '|', '*', '+');

    fs::path csv_file{"test.core.csv.test_file_write_and_read.csv"};
    write_csv(csv_file, df1, '|', '*', '+');
    const dataframe<> df2 = read_csv(csv_file, 0, true, '|', '*', '+');
    fs::remove(csv_file);
    test_ranges_equal(df1.header(), df2.header());
    std::vector<std::vector<std::string>> df1_cols = df1.raw_cols();
    std::vector<std::vector<std::string>> df2_cols = df2.raw_cols();
    BOOST_CHECK(df1_cols == df2_cols);
}

BOOST_AUTO_TEST_CASE(test_exceptions)
{
    for (auto& invalid_csv : invalid_csvs) {
        std::istringstream invalid_csv_ss{invalid_csv};
        BOOST_CHECK_THROW(read_csv(invalid_csv_ss), std::ios_base::failure);
    }
}
