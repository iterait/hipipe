/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE dataframe_test

#include "common.hpp"

#include <cxtream/core/dataframe.hpp>

#include <boost/test/unit_test.hpp>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace cxtream;

const dataframe<> simple_df{
    // columns
    std::make_tuple(
      std::vector<int>{1, 2, 3},
      std::vector<std::string>{"a1", "a2", "a3"},
      std::vector<std::string>{"1.1", "1.2", "1.3"}
    ),
    // header
    std::vector<std::string>{"Id", "A", "B"}
};

BOOST_AUTO_TEST_CASE(test_column_insertion)
{
    dataframe<> df{simple_df};
    df.col_insert(std::vector<int>{5, 6, 7}, "C");
    BOOST_TEST(df.n_cols() == 4UL);
    BOOST_TEST(df.n_rows() == 3UL);
    test_ranges_equal(df.header(), std::vector<std::string>{"Id", "A", "B", "C"});
    test_ranges_equal(df.raw_cols()[3], std::vector<std::string>{"5", "6", "7"});
    BOOST_TEST(df.raw_rows()[0][3] == "5");
}

BOOST_AUTO_TEST_CASE(test_column_drop)
{
    dataframe<> df{simple_df};
    df.col_drop("B");
    BOOST_TEST(df.n_cols() == 2UL);
    BOOST_TEST(df.n_rows() == 3UL);
    test_ranges_equal(df.header(), std::vector<std::string>{"Id", "A"});
    test_ranges_equal(df.raw_rows()[2], std::vector<std::string>{"3", "a3"});
    df.icol_drop(1);
    BOOST_TEST(df.n_cols() == 1UL);
    BOOST_TEST(df.n_rows() == 3UL);
    test_ranges_equal(df.header(), std::vector<std::string>{"Id"});
    test_ranges_equal(df.raw_rows()[2], std::vector<std::string>{"3"});
}

BOOST_AUTO_TEST_CASE(test_row_drop)
{
    dataframe<> df{simple_df};
    df.row_drop(1);
    BOOST_TEST(df.n_cols() == 3UL);
    BOOST_TEST(df.n_rows() == 2UL);
    test_ranges_equal(df.header(), std::vector<std::string>{"Id", "A", "B"});
    test_ranges_equal(df.raw_icol(1), std::vector<std::string>{"a1", "a3"});
}

BOOST_AUTO_TEST_CASE(test_row_insertion)
{
    dataframe<> df{simple_df};
    df.row_insert(std::make_tuple(4, "a3", true));
    BOOST_TEST(df.n_cols() == 3UL);
    BOOST_TEST(df.n_rows() == 4UL);
    test_ranges_equal(df.raw_rows()[3], std::vector<std::string>{"4", "a3", "true"});
}

BOOST_AUTO_TEST_CASE(test_raw_col_read)
{
    const dataframe<> df{simple_df};
    for (int i = 0; i < 3; ++i) {
        BOOST_TEST(df.raw_icol(i).size() == 3UL);
        // compare index based and name based access
        test_ranges_equal(df.raw_icol(i), df.raw_col(df.header()[i]));
    }
    auto id_col = df.raw_col("Id");
    test_ranges_equal(id_col, std::vector<std::string>{"1", "2", "3"});
}

BOOST_AUTO_TEST_CASE(test_raw_col_write)
{
    dataframe<> df{simple_df};
    auto id_col = df.raw_col("Id");
    test_ranges_equal(id_col, std::vector<std::string>{"1", "2", "3"});
    id_col = df.raw_icol(0);
    id_col[1] = "5";
    test_ranges_equal(df.raw_col("Id"), std::vector<std::string>{"1", "5", "3"});
}

BOOST_AUTO_TEST_CASE(test_col)
{
    const dataframe<> df{simple_df};
    auto b_col = df.col<double>("B");
    test_ranges_equal(b_col, std::vector<double>{1.1, 1.2, 1.3});
    test_ranges_equal(b_col, df.icol<double>(2));
}

BOOST_AUTO_TEST_CASE(test_raw_cols_read)
{
    const dataframe<> df{simple_df};
    test_ranges_equal(df.raw_icols({0, 2})[0], std::vector<std::string>{"1", "2", "3"});
    test_ranges_equal(df.raw_cols()[1], std::vector<std::string>{"a1", "a2", "a3"});
    test_ranges_equal(df.raw_cols({"Id", "B"})[1], std::vector<std::string>{"1.1", "1.2", "1.3"});
}

BOOST_AUTO_TEST_CASE(test_raw_cols_write)
{
    dataframe<> df{simple_df};
    auto cols = df.raw_cols();
    for (int i = 0; i < 3; ++i) {
        cols[1][i][0] = 'c';
    }
    test_ranges_equal(df.raw_icols({0, 2})[0], std::vector<std::string>{"1", "2", "3"});
    test_ranges_equal(df.raw_cols()[1], std::vector<std::string>{"c1", "c2", "c3"});
    test_ranges_equal(df.raw_cols({"Id", "B"})[1], std::vector<std::string>{"1.1", "1.2", "1.3"});
}

BOOST_AUTO_TEST_CASE(test_cols)
{
    const dataframe<> df{simple_df};
    auto cols = df.cols<std::string, double>({"A", "B"});
    test_ranges_equal(std::get<1>(cols), std::vector<double>{1.1, 1.2, 1.3});
    test_ranges_equal(std::get<0>(cols), std::vector<std::string>{"a1", "a2", "a3"});
    test_ranges_equal(std::get<0>(cols), std::get<0>(df.icols<std::string, double>({1, 2})));
    test_ranges_equal(std::get<1>(cols), std::get<1>(df.icols<std::string, double>({1, 2})));
}

BOOST_AUTO_TEST_CASE(test_raw_rows_read)
{
    const dataframe<> df{simple_df};
    test_ranges_equal(df.raw_rows()[0], std::vector<std::string>{"1", "a1", "1.1"});
    test_ranges_equal(df.raw_rows({"A", "B"})[1], std::vector<std::string>{"a2", "1.2"});
    test_ranges_equal(df.raw_irows({2, 0})[2], std::vector<std::string>{"1.3", "3"});
}

BOOST_AUTO_TEST_CASE(test_raw_rows_write)
{
    dataframe<> df{simple_df};
    auto rows = df.raw_rows();
    for (int i = 0; i < 3; ++i) {
        rows[1][i][0] = 'c';
    }
    test_ranges_equal(df.raw_rows()[0], std::vector<std::string>{"1", "a1", "1.1"});
    test_ranges_equal(df.raw_rows({"B", "Id"})[1], std::vector<std::string>{"c.2", "c"});
    test_ranges_equal(df.raw_irows({2, 0})[2], std::vector<std::string>{"1.3", "3"});
}

BOOST_AUTO_TEST_CASE(test_rows)
{
    const dataframe<> df{simple_df};
    auto rows = df.rows<std::string, double>({"A", "B"});
    BOOST_TEST(std::get<0>(rows[0]) == "a1");
    BOOST_TEST(std::get<1>(rows[0]) == 1.1);
    BOOST_TEST(std::get<0>(rows[1]) == std::get<0>(df.irows<std::string, double>({1, 2})[1]));
    BOOST_TEST(std::get<1>(rows[2]) == std::get<1>(df.irows<std::string, double>({1, 2})[2]));
}

BOOST_AUTO_TEST_CASE(test_print)
{
    const dataframe<> df{simple_df};
    std::ostringstream ss;
    ss << df;
    BOOST_TEST(ss.str() == "  Id|   A|    B\n"
                           "----+----+-----\n"
                           "   1|  a1|  1.1\n"
                           "   2|  a2|  1.2\n"
                           "   3|  a3|  1.3\n");
}

BOOST_AUTO_TEST_CASE(test_no_header)
{
    dataframe<> df{
      std::make_tuple(
        std::vector<int>{1, 2, 3},
        std::vector<std::string>{"a1", "a2", "a3"},
        std::vector<std::string>{"1.1", "1.2", "1.3"}
    )};
    BOOST_TEST(df.header().size() == 0);
    test_ranges_equal(df.raw_rows()[0], std::vector<std::string>{"1", "a1", "1.1"});
}

BOOST_AUTO_TEST_CASE(test_index_col)
{
    const dataframe<> df{simple_df};
    std::unordered_map<int, double> indexed_irow = df.index_icol<int, double>(0, 2);
    std::unordered_map<int, double> indexed_row = df.index_col<int, double>("Id", "B");
    std::unordered_map<int, double> desired{
      {1, 1.1},
      {2, 1.2},
      {3, 1.3},
    };
    BOOST_CHECK(indexed_irow == desired);
    BOOST_CHECK(indexed_row == desired);
}

BOOST_AUTO_TEST_CASE(test_index_cols)
{
    const dataframe<> df{simple_df};
    std::map<int, std::tuple<std::string, double>> indexed_irows =
      df.index_icols<int, std::string, double>(0, {1, 2});
    std::map<int, std::tuple<std::string, double>> indexed_rows =
      df.index_cols<int, std::string, double>("Id", {"A", "B"});
    std::map<int, std::tuple<std::string, double>> desired{
      {1, {"a1", 1.1}},
      {2, {"a2", 1.2}},
      {3, {"a3", 1.3}},
    };
    BOOST_CHECK(indexed_irows == desired);
    BOOST_CHECK(indexed_rows == desired);
}
