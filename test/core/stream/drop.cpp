/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE column_drop_test

#include "../common.hpp"

#include <cxtream/core/stream/drop.hpp>

#include <boost/test/unit_test.hpp>
#include <range/v3/view/move.hpp>

#include <tuple>
#include <vector>

using namespace cxtream::stream;

CXTREAM_DEFINE_COLUMN(Unique2, std::unique_ptr<int>)

BOOST_AUTO_TEST_CASE(test_int_column)
{
    // drop column
    std::vector<std::tuple<Int, Double>> data = {{3, 5.}, {1, 2.}};

    auto generated = data | drop<Int>;

    std::vector<std::tuple<Double>> desired = {5., 2.};
    test_ranges_equal(generated, desired);
}

BOOST_AUTO_TEST_CASE(test_move_only_column)
{
    // drop a move-only column
    std::vector<std::tuple<Unique, Unique2>> data;
    data.emplace_back(std::make_unique<int>(1), std::make_unique<int>(5));
    data.emplace_back(std::make_unique<int>(2), std::make_unique<int>(6));

    auto generated = data
      | ranges::view::move
      | drop<Unique>
      | ranges::view::transform([](auto t) { return *(std::get<0>(std::move(t)).value()[0]); });

    test_ranges_equal(generated, std::vector<int>{5, 6});
}

BOOST_AUTO_TEST_CASE(test_multiple_columns)
{
    // drop multiple columns
    std::vector<std::tuple<Int, Unique, Unique2>> data;
    data.emplace_back(-1, std::make_unique<int>(1), std::make_unique<int>(5));
    data.emplace_back(-2, std::make_unique<int>(2), std::make_unique<int>(6));
  
    auto generated = data
      | ranges::view::move
      | drop<Int, Unique>
      | ranges::view::transform([](auto t) { return *(std::get<0>(std::move(t)).value()[0]); })
      | ranges::to_vector;
  
    test_ranges_equal(generated, std::vector<int>{5, 6});
}
