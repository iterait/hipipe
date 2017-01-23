/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE column_create_test

#include "../common.hpp"

#include <cxtream/core/stream/create.hpp>

#include <boost/test/unit_test.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/move.hpp>

#include <tuple>
#include <vector>

using namespace cxtream::stream;

CXTREAM_DEFINE_COLUMN(Unique2, std::unique_ptr<int>)

BOOST_AUTO_TEST_CASE(test_int_column)
{
    // create a new column
    auto generated = ranges::view::iota(0, 10) | create<Int>();
    std::vector<std::tuple<Int>> desired = ranges::view::iota(0, 10);
    test_ranges_equal(generated, desired);
}

BOOST_AUTO_TEST_CASE(test_one_batch_column)
{
    // create a new column with a single batch
    auto generated = ranges::view::iota(0, 10) | create<Int>(50);
    BOOST_TEST(ranges::distance(generated) == 1);
    std::vector<int> generated_batch0 = std::get<Int>(*generated.begin()).value();
    std::vector<int> desired_batch0 = ranges::view::iota(0, 10);
    test_ranges_equal(generated_batch0, desired_batch0);
}

BOOST_AUTO_TEST_CASE(test_two_batch_column)
{
    // create a new column with two batches
    auto generated = ranges::view::iota(0, 10) | create<Int>(5);
    BOOST_TEST(ranges::distance(generated) == 2);
    auto it = generated.begin();
    std::vector<int> generated_batch0 = std::get<Int>(*it).value();
    std::vector<int> desired_batch0 = ranges::view::iota(0, 5);
    test_ranges_equal(generated_batch0, desired_batch0);
    std::vector<int> generated_batch1 = std::get<Int>(*++it).value();
    std::vector<int> desired_batch1 = ranges::view::iota(5, 10);
    test_ranges_equal(generated_batch1, desired_batch1);
}

BOOST_AUTO_TEST_CASE(test_move_only_column)
{
    // create a move-only column
    std::vector<std::unique_ptr<int>> data;
    data.emplace_back(std::make_unique<int>(5));
    data.emplace_back(std::make_unique<int>(6));

    auto generated = data
      | ranges::view::move 
      | create<Unique>(1)
      | ranges::view::transform([](auto t) {
            return *(std::get<0>(std::move(t)).value()[0]);
        });

    test_ranges_equal(generated, std::vector<int>{5, 6});
}
  
BOOST_AUTO_TEST_CASE(test_multiple_columns)
{
    // create multiple columns
    std::vector<std::tuple<std::unique_ptr<int>, std::unique_ptr<int>>> data;
    data.emplace_back(std::make_unique<int>(1), std::make_unique<int>(5));
    data.emplace_back(std::make_unique<int>(2), std::make_unique<int>(6));

    auto generated = data
      | ranges::view::move
      | create<Unique, Unique2>(1)
      | ranges::view::transform([](auto t) {
          return std::make_tuple(*(std::get<0>(std::move(t)).value()[0]),
                                 *(std::get<1>(std::move(t)).value()[0]));
        });

    test_ranges_equal(generated, std::vector<std::tuple<int, int>>{{1, 5}, {2, 6}});
}
