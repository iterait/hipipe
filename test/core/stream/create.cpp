/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE column_create_test

#include "common.hpp"

#include <hipipe/core/stream/create.hpp>

#include <range/v3/view/iota.hpp>

#include <tuple>


BOOST_AUTO_TEST_CASE(test_int_column)
{
    std::vector<hipipe::stream::batch_t> stream = ranges::view::iota(0, 10)
      | hipipe::stream::create<Int>();
    BOOST_TEST(stream.size() == 10);
    for (int i = 0; i < (int)stream.size(); ++i) {
        BOOST_TEST(stream.at(i).contains<Int>());
        std::vector<int> generated = stream.at(i).extract<Int>();
        BOOST_TEST(generated == (std::vector<int>{i}));
    }
}


BOOST_AUTO_TEST_CASE(test_one_batch_column)
{
    // create a new column with a single batch
    std::vector<hipipe::stream::batch_t> stream = ranges::view::iota(0, 10)
      | hipipe::stream::create<Int>(50);
    BOOST_TEST(stream.size() == 1);
    std::vector<int> desired_batch0 = ranges::view::iota(0, 10);
    std::vector<int> generated_batch0 = stream.front().extract<Int>();
    BOOST_TEST(generated_batch0 == desired_batch0);
}


BOOST_AUTO_TEST_CASE(test_two_batch_column)
{
    // create a new column with two batches
    std::vector<hipipe::stream::batch_t> stream = ranges::view::iota(0, 10)
      | hipipe::stream::create<Int>(5);
    BOOST_TEST(stream.size() == 2);
    std::vector<int> generated_batch0 = stream.at(0).extract<Int>();
    std::vector<int> desired_batch0 = ranges::view::iota(0, 5);
    BOOST_TEST(generated_batch0 == desired_batch0, boost::test_tools::per_element());
    std::vector<int> generated_batch1 = stream.at(1).extract<Int>();
    std::vector<int> desired_batch1 = ranges::view::iota(5, 10);
    BOOST_TEST(generated_batch1 == desired_batch1);
}


BOOST_AUTO_TEST_CASE(test_move_only_column)
{
    // create a move-only column
    std::vector<std::unique_ptr<int>> data;
    data.emplace_back(std::make_unique<int>(5));
    data.emplace_back(std::make_unique<int>(6));

    std::vector<int> generated = data
      | ranges::view::move 
      | hipipe::stream::create<Unique>(1)
      | ranges::view::transform([](const hipipe::stream::batch_t& batch) -> int {
            return *batch.extract<Unique>().at(0);
        });

    BOOST_TEST(generated == (std::vector<int>{5, 6}));
}

  
BOOST_AUTO_TEST_CASE(test_multiple_columns)
{
    // create multiple columns
    std::vector<std::tuple<std::unique_ptr<int>, int>> data;
    data.emplace_back(std::make_unique<int>(1), 5);
    data.emplace_back(std::make_unique<int>(2), 6);

    std::vector<std::tuple<int, int>> generated = data
      | ranges::view::move
      | hipipe::stream::create<Unique, Int>(1)
      | ranges::view::transform([](const hipipe::stream::batch_t& batch) -> std::tuple<int, int> {
          return std::make_tuple(*batch.extract<Unique>().at(0),
                                  batch.extract<Int>().at(0));
        });

    BOOST_TEST(generated == (std::vector<std::tuple<int, int>>{{1, 5}, {2, 6}}));
}
