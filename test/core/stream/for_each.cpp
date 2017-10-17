/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE for_each_test

#include "../common.hpp"

#include <cxtream/core/stream/create.hpp>
#include <cxtream/core/stream/for_each.hpp>

#include <boost/test/unit_test.hpp>
#include <range/v3/to_container.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/move.hpp>

#include <memory>
#include <tuple>
#include <vector>

using namespace cxtream::stream;

BOOST_AUTO_TEST_CASE(test_for_each_of_two)
{
    // for_each of two columns
    std::vector<std::tuple<Int, Double>> data = {{{1, 3}, {5., 6.}}, {1, 2.}};
    int sum = 0;
    auto generated = data
      | for_each(from<Int, Double>,
          [&sum](const int& v, double c) { sum += v; })
      | ranges::to_vector;
    std::vector<std::tuple<Int, Double>> desired = {{{1, 3}, {5., 6.}}, {1, 2.}};
    BOOST_TEST(sum == 5);
    test_ranges_equal(generated, desired);
}

BOOST_AUTO_TEST_CASE(test_for_each_mutable)
{
    std::vector<std::tuple<Int>> data = {{{1, 3}}, {{5, 7}}};
    struct {
        int i = 0;
        std::shared_ptr<int> i_ptr = std::make_shared<int>(0);
        void operator()(const int&) { *i_ptr = ++i; }
    } func;

    auto generated = data
      | for_each(from<Int>, func)
      | ranges::to_vector;
    BOOST_TEST(*(func.i_ptr) == 4);
}

BOOST_AUTO_TEST_CASE(test_for_each_move_only)
{
    // for_each of a move-only column
    std::vector<std::tuple<Int, Unique>> data;
    data.emplace_back(3, std::make_unique<int>(5));
    data.emplace_back(1, std::make_unique<int>(2));
  
    std::vector<int> generated;
    data
      | ranges::view::move
      | for_each(from<Int, Unique>,
          [&generated](const int& v, const std::unique_ptr<int>& p) {
              generated.push_back(v + *p);
        })
      | ranges::to_vector;
  
    test_ranges_equal(generated, std::vector<int>{8, 3});
}

BOOST_AUTO_TEST_CASE(test_for_each_dim0)
{
    std::vector<std::tuple<Int, Unique>> data;
    data.emplace_back(3, std::make_unique<int>(5));
    data.emplace_back(1, std::make_unique<int>(2));
  
    std::vector<int> generated;
    data
      | ranges::view::move
      | for_each(from<Int, Unique>,
          [&generated](const std::vector<int>& int_batch,
                       const std::vector<std::unique_ptr<int>>& ptr_batch) {
              generated.push_back(*(ptr_batch[0]) + int_batch[0]);
        }, dim<0>)
      | ranges::to_vector;
  
    test_ranges_equal(generated, std::vector<int>{8, 3});
}

BOOST_AUTO_TEST_CASE(test_for_each_dim2_move_only)
{
    auto data = generate_move_only_data();

    std::vector<int> generated;
    auto rng = data
      | ranges::view::move
      | create<Int, UniqueVec>(2)
      | cxtream::stream::for_each(from<UniqueVec>,
          [&generated](std::unique_ptr<int>& ptr) {
              generated.push_back(*ptr + 1);
          }, dim<2>)
      | ranges::to_vector;

    std::vector<int> desired = {2, 5, 9, 3, 3, 6};
    BOOST_CHECK(generated == desired);
}

BOOST_AUTO_TEST_CASE(test_for_each_dim2_move_only_mutable)
{
    auto data = generate_move_only_data();

    std::vector<int> generated;
    auto rng = data
      | ranges::view::move
      | create<Int, UniqueVec>(2)
      | cxtream::stream::for_each(from<UniqueVec>,
          [&generated, i = 4](std::unique_ptr<int>&) mutable {
              generated.push_back(i++);
          }, dim<2>)
      | ranges::to_vector;

    std::vector<int> desired = {4, 5, 6, 7, 8, 9};
    BOOST_CHECK(generated == desired);
}
