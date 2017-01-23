/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

// The tests for stream::filter are split to multiple
// files to speed up compilation in case of multiple CPUs.
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE filter1_test

#include "filter.hpp"

using namespace cxtream::stream;

BOOST_AUTO_TEST_CASE(test_dim0)
{
    const std::vector<std::tuple<int, double>> data = {
      {{3, 5.}, {1, 2.}, {7, 3.}, {8, 1.}, {2, 4.}, {6, 5.}}};

    std::size_t i = 0;
    data
      | create<Int, Double>(2)
      // for dim0, `from` is ignored anyway
      | filter(from<Int, Double>, by<Double>,
          [](const std::vector<double>& v) { return v.at(0) > 3.; }, dim<0>)
      | for_each(from<Int, Double>, [&i](auto& ints, auto& doubles) {
            switch (i++) {
            case 0: BOOST_TEST(ints    == (std::vector<int>{3, 1}));
                    BOOST_TEST(doubles == (std::vector<double>{5., 2.}));
                    break;
            case 1: BOOST_TEST(ints    == (std::vector<int>{2, 6}));
                    BOOST_TEST(doubles == (std::vector<double>{4., 5.}));
                    break;
            }
        }, dim<0>)
      | ranges::to_vector;
    BOOST_TEST(i == 2);
}

BOOST_AUTO_TEST_CASE(test_dim0_move_only)
{
    std::vector<std::tuple<Int, Unique>> data;
    data.emplace_back(3, std::make_unique<int>(5));
    data.emplace_back(1, std::make_unique<int>(2));

    std::size_t i = 0;
    data
      | ranges::view::move
      // for dim0, `from` is ignored anyway
      | filter(from<>, by<Unique>,
          [](const std::vector<std::unique_ptr<int>>& v) { return *(v.at(0)) >= 3; }, dim<0>)
      | for_each(from<Int, Unique>, [&i](auto& ints, auto& uniques) {
            switch (i++) {
            case 0: BOOST_TEST(ints == (std::vector<int>{3}));
                    BOOST_TEST(uniques.size() == 1);
                    BOOST_TEST(*(uniques.at(0)) == 5);
                    break;
            }
        }, dim<0>)
      | ranges::to_vector;
    BOOST_TEST(i == 1);
}

BOOST_AUTO_TEST_CASE(test_dim1)
{
    const std::vector<std::tuple<int, double>> data = {
      {{3, 5.}, {1, 2.}, {2, 4.}, {6, 5.}}};

    std::size_t i = 0;
    data
      | create<Int, Double>(2)
      | filter(from<Int, Double>, by<Double>, [](double v) { return v >= 5.; }, dim<1>)
      | for_each(from<Int, Double>, [&i](auto& ints, auto& doubles) {
            switch (i++) {
            case 0: BOOST_TEST(ints    == (std::vector<int>{3}));
                    BOOST_TEST(doubles == (std::vector<double>{5.}));
                    break;
            case 1: BOOST_TEST(ints    == (std::vector<int>{6}));
                    BOOST_TEST(doubles == (std::vector<double>{5.}));
                    break;
            }
        }, dim<0>)
      | ranges::to_vector;
    BOOST_TEST(i == 2);
}

BOOST_AUTO_TEST_CASE(test_dim1_partial)
{
    const std::vector<std::tuple<int, double>> data = {
      {{3, 5.}, {1, 2.}, {2, 4.}, {6, 5.}}};

    std::size_t i = 0;
    data
      | create<Int, Double>(2)
      | filter(from<Double>, by<Double>, [](double v) { return v >= 5.; }, dim<1>)
      | for_each(from<Int, Double>, [&i](auto& a, auto& b) {
            switch (i++) {
            case 0: BOOST_CHECK(a == (std::vector<int>{3, 1}));
                    BOOST_CHECK(b == (std::vector<double>{5.}));
                    break;
            case 1: BOOST_CHECK(a == (std::vector<int>{2, 6}));
                    BOOST_CHECK(b == (std::vector<double>{5.}));
                    break;
            }
        }, dim<0>)
      | ranges::to_vector;
    BOOST_TEST(i == 2);
}

BOOST_AUTO_TEST_CASE(test_dim1_move_only)
{
    std::vector<std::tuple<Int, Unique>> data;
    data.emplace_back(3, std::make_unique<int>(5));
    data.emplace_back(1, std::make_unique<int>(2));
    data.emplace_back(2, std::make_unique<int>(4));
    data.emplace_back(6, std::make_unique<int>(5));

    std::size_t i = 0;
    data
      | ranges::view::move
      | filter(from<Unique>, by<Unique>, [](auto& ptr) { return *ptr >= 5.; }, dim<1>)
      | for_each(from<Int, Unique>, [&i](auto& a, auto& b) {
            switch (i++) {
            case 0: BOOST_CHECK(a == (std::vector<int>{3}));
                    BOOST_TEST(b.size() == 1);
                    BOOST_CHECK(*(b.at(0)) == 5.);
                    break;
            case 1: BOOST_CHECK(a == (std::vector<int>{1}));
                    BOOST_TEST(b.size() == 0);
                    break;
            case 2: BOOST_CHECK(a == (std::vector<int>{2}));
                    BOOST_TEST(b.size() == 0);
                    break;
            case 3: BOOST_CHECK(a == (std::vector<int>{6}));
                    BOOST_TEST(b.size() == 1);
                    BOOST_CHECK(*(b.at(0)) == 5.);
                    break;
            }
        }, dim<0>)
      | ranges::to_vector;
    BOOST_TEST(i == 4);
}

BOOST_AUTO_TEST_CASE(test_dim2)
{
    CXTREAM_DEFINE_COLUMN(IntVec1, std::vector<int>)
    CXTREAM_DEFINE_COLUMN(IntVec2, std::vector<int>)
    const std::vector<std::tuple<std::vector<int>, std::vector<int>>> data = {
      {{{3, 2}, {1, 5}}, {{1, 5}, {2, 4}}, {{2, 4}, {7, 1}}, {{6, 4}, {3, 5}}}};

    std::size_t i = 0;
    auto generated = data
      | create<IntVec1, IntVec2>(2)
      | filter(from<IntVec1, IntVec2>, by<IntVec2>, [](int v) { return v >= 4; }, dim<2>)
      | for_each(from<IntVec1, IntVec2>, [&i](auto& iv1, auto& iv2) {
            switch (i++) {
            case 0: BOOST_TEST(iv1 == (std::vector<std::vector<int>>{{2}, {5}}));
                    BOOST_TEST(iv2 == (std::vector<std::vector<int>>{{5}, {4}}));
                    break;
            case 1: BOOST_TEST(iv1 == (std::vector<std::vector<int>>{{2}, {4}}));
                    BOOST_TEST(iv2 == (std::vector<std::vector<int>>{{7}, {5}}));
                    break;
            }
        }, dim<0>)
      | ranges::to_vector;
    BOOST_TEST(i == 2);
}
