/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE filter2_test

#include "filter.hpp"

using namespace cxtream::stream;

BOOST_AUTO_TEST_CASE(test_dim2_partial)
{
    CXTREAM_DEFINE_COLUMN(IntVec, std::vector<int>)
    const std::vector<std::tuple<int, std::vector<int>>> data = {
      {{3, {1, 5}}, {1, {2, 4}}, {2, {7, 1}}, {6, {3, 5}}}};

    std::size_t i = 0;
    auto generated = data
      | create<Int, IntVec>(2)
      | filter(from<IntVec>, by<IntVec>, [](int v) { return v >= 4; }, dim<2>)
      | for_each(from<Int, IntVec>, [&i](auto& ints, auto& intvecs) {
            switch (i++) {
            case 0: BOOST_TEST(ints    == (std::vector<int>{3, 1}));
                    BOOST_TEST(intvecs == (std::vector<std::vector<int>>{{5}, {4}}));
                    break;
            case 1: BOOST_TEST(ints    == (std::vector<int>{2, 6}));
                    BOOST_TEST(intvecs == (std::vector<std::vector<int>>{{7}, {5}}));
                    break;
            }
        }, dim<0>)
      | ranges::to_vector;
    BOOST_TEST(i == 2);
}

BOOST_AUTO_TEST_CASE(test_dim2_move_only)
{
    std::vector<std::tuple<UniqueVec>> data;
    std::vector<std::unique_ptr<int>> v1;
    std::vector<std::unique_ptr<int>> v2;
    std::vector<std::unique_ptr<int>> v3;
    v1.emplace_back(std::make_unique<int>(5));
    v1.emplace_back(std::make_unique<int>(3));
    v2.emplace_back(std::make_unique<int>(2));
    v2.emplace_back(std::make_unique<int>(4));
    v3.emplace_back(std::make_unique<int>(1));
    v3.emplace_back(std::make_unique<int>(6));
    data.emplace_back(std::move(v1));
    data.emplace_back(std::move(v2));
    data.emplace_back(std::move(v3));

    std::size_t i = 0;
    data
      | ranges::view::move
      | filter(from<UniqueVec>, by<UniqueVec>, [](auto& ptr) { return *ptr >= 4.; }, dim<2>)
      | for_each(from<UniqueVec>, [&i](auto& unique_vec) {
            switch (i++) {
            BOOST_TEST(unique_vec.size() == 1);
            BOOST_TEST(unique_vec.at(0).size() == 1);
            case 0: BOOST_TEST(*(unique_vec.at(0).at(0)) == 5);
                    break;
            case 1: BOOST_TEST(*(unique_vec.at(0).at(0)) == 4);
                    break;
            case 2: BOOST_TEST(*(unique_vec.at(0).at(0)) == 6);
                    break;
            }
        }, dim<0>)
      | ranges::to_vector;
    BOOST_TEST(i == 3);
}
